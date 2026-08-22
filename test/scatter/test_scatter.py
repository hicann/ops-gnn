# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import inspect
import itertools
import os
import warnings

import pytest
import torch

import ops_gnn as torch_scatter
from ops_gnn import (
    scatter,
    scatter_add,
    scatter_max,
    scatter_mean,
    scatter_min,
    scatter_mul,
    scatter_sum,
)


DEVICE_ID = int(os.environ.get("NPU_DEVICE_ID", "0"))
REDUCTIONS = ("sum", "add", "mul", "mean", "min", "max")
DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


@pytest.fixture(scope="module", autouse=True)
def _select_npu_device():
    """Select the requested NPU without making CPU-only tests require one."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(DEVICE_ID)


def _assert_equal(actual, expected):
    if actual.dtype.is_floating_point:
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    else:
        assert torch.equal(actual, expected)


def _reduce_selected(selected, reduce):
    values = torch.stack(selected)
    if reduce in ("sum", "add"):
        return values.sum()
    if reduce == "mul":
        return values.prod()
    if reduce == "mean":
        return values.mean()
    if reduce == "min":
        return values.min()
    return values.max()


def test_public_api_and_signatures():
    expected = {
        "src": inspect.Parameter.empty,
        "index": inspect.Parameter.empty,
        "dim": -1,
        "out": None,
        "dim_size": None,
    }
    for fn in (
        scatter_sum,
        scatter_add,
        scatter_mul,
        scatter_mean,
        scatter_min,
        scatter_max,
    ):
        assert {name: p.default for name, p in inspect.signature(fn).parameters.items()} == expected

    scatter_defaults = dict(expected)
    scatter_defaults["reduce"] = "sum"
    assert {
        name: p.default for name, p in inspect.signature(scatter).parameters.items()
    } == scatter_defaults
    assert set(torch_scatter.__all__) >= {
        "scatter", "scatter_sum", "scatter_add", "scatter_mul",
        "scatter_mean", "scatter_min", "scatter_max",
    }


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc01_one_dimensional_all_reductions(reduce):
    src = torch.tensor([1, 3, 2, 4, 5, 6], dtype=torch.float32)
    index = torch.tensor([0, 1, 0, 1, 1, 3], dtype=torch.int64)
    values = {
        "sum": [3, 12, 0, 6],
        "add": [3, 12, 0, 6],
        "mul": [2, 60, 1, 6],
        "mean": [1.5, 4, 0, 6],
        "min": [1, 3, 0, 6],
        "max": [2, 5, 0, 6],
    }
    actual = scatter(src, index, reduce=reduce)
    _assert_equal(actual, torch.tensor(values.get(reduce), dtype=src.dtype))

    if reduce == "min":
        out, arg = scatter_min(src, index)
        _assert_equal(out, actual)
        assert torch.equal(arg, torch.tensor([0, 1, 6, 5]))
    if reduce == "max":
        out, arg = scatter_max(src, index)
        _assert_equal(out, actual)
        assert torch.equal(arg, torch.tensor([2, 4, 6, 5]))


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc02_two_dimensional_index_broadcast_dim_zero(reduce):
    src = torch.tensor(
        [[1, 2], [5, 6], [3, 4], [7, 8], [9, 10], [11, 12]],
        dtype=torch.float32,
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3])
    expected = {
        "sum": [[4, 6], [21, 24], [0, 0], [11, 12]],
        "add": [[4, 6], [21, 24], [0, 0], [11, 12]],
        "mul": [[3, 8], [315, 480], [1, 1], [11, 12]],
        "mean": [[2, 3], [7, 8], [0, 0], [11, 12]],
        "min": [[1, 2], [5, 6], [0, 0], [11, 12]],
        "max": [[3, 4], [9, 10], [0, 0], [11, 12]],
    }
    _assert_equal(
        scatter(src, index, dim=0, reduce=reduce),
        torch.tensor(expected.get(reduce), dtype=src.dtype),
    )


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc03_two_dimensional_dim_one(reduce):
    src = torch.tensor(
        [[1, 5, 3, 7, 9, 11], [2, 4, 8, 6, 10, 12]], dtype=torch.int32
    )
    index = torch.tensor([[0, 1, 0, 1, 1, 3], [0, 0, 1, 0, 1, 2]])
    expected = {
        "sum": [[4, 21, 0, 11], [12, 18, 12, 0]],
        "add": [[4, 21, 0, 11], [12, 18, 12, 0]],
        "mul": [[3, 315, 1, 11], [48, 80, 12, 1]],
        "mean": [[2, 7, 0, 11], [4, 9, 12, 0]],
        "min": [[1, 5, 0, 11], [2, 8, 12, 0]],
        "max": [[3, 9, 0, 11], [6, 10, 12, 0]],
    }
    _assert_equal(
        scatter(src, index, dim=1, reduce=reduce),
        torch.tensor(expected.get(reduce), dtype=src.dtype),
    )


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc04_three_dimensional_src_two_dimensional_index(reduce):
    src = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ],
        dtype=torch.float32,
    )
    index = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int64)
    expected = {
        "sum": [[[6, 8], [3, 4], [0, 0]], [[9, 10], [18, 20], [0, 0]]],
        "add": [[[6, 8], [3, 4], [0, 0]], [[9, 10], [18, 20], [0, 0]]],
        "mul": [[[5, 12], [3, 4], [1, 1]], [[9, 10], [77, 96], [1, 1]]],
        "mean": [[[3, 4], [3, 4], [0, 0]], [[9, 10], [9, 10], [0, 0]]],
        "min": [[[1, 2], [3, 4], [0, 0]], [[9, 10], [7, 8], [0, 0]]],
        "max": [[[5, 6], [3, 4], [0, 0]], [[9, 10], [11, 12], [0, 0]]],
    }
    _assert_equal(
        scatter(src, index, dim=1, dim_size=3, reduce=reduce),
        torch.tensor(expected.get(reduce), dtype=src.dtype),
    )


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc05_index_collapsed_along_scatter_dimension(reduce):
    src = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ],
        dtype=torch.float32,
    )
    index = torch.tensor([[[0, 1]], [[1, 0]]], dtype=torch.int64)
    expected = {
        "sum": [[[9, 0], [0, 12]], [[0, 30], [27, 0]]],
        "add": [[[9, 0], [0, 12]], [[0, 30], [27, 0]]],
        "mul": [[[15, 1], [1, 48]], [[1, 960], [693, 1]]],
        # torch_scatter 2.1.2 builds mean counts from index.shape before
        # broadcasting, so an explicitly collapsed scatter dimension counts
        # once and this upstream corner case equals sum.
        "mean": [[[9, 0], [0, 12]], [[0, 30], [27, 0]]],
        "min": [[[1, 0], [0, 2]], [[0, 8], [7, 0]]],
        "max": [[[5, 0], [0, 6]], [[0, 12], [11, 0]]],
    }
    _assert_equal(
        scatter(src, index, dim=1, dim_size=2, reduce=reduce),
        torch.tensor(expected.get(reduce), dtype=src.dtype),
    )


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc06_one_dimensional_index_broadcast_to_four_dimensions(reduce):
    torch.manual_seed(7)
    src = torch.randn(4, 3, 8, 8)
    index = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    actual = scatter(src, index, dim=2, dim_size=8, reduce=reduce)
    assert actual.shape == (4, 3, 8, 8)

    for b, c, w in itertools.product(range(4), range(3), range(8)):
        selected = [src[b, c, h, w] for h in range(8) if index[h].item() == 0]
        expected = _reduce_selected(selected, reduce)
        torch.testing.assert_close(actual[b, c, 0, w], expected)

    if reduce == "mean":
        # Upstream scatter_mean counts from the original index shape.  A 1D
        # singleton index at dim > 0 therefore has count=1 even though it
        # broadcasts across the complete scatter dimension.
        singleton_src = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        singleton_index = torch.tensor([0], dtype=torch.int64)
        singleton_expected = torch.tensor([[[6.0], [15.0]]])
        torch.testing.assert_close(
            scatter_mean(singleton_src, singleton_index, dim=2),
            singleton_expected,
        )
        torch.testing.assert_close(
            scatter(singleton_src, singleton_index, dim=2, reduce="mean"),
            singleton_expected,
        )


def test_tc07_provided_out_in_place_semantics():
    src = torch.tensor([2.0, 3.0, 4.0])
    index = torch.tensor([0, 0, 1])

    out = torch.tensor([10.0, 20.0, 30.0])
    assert scatter_sum(src, index, out=out) is out
    assert torch.equal(out, torch.tensor([15.0, 24.0, 30.0]))

    out = torch.tensor([10.0, 20.0, 30.0])
    assert scatter_mul(src, index, out=out) is out
    assert torch.equal(out, torch.tensor([60.0, 80.0, 30.0]))

    out = torch.tensor([10.0, 20.0, 30.0])
    assert scatter_mean(src, index, out=out) is out
    assert torch.equal(out, torch.tensor([7.5, 24.0, 30.0]))

    out = torch.tensor([10.0, 20.0, 30.0])
    result, arg = scatter_min(src, index, out=out)
    assert result is out
    assert torch.equal(out, torch.tensor([2.0, 4.0, 30.0]))
    assert torch.equal(arg, torch.tensor([0, 2, 3]))

    out = torch.tensor([1.0, 20.0, 30.0])
    result, arg = scatter_max(src, index, out=out)
    assert result is out
    assert torch.equal(out, torch.tensor([3.0, 20.0, 30.0]))
    assert torch.equal(arg, torch.tensor([1, 3, 3]))


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc08_non_contiguous_src_index_and_out(reduce):
    src = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).t()
    index = torch.tensor([[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]).t()
    assert not src.is_contiguous()
    assert not index.is_contiguous()

    expected = scatter(src.contiguous(), index.contiguous(), dim=0, reduce=reduce)
    base = torch.empty(expected.shape[::-1], dtype=src.dtype)
    out = base.t()
    assert not out.is_contiguous()
    # A provided out participates in reduction.  Exercise non-contiguous out
    # directly for sum/add and isolate src/index contiguity for other modes.
    if reduce in ("sum", "add"):
        out.zero_()
        actual = scatter(src, index, dim=0, out=out, reduce=reduce)
    else:
        actual = scatter(src, index, dim=0, reduce=reduce)
        out.copy_(actual)
        actual = out
    _assert_equal(actual, expected)


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc09_empty_tensor_with_explicit_dim_size(reduce):
    src = torch.empty((0, 2), dtype=torch.float32)
    index = torch.empty((0,), dtype=torch.int64)
    actual = scatter(src, index, dim=0, dim_size=3, reduce=reduce)
    assert actual.shape == (3, 2)
    assert torch.equal(actual, torch.zeros(3, 2))


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc10_dim_size_larger_than_index_max(reduce):
    src = torch.tensor([1, 2, 3], dtype=torch.int32)
    index = torch.tensor([0, 1, 0])
    actual = scatter(src, index, dim_size=5, reduce=reduce)
    assert actual.shape == (5,)


def test_tc11_high_conflict_sum_and_mean():
    src = torch.arange(1, 10001, dtype=torch.float32)
    index = torch.zeros(src.numel(), dtype=torch.int64)
    torch.testing.assert_close(
        scatter_sum(src, index), src.sum().reshape(1), rtol=1e-4, atol=1e-3
    )
    torch.testing.assert_close(
        scatter_mean(src, index), src.mean().reshape(1), rtol=1e-4, atol=1e-4
    )


def test_tc12_min_max_arg_out_and_later_writer_tie():
    src = torch.tensor([4.0, 2.0, 2.0, 5.0, 5.0])
    index = torch.tensor([0, 0, 0, 1, 1])
    min_out, arg_min = scatter_min(src, index)
    max_out, arg_max = scatter_max(src, index)
    assert torch.equal(min_out, torch.tensor([2.0, 5.0]))
    assert torch.equal(max_out, torch.tensor([4.0, 5.0]))
    assert torch.equal(arg_min, torch.tensor([2, 4]))
    assert torch.equal(arg_max, torch.tensor([0, 4]))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc13_all_documented_dtypes(dtype, reduce):
    src = torch.tensor([1, 2, 3, 4], dtype=dtype)
    index = torch.tensor([0, 0, 1, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = scatter(src, index, dim=0, dim_size=3, reduce=reduce)
    assert actual.dtype == dtype
    assert actual.shape == (3,)


def test_tc14_negative_dims_and_integer_floor_mean():
    src = torch.tensor([[-3, -2, 4], [-2, -1, 5]], dtype=torch.int32)
    index = torch.tensor([0, 0, 1])
    result = scatter_mean(src, index, dim=-1)
    assert torch.equal(result, torch.tensor([[-3, 4], [-2, 5]], dtype=torch.int32))


@pytest.mark.parametrize("rank", range(1, 9))
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_rank_1_through_8_with_compact_dim_index(rank, reduce):
    dim = rank // 2
    shape = [1] * rank
    shape[dim] = 3
    src = torch.tensor([1.0, 2.0, 3.0]).reshape(shape)
    index = torch.tensor([0, 1, 0], dtype=torch.int64)
    actual = scatter(src, index, dim=dim, dim_size=2, reduce=reduce)
    expected_values = {
        "sum": [4.0, 2.0],
        "add": [4.0, 2.0],
        "mul": [3.0, 2.0],
        "mean": [2.0, 2.0],
        "min": [1.0, 2.0],
        "max": [3.0, 2.0],
    }
    expected_shape = list(shape)
    expected_shape[dim] = 2
    expected = torch.tensor(expected_values.get(reduce)).reshape(expected_shape)
    _assert_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc15_tc16_l2_cpu_fallback(dtype, reduce):
    src = torch.tensor([2, 3, 4, 5], dtype=dtype)
    index = torch.tensor([0, 0, 1, 1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = scatter(src, index, reduce=reduce)
    assert result.device.type == "cpu"
    assert result.dtype == dtype
    assert not any("CPU fallback" in str(item.message) for item in caught)


def test_invalid_inputs_are_rejected_before_kernel_launch():
    src = torch.tensor([1.0, 2.0])
    with pytest.raises(TypeError, match="int64"):
        scatter(src, torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(IndexError, match="negative"):
        scatter(src, torch.tensor([0, -1]))
    with pytest.raises(IndexError, match="out of bounds"):
        scatter(src, torch.tensor([0, 2]), dim_size=2)
    with pytest.raises(IndexError, match="Dimension out of range"):
        scatter(src, torch.tensor([0, 1]), dim=1)
    with pytest.raises(ValueError, match="reduce"):
        scatter(src, torch.tensor([0, 1]), reduce="median")
    with pytest.raises(TypeError, match="src dtype"):
        scatter(torch.tensor([True]), torch.tensor([0]))
