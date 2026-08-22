# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import math
import warnings
from dataclasses import dataclass

import pytest
import torch
import ops_gnn
from ops_gnn.scatter import _warned_l2

DEVICE_ID = int(os.environ.get("NPU_DEVICE_ID", "0"))
pytestmark = pytest.mark.npu

# Ecology operator precision standard (mixed tolerance):
# https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
FLOAT_TOLERANCES = {
    torch.float16: {"rtol": 2 ** -9, "atol": 2 ** -9, "max_abs": 1e-1},
    torch.bfloat16: {"rtol": 2 ** -6, "atol": 2 ** -6, "max_abs": 1.0},
    torch.float32: {"rtol": 2 ** -10, "atol": 2 ** -16, "max_abs": 1e-2},
}
REQUIRED_MATCHED_RATIO = 0.99

SUPPORTED_DTYPES = [torch.float16, torch.bfloat16, torch.float32,
                    torch.int8, torch.int16, torch.int32, torch.uint8]

REDUCES = ["sum", "add", "mul", "mean", "min", "max"]

TEST_SHAPES = [(128,), (1024,), (4096,), (8192,),
               (32, 512), (64, 768), (8, 16, 64), (4, 128, 256)]

GENERAL_SHAPES = [(1,), (2,), (4,), (2, 2), (1, 128),
                  (512, 768), (512, 1024), (1024, 768), (1024, 1024)]


@dataclass(frozen=True)
class CpuScatterRequest:
    dim: int = -1
    out: object = None
    dim_size: int = None
    reduce: str = "sum"


@dataclass
class CpuScatterViews:
    src: torch.Tensor
    index: torch.Tensor
    out: torch.Tensor
    count: object
    arg: object
    src_dim_size: int
    flat_after: int


def _new_cpu_output(src, request, out_size):
    if request.out is not None:
        return request.out.cpu().clone()
    if request.reduce == "mul":
        return src.new_ones(out_size)
    if request.reduce == "min":
        initial = (torch.finfo(src.dtype).max if src.dtype.is_floating_point
                   else torch.iinfo(src.dtype).max)
        return src.new_full(out_size, initial)
    if request.reduce == "max":
        initial = (torch.finfo(src.dtype).min if src.dtype.is_floating_point
                   else torch.iinfo(src.dtype).min)
        return src.new_full(out_size, initial)
    return src.new_zeros(out_size)


def _cpu_views(src, index, out, dim, request):
    flat_before = math.prod(src.shape[:dim])
    flat_after = math.prod(src.shape[dim + 1:])
    src_dim_size = src.size(dim)
    src_view = src.reshape(flat_before, src_dim_size, flat_after)
    index_view = index.reshape(flat_before, src_dim_size, flat_after)
    out_view = out.reshape(flat_before, request.dim_size, flat_after)
    count = (
        torch.zeros(out.shape, dtype=torch.int64)
        if request.reduce == "mean"
        else None
    )
    arg = out.new_full(out.shape, src_dim_size, dtype=torch.long)
    if request.reduce not in ("min", "max"):
        arg = None
    return CpuScatterViews(
        src_view,
        index_view,
        out_view,
        count.reshape(out_view.shape) if count is not None else None,
        arg.reshape(out_view.shape) if arg is not None else None,
        src_dim_size,
        flat_after,
    ), arg


def _apply_cpu_value(views, offset, reduce):
    row_size = views.src_dim_size * views.flat_after
    before = offset // row_size
    position = (offset % row_size) // views.flat_after
    inner = offset % views.flat_after
    target = int(views.index[before, position, inner].item())
    value = views.src[before, position, inner].item()
    key = (before, target, inner)
    if reduce in ("sum", "add", "mean"):
        views.out[key] += value
        if views.count is not None:
            views.count[key] += 1
    elif reduce == "mul":
        views.out[key] *= value
    elif reduce == "min" and value <= views.out[key]:
        views.out[key] = value
        views.arg[key] = position
    elif reduce == "max" and value >= views.out[key]:
        views.out[key] = value
        views.arg[key] = position


def _finalize_cpu_output(src, out, arg, views, request):
    if views.count is not None:
        nonzero = views.count > 0
        if src.dtype in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        ):
            quotient = views.out[nonzero].float() / views.count[nonzero].float()
            views.out[nonzero] = quotient.floor().to(out.dtype)
        else:
            views.out[nonzero] /= views.count[nonzero]
    if arg is None:
        return out
    if request.out is None:
        empty = arg == src.size(request.dim)
        out[empty] = 0
    return out, arg


def scatter_cpu(src, index, **kwargs):
    request = CpuScatterRequest(**kwargs)
    sc = src.cpu()
    ic = index.cpu().long()
    dim = request.dim % sc.dim()
    dim_size = request.dim_size
    if dim_size is None:
        dim_size = int(ic.max().item()) + 1 if ic.numel() > 0 else 0
    request = CpuScatterRequest(dim, request.out, dim_size, request.reduce)
    out_size = list(sc.shape)
    out_size[dim] = dim_size
    oc = _new_cpu_output(sc, request, out_size)
    sbc = sc.expand_as(ic) if sc.size(dim) == 1 else sc
    views, arg = _cpu_views(sbc, ic, oc, dim, request)
    for offset in range(sbc.numel()):
        _apply_cpu_value(views, offset, request.reduce)
    return _finalize_cpu_output(sc, oc, arg, views, request)


def _scatter_fn(reduce):
    return getattr(ops_gnn, f'scatter_{reduce}', None)


def _call_scatter(src, index, reduce, dim=-1):
    fn = _scatter_fn(reduce)
    if fn is None:
        return ops_gnn.scatter(src, index, dim=dim, reduce=reduce)
    return fn(src, index, dim=dim)


def _is_float_dtype(dtype):
    return dtype in (torch.float16, torch.bfloat16, torch.float32)


def _assert_match(result, expected, dtype):
    if _is_float_dtype(dtype):
        actual = result.cpu().float()
        golden = expected.cpu().float()
        tolerance = FLOAT_TOLERANCES[dtype]
        abs_error = (actual - golden).abs()
        allowed = tolerance["atol"] + tolerance["rtol"] * golden.abs()
        matched_ratio = (abs_error <= allowed).float().mean().item()
        max_abs_error = abs_error.max().item() if abs_error.numel() else 0.0
        assert matched_ratio >= REQUIRED_MATCHED_RATIO, (
            f"matched_ratio={matched_ratio:.8f} < {REQUIRED_MATCHED_RATIO:.8f}; "
            f"dtype={dtype}; max_abs_error={max_abs_error:.8g}"
        )
        assert max_abs_error <= tolerance["max_abs"], (
            f"max_abs_error={max_abs_error:.8g} > {tolerance['max_abs']:.8g}; "
            f"dtype={dtype}; matched_ratio={matched_ratio:.8f}"
        )
    else:
        assert torch.equal(result.cpu(), expected)


def _value_only(result):
    return result[0] if isinstance(result, tuple) else result


def _assert_npu_case(result, expected, dtype):
    actual = _value_only(result)
    golden = _value_only(expected)
    assert actual.device.type == "npu"
    _assert_match(actual, golden, dtype)


class TestScatter:
    @staticmethod
    def test_large():
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(512, 768, device='cpu')
        index = torch.randint(0, 1024, (512, 768), device='cpu')
        result = ops_gnn.scatter_sum(src.npu(), index.npu(), dim=1, dim_size=1024)
        expected = scatter_cpu(src, index, dim=1, dim_size=1024, reduce="sum")
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    @staticmethod
    def test_sum_add_equivalent():
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.randint(0, 32, (128,), device='cpu')
        r1 = ops_gnn.scatter_sum(src.npu(), index.npu())
        r2 = ops_gnn.scatter_add(src.npu(), index.npu())
        assert torch.allclose(r1.cpu(), r2.cpu())

    @staticmethod
    def test_dim_size_one():
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.zeros(128, dtype=torch.long)
        result = ops_gnn.scatter_sum(src.npu(), index.npu(), dim_size=1)
        expected = scatter_cpu(src, index, dim_size=1, reduce="sum")
        assert result.shape[0] == 1
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    @staticmethod
    def test_dim_size_large():
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.randint(0, 32, (128,), device='cpu')
        result = ops_gnn.scatter_sum(src.npu(), index.npu(), dim_size=10000)
        expected = scatter_cpu(src, index, dim_size=10000, reduce="sum")
        assert result.shape[0] == 10000
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    @staticmethod
    def test_empty_index():
        torch.npu.set_device(DEVICE_ID)
        src = torch.randn(0, device='cpu')
        index = torch.empty(0, dtype=torch.long)
        result = ops_gnn.scatter_sum(src.npu(), index.npu())
        assert result.numel() == 0

    @staticmethod
    def test_mean_int_floor():
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randint(0, 10, (64,), dtype=torch.int32)
        index = torch.randint(0, 8, (64,))
        result = ops_gnn.scatter_mean(src.npu(), index.npu())
        expected = scatter_cpu(src, index, reduce="mean")
        assert torch.equal(result.cpu(), expected)

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_shapes(self, shape, reduce, dtype):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        dim = -1
        if len(shape) >= 2:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, max(1, shape[dim] // 2), shape)
        else:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, 32, shape)
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=dim)
        if isinstance(result, tuple):
            result = result[0]
        expected = scatter_cpu(src, index, dim=dim, reduce=reduce)
        if isinstance(expected, tuple):
            expected = expected[0]
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)

    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_general_shapes(self, shape, reduce, dtype):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        if len(shape) >= 2:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, max(1, shape[-1] // 2), shape)
        else:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, max(1, shape[0] // 4), shape)
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=-1)
        if isinstance(result, tuple):
            result = result[0]
        expected = scatter_cpu(src, index, dim=-1, reduce=reduce)
        if isinstance(expected, tuple):
            expected = expected[0]
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_dim0(self, reduce):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, 32, device='cpu')
        index = torch.randint(0, 64, (128, 32), device='cpu')
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=0)
        expected = scatter_cpu(src, index, dim=0, reduce=reduce)
        _assert_npu_case(result, expected, torch.float32)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_dim1(self, reduce):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, 32, device='cpu')
        index = torch.randint(0, 64, (128, 32), device='cpu')
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=1)
        expected = scatter_cpu(src, index, dim=1, reduce=reduce)
        _assert_npu_case(result, expected, torch.float32)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_with_out(self, reduce):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(32, 128, device='cpu')
        index = torch.randint(0, 256, (32, 128), device='cpu')
        out_cpu = torch.ones(32, 256, device='cpu')
        expected = scatter_cpu(
            src, index, dim=1, out=out_cpu, dim_size=256, reduce=reduce
        )
        out = out_cpu.npu()
        fn = _scatter_fn(reduce) or ops_gnn.scatter
        kwargs = {'dim': 1, 'out': out}
        if fn == ops_gnn.scatter:
            kwargs['reduce'] = reduce
        result = fn(src.npu(), index.npu(), **kwargs)
        if isinstance(result, tuple):
            assert result[0] is out
            _assert_match(result[0], expected[0], torch.float32)
            assert torch.equal(result[1].cpu(), expected[1])
        else:
            assert result is out
            _assert_match(result, expected, torch.float32)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_high_conflict(self, reduce):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(1024, 64, device='cpu')
        index = torch.zeros(1024, 64, dtype=torch.long)
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=1)
        expected = scatter_cpu(src, index, dim=1, reduce=reduce)
        _assert_npu_case(result, expected, torch.float32)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_sparse_index(self, reduce):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.randint(0, 10000, (128,), device='cpu')
        result = _call_scatter(src.npu(), index.npu(), reduce)
        expected = scatter_cpu(src, index, reduce=reduce)
        _assert_npu_case(result, expected, torch.float32)


@pytest.mark.parametrize(
    "dtype,length",
    [
        (torch.int8, 128),
        (torch.int16, 40000),
        (torch.uint8, 300),
    ],
)
def test_integer_mean_uses_wide_count(dtype, length):
    """Counts must not wrap in the input dtype before integer floor division."""
    torch.npu.set_device(DEVICE_ID)
    src = torch.ones(length, dtype=dtype)
    index = torch.zeros(length, dtype=torch.long)
    actual = ops_gnn.scatter_mean(src.npu(), index.npu())
    expected = scatter_cpu(src, index, reduce="mean")
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("reduce", ["min", "max"])
def test_integer_minmax_value_and_later_writer_arg(dtype, reduce):
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([4, 2, 2, 5, 5], dtype=dtype)
    index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    result = _call_scatter(src.npu(), index.npu(), reduce)
    expected = scatter_cpu(src, index, reduce=reduce)
    assert torch.equal(result[0].cpu(), expected[0])
    assert torch.equal(result[1].cpu(), expected[1])


@pytest.mark.parametrize("rank", range(1, 9))
@pytest.mark.parametrize("reduce", REDUCES)
def test_compact_index_rank_1_through_8(rank, reduce):
    """Keep compact 1D index broadcasting on the NPU path for every rank."""
    torch.npu.set_device(DEVICE_ID)
    dim = rank // 2
    shape = [1] * rank
    shape[dim] = 3
    src = torch.tensor([1.0, 2.0, 3.0]).reshape(shape)
    index = torch.tensor([0, 1, 0], dtype=torch.long)
    expected_values = (
        [4.0, 2.0], [4.0, 2.0], [3.0, 2.0],
        [2.0, 2.0], [1.0, 2.0], [3.0, 2.0],
    )[REDUCES.index(reduce)]
    expected_shape = list(shape)
    expected_shape[dim] = 2
    expected = torch.tensor(expected_values).reshape(expected_shape)
    actual = _call_scatter(src.npu(), index.npu(), reduce, dim=dim)
    _assert_match(_value_only(actual), expected, torch.float32)


@pytest.mark.parametrize("reduce", REDUCES)
def test_compact_index_broadcast_to_four_dimensions(reduce):
    torch.npu.set_device(DEVICE_ID)
    src = torch.arange(1, 2 * 3 * 8 * 4 + 1, dtype=torch.float32).reshape(
        2, 3, 8, 4
    )
    index = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    expanded_index = index.reshape(1, 1, 8, 1).expand_as(src)
    expected = scatter_cpu(
        src, expanded_index, dim=2, dim_size=6, reduce=reduce
    )
    actual = ops_gnn.scatter(
        src.npu(), index.npu(), dim=2, dim_size=6, reduce=reduce
    )
    _assert_npu_case(actual, expected, torch.float32)


def test_compact_singleton_mean_uses_original_index_count():
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32
    )
    index = torch.tensor([0], dtype=torch.long)
    expected = torch.tensor([[[6.0], [15.0]]], dtype=torch.float32)
    actual = ops_gnn.scatter_mean(src.npu(), index.npu(), dim=2)
    _assert_match(actual, expected, torch.float32)


@pytest.mark.parametrize("reduce", REDUCES)
def test_non_contiguous_npu_inputs_and_out(reduce):
    torch.npu.set_device(DEVICE_ID)
    src = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).t()
    index = torch.tensor(
        [[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.long
    ).t()
    out_shape = (3, src.size(1))
    initial = {
        "sum": 0.0,
        "add": 0.0,
        "mul": 1.0,
        "mean": 0.0,
        "min": torch.finfo(src.dtype).max,
        "max": torch.finfo(src.dtype).min,
    }[reduce]
    out_cpu = torch.full(tuple(reversed(out_shape)), initial).t()
    expected = scatter_cpu(
        src, index, dim=0, out=out_cpu, dim_size=3, reduce=reduce
    )

    src_npu = src.npu()
    index_npu = index.npu()
    out = torch.full(
        tuple(reversed(out_shape)), initial, dtype=src.dtype, device="npu"
    ).t()
    assert not src_npu.is_contiguous()
    assert not index_npu.is_contiguous()
    assert not out.is_contiguous()
    fn = _scatter_fn(reduce) or ops_gnn.scatter
    kwargs = {"dim": 0, "out": out}
    if fn == ops_gnn.scatter:
        kwargs["reduce"] = reduce
    actual = fn(src_npu, index_npu, **kwargs)
    assert _value_only(actual) is out
    _assert_match(_value_only(actual), _value_only(expected), torch.float32)
    if isinstance(actual, tuple):
        assert torch.equal(actual[1].cpu(), expected[1])


@pytest.mark.parametrize("reduce", REDUCES)
def test_empty_npu_source_with_explicit_dim_size(reduce):
    torch.npu.set_device(DEVICE_ID)
    src = torch.empty((0, 2), dtype=torch.float32, device="npu")
    index = torch.empty((0,), dtype=torch.long, device="npu")
    actual = ops_gnn.scatter(src, index, dim=0, dim_size=3, reduce=reduce)
    assert actual.shape == (3, 2)
    assert torch.equal(actual.cpu(), torch.zeros(3, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_vector_sum_mean_fast_path(dtype, reduce):
    torch.npu.set_device(DEVICE_ID)
    src = ((torch.arange(128 * 32).reshape(128, 32) % 7) + 1).to(dtype)
    index = torch.arange(128, dtype=torch.long) % 8
    expanded_index = index.reshape(-1, 1).expand_as(src)
    expected = scatter_cpu(
        src, expanded_index, dim=0, dim_size=12, reduce=reduce
    )
    actual = ops_gnn.scatter(
        src.npu(), index.npu(), dim=0, dim_size=12, reduce=reduce
    )
    _assert_match(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduce", ["min", "max"])
def test_vector_minmax_fast_path_value_and_arg(dtype, reduce):
    torch.npu.set_device(DEVICE_ID)
    src = ((torch.arange(128 * 32).reshape(128, 32) % 17) - 8).to(dtype)
    index = torch.arange(128, dtype=torch.long) % 8
    expanded_index = index.reshape(-1, 1).expand_as(src)
    expected = scatter_cpu(
        src, expanded_index, dim=0, dim_size=12, reduce=reduce
    )
    fn = _scatter_fn(reduce)
    actual = fn(src.npu(), index.npu(), dim=0, dim_size=12)
    _assert_match(actual[0], expected[0], dtype)
    assert torch.equal(actual[1].cpu(), expected[1])


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
@pytest.mark.parametrize("reduce", REDUCES)
def test_l2_npu_fallback_out_identity_value_arg_and_warning(dtype, reduce):
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([2, 3, 4, 5], dtype=dtype)
    index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    initial = torch.tensor([10, 20, 30], dtype=dtype)
    expected = scatter_cpu(
        src, index, out=initial, dim_size=3, reduce=reduce
    )

    _warned_l2.discard(dtype)
    out = initial.npu()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        fn = _scatter_fn(reduce) or ops_gnn.scatter
        kwargs = {"out": out}
        if fn == ops_gnn.scatter:
            kwargs["reduce"] = reduce
        actual = fn(src.npu(), index.npu(), **kwargs)
    fallback_warnings = [
        item for item in caught if "CPU fallback" in str(item.message)
    ]
    assert len(fallback_warnings) == 1
    assert _value_only(actual) is out
    assert torch.equal(_value_only(actual).cpu(), _value_only(expected))
    if isinstance(actual, tuple):
        assert torch.equal(actual[1].cpu(), expected[1])


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_l2_broadcast_mean_npu_fallback(dtype):
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([[2, 4, 6], [3, 6, 9]], dtype=dtype)
    index = torch.tensor([[0], [1]], dtype=torch.long)
    expected = torch.tensor([[12, 0], [0, 18]], dtype=dtype)
    _warned_l2.discard(dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        actual = ops_gnn.scatter_mean(
            src.npu(), index.npu(), dim=1, dim_size=2
        )
    assert len([
        item for item in caught if "CPU fallback" in str(item.message)
    ]) == 1
    assert torch.equal(actual.cpu(), expected)


def test_npu_invalid_metadata_before_kernel_launch():
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([1.0, 2.0], device="npu")
    index = torch.tensor([0, 1], dtype=torch.long, device="npu")
    with pytest.raises(TypeError, match="int64"):
        ops_gnn.scatter(src, index.to(torch.int32))
    with pytest.raises(IndexError, match="Dimension out of range"):
        ops_gnn.scatter(src, index, dim=1)
    with pytest.raises(ValueError, match="non-negative"):
        ops_gnn.scatter(src, index, dim_size=-1)
    with pytest.raises(ValueError, match="reduce"):
        ops_gnn.scatter(src, index, reduce="median")
    with pytest.raises(TypeError, match="out dtype"):
        ops_gnn.scatter(
            src, index, out=torch.empty(2, dtype=torch.float16, device="npu")
        )
    with pytest.raises(ValueError, match="out shape"):
        ops_gnn.scatter(
            src.reshape(1, 2), index, dim=1,
            out=torch.empty((2, 2), device="npu")
        )
    with pytest.raises(IndexError, match="negative"):
        ops_gnn.scatter(
            src, torch.tensor([0, -1], dtype=torch.long, device="npu"),
            dim_size=2
        )
    with pytest.raises(IndexError, match="out of bounds"):
        ops_gnn.scatter(
            src, torch.tensor([0, 2], dtype=torch.long, device="npu"),
            dim_size=2
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        ops_gnn.scatter(src, index, unsupported=True)
    with pytest.raises(TypeError, match="multiple values"):
        ops_gnn.scatter(src, index, -1, None, None, "sum", reduce="sum")
    with pytest.raises(TypeError, match="positional arguments"):
        ops_gnn.scatter(src, index, -1, None, None, "sum", "extra")


def test_public_scatter_positional_option_binding():
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([1.0, 2.0], device="npu")
    index = torch.tensor([0, 1], dtype=torch.long, device="npu")
    actual = ops_gnn.scatter(src, index, -1, None, 3, "sum")
    expected = torch.tensor([1.0, 2.0, 0.0])
    _assert_match(actual, expected, torch.float32)


def test_npu_index_bounds_cache_invalidates_after_inplace_update():
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([1.0, 2.0], device="npu")
    index = torch.tensor([0, 1], dtype=torch.long, device="npu")
    ops_gnn.scatter(src, index, dim_size=2)
    index[1] = -1
    with pytest.raises(IndexError, match="negative"):
        ops_gnn.scatter(src, index, dim_size=2)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
