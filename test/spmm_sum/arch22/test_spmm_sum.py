# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import runpy
from pathlib import Path

import pytest
import torch

import conftest as test_support


spmm_sum_reference = runpy.run_path(Path(__file__).with_name("golden.py"))["spmm_sum_reference"]


@pytest.fixture
def op():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    from ops_gnn import spmm
    return lambda indptr, indices, x, out=None: spmm(
        indptr, indices, x, reduce="sum", out=out)


def test_reference():
    x = torch.tensor([[1, -2], [3, float("nan")]], dtype=torch.float16)
    ptr = torch.tensor([0, 0, 1, 3, 3])
    idx = torch.tensor([0, 0, 1])
    expected = torch.tensor([[0, 0], [1, -2], [4, float("nan")], [0, 0]],
                            dtype=torch.float16)
    torch.testing.assert_close(spmm_sum_reference(ptr, idx, x), expected, equal_nan=True)


@pytest.mark.parametrize("n", [1, 15, 16, 17, 31, 32, 33, 127, 1024])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_features(op, n, dtype):
    ptr = torch.tensor([0, 0, 3, 4], dtype=dtype)
    idx = torch.tensor([0, 2, 1, 2], dtype=dtype)
    x = (torch.arange(3 * n, dtype=torch.float16).reshape(3, n) % 7) - 3
    expected = spmm_sum_reference(ptr, idx, x)
    torch.testing.assert_close(op(ptr.npu(), idx.npu(), x.npu()).cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("m,k", [(0, 0), (0, 2), (1, 0), (5, 0), (5, 2)])
def test_empty(op, m, k):
    test_support.assert_empty_case(op, m, k)


def test_special_values_and_out(op):
    test_support.assert_special_values_and_out(op, spmm_sum_reference)


def test_strides_out_and_repetition(op):
    ptr = torch.tensor([0, 99, 2, 99, 3, 99], device="npu")[::2]
    idx = torch.tensor([0, 99, 1, 99, 0, 99], device="npu")[::2]
    x = torch.arange(34, device="npu", dtype=torch.float16).reshape(17, 2).t()
    out = torch.empty(2, 17, device="npu", dtype=torch.float16)
    expected = spmm_sum_reference(ptr.cpu(), idx.cpu(), x.cpu())
    for _ in range(10):
        got = op(ptr, idx, x, out)
        assert got.data_ptr() == out.data_ptr()
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="contiguous"):
        op(ptr, idx, x, torch.empty(17, 2, device="npu", dtype=torch.float16).t())


def test_multiple_batches(op):
    ptr = torch.tensor([0, 10001, 10001, 10002])
    idx = torch.arange(10002) % 32
    x = ((torch.arange(32 * 33).reshape(32, 33) % 5) - 2).to(torch.float16)
    expected = spmm_sum_reference(ptr, idx, x)
    torch.testing.assert_close(op(ptr.npu(), idx.npu(), x.npu()).cpu(),
                               expected, rtol=0, atol=0)


@pytest.mark.parametrize("ptr,idx", test_support.SPMM_BAD_CSR_CASES)
def test_bad_csr(op, ptr, idx):
    test_support.assert_bad_csr(op, ptr, idx)


@pytest.mark.parametrize("case", test_support.SPMM_BAD_METADATA_CASES)
def test_bad_metadata(op, case):
    test_support.assert_bad_metadata(op, case)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("batches,features", test_support.SPMM_BATCH_CASES)
def test_bspmm(dtype, batches, features):
    test_support.assert_bspmm_case("sum", spmm_sum_reference, (dtype, batches, features), (1e-2, 5e-2))


def test_bspmm_empty_batch():
    from ops_gnn import bspmm
    ptr = torch.tensor([0, 0, 0], device="npu")
    idx = torch.empty(0, dtype=torch.int64, device="npu")
    x = torch.empty((4, 0, 17), dtype=torch.float16, device="npu")
    got = bspmm(ptr, idx, x, reduce="sum")
    assert got.shape == (2, 0, 17)


def test_bspmm_special_values():
    test_support.assert_bspmm_special_values("sum", spmm_sum_reference, 1e-2, 5e-2)


def test_stream(op):
    test_support.assert_stream(op)


def test_ub_boundary(op):
    test_support.assert_ub_boundary(op)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_large_random_graph_reference(op, dtype):
    test_support.assert_large_random_graph(op, spmm_sum_reference, dtype, 1e-2, 5e-2)


def test_bspmm_binary_mean():
    from ops_gnn import bspmm
    ptr = torch.tensor([0, 2, 3])
    idx = torch.tensor([0, 2, 1])
    lhs = torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3) + 1
    rhs = torch.tensor([1.0, 2.0, 4.0]).reshape(3, 1, 1)
    messages = lhs[idx] * rhs
    expected = torch.stack([messages[:2].mean(0), messages[2:].mean(0)])
    got = bspmm(ptr.npu(), idx.npu(), lhs.npu(), op="mul", reduce="mean",
                rhs=rhs.npu())
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
