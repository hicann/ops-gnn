# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os

import pytest
import torch
from golden import spmm_max_reference


@pytest.fixture
def op():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    from ops_gnn import spmm_max_csr
    return spmm_max_csr


def test_reference():
    x = torch.tensor([[-float("inf"), -2], [-3, float("nan")]], dtype=torch.float16)
    p, i = torch.tensor([0, 0, 1, 3, 3]), torch.tensor([0, 0, 1])
    expected = torch.tensor([[0, 0], [-float("inf"), -2], [-3, float("nan")], [0, 0]],
                            dtype=torch.float16)
    torch.testing.assert_close(spmm_max_reference(p, i, x), expected, equal_nan=True)


@pytest.mark.parametrize("n", [1, 15, 16, 17, 31, 32, 33, 127, 1024])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_features(op, n, dtype):
    p = torch.tensor([0, 0, 3, 3, 4, 7, 7], dtype=dtype)
    i = torch.tensor([0, 2, 0, 1, 3, 1, 2], dtype=dtype)
    x = -torch.rand(4, n, dtype=torch.float16)
    expected = spmm_max_reference(p, i, x)
    got = op(p.npu(), i.npu(), x.npu())
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(expected, spmm_max_reference(p, i, x, fp32=False))


@pytest.mark.parametrize("m,k", [(0, 0), (0, 2), (1, 0), (5, 0), (5, 2)])
def test_empty(op, m, k):
    got = op(torch.zeros(m + 1, dtype=torch.int64, device="npu"),
             torch.empty(0, dtype=torch.int64, device="npu"),
             torch.empty(k, 17, dtype=torch.float16, device="npu"))
    torch.testing.assert_close(got.cpu(), torch.zeros(m, 17, dtype=torch.float16))


def test_special_values(op):
    x = torch.tensor([[-float("inf"), -65504, float("nan"), 1, -2],
                      [-float("inf"), float("inf"), 4, float("nan"), -1]], dtype=torch.float16)
    # Test both NaN ordering directions and repeated edges.
    p, i = torch.tensor([0, 2, 4, 5]), torch.tensor([0, 1, 1, 0, 0])
    torch.testing.assert_close(op(p.npu(), i.npu(), x.npu()).cpu(),
                               spmm_max_reference(p, i, x), equal_nan=True, rtol=0, atol=0)


def test_strides_out_and_repetition(op):
    p = torch.tensor([0, 99, 2, 99, 3, 99], device="npu")[::2]
    i = torch.tensor([0, 99, 1, 99, 0, 99], device="npu")[::2]
    x = torch.randn(17, 2, device="npu", dtype=torch.float16).t()
    out = torch.empty(2, 17, device="npu", dtype=torch.float16)
    expected = spmm_max_reference(p.cpu(), i.cpu(), x.cpu())
    for _ in range(100):
        got = op(p, i, x, out)
        assert got.data_ptr() == out.data_ptr()
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="contiguous"):
        op(p, i, x, torch.empty(17, 2, device="npu", dtype=torch.float16).t())


def test_multiple_batches(op):
    torch.manual_seed(42)
    p = torch.tensor([0, 10001, 10001, 10002])
    i = torch.randint(0, 32, (10002,))
    x = torch.randn(32, 33, dtype=torch.float16)
    torch.testing.assert_close(op(p.npu(), i.npu(), x.npu()).cpu(),
                               spmm_max_reference(p, i, x), rtol=0, atol=0)


@pytest.mark.parametrize("p,i", [([1, 1], [0]), ([0, 2], [0]),
                                ([0, 2, 1, 2], [0, 0]), ([0, -1, 1], [0]),
                                ([0, 1], [-1]), ([0, 1], [2]),
                                ([0, 1], [2**32]), ([0, 2**32], [0])])
def test_bad_csr(op, p, i):
    with pytest.raises(RuntimeError, match="spmm_max_csr"):
        op(torch.tensor(p, device="npu"), torch.tensor(i, device="npu"),
           torch.ones(2, 17, dtype=torch.float16, device="npu"))


@pytest.mark.parametrize("case", ["cpu", "fp32", "bf16", "rank", "dtype", "ptr_rank",
                                  "empty_ptr", "zero_dim", "large_dim", "out_shape",
                                  "out_dtype", "out_cpu", "alias", "grad", "out_grad"])
def test_bad_metadata(op, case):
    p = torch.tensor([0, 1, 2], device="npu")
    i = torch.tensor([0, 1], device="npu")
    x = torch.ones(2, 17, device="npu", dtype=torch.float16)
    out = None
    if case == "cpu":
        x = x.cpu()
    if case == "fp32":
        x = x.float()
    if case == "bf16":
        x = x.bfloat16()
    if case == "rank":
        x = x.unsqueeze(0)
    if case == "dtype":
        i = i.int()
    if case == "ptr_rank":
        p = p.unsqueeze(0)
    if case == "empty_ptr":
        p = p[:0]
    if case == "zero_dim":
        x = x[:, :0]
    if case == "large_dim":
        x = torch.empty(2, 2**20, device="npu", dtype=torch.float16)
    if case == "out_shape":
        out = torch.empty(1, 17, device="npu", dtype=torch.float16)
    if case == "out_dtype":
        out = x.float()
    if case == "out_cpu":
        out = x.cpu()
    if case == "alias":
        out = x
    if case == "grad":
        x.requires_grad_()
    if case == "out_grad":
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device, requires_grad=True)
    with pytest.raises(RuntimeError, match="spmm_max_csr"):
        op(p, i, x, out)


def test_stream(op):
    stream = torch.npu.Stream()
    p = torch.tensor([0, 1, 2], device="npu")
    i = torch.tensor([1, 0], device="npu")
    x = torch.ones(2, 17, device="npu", dtype=torch.float16)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        x = x * -3
        got = op(p, i, x) + 2
    torch.npu.current_stream().wait_stream(stream)
    torch.testing.assert_close(got.cpu(), torch.full((2, 17), -1, dtype=torch.float16))


def test_ub_boundary(op):
    # Discover the device-specific boundary using the public validation error.
    import re
    p, i = torch.tensor([0, 1], device="npu"), torch.tensor([0], device="npu")
    with pytest.raises(RuntimeError, match="UB limit") as exc:
        op(p, i, torch.empty(1, 2**20, dtype=torch.float16, device="npu"))
    limit = int(re.search(r"UB limit (\d+)", str(exc.value)).group(1))
    for n in (limit - 1, limit):
        x = torch.randn(1, n, dtype=torch.float16, device="npu")
        torch.testing.assert_close(op(p, i, x).cpu(), x.cpu(), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="UB limit"):
        op(p, i, torch.empty(1, limit + 1, dtype=torch.float16, device="npu"))
