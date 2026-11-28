# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for csr_row_wise_topk on Ascend A2/A3."""
import os

import pytest
import torch


def expected_weights(indptr, rows, weight, k, ascending):
    ptr = indptr.tolist()
    w = weight.tolist()
    out = {}
    for rid in rows.tolist():
        if rid < 0 or rid >= len(ptr) - 1:
            out[rid] = []
            continue
        begin, end = ptr[rid], ptr[rid + 1]
        row_w = w[begin:end]
        if k == -1:
            out[rid] = sorted(row_w)
        elif k == 0:
            out[rid] = []
        else:
            take = min(k, len(row_w))
            if ascending:
                out[rid] = sorted(sorted(row_w)[:take])
            else:
                out[rid] = sorted(sorted(row_w)[-take:]) if take else []
    return out


def check_result(out_rows, out_cols, out_data, indptr, indices, rows, weight,
                 k, ascending, data=None):
    r = out_rows.cpu().tolist()
    c = out_cols.cpu().tolist()
    d = out_data.cpu().tolist()
    csr_ptr = indptr.cpu().tolist()
    csr_idx = indices.cpu().tolist()
    w = weight.float().cpu().tolist()
    expected = expected_weights(indptr, rows, weight, k, ascending)

    actual_weights = {}
    counts = {}
    for rr, cc, _ in zip(r, c, d):
        assert 0 <= rr < len(csr_ptr) - 1, rr
        begin, end = csr_ptr[rr], csr_ptr[rr + 1]
        assert cc in csr_idx[begin:end], (rr, cc)
        counts[rr] = counts.get(rr, 0) + 1
    for rid, exp in expected.items():
        assert counts.get(rid, 0) == len(exp), (rid, counts.get(rid, 0), len(exp))
    for rr in counts:
        assert rr in expected, rr

    if data is None:
        for rr, cc, dd in zip(r, c, d):
            begin, end = csr_ptr[rr], csr_ptr[rr + 1]
            assert begin <= dd < end, (rr, dd, begin, end)
            assert csr_idx[dd] == cc, (rr, cc, dd)
            actual_weights.setdefault(rr, []).append(w[dd])
        for rid, exp in expected.items():
            got = sorted(actual_weights.get(rid, []))
            assert len(got) == len(exp), (rid, got, exp)
            assert all(abs(a - b) <= 1e-5 * max(1.0, abs(b)) for a, b in zip(got, exp)), (rid, got, exp)
    else:
        dat = data.cpu().tolist()
        for rr, cc, dd in zip(r, c, d):
            begin, end = csr_ptr[rr], csr_ptr[rr + 1]
            eids = [e for e in range(begin, end) if dat[e] == dd]
            assert eids, (rr, dd)
            assert any(csr_idx[e] == cc for e in eids), (rr, cc, dd)
            actual_weights.setdefault(rr, []).append(w[eids[0]])
        for rid, exp in expected.items():
            got = sorted(actual_weights.get(rid, []))
            assert len(got) == len(exp), (rid, got, exp)
            assert all(abs(a - b) <= 1e-5 * max(1.0, abs(b)) for a, b in zip(got, exp)), (rid, got, exp)
    return expected



@pytest.fixture
def op():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    from ops_gnn import csr_row_wise_topk
    return csr_row_wise_topk


def make_csr(dtype=torch.int32, degrees=(4, 0, 6, 3), seed=7):
    gen = torch.Generator().manual_seed(seed)
    ptr = [0]
    for degree in degrees:
        ptr.append(ptr[-1] + degree)
    idx = [int(torch.randint(0, 40, (1,), generator=gen)) for _ in range(ptr[-1])]
    values = list(range(ptr[-1]))
    weight = torch.rand(ptr[-1], generator=gen)
    return (torch.tensor(ptr, dtype=dtype), torch.tensor(idx, dtype=dtype),
            torch.tensor(values, dtype=dtype), weight)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("k,ascending", [(-1, False), (0, False), (1, False),
                                         (3, False), (3, True), (8, False)])
def test_topk_modes(op, dtype, k, ascending):
    ptr, idx, _, weight = make_csr(dtype)
    rows = torch.tensor([0, 1, 2, 3], dtype=dtype)
    out = op(ptr.npu(), idx.npu(), rows.npu(), k, weight.npu(), ascending)
    check_result(*out, ptr, idx, rows, weight, k, ascending)
    assert out[0].dtype == dtype


def test_with_data(op):
    ptr, idx, val, weight = make_csr()
    rows = torch.tensor([0, 2, 3], dtype=ptr.dtype)
    out = op(ptr.npu(), idx.npu(), rows.npu(), 2, weight.npu(), False, val.npu())
    check_result(*out, ptr, idx, rows, weight, 2, False, val)


def test_invalid_rows_are_empty(op):
    ptr, idx, _, weight = make_csr()
    rows = torch.tensor([-1, 99, 1, 4], dtype=torch.int32)
    out = op(ptr.npu(), idx.npu(), rows.npu(), -1, weight.npu())
    expected = check_result(*out, ptr, idx, rows, weight, -1, False)
    assert expected == {-1: [], 99: [], 1: [], 4: []}


def test_empty_rows_tensor(op):
    ptr, idx, _, weight = make_csr()
    rows = torch.tensor([], dtype=ptr.dtype)
    out = op(ptr.npu(), idx.npu(), rows.npu(), 4, weight.npu())
    assert all(t.numel() == 0 for t in out)


@pytest.mark.parametrize("wdtype", [torch.float64, torch.int32, torch.int64])
def test_weight_dtypes(op, wdtype):
    ptr, idx, _, weight = make_csr(degrees=(5, 2, 0, 7))
    if wdtype in (torch.int32, torch.int64):
        scaled = (weight * 100).to(wdtype)
    else:
        scaled = weight.to(wdtype)
    rows = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    for k, asc in [(3, False), (2, True), (-1, False)]:
        out = op(ptr.npu(), idx.npu(), rows.npu(), k, scaled.npu(), asc)
        check_result(*out, ptr, idx, rows, scaled.float(), k, asc)


def test_large_degree(op):
    ptr, idx, _, weight = make_csr(degrees=(900, 4, 0, 512), seed=9)
    rows = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    for k, asc in [(100, False), (100, True), (-1, False)]:
        out = op(ptr.npu(), idx.npu(), rows.npu(), k, weight.npu(), asc)
        check_result(*out, ptr, idx, rows, weight, k, asc)


def test_oversize_row_rejected(op):
    ptr, idx, _, weight = make_csr(degrees=(1200,), seed=3)
    rows = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(RuntimeError):
        op(ptr.npu(), idx.npu(), rows.npu(), 4, weight.npu())


def test_deterministic(op):
    ptr, idx, _, weight = make_csr(degrees=(20, 20, 20))
    rows = torch.tensor([0, 1, 2], dtype=torch.int32)
    first = op(ptr.npu(), idx.npu(), rows.npu(), 5, weight.npu())
    second = op(ptr.npu(), idx.npu(), rows.npu(), 5, weight.npu())
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_bad_weight(op):
    ptr, idx, _, weight = make_csr()
    rows = torch.tensor([0], dtype=torch.int32)
    with pytest.raises((ValueError, RuntimeError)):
        op(ptr.npu(), idx.npu(), rows.npu(), 2, weight.to(torch.float16).npu())
    with pytest.raises((ValueError, RuntimeError)):
        op(ptr.npu(), idx.npu(), rows.npu(), 2, weight[:-1].npu())


def test_dtype_mismatch_rejected(op):
    ptr, idx, _, weight = make_csr()
    rows = torch.tensor([0], dtype=torch.int64)
    with pytest.raises((ValueError, RuntimeError)):
        op(ptr.npu(), idx.npu(), rows.npu(), 2, weight.npu())
