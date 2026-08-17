# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stream usage tests for ind2ptr / ptr2ind (aligned with pytorch_sparse convert).

API matches torch_sparse: ind2ptr(ind, num_rows) / ptr2ind(ptr, num_edges) with no extra args.
Async launch on current NPU stream; sync via torch.npu.synchronize() / stream.synchronize().
"""

import torch
import ops_gnn

from test_convert import assert_exact, ind2ptr_cpu, ptr2ind_cpu


def _set_device(device_id: int = 0):
    if hasattr(torch, 'npu'):
        torch.npu.set_device(device_id)


def test_ind2ptr_async_default_stream():
    _set_device()
    row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
    out = ops_gnn.ind2ptr(row, 8)
    torch.npu.current_stream().synchronize()
    assert_exact(out, torch.tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], dtype=torch.long))


def test_ptr2ind_async_default_stream():
    _set_device()
    ptr = torch.tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], dtype=torch.long, device='npu')
    out = ops_gnn.ptr2ind(ptr, 6)
    torch.npu.synchronize()
    assert_exact(out, torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long))


def test_ind2ptr_custom_stream():
    _set_device()
    row = torch.tensor([1, 2, 2], dtype=torch.long, device='npu')
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        out = ops_gnn.ind2ptr(row, 4)
    stream.synchronize()
    assert_exact(out, ind2ptr_cpu(row.cpu(), 4))


def test_ptr2ind_custom_stream():
    _set_device()
    ptr = torch.tensor([0, 0, 1, 3, 3], dtype=torch.long, device='npu')
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        out = ops_gnn.ptr2ind(ptr, 3)
    stream.synchronize()
    assert_exact(out, ptr2ind_cpu(ptr.cpu(), 3))


def test_ind2ptr_async_after_arange():
    """Preceding torch_npu op on same stream must be visible."""
    _set_device()
    num_rows = 128
    row = torch.arange(num_rows, dtype=torch.long, device='npu')
    out = ops_gnn.ind2ptr(row, num_rows)
    torch.npu.synchronize()
    assert_exact(out, ind2ptr_cpu(row.cpu(), num_rows), msg='arange async')


def test_roundtrip_same_stream():
    _set_device()
    row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
    ptr = ops_gnn.ind2ptr(row, 8)
    out = ops_gnn.ptr2ind(ptr, 6)
    torch.npu.synchronize()
    assert_exact(out, row.cpu())
