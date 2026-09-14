# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU reference implementations for sparse index conversions."""

import torch


def ind2ptr_cpu(ind, num_rows):
    """CPU golden for ind2ptr, matching torch_sparse/csrc/cpu/convert_cpu.cpp."""
    if ind.dim() != 1:
        raise ValueError('ind must be 1-D')
    if ind.dtype != torch.long:
        raise ValueError('ind must be torch.long')
    out = torch.empty(num_rows + 1, dtype=torch.long)
    numel = ind.numel()
    if numel == 0:
        return out.zero_()

    ind_data = ind.tolist()
    out_data = [0] * (num_rows + 1)
    for i in range(ind_data[0] + 1):
        out_data[i] = 0
    for i in range(numel - 1):
        for k in range(ind_data[i], ind_data[i + 1]):
            out_data[k + 1] = i + 1
    for i in range(ind_data[-1] + 1, num_rows + 1):
        out_data[i] = numel
    return torch.tensor(out_data, dtype=torch.long)


def ptr2ind_cpu(ptr, num_edges):
    """CPU golden for ptr2ind, matching torch_sparse/csrc/cpu/convert_cpu.cpp."""
    if ptr.dim() != 1:
        raise ValueError('ptr must be 1-D')
    if ptr.dtype != torch.long:
        raise ValueError('ptr must be torch.long')
    out = torch.empty(num_edges, dtype=torch.long)
    if num_edges == 0:
        return out
    ptr_data = ptr.tolist()
    out_data = [0] * num_edges
    for i in range(ptr.numel() - 1):
        for edge_idx in range(ptr_data[i], ptr_data[i + 1]):
            out_data[edge_idx] = i
    return torch.tensor(out_data, dtype=torch.long)


def make_sorted_row_indices(num_edges, num_rows, seed=0):
    """Generate non-decreasing row indices in [0, num_rows)."""
    generator = torch.Generator().manual_seed(seed)
    if num_edges == 0:
        return torch.empty(0, dtype=torch.long)
    vals = torch.randint(0, num_rows, (num_edges,), generator=generator, dtype=torch.long)
    return torch.sort(vals).values
