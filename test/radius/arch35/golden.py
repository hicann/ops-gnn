# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU reference helpers for radius functional tests."""

import torch


def radius_cpu(x, y, r, max_num_neighbors=32, ignore_same_index=False):
    xc = x.cpu().float()
    yc = y.cpu().float()
    rows = []
    cols = []
    for i in range(yc.size(0)):
        diff = xc - yc[i].unsqueeze(0)
        dist = diff.pow(2).sum(1)
        neighbors = (dist <= r * r).nonzero(as_tuple=False).flatten()
        if ignore_same_index:
            neighbors = neighbors[neighbors != i]
        for n in neighbors[:max_num_neighbors].tolist():
            rows.append(i)
            cols.append(int(n))
    if not rows:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def full_neighbors(x, y, r, ignore_same_index, batches=None):
    """Return every in-radius x index per query in ascending order."""
    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    n, m = x.shape[0], y.shape[0]
    if batches is None:
        bx = torch.zeros(n, dtype=torch.long)
        by = torch.zeros(m, dtype=torch.long)
    else:
        bx, by = batches
    full = [[] for _ in range(m)]
    r2 = float(r) * float(r)
    for q in range(m):
        idx = torch.arange(n, dtype=torch.long)
        mask = bx == by[q].item()
        if ignore_same_index:
            mask = mask & (idx != q)
        if mask.sum().item() == 0:
            continue
        valid_idx = torch.nonzero(mask).flatten()
        diff = x[valid_idx].double() - y[q].double()
        dist2 = (diff * diff).sum(dim=-1)
        full[q] = valid_idx[dist2 <= r2].tolist()
    return full
