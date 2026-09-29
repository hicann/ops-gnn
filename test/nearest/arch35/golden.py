# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU-only reference. This module is never imported by ops_gnn."""

import torch
from scipy.cluster.vq import vq


def nearest_scipy(x, y, batch_x=None, batch_y=None):
    """Run scipy vq on FP32 coordinates independently inside each batch.

    Preserve each complete batch: splitting either axis can change BLAS
    dispatch and FP32 rounding, hence the integer reference itself. The
    largest task shape requires about 4 GiB for scipy's distance workspace.
    """
    x = x.detach().cpu().float()
    y = y.detach().cpu().float()
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    n, m = x.size(0), y.size(0)
    out = torch.empty(n, dtype=torch.long)
    if not n:
        return out
    bx = torch.zeros(n, dtype=torch.long) if batch_x is None else batch_x.cpu()
    by = torch.zeros(m, dtype=torch.long) if batch_y is None else batch_y.cpu()
    for label in bx.unique_consecutive():
        ix = (bx == label).nonzero().flatten()
        iy = (by == label).nonzero().flatten()
        codebook = y[iy].numpy()
        codes = vq(x[ix].numpy(), codebook)[0]
        out[ix] = iy[torch.from_numpy(codes).long()]
    return out


def nearest_golden(x, y, batch_x=None, batch_y=None):
    """Use the task-text CPU/scipy reference for both input dtypes (issue #1)."""
    return nearest_scipy(x, y, batch_x, batch_y)


def original_attachment_cpu(x, y):
    """Original attachment oracle, only for regression counterexamples."""
    xc = x.cpu()
    yc = y.cpu()
    if xc.dim() == 1:
        xc = xc.view(-1, 1)
    if yc.dim() == 1:
        yc = yc.view(-1, 1)
    point_count = xc.size(0)
    thread_count = 1024
    out = torch.empty(point_count, dtype=torch.long)
    for i in range(point_count):
        dist = torch.zeros(yc.size(0), dtype=xc.dtype)
        for d in range(xc.size(1)):
            diff = yc[:, d] - xc[i, d]
            dist += diff * diff
        cand = (dist == dist.min()).nonzero().flatten()
        if cand.numel() > 1:
            r = cand % thread_count
            t = cand // thread_count
            key = r * (t.max() + 1) + t
            out[i] = cand[torch.argmin(key)]
        else:
            out[i] = cand[0]
    return out
