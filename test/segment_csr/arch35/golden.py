# Copyright (c) 2026 Starlink_. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""

CPU golden reference for segment_csr: a 1:1 replica of the reference
implementation shipped with the official test suite (the acceptance
baseline), plus a first-occurrence arg reference for min/max.
"""

import torch


def segment_csr_cpu(src, indptr, reduce="sum"):
    sc = src.cpu()
    ipc = indptr.cpu().long()
    dim = ipc.dim() - 1
    sizes = list(ipc.shape)
    for i in range(dim):
        sizes[i] = sc.size(i)
    ipc = ipc.expand(sizes)

    ns = ipc.size(dim) - 1
    seg = sc.size(dim)
    out_size = list(sc.shape)
    out_size[dim] = ns
    sc_m = sc.movedim(dim, -1).contiguous().reshape(-1, seg)
    oc_m = sc.new_zeros(out_size).movedim(dim, -1).contiguous()
    oc_flat = oc_m.reshape(-1, ns)
    cnt_m = sc.new_zeros(oc_m.shape) if reduce == "mean" else None
    cnt_flat = cnt_m.reshape(-1, ns) if cnt_m is not None else None
    ipf = ipc.reshape(-1, ipc.size(-1))

    for b in range(sc_m.size(0)):
        bi = min(b, ipf.size(0) - 1)
        ip = ipf[bi]
        for sgm in range(ns):
            s = int(ip[sgm].item())
            e = int(ip[sgm + 1].item())
            if s >= e or s >= seg:
                continue
            e = min(e, seg)
            vals = sc_m[b, s:e]
            if vals.numel() == 0:
                continue
            if reduce in ("sum", "add"):
                oc_flat[b, sgm] += vals.sum()
            elif reduce == "mean":
                oc_flat[b, sgm] += vals.sum()
                cnt_flat[b, sgm] = e - s
            elif reduce == "min":
                oc_flat[b, sgm] = vals.min()
            elif reduce == "max":
                oc_flat[b, sgm] = vals.max()
    if reduce == "mean" and cnt_flat is not None:
        nz = cnt_flat > 0
        mean = oc_flat[nz].float() / cnt_flat[nz].float()
        oc_flat[nz] = mean.to(sc.dtype)
    return oc_m.movedim(-1, dim).contiguous()


def segment_argminmax_cpu(src, indptr, reduce):
    """Return first-occurrence extremum indices in the output tensor shape.

    Each index refers to the original input reduction dimension.
    """
    if not (reduce in ("min", "max")):
        raise AssertionError("Validation condition failed")
    sc = src.cpu()
    dim = indptr.dim() - 1
    ns = indptr.numel() - 1
    m = sc.size(dim)
    moved = sc.movedim(dim, 0).contiguous()
    rest_shape = list(moved.shape[1:])
    rest_num = int(torch.tensor(rest_shape).prod().item()) if rest_shape else 1
    flat = moved.reshape(m, rest_num)
    argflat = torch.full((ns, rest_num), m, dtype=torch.long)
    for s in range(ns):
        lo = int(indptr[s].item())
        hi = min(int(indptr[s + 1].item()), m)
        if lo >= hi:
            continue
        vals = flat[lo:hi]
        v = vals.min(0).values if reduce == "min" else vals.max(0).values
        first = (vals == v.unsqueeze(0)).to(torch.long).argmax(0)
        argflat[s] = lo + first
    return argflat.reshape([ns] + rest_shape).movedim(0, dim)
