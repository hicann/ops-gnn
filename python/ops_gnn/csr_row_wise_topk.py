# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CSR row-wise top-k selection (DGL CSRRowWiseTopk)."""
__all__ = ["csr_row_wise_topk"]

from typing import Optional, Tuple

import torch

from . import _pybind

_ID_DTYPES = (torch.int32, torch.int64)


def _check_id(name: str, value: torch.Tensor, ndim: int = 1) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"csr_row_wise_topk: {name} must be a Tensor")
    if value.device.type != "npu":
        raise ValueError(f"csr_row_wise_topk: {name} must be an NPU tensor")
    if value.dim() != ndim:
        raise ValueError(f"csr_row_wise_topk: {name} must be {ndim}-D")
    if value.dtype not in _ID_DTYPES:
        raise ValueError(f"csr_row_wise_topk: {name} dtype must be int32 or int64")


def _normalize_weight(weight: torch.Tensor) -> torch.Tensor:
    """Normalize weight to float32 preserving the in-row ordering.

    The AscendC sort network is float-only. float32 passes through; float64,
    int32 and int64 are shifted by their global minimum (computed in double)
    and cast to float32, which keeps the relative order of every row.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError("csr_row_wise_topk: weight must be a Tensor")
    if weight.device.type != "npu":
        raise ValueError("csr_row_wise_topk: weight must be an NPU tensor")
    if weight.dim() != 1:
        raise ValueError("csr_row_wise_topk: weight must be 1-D")
    if weight.dtype == torch.float32:
        return weight.contiguous()
    if weight.dtype not in (torch.float64, torch.int32, torch.int64):
        raise ValueError(
            "csr_row_wise_topk: weight dtype must be float32, float64, int32 or int64")
    host = weight.detach().to(torch.float64).cpu()
    if host.numel() > 0:
        host = host - host.min()
    return host.to(torch.float32).to(weight.device).contiguous()


def csr_row_wise_topk(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    rows: torch.Tensor,
    k: int,
    weight: torch.Tensor,
    ascending: bool = False,
    data: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the top-k weighted neighbors of each requested CSR row.

    DGL ``CSRRowWiseTopk`` parity. For every row in ``rows``, the ``k`` edges
    with the largest (``ascending=False``) or smallest (``ascending=True``)
    weight are returned. ``k == -1`` selects every edge, ``k == 0`` returns an
    empty result, and any other value is the fanout (capped by the degree).

    ``weight`` is indexed by edge id; it may be ``float32``, ``float64``,
    ``int32`` or ``int64`` (non-float32 weights are shifted to non-negative and
    evaluated in float32, preserving the per-row ordering). Out-of-range seed
    rows are treated as empty rows.

    Returns ``(row, col, data)``: selected COO triples. ``data`` holds the CSR
    data value when supplied, otherwise the global edge id. All tensors use
    ``indptr``'s dtype and device.
    """
    _check_id("indptr", indptr)
    _check_id("indices", indices)
    _check_id("rows", rows)
    if indptr.dtype != indices.dtype or indptr.dtype != rows.dtype:
        raise ValueError("csr_row_wise_topk: indptr/indices/rows dtypes must match")
    if indptr.device != indices.device or indptr.device != rows.device:
        raise ValueError("csr_row_wise_topk: inputs must share one NPU device")
    if indptr.numel() == 0:
        raise ValueError("csr_row_wise_topk: indptr must not be empty")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("csr_row_wise_topk: k must be an int")
    if k < -1:
        raise ValueError("csr_row_wise_topk: k must be >= -1")
    if not isinstance(ascending, bool):
        raise TypeError("csr_row_wise_topk: ascending must be a bool")
    if weight.numel() < indices.numel():
        raise ValueError("csr_row_wise_topk: weight length must cover the matrix nnz")
    if data is not None:
        _check_id("data", data)
        if data.dtype != indptr.dtype or data.device != indptr.device:
            raise ValueError("csr_row_wise_topk: data must match indptr dtype and device")
        if data.numel() != indices.numel():
            raise ValueError("csr_row_wise_topk: data length must match indices")

    weights = _normalize_weight(weight)
    return _pybind.csr_row_wise_topk(
        indptr.contiguous(), indices.contiguous(), rows.contiguous(), k,
        weights, ascending,
        None if data is None else data.contiguous())
