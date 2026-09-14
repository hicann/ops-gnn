# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Forward-only FP16 CSR maximum aggregation on Ascend NPU."""
from typing import Optional

import torch
from . import _pybind


def spmm_max_csr(indptr: torch.Tensor, indices: torch.Tensor, x: torch.Tensor,
                 out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Aggregate source features using CSR rows corresponding to destinations.

    ``out[v, f] = max(x[indices[indptr[v]:indptr[v+1]], f])``.
    Empty rows are zero. NaNs propagate; infinities are preserved.
    ``x`` must be a 2D float16 NPU tensor. CSR tensors must be 1D,
    have matching int32/int64 dtypes and reside on the same NPU as x.
    Noncontiguous inputs are copied. Optional out must be contiguous,
    have shape (indptr.numel()-1, x.size(1)), and not alias any input.
    Gradients are unsupported, including when an input requires gradients.
    CSR validation synchronizes scalar results. No CPU fallback is provided.
    The maximum feature dimension is determined by the device's UB capacity.
    """
    for name, value in (("indptr", indptr), ("indices", indices), ("x", x), ("out", out)):
        if value is None and name == "out":
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"spmm_max_csr: {name} must be a Tensor")
    return _pybind.spmm_max_csr(indptr, indices, x, out)
