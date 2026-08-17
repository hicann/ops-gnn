# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it
# under the terms and conditions of CANN Open Software License Agreement
# Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except
# in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text.

"""Public Gather COO API."""

from typing import Optional

from . import _pybind
from .typing import Tensor


def gather_coo(
    src: Tensor,
    index: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Gather rows from ``src`` according to a COO-style index tensor.

    ``index`` has the same prefix shape as ``src`` and uses its last
    dimension as the entry dimension.  The corresponding dimension of
    ``src`` is replaced by ``index.size(-1)``; all suffix dimensions are
    copied unchanged.  Indices must be int64, sorted, and within the source
    row range as required by the task contract.  The operation is a raw
    element-wise copy and does not provide an autograd implementation.

    Args:
        src: NPU tensor with rank 1 through 8.
        index: NPU int64 tensor.  Its rank is at most ``src.dim()`` and its
            prefix dimensions must match ``src``.
        out: Optional destination tensor with the exact output shape and
            matching dtype/device.  ``None`` is distinct from a valid empty
            tensor and is forwarded explicitly to the native binding.

    Returns:
        ``out`` when supplied, otherwise a newly allocated NPU tensor.
    """
    return _pybind.gather_coo(src, index, out)
