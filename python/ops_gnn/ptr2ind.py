# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public ptr2ind API."""

from . import _pybind
from .typing import Tensor


def ptr2ind(ptr: Tensor, num_edges: int) -> Tensor:
    """Convert a CSR row-pointer tensor to row indices.

    Drop-in replacement for ``torch.ops.torch_sparse.ptr2ind(ptr, E)`` on Ascend NPU.
    The second argument is the edge/nnz count (historically named ``E`` in torch_sparse).

    Launches asynchronously on the current NPU stream (same as pytorch_sparse CUDA).
    Synchronize with ``torch.npu.synchronize()`` / ``torch.npu.current_stream().synchronize()``
    before reading results on host if needed.

    Args:
        ptr (Tensor): 1-D ``torch.long`` CSR pointer on NPU, length ``num_rows + 1``.
        num_edges (int): Number of edges / non-zeros. Output length is ``num_edges``.

    Returns:
        Tensor: 1-D ``torch.long`` row indices of shape ``[num_edges]`` on NPU.

    Example:
        >>> import torch
        >>> import ops_gnn
        >>> rowptr = torch.tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], dtype=torch.long, device='npu')
        >>> ops_gnn.ptr2ind(rowptr, 6)
        tensor([2, 2, 4, 5, 5, 6], device='npu:0')
    """
    if not ptr.is_contiguous():
        ptr = ptr.contiguous()
    return _pybind.ptr2ind(ptr, int(num_edges))
