"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

from typing import Optional

import torch

from . import _pybind
from .typing import Tensor, OptTensor


def segment_max_csr(
    src: Tensor,
    indptr: Tensor,
    optional_out: Optional[Tensor] = None,
) -> Tensor:
    """
    Computes the maximum value of elements in each segment of a CSR matrix.

    Args:
        src (Tensor): The input tensor. Must be on NPU device.
        indptr (Tensor): The index pointer tensor of type int32. Must be on NPU device.
            The last dimension of indptr determines the number of segments.
        optional_out (Optional[Tensor]): Optional output tensor for initial values.
            If provided, segments will compute max(src, optional_out).

    Returns:
        Tensor: The result tensor with the same dtype as src. The shape is the same
            as src, except the dimension corresponding to indptr's last dimension
            is reduced to (indptr_last_dim - 1).

    Example:
        >>> import ops_gnn
        >>> src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
        >>> indptr = torch.tensor([[0, 2, 4]], dtype=torch.int32, device='npu')
        >>> result = ops_gnn.segment_max_csr(src, indptr)
        >>> result
        tensor([[3., 4.],
                [7., 8.]], device='npu:0')
    """
    if optional_out is None:
        optional_out = torch.tensor([])
    return _pybind.segment_max_csr(src, indptr, optional_out)