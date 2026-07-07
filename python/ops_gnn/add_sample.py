"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

from typing import Tuple

import torch

from . import _pybind
from .typing import Tensor


def add_sample(
    src1: Tensor,
    src2: Tensor,
) -> Tensor:
    """
    Performs element-wise addition of two tensors on NPU.

    Args:
        src1 (Tensor): The first input tensor. Must be on NPU device.
        src2 (Tensor): The second input tensor. Must be on NPU device and have
            the same shape as :obj:`src1`.

    Returns:
        Tensor: The element-wise sum of :obj:`src1` and :obj:`src2`, on NPU device.

    Example:
        >>> import ops_gnn
        >>> src1 = torch.tensor([1, 2, 3], dtype=torch.uint8, device='npu')
        >>> src2 = torch.tensor([4, 5, 6], dtype=torch.uint8, device='npu')
        >>> result = ops_gnn.add_sample(src1, src2)
        >>> result
        tensor([5, 7, 9], device='npu:0', dtype=torch.uint8)
    """
    return _pybind.add_sample(src1, src2)
