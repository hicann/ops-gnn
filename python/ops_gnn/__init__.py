"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from .add_sample import add_sample
from .graclus_cluster import graclus_cluster
from .gather_coo import gather_coo
from .random_walk import random_walk
from .segment_max_csr import segment_max_csr
from .typing import Tensor, OptTensor

__version__ = '0.1.0'

__all__ = [
    'add_sample',
    'graclus_cluster',
    'gather_coo',
    'random_walk',
    'segment_max_csr',
    'Tensor',
    'OptTensor',
]
