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
from .gather_csr import gather_csr
from .graclus_cluster import graclus_cluster
from .gather_coo import gather_coo
from .ind2ptr import ind2ptr
from .ptr2ind import ptr2ind
from .random_walk import random_walk
from .scatter import (
    native_available,
    scatter,
    scatter_add,
    scatter_max,
    scatter_mean,
    scatter_min,
    scatter_mul,
    scatter_sum,
)
from .segment_max_csr import segment_max_csr
from .radius import radius, radius_graph
from .typing import Tensor, OptTensor

__version__ = '0.1.0'

__all__ = [
    'add_sample',
    'gather_csr',
    'graclus_cluster',
    'gather_coo',
    'random_walk',
    'scatter',
    'scatter_sum',
    'scatter_add',
    'scatter_mul',
    'scatter_mean',
    'scatter_min',
    'scatter_max',
    'native_available',
    'segment_max_csr',
    'ind2ptr',
    'ptr2ind',
    'radius',
    'radius_graph',
    'Tensor',
    'OptTensor',
]
