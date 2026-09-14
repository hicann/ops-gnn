# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent PyTorch reference, usable on CPU without ops_gnn."""
import torch


def spmm_max_reference(indptr, indices, x, fp32=True):
    m, n = indptr.numel() - 1, x.size(1)
    degrees = (indptr[1:] - indptr[:-1]).long()
    rows = torch.repeat_interleave(torch.arange(m, device=x.device), degrees)
    values = x.float() if fp32 else x
    result = torch.full((m, n), -float("inf"), dtype=values.dtype, device=x.device)
    result.scatter_reduce_(0, rows[:, None].expand(-1, n),
                           values[indices.long()], reduce="amax", include_self=True)
    result[degrees == 0] = 0
    return result.to(x.dtype)
