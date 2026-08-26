# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU reference implementation for segment_max_csr."""

import torch


def segment_max_csr_cpu(src, indptr):
    """Compute the segment_max_csr reference result on CPU."""
    indptr_dim = indptr.dim()
    n_segments = indptr.shape[-1] - 1
    result_shape = list(src.shape)
    result_shape[indptr_dim - 1] = n_segments
    result = torch.empty(result_shape, dtype=src.dtype)

    src_cpu = src.cpu()
    indptr_cpu = indptr.cpu()
    fill_values = {
        torch.float32: float("-inf"),
        torch.float16: -65504,
        torch.int32: -2147483648,
        torch.int16: -32768,
    }
    fill_value = fill_values.get(src.dtype, -32768)

    for batch_idx in range(src.shape[0]):
        for seg in range(n_segments):
            start = int(indptr_cpu[0, seg]) if indptr_dim > 1 else int(indptr_cpu[seg])
            end = int(indptr_cpu[0, seg + 1]) if indptr_dim > 1 else int(indptr_cpu[seg + 1])
            if start >= end:
                result[batch_idx, seg] = fill_value
            elif indptr_dim == 1:
                result[seg] = src_cpu[start:end].max(dim=0).values
            else:
                result[batch_idx, seg] = src_cpu[batch_idx, start:end].max(dim=0).values
    return result
