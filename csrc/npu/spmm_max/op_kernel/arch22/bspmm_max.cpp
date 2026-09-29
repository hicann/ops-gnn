/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bspmm_max_kernel.h"
#include "bspmm_max.h"

namespace opsgnn {

void BspmmMax(uint32_t blocks, aclrtStream stream, void* x, void* out,
                 void* ptr, void* idx, void* split, uint32_t m, uint32_t k,
                 uint32_t n, uint32_t nnz, uint32_t batches,
                 uint32_t ubBytes, uint32_t hasNan)
{
    bspmm_max<<<blocks, nullptr, stream>>>(static_cast<uint8_t*>(x),
        static_cast<uint8_t*>(out), static_cast<uint8_t*>(ptr),
        static_cast<uint8_t*>(idx), static_cast<uint8_t*>(split),
        m, k, n, nnz, batches, ubBytes, hasNan);
}

}  // namespace opsgnn
