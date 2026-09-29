/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <acl/acl.h>

struct NearestTiling {
    uint32_t n;
    uint32_t m;
    uint32_t features;
    uint32_t batches;
    uint32_t cores;
    uint64_t ub_bytes;
    uint32_t tile_x;
    uint32_t tile_y;
};

namespace opsgnn {

void Nearest(const void* x, const void* y, const int64_t* ptr_x,
             const int64_t* ptr_y, float* workspace, int64_t* out,
             bool is_half, const NearestTiling& tiling, aclrtStream stream);

}  // namespace opsgnn
