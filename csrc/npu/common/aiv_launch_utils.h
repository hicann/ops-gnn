/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SPARSE_COMMON_AIV_LAUNCH_UTILS_H
#define OPS_GNN_SPARSE_COMMON_AIV_LAUNCH_UTILS_H

#include <cstdint>

#include "tiling/platform/platform_ascendc.h"

namespace ops_gnn {
namespace sparse {

// Cap requested AIV count by platform AIV number; never return 0.
inline uint32_t ResolveAivCoreNum(uint32_t needCoreNum)
{
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    if (needCoreNum < coreNum) {
        coreNum = needCoreNum;
    }
    return coreNum;
}

}  // namespace sparse
}  // namespace ops_gnn

#endif  // OPS_GNN_SPARSE_COMMON_AIV_LAUNCH_UTILS_H
