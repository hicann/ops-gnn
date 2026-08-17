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

constexpr uint32_t GATHER_CSR_SEGMENT_MAJOR = 0;
constexpr uint32_t GATHER_CSR_OUTPUT_MAJOR = 1;

struct GatherCsrTilingData {
    uint64_t batchCount;
    uint64_t segmentCount;
    uint64_t outputRows;
    uint64_t featureBytes;
    uint64_t totalSegments;
    uint32_t tileBytes;
    uint32_t activeCoreNum;
    uint32_t scheduleMode;
};
