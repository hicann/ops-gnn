/*
 * Copyright (c) 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdint>

enum SegmentCooReduce : int32_t {
    SEGMENT_COO_SUM = 0,
    SEGMENT_COO_MEAN = 1,
    SEGMENT_COO_MIN = 2,
    SEGMENT_COO_MAX = 3,
};

enum SegmentCooDtype : int32_t {
    SEGMENT_COO_FLOAT32 = 0,
    SEGMENT_COO_FLOAT16 = 1,
    SEGMENT_COO_BFLOAT16 = 2,
    SEGMENT_COO_INT8 = 3,
    SEGMENT_COO_UINT8 = 4,
    SEGMENT_COO_INT32 = 5,
    SEGMENT_COO_INT64 = 6,
};

constexpr int64_t kSegmentCooPackedElementLimit = 2147483647LL;
constexpr int64_t kSegmentCooPackedChannelAlignment = 8;

struct SegmentCooPlan {
    int64_t batches, length, channels, segments, indexBatches;
    SegmentCooReduce reduce;
    SegmentCooDtype dtype;
    int32_t hasOut, blocks;
    int32_t columnShift = 0, segmentShift = 0;
    int32_t usePacked = 0, directIndex = 0;
};

// Called once while constructing the host plan. Allocation and dispatch consume
// these flags together: directIndex means the reduction reads index, not bounds.
inline void ConfigureSegmentCooPaths(SegmentCooPlan &plan) {
    plan.usePacked = plan.channels % kSegmentCooPackedChannelAlignment == 0 &&
                     plan.batches * plan.length * plan.channels < kSegmentCooPackedElementLimit &&
                     plan.batches * plan.segments * plan.channels < kSegmentCooPackedElementLimit;
    plan.directIndex = plan.usePacked && plan.dtype == SEGMENT_COO_INT64 &&
                       plan.reduce == SEGMENT_COO_MEAN;
}
