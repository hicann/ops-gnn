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

#include <acl/acl.h>
#include <cstdint>

struct RandomWalkLaunchParams {
    uint64_t startCount;
    uint32_t walkLength;
    uint64_t seed;
    uint64_t offset;
    uint32_t returnThreshold;
    uint32_t neighborThreshold;
    uint32_t distantThreshold;
    bool node2vec;
    bool writeEdge;
    bool neighborsSorted;
};

void RandomWalk(const int64_t* rowptr, const int64_t* col, const int64_t* start,
                            int64_t* nodeOut, int64_t* edgeOut, const RandomWalkLaunchParams& params,
                            aclrtStream stream);
