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

// Scalar tiling contract. Grid metadata is passed through a GM config buffer,
// not through this struct, to keep tiling buffers small and aligned.
struct RadiusTilingData {
    int64_t n;
    int64_t m;
    int64_t feature_dim;
    int64_t batch_size;
    int64_t max_num_neighbors;
    float r2;
    int64_t ignore_same_index;
    int64_t core_num;
    int64_t workspace_pairs;
    int64_t use_grid;
    int64_t unique_cells;
    int64_t config_ptr;
};

struct RadiusGridConfig {
    float cell_size;
    float origin[3];
    int64_t dims[3];
};
