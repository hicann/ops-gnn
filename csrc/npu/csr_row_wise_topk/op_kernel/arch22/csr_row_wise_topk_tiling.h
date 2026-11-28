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

#ifndef CSR_ROW_WISE_TOPK_TILING_H
#define CSR_ROW_WISE_TOPK_TILING_H

#include <cstdint>

// Tiling layout for the multi-core CSR row-wise topk kernel.
// Field order must match KernelCsrRowWiseTopk::Init.
constexpr uint32_t kTilingHeaderWordsTopk = 8;

struct CsrRowWiseTopkTiling {
  uint32_t num_rows;        // number of seed rows to pick from
  uint32_t k;               // top-k (0 when select_all)
  uint32_t select_all;      // 1 = k == -1 (pick every edge)
  uint32_t ascending;       // 1 = pick the k smallest (tail of sort)
  uint32_t has_data;        // 1 = CSR data array present (eid mapping)
  uint32_t num_total_rows;  // total rows of the CSR matrix (bounds check)
  uint32_t ub_available;    // per-core UB budget in bytes (runtime query)
  uint32_t reserved;        // padding (must stay 8 words)
};

// Hardware parameters are queried at runtime via aclrtGetDeviceInfo /
// platform_ascendc and passed through the tiling block — never hard-coded.
// The values below are only fallbacks for when the query fails.
constexpr uint32_t kDefaultVectorCoreCountTopk = 40;  // fallback, 910B
constexpr uint32_t kDefaultUbBytesTopk = 192 * 1024;  // fallback, 910B
constexpr uint32_t kUbReservedBytesTopk = 2 * 1024;   // reserved tail

// One Sort call covers repeatTimes * 32 elements with repeatTimes capped at
// 255; the window formula must stay below this bound (launcher CHECK).
constexpr uint32_t kSortMaxElemsPerCallTopk = 255 * 32;

// Device-verified envelope on 910B3: the UB sort path is exact up to
// 992-element windows.
constexpr uint32_t kVerifiedWindowElemsTopk = 992;

#endif  // CSR_ROW_WISE_TOPK_TILING_H
