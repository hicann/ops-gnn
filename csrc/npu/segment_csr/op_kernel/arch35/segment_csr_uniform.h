/*
 * Copyright (c) 2026 Starlink_. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"

namespace ops_gnn {
namespace uniform_ptr {
namespace R = AscendC::Reg;

// The caller bounds the affine sequence to INT32_MAX and provides a pointer
// allocation rounded up to 64 int64 entries, plus a 256-byte flag allocation.
__simd_vf__ inline void ValidateUniformPtr(
    __ubuf__ int64_t* ptr, __ubuf__ int32_t* flag, uint32_t count, uint32_t base, uint32_t length)
{
    auto fullMask = R::CreateMask<int32_t, R::MaskPattern::ALL>();
    R::RegTensor<int32_t> low, high, expected, invalid, one, reduced;
    R::Duplicate(invalid, int32_t(0), fullMask);
    R::Duplicate(one, int32_t(1), fullMask);
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += 64) {
        auto mask = R::UpdateMask<int32_t>(remaining);
        R::LoadAlign<int32_t, R::LoadDist::DIST_DINTLV_B32>(low, high, (__ubuf__ int32_t*)(ptr + offset));
        R::Arange(expected, static_cast<int32_t>(offset));
        R::Muls(expected, expected, static_cast<int32_t>(length), mask);
        R::Adds(expected, expected, static_cast<int32_t>(base), mask);
        R::MaskReg lowBad, highBad, bad;
        R::Compare<int32_t, AscendC::CMPMODE::NE>(lowBad, low, expected, mask);
        R::Compares<int32_t, AscendC::CMPMODE::NE>(highBad, high, int32_t(0), mask);
        R::Or(bad, lowBad, highBad, fullMask);
        R::Select(invalid, one, invalid, bad);
    }
    R::Reduce<R::ReduceType::MAX>(reduced, invalid, fullMask);
    auto firstMask = R::CreateMask<int32_t, R::MaskPattern::VL1>();
    R::StoreAlign(flag, reduced, firstMask);
}

} // namespace uniform_ptr
} // namespace ops_gnn
