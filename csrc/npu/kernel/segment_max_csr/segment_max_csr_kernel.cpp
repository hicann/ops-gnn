/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "segment_max_csr_kernel_impl.h"
#include "segment_max_csr_kernel.h"

template <typename T>
void LaunchSegmentMaxCsrKernel(T* src, int32_t* indptr, T* optional_out, T* out,
                               const SegmentMaxCsrTilingData& tiling, aclrtStream stream)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    segment_max_csr_kernel<T><<<coreNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)indptr, (GM_ADDR)optional_out, (GM_ADDR)out, tiling);
}

template void LaunchSegmentMaxCsrKernel<float>(float* src, int32_t* indptr, float* optional_out, float* out, const SegmentMaxCsrTilingData& tiling, aclrtStream stream);
template void LaunchSegmentMaxCsrKernel<int16_t>(int16_t* src, int32_t* indptr, int16_t* optional_out, int16_t* out, const SegmentMaxCsrTilingData& tiling, aclrtStream stream);
template void LaunchSegmentMaxCsrKernel<int32_t>(int32_t* src, int32_t* indptr, int32_t* optional_out, int32_t* out, const SegmentMaxCsrTilingData& tiling, aclrtStream stream);
template void LaunchSegmentMaxCsrKernel<uint16_t>(uint16_t* src, int32_t* indptr, uint16_t* optional_out, uint16_t* out, const SegmentMaxCsrTilingData& tiling, aclrtStream stream);
