/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"

__simt_vf__ __aicore__ LAUNCH_BOUND(1024) void AddSampleKernelSimt(__gm__ uint8_t* src1, __gm__ uint8_t* src2, 
                                                                  __gm__ uint8_t* dst, uint32_t valueNum)
{
    uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
    uint32_t blockNumber = AscendC::Simt::GetBlockNum();
    uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();
    for (uint32_t i = globalThreadIdx; i < valueNum; i += totalThreadNum)
    {
        dst[i] = src1[i] + src2[i];
    }
}

__attribute__((aiv)) __global__ __aicore__ void AddSampleKernel(__gm__ uint8_t* src1, __gm__ uint8_t* src2, 
                                                              __gm__ uint8_t* dst, uint32_t valueNum)
{
    AscendC::Simt::VF_CALL<AddSampleKernelSimt>(AscendC::Simt::Dim3{1024}, src1, src2, dst, valueNum);
}

void LaunchAddSampleKernel(uint8_t* src1, uint8_t* src2, uint8_t* dst, uint32_t valueNum, aclrtStream stream)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    AddSampleKernel<<<coreNum, nullptr, stream>>>(src1, src2, dst, valueNum);
}