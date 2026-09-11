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
#include "common/aiv_launch_utils.h"
#include "ind2ptr/op_kernel/arch35/ind2ptr.h"

// SIMT threads launched per AIV via VF_CALL Dim3.
constexpr uint32_t IND2PTR_SIMT_THREADS = 256;

__simt_vf__ __aicore__ LAUNCH_BOUND(256) void Ind2PtrKernelSimt(__gm__ int64_t* ind, __gm__ int64_t* out,
                                                                int64_t M, int64_t numel)
{
    const int64_t threadNum = AscendC::Simt::GetThreadNum();
    const int64_t blockNum = AscendC::Simt::GetBlockNum();
    const int64_t globalThreadIdx =
        AscendC::Simt::GetBlockIdx() * threadNum + AscendC::Simt::GetThreadIdx();
    const int64_t totalThreadNum = blockNum * threadNum;

    // Logical work indices are [0, numel]. Stride over all AIV SIMT threads when
    // workItems > coreNum * IND2PTR_SIMT_THREADS.
    for (int64_t workIdx = globalThreadIdx; workIdx <= numel; workIdx += totalThreadNum) {
        if (workIdx == 0) {
            for (int64_t i = 0; i <= ind[0]; i++) {
                out[i] = 0;
            }
        } else if (workIdx < numel) {
            for (int64_t i = ind[workIdx - 1]; i < ind[workIdx]; i++) {
                out[i + 1] = workIdx;
            }
        } else {
            for (int64_t i = ind[numel - 1] + 1; i < M + 1; i++) {
                out[i] = numel;
            }
        }
    }
}

__attribute__((aiv)) __global__ __aicore__ void Ind2PtrKernel(__gm__ int64_t* ind, __gm__ int64_t* out,
                                                              int64_t M, int64_t numel)
{
    AscendC::Simt::VF_CALL<Ind2PtrKernelSimt>(AscendC::Simt::Dim3{IND2PTR_SIMT_THREADS}, ind, out, M,
                                              numel);
}

void Ind2Ptr(const int64_t* ind, int64_t* out, int64_t M, int64_t numel, aclrtStream stream)
{
    // AscendC SIMT: each AIV runs IND2PTR_SIMT_THREADS threads; cover workIdx in [0, numel].
    const uint64_t workItems = static_cast<uint64_t>(numel) + 1;
    const uint32_t needCoreNum =
        static_cast<uint32_t>((workItems + IND2PTR_SIMT_THREADS - 1) / IND2PTR_SIMT_THREADS);
    const uint32_t coreNum = ops_gnn::sparse::ResolveAivCoreNum(needCoreNum);

    Ind2PtrKernel<<<coreNum, nullptr, stream>>>(const_cast<int64_t*>(ind), out, M, numel);
}
