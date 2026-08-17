/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0.
 * Please refer to the License for details.
 */

#include "gather_coo_kernel_impl.h"
#include "gather_coo_kernel.h"

template <typename T>
void LaunchGatherCooKernel(
    T* src,
    int64_t* index,
    T* out,
    const GatherCooTilingData& tiling,
    aclrtStream stream)
{
    gather_coo_kernel<T><<<tiling.coreNum, nullptr, stream>>>(
        reinterpret_cast<uint8_t*>(src),
        reinterpret_cast<uint8_t*>(index),
        reinterpret_cast<uint8_t*>(out),
        tiling);
}

template void LaunchGatherCooKernel<uint8_t>(
    uint8_t*, int64_t*, uint8_t*, const GatherCooTilingData&, aclrtStream);
template void LaunchGatherCooKernel<uint16_t>(
    uint16_t*, int64_t*, uint16_t*, const GatherCooTilingData&, aclrtStream);
template void LaunchGatherCooKernel<uint32_t>(
    uint32_t*, int64_t*, uint32_t*, const GatherCooTilingData&, aclrtStream);
