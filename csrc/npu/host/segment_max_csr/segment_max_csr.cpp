/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "segment_max_csr.h"
#include "kernel/segment_max_csr/segment_max_csr_kernel.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel/segment_max_csr/segment_max_csr_tiling.h"
#include <acl/acl_base.h>

static uint32_t GetCeilInt(uint64_t value1, uint64_t value2)
{
    if (value2 == 0) {
        return static_cast<uint32_t>(value1);
    }
    return static_cast<uint32_t>((value1 + value2 - 1) / value2);
}

torch::Tensor segment_max_csr(torch::Tensor src, torch::Tensor indptr, torch::Tensor optional_out)
{
    auto srcDim = src.dim();
    auto indptrDim = indptr.dim();

    uint32_t E_1 = 1;
    for (uint32_t d = 0; d < indptrDim - 1; ++d) {
        E_1 *= src.size(d);
    }

    uint32_t M = src.size(indptrDim - 1);
    uint32_t indptrLastDim = indptr.size(indptrDim - 1);
    uint32_t nSegments = indptrLastDim - 1;

    uint32_t K = 1;
    for (uint32_t d = indptrDim; d < srcDim; ++d) {
        K *= src.size(d);
    }

    uint32_t srcLength = src.numel();
    uint32_t indptrPhysicalNum = indptr.numel();

    uint32_t indptrPreProduct = 1;
    for (uint32_t d = 0; d < indptrDim - 1; ++d) {
        indptrPreProduct *= indptr.size(d);
    }
    uint32_t strideIndptr = (indptrPreProduct == E_1) ? indptrLastDim : 0;

    uint32_t hasOptionalOut = (optional_out.defined() && optional_out.numel() > 0) ? 1 : 0;

    std::vector<int64_t> outShape(src.sizes().begin(), src.sizes().end());
    outShape[indptrDim - 1] = nSegments;
    torch::Tensor out = torch::empty(outShape, src.options());

    uint32_t sizeofdatatype;
    auto dt = src.dtype();
    if (dt == torch::kFloat16 || dt == torch::kInt16) {
        sizeofdatatype = 2;
    } else {
        sizeofdatatype = 4;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivCanUseNum = ascendcPlatform->GetCoreNumAiv();

    uint32_t ALIGN_NUM = 32 / sizeofdatatype;

    uint32_t aivNum = (aivCanUseNum < E_1) ? aivCanUseNum : E_1;
    aivNum = aivNum >= 1 ? aivNum : 1;

    uint32_t rowsPerAiv = GetCeilInt(E_1, aivNum);
    uint32_t rowsLastAiv = (E_1 % aivNum == 0) ? rowsPerAiv : (E_1 % aivNum);

    uint32_t coreDataNum = 2048;
    uint32_t coreTailDataNum = (K % 2048 == 0) ? 2048 : (K % 2048);

    uint32_t KloopTime = GetCeilInt(K, 2048);

    SegmentMaxCsrTilingData tiling;
    tiling.srcLength = srcLength;
    tiling.E_1 = E_1;
    tiling.M = M;
    tiling.nSegments = nSegments;
    tiling.K = K;
    tiling.indptrPhysicalNum = indptrPhysicalNum;
    tiling.coreDataNum = coreDataNum;
    tiling.coreTailDataNum = coreTailDataNum;
    tiling.ALIGN_NUM = ALIGN_NUM;
    tiling.aivNum = aivNum;
    tiling.rowsPerAiv = rowsPerAiv;
    tiling.rowsLastAiv = rowsLastAiv;
    tiling.KloopTime = KloopTime;
    tiling.hasOptionalOut = hasOptionalOut;
    tiling.indptrDimNum = indptrDim;
    tiling.strideIndptr = strideIndptr;

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    auto scalar_type = src.scalar_type();
    if (scalar_type == at::ScalarType::Float) {
        LaunchSegmentMaxCsrKernel(
            src.data_ptr<float>(),
            indptr.data_ptr<int32_t>(),
            hasOptionalOut ? optional_out.data_ptr<float>() : nullptr,
            out.data_ptr<float>(),
            tiling,
            stream
        );
    } else if (scalar_type == at::ScalarType::Half) {
        LaunchSegmentMaxCsrKernel(
            reinterpret_cast<uint16_t*>(src.data_ptr<c10::Half>()),
            indptr.data_ptr<int32_t>(),
            hasOptionalOut ? reinterpret_cast<uint16_t*>(optional_out.data_ptr<c10::Half>()) : nullptr,
            reinterpret_cast<uint16_t*>(out.data_ptr<c10::Half>()),
            tiling,
            stream
        );
    } else if (scalar_type == at::ScalarType::Int) {
        LaunchSegmentMaxCsrKernel(
            src.data_ptr<int32_t>(),
            indptr.data_ptr<int32_t>(),
            hasOptionalOut ? optional_out.data_ptr<int32_t>() : nullptr,
            out.data_ptr<int32_t>(),
            tiling,
            stream
        );
    } else if (scalar_type == at::ScalarType::Short) {
        LaunchSegmentMaxCsrKernel(
            src.data_ptr<int16_t>(),
            indptr.data_ptr<int32_t>(),
            hasOptionalOut ? optional_out.data_ptr<int16_t>() : nullptr,
            out.data_ptr<int16_t>(),
            tiling,
            stream
        );
    } else {
        TORCH_CHECK(false, "segment_max_csr only supports float, half, int32, and int16 types");
    }

    aclrtSynchronizeStream(stream);
    
    // aclrtFree(tilingDev);
    aclrtDestroyStream(stream);

    return out;
}
