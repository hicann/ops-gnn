/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_sample.h"
#include "add_sample_kernel.h"

torch::Tensor add_sample(torch::Tensor src1, torch::Tensor src2)
{
    uint32_t valueNum = src1.numel();
    torch::Tensor dst = torch::zeros_like(src1);
    aclrtStream stream = nullptr;

    aclrtCreateStream(&stream);
    LaunchAddSampleKernel(src1.data_ptr<uint8_t>(), src2.data_ptr<uint8_t>(), dst.data_ptr<uint8_t>(), valueNum, stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    return dst;
}
