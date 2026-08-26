/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graclus_cluster.h"
#include "graclus_cluster/op_kernel/arch35/graclus_cluster_kernel.h"
#include <acl/acl_base.h>
#include <limits>
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {
void CheckTensor(const torch::Tensor& tensor, const char* name, at::ScalarType dtype)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has invalid dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}
}

torch::Tensor graclus_cluster_npu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor weight,
                                  torch::Tensor node_perm, int64_t num_nodes, bool has_weight,
                                  int64_t weight_mode)
{
    TORCH_CHECK(num_nodes >= 0, "num_nodes must be non-negative");
    // Kernel launch passes num_nodes as uint32_t. Reject overflow before casting.
    TORCH_CHECK(num_nodes <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "num_nodes exceeds uint32_t kernel launch limit");
    CheckTensor(rowptr, "rowptr", at::ScalarType::Long);
    CheckTensor(col, "col", at::ScalarType::Long);
    CheckTensor(node_perm, "node_perm", at::ScalarType::Long);
    if (has_weight) {
        CheckTensor(weight, "weight", at::ScalarType::Float);
    }
    TORCH_CHECK(rowptr.numel() == num_nodes + 1, "rowptr length must be num_nodes + 1");
    TORCH_CHECK(node_perm.numel() == num_nodes, "node_perm length must be num_nodes");
    if (has_weight) {
        TORCH_CHECK(weight.numel() == col.numel(), "weight length must match col length");
    }
    TORCH_CHECK(weight_mode >= 0 && weight_mode <= 3, "weight_mode must be 0, 1, 2, or 3");

    torch::Tensor cluster = torch::empty({num_nodes}, rowptr.options());
    aclrtStream stream = c10_npu::getCurrentNPUStream(rowptr.device().index()).stream();
    LaunchGraclusClusterKernel(
        rowptr.data_ptr<int64_t>(),
        col.data_ptr<int64_t>(),
        has_weight ? weight.data_ptr<float>() : nullptr,
        node_perm.data_ptr<int64_t>(),
        cluster.data_ptr<int64_t>(),
        static_cast<uint32_t>(num_nodes),
        has_weight ? 1U : 0U,
        static_cast<uint32_t>(weight_mode),
        stream);
    aclError ret = aclrtSynchronizeStream(stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed, error code: ", ret);
    return cluster;
}
