/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "spmm_max.h"
#include <algorithm>
#include <limits>
#include <c10/core/DeviceGuard.h>
#include "spmm_max/op_kernel/arch22/spmm_max_kernel.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

void CheckInputTensors(const torch::Tensor& indptr, const torch::Tensor& indices,
                       const torch::Tensor& x)
{
    for (const auto& t : {indptr, indices, x}) {
        TORCH_CHECK(t.defined() && t.device().type() == c10::DeviceType::PrivateUse1,
                    "spmm_max_csr: inputs must be NPU tensors");
        TORCH_CHECK(t.device() == x.device(), "spmm_max_csr: inputs must be on the same NPU");
    }
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kHalf,
                "spmm_max_csr: x must be two-dimensional float16");
    TORCH_CHECK(!x.requires_grad(), "spmm_max_csr: autograd is not supported");
    TORCH_CHECK(indptr.dim() == 1 && indices.dim() == 1 && indptr.numel() >= 1,
                "spmm_max_csr: CSR tensors must be one-dimensional; indptr must be nonempty");
    TORCH_CHECK((indptr.scalar_type() == at::kInt || indptr.scalar_type() == at::kLong) &&
                indices.scalar_type() == indptr.scalar_type(),
                "spmm_max_csr: CSR tensors must have the same int32/int64 dtype");
}

void CheckSparseLimits(int64_t m, int64_t k, int64_t n, int64_t nnz, int64_t indptrNumel)
{
    const int64_t limit = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(m < limit && k <= limit && n > 0 && n <= limit / 2 && nnz <= limit,
                "spmm_max_csr: dimensions exceed uint32 limits or feature_dim is zero");
    TORCH_CHECK(k <= limit / (2 * n) && m <= limit / (2 * n) &&
                indptrNumel <= limit / 4 && nnz <= limit / 4,
                "spmm_max_csr: address byte offsets exceed uint32 limits");
}

void CheckOutTensor(const std::optional<torch::Tensor>& out, const torch::Tensor& x,
                    const torch::Tensor& indptr, const torch::Tensor& indices,
                    int64_t m, int64_t n)
{
    if (!out) {
        return;
    }
    TORCH_CHECK(out->defined() && out->device() == x.device() &&
                out->scalar_type() == at::kHalf && out->dim() == 2 &&
                out->size(0) == m && out->size(1) == n,
                "spmm_max_csr: out must match output shape, dtype and device");
    TORCH_CHECK(out->is_contiguous() && !out->requires_grad(),
                "spmm_max_csr: out must be contiguous and must not require gradients");
    TORCH_CHECK(!out->is_alias_of(x) && !out->is_alias_of(indptr) && !out->is_alias_of(indices),
                "spmm_max_csr: out must not alias any input storage");
}

void CheckCsrValues(const torch::Tensor& indptr, const torch::Tensor& indices,
                    int64_t m, int64_t k, int64_t nnz)
{
    // Only scalar validation results reach the host, never the complete CSR.
    TORCH_CHECK(indptr.select(0, 0).item<int64_t>() == 0 &&
                indptr.select(0, m).item<int64_t>() == nnz,
                "spmm_max_csr: indptr must start at zero and end at nnz");
    TORCH_CHECK(indptr.ge(0).logical_and(indptr.le(nnz)).all().item<bool>(),
                "spmm_max_csr: indptr values out of range");
    if (m) {
        TORCH_CHECK(indptr.slice(0, 1).ge(indptr.slice(0, 0, m)).all().item<bool>(),
                    "spmm_max_csr: indptr must be non-decreasing");
    }
    if (nnz) {
        TORCH_CHECK(indices.ge(0).logical_and(indices.lt(k)).all().item<bool>(),
                    "spmm_max_csr: indices out of range");
    }
}

uint64_t CheckPlatformAndUb(int64_t m, int64_t n, uint32_t& blocks)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "spmm_max_csr: platform information unavailable");
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    // Two accumulation buffers and two feature buffers; reserve pipe overhead.
    TORCH_CHECK(ubBytes > 2048 && ubBytes <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                "spmm_max_csr: invalid platform UB capacity");
    const uint64_t maxRowBytes = ((ubBytes - 2048) / 4 / 32) * 32;
    TORCH_CHECK(static_cast<uint64_t>(n) <= maxRowBytes / 2,
                "spmm_max_csr: feature_dim exceeds UB limit ", maxRowBytes / 2);
    const uint32_t cores = platform->GetCoreNumAiv();
    TORCH_CHECK(cores > 0, "spmm_max_csr: no AIV cores available");
    blocks = static_cast<uint32_t>(std::min<int64_t>(cores, m));
    return ubBytes;
}

}  // namespace

torch::Tensor SpmmMaxCsr(const torch::Tensor& indptr, const torch::Tensor& indices,
                        const torch::Tensor& x, const std::optional<torch::Tensor>& out)
{
    CheckInputTensors(indptr, indices, x);
    c10::DeviceGuard guard(x.device());
    const int64_t m = indptr.numel() - 1, k = x.size(0), n = x.size(1), nnz = indices.numel();
    CheckSparseLimits(m, k, n, nnz, indptr.numel());
    CheckOutTensor(out, x, indptr, indices, m, n);
    CheckCsrValues(indptr, indices, m, k, nnz);
    uint32_t blocks = 0;
    uint64_t ubBytes = CheckPlatformAndUb(m, n, blocks);
    auto result = out ? *out : torch::empty({m, n}, x.options());
    if (m == 0) {
        return result;
    }
    // Generate on the current stream. int32 storage has identical uint32 bits;
    // checked bounds here ensure every split and index is also signed-safe.
    auto split = torch::arange(static_cast<int64_t>(blocks) + 1,
                              indptr.options().dtype(at::kLong));
    split = at::floor_divide(split * m, blocks).to(at::kInt);
    auto ptr = indptr.to(at::kInt).contiguous();
    auto idx = indices.to(at::kInt).contiguous();
    auto features = x.contiguous();
    const uint32_t hasNan = x.numel() && x.isnan().any().item<bool>();
    auto stream = c10_npu::getCurrentNPUStream(x.get_device()).stream();
    LaunchSpmmMax(blocks, stream, features.data_ptr(), result.data_ptr(), ptr.data_ptr(),
                  idx.data_ptr(), split.data_ptr(), m, k, n, nnz, ubBytes, hasNan);
    return result;
}
