/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "csr_row_wise_topk.h"

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "csr_row_wise_topk/op_kernel/arch22/csr_row_wise_topk.h"
#include "csr_row_wise_topk/op_kernel/arch22/csr_row_wise_topk_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {
namespace csr_topk {
namespace {

void CheckNpuTensor(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), "csr_row_wise_topk: ", name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "csr_row_wise_topk: ", name, " must be an NPU tensor");
    TORCH_CHECK(tensor.dim() == 1, "csr_row_wise_topk: ", name, " must be 1-D");
}

std::vector<uint32_t> BuildBalancedPartitions(
    const std::vector<uint32_t>& weights, uint32_t numParts)
{
    std::vector<uint32_t> boundaries(numParts + 1, 0);
    if (numParts == 0) {
        return boundaries;
    }
    const uint32_t itemCount = static_cast<uint32_t>(weights.size());
    boundaries[numParts] = itemCount;
    if (itemCount == 0) {
        return boundaries;
    }
    if (itemCount <= numParts) {
        for (uint32_t i = 0; i <= itemCount; ++i) {
            boundaries[i] = i;
        }
        for (uint32_t i = itemCount + 1; i <= numParts; ++i) {
            boundaries[i] = itemCount;
        }
        return boundaries;
    }
    std::vector<double> prefix(itemCount + 1, 0.0);
    for (uint32_t i = 0; i < itemCount; ++i) {
        prefix[i + 1] = prefix[i] + weights[i];
    }
    const double totalWeight = prefix[itemCount];
    if (totalWeight <= 0.0) {
        for (uint32_t part = 1; part < numParts; ++part) {
            boundaries[part] = part * itemCount / numParts;
        }
        return boundaries;
    }
    for (uint32_t part = 1; part < numParts; ++part) {
        const double target = totalWeight * part / numParts;
        auto it = std::lower_bound(prefix.begin(), prefix.end(), target);
        boundaries[part] = static_cast<uint32_t>(it - prefix.begin());
    }
    for (uint32_t part = 1; part < numParts; ++part) {
        if (boundaries[part] < boundaries[part - 1]) {
            boundaries[part] = boundaries[part - 1];
        }
        if (boundaries[part] > itemCount) {
            boundaries[part] = itemCount;
        }
    }
    return boundaries;
}

template <typename T>
torch::Tensor Uint32ToTensor(const std::vector<uint32_t>& values, const torch::Tensor& like)
{
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto cpu = torch::from_blob(const_cast<uint32_t*>(values.data()),
        {static_cast<int64_t>(values.size())}, options).clone();
    return cpu.to(like.options().dtype(c10::CppTypeToScalarType<T>::value));
}

torch::Tensor MakeTiling(const std::array<uint32_t, kTilingHeaderWordsTopk>& values,
    const torch::Tensor& like)
{
    auto cpu = torch::from_blob(const_cast<uint32_t*>(values.data()),
        {static_cast<int64_t>(values.size())},
        torch::TensorOptions().dtype(torch::kInt32)).clone();
    return cpu.to(like.device());
}

struct TopkLaunchConfig {
    uint32_t blocks = 1;
    uint32_t ubAvailable = 0;
    bool isInt32 = false;
    aclrtStream stream = nullptr;
};

TopkLaunchConfig ResolveTopkLaunchConfig(const torch::Tensor& indptr)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr,
                "csr_row_wise_topk: failed to initialize AscendC platform");
    TopkLaunchConfig cfg;
    cfg.blocks = std::max<uint32_t>(platform->GetCoreNumAiv(), 1);
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    cfg.ubAvailable = ubBytes > kUbReservedBytesTopk
        ? static_cast<uint32_t>(ubBytes - kUbReservedBytesTopk) : 0;
    TORCH_CHECK(cfg.ubAvailable > 0, "csr_row_wise_topk: invalid UB size");
    cfg.isInt32 = indptr.scalar_type() == at::ScalarType::Int;
    cfg.stream = c10_npu::getCurrentNPUStream(indptr.get_device()).stream();
    return cfg;
}

struct TopkRowPlan {
    std::vector<uint32_t> rowSplit;
    std::vector<uint32_t> outStarts;
    std::vector<uint32_t> degrees;
    int64_t maxOutput = 0;
};

template <typename IdType>
TopkRowPlan BuildTopkRowPlan(const torch::Tensor& indptr, const torch::Tensor& rows,
    int64_t k, uint32_t blocks)
{
    const bool selectAll = k == -1;
    const int64_t numRows = rows.numel();
    const int64_t numTotalRows = indptr.numel() - 1;
    auto indptrCpu = indptr.to(torch::kCPU).contiguous();
    auto rowsCpu = rows.to(torch::kCPU).contiguous();
    const IdType* indptrPtr = indptrCpu.data_ptr<IdType>();
    const IdType* rowsPtr = rowsCpu.data_ptr<IdType>();
    const uint32_t k32 = selectAll ? 0u : static_cast<uint32_t>(k);
    std::vector<uint32_t> picks(static_cast<size_t>(numRows));
    std::vector<uint32_t> degrees(static_cast<size_t>(numRows));
    std::vector<uint64_t> prefix(numRows + 1, 0);
    for (int64_t i = 0; i < numRows; ++i) {
        const IdType r = rowsPtr[i];
        uint32_t d = 0;
        if (r >= 0 && static_cast<int64_t>(r) < numTotalRows) {
            d = static_cast<uint32_t>(indptrPtr[static_cast<int64_t>(r) + 1] -
                                      indptrPtr[static_cast<int64_t>(r)]);
        }
        degrees[static_cast<size_t>(i)] = d;
        picks[static_cast<size_t>(i)] = selectAll ? d : std::min(k32, d);
        prefix[static_cast<size_t>(i) + 1] =
            prefix[static_cast<size_t>(i)] + picks[static_cast<size_t>(i)];
    }
    TORCH_CHECK(prefix.back() <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                "csr_row_wise_topk: output size exceeds int32 range");

    TopkRowPlan plan;
    plan.maxOutput = static_cast<int64_t>(prefix.back());
    plan.degrees = std::move(degrees);
    plan.rowSplit = BuildBalancedPartitions(picks, blocks);
    plan.outStarts.assign(static_cast<size_t>(blocks) + 1, 0);
    for (uint32_t b = 0; b <= blocks; ++b) {
        plan.outStarts[static_cast<size_t>(b)] = static_cast<uint32_t>(prefix[plan.rowSplit[b]]);
    }
    return plan;
}

template <typename IdType>
uint32_t ResolveTopkWindow(uint32_t ubAvailable, const std::vector<uint32_t>& degrees)
{
    const uint32_t bytesPerElem = 48u + 7u * static_cast<uint32_t>(sizeof(IdType));
    uint32_t window = std::min<uint32_t>(ubAvailable / bytesPerElem,
                                         kSortMaxElemsPerCallTopk);
    window = std::min(window, kVerifiedWindowElemsTopk);
    window = window / 32u * 32u;
    TORCH_CHECK(window >= 32, "csr_row_wise_topk: UB budget too small for the topk window");
    for (uint32_t d : degrees) {
        TORCH_CHECK(d <= window,
                    "csr_row_wise_topk: select_topk on Ascend does not yet support rows wider "
                    "than the UB window (", d, " > ", window, ")");
    }
    return window;
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RunTopk(const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::Tensor& rows, int64_t k, const torch::Tensor& weight,
    bool ascending, const c10::optional<torch::Tensor>& data)
{
    auto options = indptr.options();
    auto empty = torch::empty({0}, options);
    const bool selectAll = k == -1;
    const int64_t numRows = rows.numel();
    const int64_t numTotalRows = indptr.numel() - 1;
    if (numRows == 0 || numTotalRows <= 0 || k == 0) {
        return std::make_tuple(empty, torch::empty({0}, options), torch::empty({0}, options));
    }

    const auto cfg = ResolveTopkLaunchConfig(indptr);
    const auto plan = BuildTopkRowPlan<IdType>(indptr, rows, k, cfg.blocks);
    ResolveTopkWindow<IdType>(cfg.ubAvailable, plan.degrees);
    const bool hasData = data.has_value();
    void* dataPtr = hasData ? data.value().data_ptr() : nullptr;
    const uint32_t k32 = selectAll ? 0u : static_cast<uint32_t>(k);
    std::array<uint32_t, kTilingHeaderWordsTopk> tiling = {
        static_cast<uint32_t>(numRows), k32,
        static_cast<uint32_t>(selectAll ? 1 : 0),
        static_cast<uint32_t>(ascending ? 1 : 0),
        static_cast<uint32_t>(hasData ? 1 : 0),
        static_cast<uint32_t>(numTotalRows), cfg.ubAvailable, 0u,
    };
    if (plan.maxOutput == 0) {
        return std::make_tuple(empty, torch::empty({0}, options), torch::empty({0}, options));
    }

    auto rowSplit = Uint32ToTensor<uint32_t>(plan.rowSplit, indptr);
    auto outStartsTensor = Uint32ToTensor<uint32_t>(plan.outStarts, indptr);
    auto outRows = torch::empty({plan.maxOutput}, options);
    auto outCols = torch::empty({plan.maxOutput}, options);
    auto outData = torch::empty({plan.maxOutput}, options);
    auto tilingDev = MakeTiling(tiling, indptr);
    TORCH_CHECK(aclrtSynchronizeDevice() == ACL_SUCCESS,
                "csr_row_wise_topk: pre-launch sync failed");
    const aclError launchStatus = LaunchTopk(cfg.isInt32, cfg.blocks, cfg.stream,
        indptr.data_ptr(), indices.data_ptr(),
        dataPtr, rows.data_ptr(), weight.data_ptr(), outRows.data_ptr(),
        outCols.data_ptr(), outData.data_ptr(), rowSplit.data_ptr(),
        outStartsTensor.data_ptr(), tilingDev.data_ptr());
    TORCH_CHECK(launchStatus == ACL_SUCCESS,
                "csr_row_wise_topk: kernel launch failed, aclError=",
                static_cast<int>(launchStatus));
    TORCH_CHECK(aclrtSynchronizeStream(cfg.stream) == ACL_SUCCESS,
                "csr_row_wise_topk: topk sync failed");
    return std::make_tuple(outRows, outCols, outData);
}

}  // namespace
}  // namespace csr_topk

using namespace csr_topk;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CsrRowWiseTopk(const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::Tensor& rows, int64_t k, const torch::Tensor& weight,
    bool ascending, const c10::optional<torch::Tensor>& data)
{
    CheckNpuTensor(indptr, "indptr");
    CheckNpuTensor(indices, "indices");
    CheckNpuTensor(rows, "rows");
    CheckNpuTensor(weight, "weight");
    TORCH_CHECK(indptr.scalar_type() == at::ScalarType::Int ||
                indptr.scalar_type() == at::ScalarType::Long,
                "csr_row_wise_topk: indptr dtype must be int32 or int64");
    TORCH_CHECK(indices.scalar_type() == indptr.scalar_type() &&
                rows.scalar_type() == indptr.scalar_type(),
                "csr_row_wise_topk: indptr/indices/rows dtypes must match");
    TORCH_CHECK(indptr.device() == indices.device() &&
                indptr.device() == rows.device() &&
                indptr.device() == weight.device(),
                "csr_row_wise_topk: all inputs must be on the same NPU");
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float,
                "csr_row_wise_topk: weight must be float32 (normalize on the Python side)");
    TORCH_CHECK(weight.numel() >= indices.numel(),
                "csr_row_wise_topk: weight length must cover the matrix nnz");
    TORCH_CHECK(k >= -1, "csr_row_wise_topk: k must be >= -1");
    TORCH_CHECK(indptr.numel() >= 1,
                "csr_row_wise_topk: indptr must have at least one element");

    if (data.has_value()) {
        CheckNpuTensor(data.value(), "data");
        TORCH_CHECK(data.value().scalar_type() == indptr.scalar_type(),
                    "csr_row_wise_topk: data dtype must match indptr");
        TORCH_CHECK(data.value().device() == indptr.device(),
                    "csr_row_wise_topk: data must be on the same NPU");
        TORCH_CHECK(data.value().numel() == indices.numel(),
                    "csr_row_wise_topk: data length must match indices");
    }

    if (indptr.scalar_type() == at::ScalarType::Int) {
        return RunTopk<int32_t>(indptr, indices, rows, k, weight, ascending, data);
    }
    return RunTopk<int64_t>(indptr, indices, rows, k, weight, ascending, data);
}

}  // namespace opsgnn
