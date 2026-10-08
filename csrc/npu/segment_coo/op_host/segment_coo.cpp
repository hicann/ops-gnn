/*
 * Copyright (c) 2026.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "segment_coo.h"
#include "segment_coo/op_kernel/arch35/segment_coo.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include <algorithm>
#include <c10/core/DeviceGuard.h>

namespace {
constexpr uintptr_t kSegmentCooPackedAlignment = 32;

struct SegmentShape {
    int64_t dim;
    int64_t before;
    int64_t after;
    int64_t length;
    bool sharedIndex;
};

void ValidateInputs(const torch::Tensor &src, const torch::Tensor &index, int64_t reduce) {
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "src must be NPU");
    TORCH_CHECK(index.device() == src.device(), "index and src must be on the same NPU");
    TORCH_CHECK(index.scalar_type() == at::kLong, "index must be int64");
    TORCH_CHECK(src.dim() >= 1 && src.dim() <= 8 && index.dim() >= 1 && index.dim() <= src.dim(),
                "invalid rank");
    TORCH_CHECK(reduce >= SEGMENT_COO_SUM && reduce <= SEGMENT_COO_MAX, "invalid reduction");
}

SegmentCooDtype ResolveDtype(at::ScalarType type) {
    switch (type) {
    case at::kFloat:
        return SEGMENT_COO_FLOAT32;
    case at::kHalf:
        return SEGMENT_COO_FLOAT16;
    case at::kBFloat16:
        return SEGMENT_COO_BFLOAT16;
    case at::kChar:
        return SEGMENT_COO_INT8;
    case at::kByte:
        return SEGMENT_COO_UINT8;
    case at::kInt:
        return SEGMENT_COO_INT32;
    case at::kLong:
        return SEGMENT_COO_INT64;
    default:
        TORCH_CHECK(false, "unsupported src dtype");
    }
}

SegmentShape PrepareIndex(const torch::Tensor &src, torch::Tensor &index) {
    SegmentShape shape{index.dim() - 1, 1, 1, src.size(index.dim() - 1), true};
    std::vector<int64_t> indexShape;
    for (int64_t d = 0; d < shape.dim; ++d) {
        TORCH_CHECK(index.size(d) == 1 || index.size(d) == src.size(d), "index broadcast mismatch");
        shape.before *= src.size(d);
        indexShape.push_back(src.size(d));
        shape.sharedIndex = shape.sharedIndex && index.size(d) == 1;
    }
    TORCH_CHECK(index.size(shape.dim) == shape.length,
                "index length must match reduction dimension");
    indexShape.push_back(shape.length);
    for (int64_t d = shape.dim + 1; d < src.dim(); ++d) {
        shape.after *= src.size(d);
    }
    if (!shape.sharedIndex) {
        index = index.expand(indexShape);
    }
    index = index.contiguous();
    return shape;
}

torch::Tensor AlignedContiguous(torch::Tensor value) {
    value = value.contiguous();
    // The widest packed access is 32 bytes; contiguous offset views may be
    // unaligned.
    if (reinterpret_cast<uintptr_t>(value.data_ptr()) % kSegmentCooPackedAlignment != 0) {
        value = value.clone();
    }
    return value;
}

int64_t ResolveSegments(const torch::Tensor &src, const torch::Tensor &index,
                        const std::optional<torch::Tensor> &optionalOut,
                        std::optional<int64_t> dimSize, const SegmentShape &shape) {
    if (optionalOut.has_value()) {
        const auto &out = *optionalOut;
        TORCH_CHECK(out.device() == src.device() && out.scalar_type() == src.scalar_type(),
                    "out device and dtype must match src");
        TORCH_CHECK(out.dim() == src.dim(), "out rank mismatch");
        for (int64_t d = 0; d < src.dim(); ++d) {
            if (d != shape.dim) {
                TORCH_CHECK(out.size(d) == src.size(d), "out shape mismatch");
            }
        }
        return out.size(shape.dim);
    }
    if (dimSize.has_value()) {
        return *dimSize;
    }
    if (index.numel() == 0) {
        return 0;
    }
    return index.select(shape.dim, shape.length - 1).max().item<int64_t>() + 1;
}

SegmentCooPlan MakePlan(const SegmentShape &shape, int64_t segments, SegmentCooDtype dtype,
                       int64_t reduce, bool hasOut) {
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int32_t blocks = std::max<uint32_t>(1, platform->GetCoreNumAiv());
    SegmentCooPlan plan{shape.before,
            shape.length,
            shape.after,
            segments,
            shape.sharedIndex ? 1 : shape.before,
            static_cast<SegmentCooReduce>(reduce),
            dtype,
            hasOut,
            blocks};
    ConfigureSegmentCooPaths(plan);
    return plan;
}

void EnqueueReduction(torch::Tensor src, torch::Tensor index, torch::Tensor out, torch::Tensor arg,
                      const SegmentCooPlan &plan) {
    auto ptr = plan.directIndex
                   ? torch::Tensor()
                   : torch::empty({plan.indexBatches * (plan.segments + 1)}, index.options());
    auto stream = c10_npu::getCurrentNPUStream(src.device().index()).stream(false);
    // Captured tensors retain all temporaries until the queued launch has
    // executed.
    at_npu::native::OpCommand::RunOpApi("segment_coo", [src, index, out, arg, ptr, plan, stream]() {
        SegmentCoo(src.data_ptr(), index.data_ptr<int64_t>(), out.data_ptr(),
                   arg.defined() && arg.numel() ? arg.data_ptr<int64_t>() : nullptr,
                   ptr.defined() ? ptr.data_ptr<int64_t>() : nullptr, plan, stream);
        return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
    });
}
} // namespace

std::tuple<torch::Tensor, torch::Tensor>
segment_coo_forward(torch::Tensor src, torch::Tensor index,
                    std::optional<torch::Tensor> optional_out, std::optional<int64_t> dim_size,
                    int64_t reduce, bool return_arg) {
    ValidateInputs(src, index, reduce);
    c10::DeviceGuard guard(src.device());
    const auto dtype = ResolveDtype(src.scalar_type());
    const auto dims = PrepareIndex(src, index);
    src = AlignedContiguous(src);
    const int64_t segments = ResolveSegments(src, index, optional_out, dim_size, dims);
    TORCH_CHECK(segments >= 0, "dim_size must be non-negative");
    auto shape = src.sizes().vec();
    shape[dims.dim] = segments;
    auto out = optional_out.has_value() ? AlignedContiguous(*optional_out)
                                        : torch::empty(shape, src.options());
    auto arg = reduce >= SEGMENT_COO_MIN && return_arg
                   ? torch::empty(shape, src.options().dtype(at::kLong))
                   : torch::Tensor();
    if (out.numel() == 0) {
        return {optional_out.has_value() ? *optional_out : out, arg};
    }
    const auto plan = MakePlan(dims, segments, dtype, reduce, optional_out.has_value());
    EnqueueReduction(src, index, out, arg, plan);
    if (optional_out.has_value() && out.data_ptr() != optional_out->data_ptr()) {
        optional_out->copy_(out);
        out = *optional_out;
    }
    return {out, arg};
}
