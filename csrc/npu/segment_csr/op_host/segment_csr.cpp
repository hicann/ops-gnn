/*
 * Copyright (c) 2026 Starlink_. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 *
 * Host layer of the segment_csr family (torch_scatter-compatible) for NPU.
 *
 * Canonical decomposition of contiguous src with dim = indptr.dim() - 1:
 *   E1 = prod(src.size(0 .. dim-1))   batch rows
 *   M  = src.size(dim)                reduce-dim length (indexed by indptr)
 *   K  = prod(src.size(dim+1 ..))     trailing channels
 *   nSeg = indptr.size(-1) - 1
 * indptr is INT64 (task-book requirement); its leading dims broadcast to
 * src's leading dims. All leading dims equal to 1 share one indptr vector
 * (stride 0), otherwise indptr is expanded+materialized per batch row.
 *
 * Launches on the CURRENT NPU stream and never synchronizes: the wall-clock
 * benchmark in the task book times the full interface call.
 *
 * dtype facts (register-path item size, vector lane count) come from the
 * kernel header's KindRegisterItemSize / KindVecLanes so this layer never
 * duplicates them.
 */

#include "segment_csr.h"

#include <c10/util/Optional.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/irange.h>
#include <limits>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include "npu/segment_csr/op_kernel/arch35/segment_csr.h"

namespace ops_gnn {
namespace {

struct SegDims {
    int64_t dim;
    int64_t e1;
    int64_t m;
    int64_t k;
    int64_t nSeg;
    int64_t indptrLast;
};

SegDims ComputeDims(const torch::Tensor& src, const torch::Tensor& indptr)
{
    SegDims d;
    d.dim = indptr.dim() - 1;
    d.e1 = 1;
    for (const auto i : c10::irange(d.dim)) {
        d.e1 *= src.size(i);
    }
    d.m = src.size(d.dim);
    d.indptrLast = indptr.size(d.dim);
    d.nSeg = d.indptrLast - 1;
    d.k = 1;
    for (const auto i : c10::irange(static_cast<int64_t>(d.dim + 1), src.dim())) {
        d.k *= src.size(i);
    }
    return d;
}

void Validate(const torch::Tensor& src, const torch::Tensor& indptr, const SegDims& d)
{
    TORCH_CHECK(indptr.scalar_type() == at::kLong, "indptr must be int64, got ", indptr.scalar_type());
    TORCH_CHECK(indptr.dim() >= 1, "indptr must have at least 1 dim");
    TORCH_CHECK(src.dim() >= indptr.dim(), "src.dim() (", src.dim(), ") must be >= indptr.dim() (", indptr.dim(), ")");
    TORCH_CHECK(d.indptrLast >= 1, "indptr last dim must be >= 1");
    for (const auto i : c10::irange(d.dim)) {
        TORCH_CHECK(
            indptr.size(i) == 1 || indptr.size(i) == src.size(i), "indptr leading dim ", i, " (", indptr.size(i),
            ") must be 1 or match src (", src.size(i), ")");
    }
    TORCH_CHECK(
        src.device().type() == c10::DeviceType::PrivateUse1, "segment_csr(NPU) requires NPU tensors, got ",
        src.device());
    TORCH_CHECK(indptr.device() == src.device(), "src and indptr must live on the same device");
}

int KindFor(const torch::Tensor& src)
{
    switch (src.scalar_type()) {
        case at::kFloat:
            return KIND_F32;
        case at::kHalf:
            return KIND_F16;
        case at::kBFloat16:
            return KIND_BF16;
        case at::kChar:
            return KIND_I8;
        case at::kByte:
            return KIND_U8;
        case at::kInt:
            return KIND_I32;
        case at::kLong:
            return KIND_I64;
        default:
            TORCH_CHECK(false, "segment_csr unsupported dtype: ", src.scalar_type());
            return -1;
    }
}

// 16-byte vector lane count for the perf dtypes; 0 keeps the scalar path.
// Lane counts live in KindVecLanes (kernel header, single source).
int VecElemsFor(int kind, uint64_t k)
{
    const int lanes = KindVecLanes(kind);
    return lanes > 0 && k % static_cast<uint64_t>(lanes) == 0 ? lanes : 0;
}


struct ReductionConfig {
    int op;
    int isMean;
};

ReductionConfig ParseReduction(const std::string& reduce)
{
    if (reduce == "sum" || reduce == "add") {
        return {SEGMENT_CSR_SUM, 0};
    }
    if (reduce == "mean") {
        return {SEGMENT_CSR_SUM, 1};
    }
    if (reduce == "min") {
        return {SEGMENT_CSR_MIN, 0};
    }
    TORCH_CHECK(reduce == "max", "reduce value '", reduce, "' is not valid (use sum, add, mean, min or max)");
    return {SEGMENT_CSR_MAX, 0};
}

torch::Tensor PreparePointers(
    const torch::Tensor& src, const torch::Tensor& indptr, const SegDims& dims, uint64_t& stride)
{
    auto result = indptr.is_contiguous() ? indptr : indptr.contiguous();
    bool shared = true;
    for (const auto i : c10::irange(dims.dim)) {
        if (indptr.size(i) != 1) {
            shared = false;
            break;
        }
    }
    stride = 0;
    if (!shared) {
        std::vector<int64_t> shape;
        for (const auto i : c10::irange(dims.dim)) {
            shape.push_back(src.size(i));
        }
        shape.push_back(dims.indptrLast);
        result = result.expand(shape).contiguous();
        stride = static_cast<uint64_t>(dims.indptrLast);
    }
    return result;
}

torch::Tensor PrepareOutput(
    const torch::Tensor& src, const std::vector<int64_t>& shape, const c10::optional<torch::Tensor>& outOpt)
{
    if (!outOpt.has_value() || !outOpt->defined()) {
        return torch::empty(shape, src.options());
    }
    const auto& out = *outOpt;
    TORCH_CHECK(out.scalar_type() == src.scalar_type(), "out dtype (", out.scalar_type(),
        ") must match src (", src.scalar_type(), ")");
    TORCH_CHECK(out.sizes() == c10::IntArrayRef(shape), "out shape ", out.sizes(),
        " must be ", c10::IntArrayRef(shape));
    TORCH_CHECK(out.device() == src.device(), "out must live on src's device");
    return out.is_contiguous() ? out : out.contiguous();
}

std::vector<torch::Tensor> ResultTensors(torch::Tensor out, torch::Tensor argOut, int op)
{
    if (op != SEGMENT_CSR_SUM) {
        return {out, argOut};
    }
    return {out};
}

std::vector<torch::Tensor> EmptyOutput(
    torch::Tensor out, torch::Tensor argOut, const c10::optional<torch::Tensor>& outOpt,
    const SegDims& dims, int op)
{
    if (!outOpt.has_value() || !outOpt->defined()) {
        out.zero_();
    }
    if (op != SEGMENT_CSR_SUM) {
        argOut.fill_(dims.m);
    }
    if (outOpt.has_value() && outOpt->defined() && !outOpt->is_contiguous()) {
        out = *outOpt;
    }
    return ResultTensors(out, argOut, op);
}

bool UseRegisterKernel(int kind, int op, const SegDims& dims, uint64_t stride)
{
    const uint32_t itemSize = static_cast<uint32_t>(KindRegisterItemSize(kind));
    const uint64_t rowBytes = static_cast<uint64_t>(dims.k) * itemSize;
    const bool minMaxAligned = op == SEGMENT_CSR_SUM || itemSize == 2 || itemSize == 4 || rowBytes % 256 == 0;
    return itemSize != 0 && dims.e1 == 1 && stride == 0 && dims.m != 0 && minMaxAligned &&
        rowBytes % 32 == 0 && dims.k <= 1024 && dims.m <= 0x7FFFFFFFll;
}

void LaunchReduction(
    const torch::Tensor& src, const torch::Tensor& ptr, torch::Tensor& out, torch::Tensor& argOut,
    const SegDims& dims, uint64_t stride, int kind, const ReductionConfig& reduction)
{
    const auto e1 = static_cast<uint64_t>(dims.e1);
    const auto m = static_cast<uint64_t>(dims.m);
    const auto nSeg = static_cast<uint32_t>(dims.nSeg);
    const auto k = static_cast<uint32_t>(dims.k);
    if (e1 == 0 || nSeg == 0 || k == 0) {
        return;
    }
    aclrtStream stream = c10_npu::getCurrentNPUStream(src.get_device()).stream();
    const void* srcPtr = src.const_data_ptr();
    const int64_t* indptrPtr = ptr.const_data_ptr<int64_t>();
    void* outPtr = out.mutable_data_ptr();
    int64_t* argPtr = argOut.defined() ? argOut.data_ptr<int64_t>() : nullptr;
    if (UseRegisterKernel(kind, reduction.op, dims, stride)) {
        SegmentCsrVector(
            kind, srcPtr, indptrPtr, outPtr, argPtr, reduction.op, m, nSeg, k, stream, reduction.isMean);
        return;
    }
    const int vecElems = VecElemsFor(kind, k);
    const uint64_t laneCount = vecElems > 0 ? k / static_cast<uint64_t>(vecElems) : k;
    const uint64_t total = e1 * nSeg * laneCount;
    TORCH_CHECK(total < 0xFFFFFFFFull, "segment_csr work items exceed 2^32 (unsupported shape)");
    SegmentCsr(kind, vecElems, srcPtr, indptrPtr, outPtr, argPtr, reduction.op,
        reduction.isMean, e1, m, nSeg, k, stride, stream);
}

} // namespace

std::vector<torch::Tensor> segment_csr(
    torch::Tensor src, torch::Tensor indptr, c10::optional<torch::Tensor> outOpt, const std::string& reduce)
{
    const auto reduction = ParseReduction(reduce);
    TORCH_CHECK(indptr.dim() >= 1, "indptr must have at least 1 dim");
    TORCH_CHECK(src.dim() >= indptr.dim(), "src.dim() must be >= indptr.dim()");
    const SegDims dims = ComputeDims(src, indptr);
    Validate(src, indptr, dims);
    TORCH_CHECK(src.layout() == at::kStrided && indptr.layout() == at::kStrided,
        "src and indptr must have strided layout");
    TORCH_CHECK(dims.nSeg < std::numeric_limits<uint32_t>::max() && dims.k <= std::numeric_limits<uint32_t>::max(),
        "segment count or channel count exceeds the kernel limit");
    const int kind = KindFor(src);
    const c10::DeviceGuard deviceGuard(src.device());
    torch::Tensor srcC = src.is_contiguous() ? src : src.contiguous();
    uint64_t stride = 0;
    torch::Tensor indptrC = PreparePointers(src, indptr, dims, stride);
    std::vector<int64_t> outShape(src.sizes().begin(), src.sizes().end());
    outShape[dims.dim] = dims.nSeg;
    torch::Tensor out = PrepareOutput(src, outShape, outOpt);
    torch::Tensor argOut;
    if (reduction.op != SEGMENT_CSR_SUM) {
        argOut = torch::empty(outShape, src.options().dtype(at::kLong));
    }
    if (src.numel() == 0) {
        return EmptyOutput(out, argOut, outOpt, dims, reduction.op);
    }
    LaunchReduction(srcC, indptrC, out, argOut, dims, stride, kind, reduction);
    if (outOpt.has_value() && outOpt->defined() && !outOpt->is_contiguous()) {
        outOpt->copy_(out);
        out = *outOpt;
    }
    return ResultTensors(out, argOut, reduction.op);
}

} // namespace ops_gnn
