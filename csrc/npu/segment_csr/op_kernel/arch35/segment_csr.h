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
 * Ascend C (SIMT) kernel launcher for the segment_csr family on
 * Ascend 950 (arch35 / dav-3510).
 *
 * Canonical shape decomposition (identical to the host layer):
 *   E1 = prod(src.size(0 .. dim-1))       batch rows
 *   M  = src.size(dim)                     reduce-dim length (indexed by indptr)
 *   K  = prod(src.size(dim+1 .. ndim-1))   trailing channels
 *   nSeg = indptr.size(-1) - 1             segments
 * element (e, j, k) of contiguous src lives at ((e * M) + j) * K + k.
 *
 * indptr is int64. indptrStride == 0 means one shared indptr vector of
 * length nSeg+1; indptrStride == nSeg+1 means a per-batch row (E1 rows).
 *
 * Kind dispatch: every runtime `kind` -> compile-time KIND expansion goes
 * through DispatchKind below, so per-dtype launch sites are written once.
 */

#ifndef OPS_GNN_SEGMENT_CSR_KERNEL_H
#define OPS_GNN_SEGMENT_CSR_KERNEL_H

#include <cstdint>
#include <acl/acl_base.h>

namespace ops_gnn {

// Kernel-side reduce selector (values must stay in sync with the host layer).
enum SegmentCsrReduce {
    SEGMENT_CSR_SUM = 0,
    SEGMENT_CSR_MIN = 1,
    SEGMENT_CSR_MAX = 2,
};

// Element kinds handled by one templated kernel body.
enum SegmentCsrKind {
    KIND_F32 = 0,  // float
    KIND_F16 = 1,  // IEEE half stored as uint16_t
    KIND_BF16 = 2, // bfloat16 stored as uint16_t
    KIND_I8 = 3,
    KIND_U8 = 4,
    KIND_I32 = 5,
    KIND_I64 = 6,
};

// Compile-time kind tag passed by DispatchKind to the instantiation functor.
template <int KIND>
struct KindConstant {
    static constexpr int value = KIND;
};

// Single kind -> template dispatcher. `func` must be callable with
// KindConstant<K>{} for every kind; unsupported kinds return false.
// Every launcher uses this instead of per-dtype switch-case expansion.
template <typename Func>
inline bool DispatchKind(int kind, Func&& func)
{
    switch (kind) {
        case KIND_F32:
            func(KindConstant<KIND_F32>{});
            return true;
        case KIND_F16:
            func(KindConstant<KIND_F16>{});
            return true;
        case KIND_BF16:
            func(KindConstant<KIND_BF16>{});
            return true;
        case KIND_I8:
            func(KindConstant<KIND_I8>{});
            return true;
        case KIND_U8:
            func(KindConstant<KIND_U8>{});
            return true;
        case KIND_I32:
            func(KindConstant<KIND_I32>{});
            return true;
        case KIND_I64:
            func(KindConstant<KIND_I64>{});
            return true;
        default:
            return false;
    }
}

// Runtime mirror of the kernel-side SegTraits facts (single source for the
// host layer): vector lane count for the 16-byte packet path, and the byte
// width of the register-engine dtypes. Both return 0 / keep the scalar path
// for kinds without that path.
inline int KindVecLanes(int kind)
{
    switch (kind) {
        case KIND_F32:
        case KIND_I32:
            return 4;
        case KIND_F16:
            return 8;
        case KIND_I64:
            return 2;
        default:
            return 0;
    }
}

inline int KindRegisterItemSize(int kind)
{
    switch (kind) {
        case KIND_F32:
        case KIND_I32:
            return 4;
        case KIND_F16:
            return 2;
        case KIND_I64:
            return 8;
        default:
            return 0; // no register-engine path: host keeps the SIMT launch
    }
}

// Launches the SIMT kernel on `stream`. Template arguments are resolved by the
// host layer: vecElems > 0 selects the 16-byte vectorized path (requires
// K % vecElems == 0), vecElems == 0 the scalar path.
void SegmentCsr(
    int kind, int vecElems, const void* src, const int64_t* indptr, void* out,
    int64_t* argOut, // nullptr unless reduce is min/max
    int reduce,      // SegmentCsrReduce
    int isMean,      // finalize SUM as mean
    uint64_t e1, uint64_t m, uint32_t nSeg, uint32_t k,
    uint64_t indptrStride, // 0 = shared, nSeg+1 = per batch
    aclrtStream stream);

// AIV vector-engine path for the perf orientation (E1 == 1, shared indptr,
// aligned K, dtype float32/float16/int32/int64). See segment_csr_vec_kernel.cpp.
void SegmentCsrVector(
    int kind, const void* src, const int64_t* indptr, void* out, int64_t* argOut,
    int reduce, // 0 sum, 1 min, 2 max
    uint64_t m, uint32_t nSeg, uint32_t k, aclrtStream stream,
    int isMeanFp = 0); // fp means divide inside the kernel

} // namespace ops_gnn

#endif // OPS_GNN_SEGMENT_CSR_KERNEL_H
