/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "nearest.h"

#include "kernel_operator.h"
#include "simt_api/asc_fp16.h"
#include "nearest_simd.h"

namespace {
constexpr uint32_t THREADS = 512;
constexpr uint32_t WARP = 32;

template <typename T>
__simt_callee__ __aicore__ inline float ToFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__simt_callee__ __aicore__ inline float ToFloat<uint16_t>(uint16_t value)
{
    return __half2float(__ushort_as_half(value));
}

// Convert FP16 exactly to FP32 and transpose to [F, N] / [F, M]. Every
// nearest warp then reads consecutive y points for one feature.
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) void Prepare(
    __gm__ T* x, __gm__ T* y, __gm__ float* work, NearestTiling t)
{
    uint64_t nx = static_cast<uint64_t>(t.n) * t.features;
    uint64_t total = nx + static_cast<uint64_t>(t.m) * t.features;
    uint64_t tid = AscendC::Simt::GetBlockIdx() * THREADS + AscendC::Simt::GetThreadIdx();
    uint64_t step = AscendC::Simt::GetBlockNum() * THREADS;
    for (uint64_t p = tid; p < total; p += step) {
        if (p < nx) {
            work[(p % t.features) * t.n + p / t.features] = ToFloat(x[p]);
        } else {
            uint64_t q = p - nx;
            work[nx + (q % t.features) * t.m + q / t.features] = ToFloat(y[q]);
        }
    }
    // Both input dtypes share the FP32 distance path, including these norms.
    if (t.features >= 5) {
        for (uint64_t row = tid; row < static_cast<uint64_t>(t.n) + t.m; row += step) {
            float norm = 0.0f;
            for (uint32_t d = 0; d < t.features; ++d) {
                float v = row < t.n ? ToFloat(x[row * t.features + d]) :
                    ToFloat(y[(row - t.n) * t.features + d]);
                volatile float square = v * v;
                norm = norm + square;
            }
            work[total + row] = norm;
        }
    }
}

__simt_callee__ __aicore__ inline uint32_t FindBatch(
    uint32_t row, __gm__ int64_t* ptr, uint32_t batches)
{
    uint32_t lo = 0;
    uint32_t hi = batches;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (ptr[mid + 1] <= row) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

struct BatchRange {
    uint32_t begin;
    uint32_t end;
    uint32_t x_begin;
    uint32_t x_count;
};

__simt_callee__ __aicore__ inline bool GetBatchRange(
    uint32_t row, __gm__ int64_t* ptr_x, __gm__ int64_t* ptr_y,
    NearestTiling t, BatchRange& range)
{
    if (t.batches == 0) return true;
    uint32_t batch = FindBatch(row, ptr_x, t.batches);
    // The Python API validates pointers. Keep the low-level dispatcher bounds checks.
    if (batch >= t.batches) return false;
    int64_t a = ptr_y[batch];
    int64_t b = ptr_y[batch + 1];
    if (a < 0 || b <= a || b > t.m) return false;
    int64_t xa = ptr_x[batch];
    int64_t xb = ptr_x[batch + 1];
    if (xa < 0 || xb <= xa || xb > t.n) return false;
    range = {static_cast<uint32_t>(a), static_cast<uint32_t>(b),
             static_cast<uint32_t>(xa), static_cast<uint32_t>(xb - xa)};
    return true;
}

__simt_callee__ __aicore__ inline float SmallDot(
    __gm__ float* x, __gm__ float* y, uint32_t row, uint32_t col,
    NearestTiling t, BatchRange range)
{
    // The pinned CPU BLAS uses 16 interleaved FMA chains for small TN matrices.
    // Preserve both addition trees, including the tail rows and columns.
    float partial[16];
    for (uint32_t r = 0; r < 16; ++r) {
        float dot = 0.0f;
        for (uint32_t d = r; d < t.features; d += 16) {
            dot = AscendC::Simt::Fma(x[static_cast<uint64_t>(d) * t.n + row],
                                   y[static_cast<uint64_t>(d) * t.m + col], dot);
        }
        partial[r] = dot;
    }
    if (row - range.x_begin < range.x_count / 4 * 4 ||
        col - range.begin < (range.end - range.begin) / 4 * 4) {
        float q0 = (partial[0] + partial[1]) + (partial[2] + partial[3]);
        float q1 = (partial[4] + partial[5]) + (partial[6] + partial[7]);
        float q2 = (partial[8] + partial[9]) + (partial[10] + partial[11]);
        float q3 = (partial[12] + partial[13]) + (partial[14] + partial[15]);
        return (q0 + q1) + (q2 + q3);
    }
    for (uint32_t r = 0; r < 8; ++r) partial[r] = partial[r] + partial[r + 8];
    for (uint32_t r = 0; r < 4; ++r) partial[r] = partial[r] + partial[r + 4];
    return (partial[0] + partial[2]) + (partial[1] + partial[3]);
}

__simt_callee__ __aicore__ inline float PanelDistance(
    __gm__ float* x, __gm__ float* y, uint32_t row, uint32_t col, NearestTiling t)
{
    float distance = 0.0f;
    for (uint32_t begin_k = 0; begin_k < t.features;) {
        uint32_t count = t.features - begin_k;
        count = count >= 896 ? 448 : (count > 448 ? (count + 1) / 2 : count);
        float dot = 0.0f;
        for (uint32_t k = 0; k < count; ++k) {
            uint32_t d = begin_k + k;
            float xv = x[static_cast<uint64_t>(d) * t.n + row];
            float yv = y[static_cast<uint64_t>(d) * t.m + col];
            dot = AscendC::Simt::Fma(xv, yv, dot);
        }
        distance = AscendC::Simt::Fma(-2.0f, dot, distance);
        begin_k += count;
    }
    return distance;
}

template <int FEATURES>
__simt_callee__ __aicore__ inline float Distance(
    __gm__ float* x, __gm__ float* y, __gm__ float* norms,
    uint32_t row, uint32_t col, NearestTiling t, BatchRange range)
{
    float distance = 0.0f;
    uint32_t features = FEATURES == 0 ? t.features : FEATURES;
    if (FEATURES == 3 || t.features < 5) {
        for (uint32_t d = 0; d < features; ++d) {
            float xv = x[static_cast<uint64_t>(d) * t.n + row];
            float yv = y[static_cast<uint64_t>(d) * t.m + col];
            float diff = yv - xv;
            volatile float square = diff * diff;
            distance = distance + square;
        }
        return distance;
    }
    uint64_t pairs = static_cast<uint64_t>(range.x_count) * (range.end - range.begin);
    if (features >= 32 && pairs <= 1200 && pairs * features <= 1000000) {
        distance = -2.0f * SmallDot(x, y, row, col, t, range);
    } else {
        distance = PanelDistance(x, y, row, col, t);
    }
    volatile float with_x_norm = distance + norms[row];
    return with_x_norm + norms[t.n + col];
}

template <int FEATURES>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) void Search(
    __gm__ float* work, __gm__ int64_t* ptr_x, __gm__ int64_t* ptr_y,
    __gm__ int64_t* out, NearestTiling t)
{
    uint32_t tid = AscendC::Simt::GetThreadIdx();
    uint32_t lane = tid % WARP;
    uint32_t first = AscendC::Simt::GetBlockIdx() * (THREADS / WARP) + tid / WARP;
    uint32_t step = AscendC::Simt::GetBlockNum() * (THREADS / WARP);
    __gm__ float* y = work + static_cast<uint64_t>(t.n) * t.features;
    __gm__ float* norms = y + static_cast<uint64_t>(t.m) * t.features;
    for (uint32_t row = first; row < t.n; row += step) {
        BatchRange range{0, t.m, 0, t.n};
        if (!GetBatchRange(row, ptr_x, ptr_y, t, range)) {
            if (lane == 0) out[row] = -1;
            continue;
        }
        float best = __builtin_inff();
        uint32_t index = range.begin;
        for (uint32_t col = range.begin + lane; col < range.end; col += WARP) {
            float distance = Distance<FEATURES>(work, y, norms, row, col, t, range);
            // Equal distances always choose the smallest global y index.
            if (distance < best || (distance == best && col < index)) {
                best = distance;
                index = col;
            }
        }
        for (uint32_t delta = WARP / 2; delta > 0; delta /= 2) {
            float other = AscendC::Simt::WarpShflDownSync(best, delta);
            uint32_t other_index = AscendC::Simt::WarpShflDownSync(index, delta);
            if (lane + delta < WARP &&
                (other < best || (other == best && other_index < index))) {
                best = other;
                index = other_index;
            }
        }
        if (lane == 0) out[row] = static_cast<int64_t>(index);
    }
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void PrepareKernel(
    __gm__ T* x, __gm__ T* y, __gm__ float* work, NearestTiling t)
{
    AscendC::Simt::VF_CALL<Prepare<T>>(AscendC::Simt::Dim3{THREADS}, x, y, work, t);
}

template <int FEATURES>
__attribute__((aiv)) __global__ __aicore__ void SearchKernel(
    __gm__ float* work, __gm__ int64_t* ptr_x, __gm__ int64_t* ptr_y,
    __gm__ int64_t* out, NearestTiling t)
{
    AscendC::Simt::VF_CALL<Search<FEATURES>>(
        AscendC::Simt::Dim3{THREADS}, work, ptr_x, ptr_y, out, t);
}
} // namespace

namespace opsgnn {

void Nearest(const void* x, const void* y, const int64_t* ptr_x,
             const int64_t* ptr_y, float* workspace, int64_t* out,
             bool is_half, const NearestTiling& t, aclrtStream stream)
{
    if (is_half) {
        PrepareKernel<uint16_t><<<t.cores, nullptr, stream>>>(
            (uint16_t*)x, (uint16_t*)y, workspace, t);
    } else {
        PrepareKernel<float><<<t.cores, nullptr, stream>>>((float*)x, (float*)y, workspace, t);
    }
    uint32_t blocks = (t.n + THREADS / WARP - 1) / (THREADS / WARP);
    blocks = blocks < t.cores ? blocks : t.cores;
    if (t.tile_y != 0) {
        uint32_t simd_blocks = (t.n + t.tile_x - 1) / t.tile_x;
        simd_blocks = simd_blocks < t.cores ? simd_blocks : t.cores;
        nearest_simd::SearchHigh<<<simd_blocks, nullptr, stream>>>(workspace, out, t);
        return;
    }
    if (t.features == 3 && t.batches == 0 && t.n >= 4096 && t.ub_bytes >= 110592) {
        uint32_t simd_blocks = (t.n + nearest_simd::TILE_X - 1) / nearest_simd::TILE_X;
        simd_blocks = simd_blocks < t.cores ? simd_blocks : t.cores;
        nearest_simd::Search3<<<simd_blocks, nullptr, stream>>>(workspace, out, t.n, t.m);
        return;
    }
    if (t.features == 3) {
        SearchKernel<3><<<blocks, nullptr, stream>>>(workspace, (int64_t*)ptr_x, (int64_t*)ptr_y, out, t);
    } else {
        SearchKernel<0><<<blocks, nullptr, stream>>>(workspace, (int64_t*)ptr_x, (int64_t*)ptr_y, out, t);
    }
}

}  // namespace opsgnn
