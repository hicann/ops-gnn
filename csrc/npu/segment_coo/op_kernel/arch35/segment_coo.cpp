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
#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_fp16.h"
using namespace AscendC;

template <typename T, typename Acc, typename Count>
__simt_callee__ inline Acc FinishMean(Acc value, Count count) {
    if constexpr (sizeof(T) == 1) {
        // CPU torch_scatter's Reducer<scalar_t, MEAN>::write narrows BOTH
        // operands to the byte dtype before the promoted integer division.
        const int32_t numerator = static_cast<int32_t>(static_cast<T>(value));
        const int32_t denominator = static_cast<int32_t>(static_cast<T>(count));
        if (denominator != 0)
            return static_cast<Acc>(numerator / denominator);
        // A nonempty count divisible by 256 gives undefined CPU division by
        // zero. Keep a safe mathematical mean on that undefined-oracle input.
    }
    return value / static_cast<Acc>(count);
}

__simt_callee__ inline uint32_t DirectLowerBound(__gm__ int64_t *index, uint32_t length,
                                                 uint32_t key) {
    if (length == 0)
        return 0;
    const int64_t first = index[0], last = index[length - 1];
    if (static_cast<int64_t>(key) <= first)
        return 0;
    if (static_cast<int64_t>(key) > last)
        return length;
    // A search-order hint from actual endpoint values, not assumed boundaries.
    // Products fit uint64_t under the packed-path address guard.
    const uint32_t range = static_cast<uint32_t>(last - first) + 1;
    const uint64_t numerator = static_cast<uint64_t>(static_cast<int64_t>(key) - first) * length;
    uint32_t guess;
    if ((range & (range - 1)) == 0)
        guess = static_cast<uint32_t>(numerator >> (31 - Simt::Clz(range)));
    else
        guess = static_cast<uint32_t>(numerator / range);
    uint32_t lo = 0, hi = length;
    if (index[guess] < static_cast<int64_t>(key)) {
        lo = guess + 1;
    } else {
        if (guess == 0 || index[guess - 1] < static_cast<int64_t>(key))
            return guess;
        hi = guess;
    }
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (index[mid] < static_cast<int64_t>(key))
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256) void BuildPointers(__gm__ int64_t *index,
                                                            __gm__ int64_t *ptr, SegmentCooPlan p) {
    int64_t tid = Simt::GetBlockIdx() * 256 + Simt::GetThreadIdx();
    int64_t stride = Simt::GetBlockNum() * 256;
    for (int64_t i = tid; i < p.indexBatches * (p.segments + 1); i += stride) {
        int64_t batch = i / (p.segments + 1), key = i % (p.segments + 1);
        int64_t lo = 0, hi = p.length;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (index[batch * p.length + mid] < key)
                lo = mid + 1;
            else
                hi = mid;
        }
        ptr[i] = lo;
    }
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void BuildPointers32(__gm__ int64_t *index,
                                                              __gm__ int64_t *ptr,
                                                              SegmentCooPlan p) {
    const uint32_t tid = Simt::GetBlockIdx() * 256 + Simt::GetThreadIdx();
    const uint32_t stride = Simt::GetBlockNum() * 256;
    const uint32_t length = p.length, entries = p.segments + 1;
    const uint32_t total = p.indexBatches * entries;
    for (uint32_t i = tid; i < total; i += stride) {
        const uint32_t batch = p.indexBatches == 1 ? 0 : i / entries;
        const uint32_t key = p.indexBatches == 1 ? i : i % entries;
        // Verify hints against current data on EVERY invocation. Empty, holes,
        // unequal segments and changed indices retain exact binary fallback.
        ptr[i] = DirectLowerBound(index + static_cast<int64_t>(batch) * length, length, key);
    }
}
__attribute__((aiv)) __global__ __aicore__ void
PointerKernel32(__gm__ int64_t *index, __gm__ int64_t *ptr, SegmentCooPlan p) {
    Simt::VF_CALL<BuildPointers32>(Simt::Dim3{256}, index, ptr, p);
}
__attribute__((aiv)) __global__ __aicore__ void
PointerKernel(__gm__ int64_t *index, __gm__ int64_t *ptr, SegmentCooPlan p) {
    Simt::VF_CALL<BuildPointers>(Simt::Dim3{256}, index, ptr, p);
}

template <typename T, typename Acc, int Reduce, bool NeedArg>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void ReduceSegments(__gm__ T *src, __gm__ T *out,
                                                             __gm__ int64_t *arg,
                                                             __gm__ int64_t *ptr,
                                                             SegmentCooPlan p) {
    int64_t tid = Simt::GetBlockIdx() * 256 + Simt::GetThreadIdx();
    int64_t stride = Simt::GetBlockNum() * 256;
    int64_t total = p.batches * p.segments * p.channels;
    for (int64_t i = tid; i < total; i += stride) {
        int64_t c = i % p.channels, seg = (i / p.channels) % p.segments;
        int64_t batch = i / (p.segments * p.channels);
        int64_t pb = p.indexBatches == 1 ? 0 : batch;
        int64_t begin = ptr[pb * (p.segments + 1) + seg];
        int64_t end = ptr[pb * (p.segments + 1) + seg + 1];
        Acc value = static_cast<Acc>(0);
        int64_t best = p.length;
        if constexpr (Reduce == SEGMENT_COO_MEAN)
            if (p.hasOut)
                value = static_cast<Acc>(out[i]);
        if constexpr (Reduce >= SEGMENT_COO_MIN) {
            if (p.hasOut)
                value = static_cast<Acc>(out[i]);
            else if (begin < end) {
                value = static_cast<Acc>(src[(batch * p.length + begin) * p.channels + c]);
                best = begin;
            }
        }
        for (int64_t row = begin; row < end; ++row) {
            Acc next = static_cast<Acc>(src[(batch * p.length + row) * p.channels + c]);
            if constexpr (Reduce < SEGMENT_COO_MIN)
                value += next;
            else if ((Reduce == SEGMENT_COO_MIN && next <= value) || (Reduce == SEGMENT_COO_MAX && next >= value)) {
                value = next;
                best = row;
            }
        }
        if constexpr (Reduce == SEGMENT_COO_MEAN) {
            if (end > begin)
                value = FinishMean<T>(value, end - begin);
        }
        out[i] = static_cast<T>(value);
        if constexpr (NeedArg)
            arg[i] = best;
    }
}
template <typename T, typename Acc, int Reduce, bool NeedArg>
__attribute__((aiv)) __global__ __aicore__ void
ReductionKernel(__gm__ T *src, __gm__ T *out, __gm__ int64_t *arg, __gm__ int64_t *ptr,
                SegmentCooPlan p) {
    Simt::VF_CALL<ReduceSegments<T, Acc, Reduce, NeedArg>>(Simt::Dim3{256}, src, out, arg, ptr, p);
}
template <typename I, bool FastCoordinates, bool DirectIndex>
__simt_callee__ __attribute__((always_inline)) inline void
PackedCoordinates(I i, I columns, I segments, I length, __gm__ int64_t *bounds, SegmentCooPlan p,
                  I &c, I &batch, I &begin, I &end) {
    I seg;
    if constexpr (FastCoordinates) {
        // Exact unsigned address decomposition for power-of-two shapes.
        // This does not alter accumulation or signed integer mean division.
        I segmentAndBatch = i >> p.columnShift;
        c = i & (columns - 1);
        seg = segmentAndBatch & (segments - 1);
        batch = segmentAndBatch >> p.segmentShift;
    } else {
        c = i % columns;
        seg = (i / columns) % segments;
        batch = i / (segments * columns);
    }
    I pb = p.indexBatches == 1 ? 0 : batch;
    if constexpr (DirectIndex) {
        begin = DirectLowerBound(bounds + pb * length, length, seg);
        end = DirectLowerBound(bounds + pb * length, length, seg + 1);
    } else {
        begin = bounds[pb * (segments + 1) + seg];
        end = bounds[pb * (segments + 1) + seg + 1];
    }
}

template <typename T, typename Acc, int Reduce, int Width, typename I, typename Pack4>
__simt_callee__ __attribute__((always_inline)) inline void
InitializePacked(__gm__ Pack4 *input, __gm__ Pack4 *output, I i, I batch, I length, I columns, I c,
                 I begin, I end, bool hasOut, Acc *val, I *best) {
#pragma unroll
    for (int j = 0; j < Width; ++j) {
        val[j] = 0;
        best[j] = length;
    }
    if constexpr (Reduce == SEGMENT_COO_MEAN) {
        if (hasOut) {
            Pack4 initial = output[i];
#pragma unroll
            for (int j = 0; j < Width; ++j)
                val[j] = static_cast<Acc>(initial[j]);
        }
    }
    if constexpr (Reduce >= SEGMENT_COO_MIN) {
        if (hasOut) {
            Pack4 initial = output[i];
#pragma unroll
            for (int j = 0; j < Width; ++j)
                val[j] = static_cast<Acc>(initial[j]);
        } else if (begin < end) {
            Pack4 initial = input[(batch * length + begin) * columns + c];
#pragma unroll
            for (int j = 0; j < Width; ++j) {
                val[j] = static_cast<Acc>(initial[j]);
                best[j] = begin;
            }
        }
    }
}

template <typename Acc, int Reduce, int Width, typename I, typename Pack4>
__simt_callee__ __attribute__((always_inline)) inline void
AccumulatePacked(__gm__ Pack4 *input, I batch, I length, I columns, I c, I begin, I end, Acc *val,
                 I *best) {
    for (I row = begin; row < end; ++row) {
        Pack4 next = input[(batch * length + row) * columns + c];
#pragma unroll
        for (int j = 0; j < Width; ++j) {
            Acc v = static_cast<Acc>(next[j]);
            if constexpr (Reduce < SEGMENT_COO_MIN)
                val[j] += v;
            else if ((Reduce == SEGMENT_COO_MIN && v <= val[j]) || (Reduce == SEGMENT_COO_MAX && v >= val[j])) {
                val[j] = v;
                best[j] = row;
            }
        }
    }
}

template <typename T, typename Acc, int Reduce, int Width, bool NeedArg, typename I, typename Pack4>
__simt_callee__ __attribute__((always_inline)) inline void
StorePacked(__gm__ Pack4 *output, __gm__ int64_t *arg, I i, I begin, I end, Acc *val, I *best) {
    Pack4 result;
#pragma unroll
    for (int j = 0; j < Width; ++j) {
        if constexpr (Reduce == SEGMENT_COO_MEAN)
            if (end > begin)
                val[j] = FinishMean<T>(val[j], end - begin);
        result[j] = static_cast<T>(val[j]);
        if constexpr (NeedArg)
            arg[i * Width + j] = best[j];
    }
    output[i] = result;
}

template <typename T, typename Acc, int Reduce, int Width, typename I, int Threads, bool NeedArg,
          bool FastCoordinates = false, bool DirectIndex = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(Threads) void ReducePacked(__gm__ T *src, __gm__ T *out,
                                                               __gm__ int64_t *arg,
                                                               __gm__ int64_t *bounds,
                                                               SegmentCooPlan p) {
    I tid = Simt::GetBlockIdx() * Threads + Simt::GetThreadIdx();
    I stride = Simt::GetBlockNum() * Threads;
    I columns = p.channels / Width, segments = p.segments, length = p.length;
    I total = p.batches * segments * columns;
    typedef T Pack4 __attribute__((ext_vector_type(Width)));
    auto input = reinterpret_cast<__gm__ Pack4 *>(src);
    auto output = reinterpret_cast<__gm__ Pack4 *>(out);
    for (I i = tid; i < total; i += stride) {
        I c, batch, begin, end;
        PackedCoordinates<I, FastCoordinates, DirectIndex>(i, columns, segments, length, bounds, p,
                                                           c, batch, begin, end);
        Acc val[Width];
        I best[Width];
        InitializePacked<T, Acc, Reduce, Width>(input, output, i, batch, length, columns, c, begin,
                                                end, p.hasOut, val, best);
        AccumulatePacked<Acc, Reduce, Width>(input, batch, length, columns, c, begin, end, val,
                                             best);
        StorePacked<T, Acc, Reduce, Width, NeedArg>(output, arg, i, begin, end, val, best);
    }
}

template <typename T, typename Acc, int Reduce, int Width, typename I, int Threads, bool NeedArg,
          bool FastCoordinates = false, bool DirectIndex = false>
__attribute__((aiv)) __global__ __aicore__ void
PackedKernel(__gm__ T *src, __gm__ T *out, __gm__ int64_t *arg, __gm__ int64_t *ptr,
             SegmentCooPlan p) {
    Simt::VF_CALL<
        ReducePacked<T, Acc, Reduce, Width, I, Threads, NeedArg, FastCoordinates, DirectIndex>>(
        Simt::Dim3{Threads}, src, out, arg, ptr, p);
}
template <typename T, typename Acc, int Reduce, int Width, typename Packed>
__simt_callee__ __attribute__((always_inline)) inline void
StoreWarpPacked(__gm__ Packed *output, uint32_t destination, uint32_t begin, uint32_t end,
                bool hasOut, Acc *sums) {
    if constexpr (Reduce == SEGMENT_COO_MEAN) {
        if (hasOut) {
            Packed initial = output[destination];
#pragma unroll
            for (int j = 0; j < Width; ++j)
                sums[j] += static_cast<Acc>(initial[j]);
        }
    }
    Packed result;
#pragma unroll
    for (int j = 0; j < Width; ++j) {
        if constexpr (Reduce == SEGMENT_COO_MEAN)
            if (end > begin)
                sums[j] = FinishMean<T>(sums[j], end - begin);
        result[j] = static_cast<T>(sums[j]);
    }
    output[destination] = result;
}

template <bool FastCoordinates>
__simt_callee__ __attribute__((always_inline)) inline void
WarpCoordinates(uint32_t tile, uint32_t columnLane, uint32_t columnTiles, uint32_t segments,
                SegmentCooPlan p, uint32_t &column, uint32_t &segment, uint32_t &batch) {
    constexpr uint32_t ColumnLanes = 4;
    if constexpr (FastCoordinates) {
        // Four packed columns per warp tile; shape guards make these
        // unsigned masks/shifts exact, including the batched case.
        const uint32_t segmentAndBatch = tile >> (p.columnShift - 2);
        column = (tile & (columnTiles - 1)) * ColumnLanes + columnLane;
        segment = segmentAndBatch & (segments - 1);
        batch = segmentAndBatch >> p.segmentShift;
    } else {
        column = (tile % columnTiles) * ColumnLanes + columnLane;
        segment = (tile / columnTiles) % segments;
        batch = tile / (columnTiles * segments);
    }
}

// Cooperative reduction: a warp covers four packed feature columns and eight
// row partitions of the ACTUAL segment. No uniform-segment assumption or
// atomics.
template <typename T, typename Acc, int Reduce, int Width, bool FastCoordinates = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) void ReduceWarpPacked(__gm__ T *src, __gm__ T *out,
                                                               __gm__ int64_t *ptr,
                                                               SegmentCooPlan p) {
    constexpr uint32_t ColumnLanes = 4;
    constexpr uint32_t RowLanes = 8;
    uint32_t thread = Simt::GetThreadIdx();
    uint32_t columnLane = thread % ColumnLanes;
    uint32_t rowLane = (thread % 32) / ColumnLanes;
    uint32_t firstTile = Simt::GetBlockIdx() * 16 + thread / 32;
    uint32_t tileStride = Simt::GetBlockNum() * 16;
    uint32_t columns = p.channels / Width;
    uint32_t columnTiles = columns / ColumnLanes;
    uint32_t segments = p.segments, length = p.length;
    uint32_t total = p.batches * segments * columnTiles;
    typedef T Packed __attribute__((ext_vector_type(Width)));
    auto input = reinterpret_cast<__gm__ Packed *>(src);
    auto output = reinterpret_cast<__gm__ Packed *>(out);
    for (uint32_t tile = firstTile; tile < total; tile += tileStride) {
        uint32_t column, segment, batch;
        WarpCoordinates<FastCoordinates>(tile, columnLane, columnTiles, segments, p, column,
                                         segment, batch);
        uint32_t pointerBatch = p.indexBatches == 1 ? 0 : batch;
        uint32_t begin = ptr[pointerBatch * (segments + 1) + segment];
        uint32_t end = ptr[pointerBatch * (segments + 1) + segment + 1];
        Acc sums[Width];
#pragma unroll
        for (int j = 0; j < Width; ++j)
            sums[j] = 0;
        for (uint32_t row = begin + rowLane; row < end; row += RowLanes) {
            Packed values = input[(batch * length + row) * columns + column];
#pragma unroll
            for (int j = 0; j < Width; ++j)
                sums[j] += static_cast<Acc>(values[j]);
        }
        // XOR only row-lane bits, preserving each lane's packed feature column.
#pragma unroll
        for (int delta = ColumnLanes; delta < 32; delta *= 2) {
#pragma unroll
            for (int j = 0; j < Width; ++j)
                sums[j] += Simt::WarpShflXorSync(sums[j], delta);
        }
        if (rowLane == 0) {
            const uint32_t destination = (batch * segments + segment) * columns + column;
            StoreWarpPacked<T, Acc, Reduce, Width>(output, destination, begin, end, p.hasOut, sums);
        }
    }
}
template <typename T, typename Acc, int Reduce, int Width, bool FastCoordinates = false>
__attribute__((aiv)) __global__ __aicore__ void
WarpPackedKernel(__gm__ T *src, __gm__ T *out, __gm__ int64_t *ptr, SegmentCooPlan p) {
    Simt::VF_CALL<ReduceWarpPacked<T, Acc, Reduce, Width, FastCoordinates>>(Simt::Dim3{512}, src,
                                                                            out, ptr, p);
}

struct LaunchContext {
    void *src;
    void *out;
    int64_t *arg;
    int64_t *ptr;
    SegmentCooPlan plan;
    aclrtStream stream;
};

template <typename T, typename Acc, bool Fast> void LaunchInt64Mean(LaunchContext context) {
    auto p = context.plan;
#define MEAN(W, TH)                                                                                \
    PackedKernel<T, Acc, SEGMENT_COO_MEAN, W, uint32_t, TH, false, Fast, true>                                    \
        <<<p.blocks, nullptr, context.stream>>>(static_cast<T *>(context.src),                     \
                                                static_cast<T *>(context.out), context.arg,        \
                                                context.ptr, p)
    if (p.length > p.segments * 16) {
        MEAN(4, 1024);
    } else if (p.channels > 32) {
        MEAN(4, 512);
    } else {
        if constexpr (Fast) {
            ++p.columnShift;
        }
        MEAN(2, 1024);
    }
#undef MEAN
}

template <typename T, typename Acc, int Reduce, bool NeedArg, bool Fast, bool Packed>
void LaunchReduction(const LaunchContext &context) {
    const auto p = context.plan;
    if constexpr (Packed) {
        constexpr int width = sizeof(T) == 8 ? 4 : 8;
        constexpr int threads =
            sizeof(T) == 8 ? (Reduce < SEGMENT_COO_MIN || !NeedArg ? 1024 : 512) : 256;
        PackedKernel<T, Acc, Reduce, width, uint32_t, threads, NeedArg, Fast>
            <<<p.blocks, nullptr, context.stream>>>(static_cast<T *>(context.src),
                                                    static_cast<T *>(context.out), context.arg,
                                                    context.ptr, p);
    } else {
        ReductionKernel<T, Acc, Reduce, NeedArg><<<p.blocks, nullptr, context.stream>>>(
            static_cast<T *>(context.src), static_cast<T *>(context.out), context.arg, context.ptr,
            p);
    }
}

template <typename T, typename Acc, bool Fast, bool Packed>
void SelectReduction(const LaunchContext &context) {
#define RUN(R, A) LaunchReduction<T, Acc, R, A, Fast, Packed>(context)
    switch (context.plan.reduce) {
    case SEGMENT_COO_SUM:
        RUN(SEGMENT_COO_SUM, false);
        break;
    case SEGMENT_COO_MEAN:
        RUN(SEGMENT_COO_MEAN, false);
        break;
    case SEGMENT_COO_MIN:
        if (context.arg) {
            RUN(SEGMENT_COO_MIN, true);
        } else {
            RUN(SEGMENT_COO_MIN, false);
        }
        break;
    case SEGMENT_COO_MAX:
        if (context.arg) {
            RUN(SEGMENT_COO_MAX, true);
        } else {
            RUN(SEGMENT_COO_MAX, false);
        }
        break;
    }
#undef RUN
}

template <typename T, typename Acc, bool Fast> void SelectPacked(const LaunchContext &context) {
    const auto p = context.plan;
    constexpr int width = sizeof(T) == 8 ? 4 : 8;
    if constexpr (sizeof(T) == 8) {
        if (p.directIndex) {
            LaunchInt64Mean<T, Acc, Fast>(context);
            return;
        }
    } else {
        if (p.reduce < SEGMENT_COO_MIN && p.channels % (width * 4) == 0 &&
            p.length >= p.segments * 16) {
#define WARP(R)                                                                                    \
    WarpPackedKernel<T, Acc, R, width, Fast><<<p.blocks, nullptr, context.stream>>>(               \
        static_cast<T *>(context.src), static_cast<T *>(context.out), context.ptr, p)
            if (p.reduce == SEGMENT_COO_SUM) {
                WARP(SEGMENT_COO_SUM);
            } else {
                WARP(SEGMENT_COO_MEAN);
            }
#undef WARP
            return;
        }
    }
    SelectReduction<T, Acc, Fast, true>(context);
}

template <typename T, typename Acc>
void Dispatch(void *src, void *out, int64_t *arg, int64_t *ptr, SegmentCooPlan p,
              aclrtStream stream) {
    if (!p.usePacked) {
        SelectReduction<T, Acc, false, false>({src, out, arg, ptr, p, stream});
        return;
    }
    constexpr int width = sizeof(T) == 8 ? 4 : 8;
    const uint32_t columns = p.channels / width;
    const uint32_t segments = p.segments;
    const bool fast =
        columns && segments && (columns & (columns - 1)) == 0 && (segments & (segments - 1)) == 0;
    if (fast) {
        p.columnShift = __builtin_ctz(columns);
        p.segmentShift = __builtin_ctz(segments);
        SelectPacked<T, Acc, true>({src, out, arg, ptr, p, stream});
    } else {
        SelectPacked<T, Acc, false>({src, out, arg, ptr, p, stream});
    }
}

void SegmentCoo(void *src, int64_t *index, void *out, int64_t *arg, int64_t *ptr, SegmentCooPlan p,
                aclrtStream stream) {
    if (p.directIndex) {
        ptr = index;
    } else if (p.length < 2147483647LL && p.segments < 2147483647LL &&
               p.indexBatches * (p.segments + 1) < 2147483647LL) {
        PointerKernel32<<<p.blocks, nullptr, stream>>>(index, ptr, p);
    } else {
        PointerKernel<<<p.blocks, nullptr, stream>>>(index, ptr, p);
    }
    switch (p.dtype) {
    case SEGMENT_COO_FLOAT32:
        Dispatch<float, float>(src, out, arg, ptr, p, stream);
        break;
    case SEGMENT_COO_FLOAT16:
        Dispatch<half, float>(src, out, arg, ptr, p, stream);
        break;
    case SEGMENT_COO_BFLOAT16:
        Dispatch<bfloat16_t, float>(src, out, arg, ptr, p, stream);
        break;
    case SEGMENT_COO_INT8:
        Dispatch<int8_t, int64_t>(src, out, arg, ptr, p, stream);
        break;
    case SEGMENT_COO_UINT8:
        Dispatch<uint8_t, int64_t>(src, out, arg, ptr, p, stream);
        break;
    case SEGMENT_COO_INT32:
        Dispatch<int32_t, int32_t>(src, out, arg, ptr, p, stream);
        break;
    case SEGMENT_COO_INT64:
        Dispatch<int64_t, int64_t>(src, out, arg, ptr, p, stream);
        break;
    }
}
