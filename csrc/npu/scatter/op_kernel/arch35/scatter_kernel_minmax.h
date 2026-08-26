/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

__attribute__((aiv)) __global__ __aicore__ void
scatter_minmax_init_kernel(GM_ADDR values, GM_ADDR argOut,
                           ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0) {
        return;
    }
    uint64_t start = tiling.outNumel * block / blocks;
    uint64_t end = tiling.outNumel * (block + 1) / blocks;
    asc_vf_call<MinMaxInitCompute>(dim3(THREAD_NUM), start, end, tiling.reduce,
                                   (__gm__ float*)values,
                                   (__gm__ int32_t*)argOut);
}

template <typename T>
__aicore__ inline void LoadMinMaxTile(
    ScatterRow row, uint64_t inner, uint32_t slot, uint32_t length,
    GlobalTensor<T>& srcGm, LocalTensor<float> storage,
    DoubleBufferEvents& events)
{
    LocalTensor<float> wide = storage[slot * PROMOTE_SLOT_FLOATS];
    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<half> raw =
            storage[slot * PROMOTE_SLOT_FLOATS + PROMOTE_TILE_ELEMENTS]
                .ReinterpretCast<half>();
        DataCopy(raw, srcGm[row.sourceBase + inner], length);
        SetFlag<HardEvent::MTE2_V>(events.load[slot]);
        WaitFlag<HardEvent::MTE2_V>(events.load[slot]);
        Cast(wide, raw, RoundMode::CAST_NONE, static_cast<int32_t>(length));
    } else {
        DataCopy(wide, srcGm[row.sourceBase + inner], length);
        SetFlag<HardEvent::MTE2_V>(events.load[slot]);
        WaitFlag<HardEvent::MTE2_V>(events.load[slot]);
    }
}

template <typename T>
__aicore__ inline void ReduceMinMaxRow(
    ScatterRow row, ScatterTilingData tiling, GlobalTensor<T>& srcGm,
    GlobalTensor<float>& valuesGm, LocalTensor<float> storage,
    DoubleBufferEvents& events, uint64_t& sequence)
{
    for (uint64_t inner = 0; inner < tiling.afterDim;
         inner += PROMOTE_TILE_ELEMENTS) {
        uint32_t slot = static_cast<uint32_t>(sequence & 1U);
        events.WaitSlot(slot);
        uint32_t length =
            TileLength(tiling.afterDim, inner, PROMOTE_TILE_ELEMENTS);
        LoadMinMaxTile<T>(row, inner, slot, length, srcGm, storage, events);
        SetFlag<HardEvent::V_MTE3>(events.compute[slot]);
        WaitFlag<HardEvent::V_MTE3>(events.compute[slot]);
        if (tiling.reduce == SCATTER_MIN) {
            SetAtomicMin<float>();
        } else {
            SetAtomicMax<float>();
        }
        LocalTensor<float> wide = storage[slot * PROMOTE_SLOT_FLOATS];
        DataCopy(valuesGm[row.outputBase + inner], wide, length);
        DisableDmaAtomic();
        SetFlag<HardEvent::MTE3_MTE2>(events.done[slot]);
        events.issued[slot] = true;
        ++sequence;
    }
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void
scatter_vector_minmax_atomic_kernel(GM_ADDR src, GM_ADDR index, GM_ADDR values,
                                    ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0 || tiling.afterDim == 0) {
        return;
    }

    GlobalTensor<T> srcGm;
    GlobalTensor<float> valuesGm;
    srcGm.SetGlobalBuffer((__gm__ T*)src, tiling.srcNumel);
    valuesGm.SetGlobalBuffer((__gm__ float*)values, tiling.outNumel);

    uint64_t sequence = 0;
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuffer;
    pipe.InitBuffer(dataBuffer, 2 * PROMOTE_SLOT_FLOATS * sizeof(float));
    LocalTensor<float> storage = dataBuffer.Get<float>();
    DoubleBufferEvents events;
    events.Init();
    uint64_t rows = tiling.beforeDim * tiling.dimLength;
    for (uint64_t row = block; row < rows; row += blocks) {
        ScatterRow mapping =
            ResolveScatterRow(row, (__gm__ int64_t*)index, tiling);
        if (!mapping.valid) {
            continue;
        }
        ReduceMinMaxRow<T>(mapping, tiling, srcGm, valuesGm, storage,
                           events, sequence);
    }
    events.Finish();
}

template <typename T>
__aicore__ inline void UpdateMinMaxArgTile(
    GlobalTensor<T>& srcGm, GlobalTensor<float>& valuesGm,
    GlobalTensor<int32_t>& argGm, LocalTensor<float> storage,
    DoubleBufferEvents& events, uint64_t source, uint64_t destination,
    int32_t element, uint32_t length, uint32_t slot)
{
    events.WaitSlot(slot);
    uint32_t base = slot * MINMAX_ARG_SLOT_FLOATS;
    LocalTensor<float> sourceWide = storage[base];
    LocalTensor<float> finalWide = storage[base + 4096];
    LocalTensor<int32_t> candidate =
        storage[base + 8192].ReinterpretCast<int32_t>();
    LocalTensor<half> sourceRaw =
        storage[base + 12288].ReinterpretCast<half>();
    LocalTensor<uint8_t> equalMask =
        storage[base + 14336].ReinterpretCast<uint8_t>();
    DataCopy(finalWide, valuesGm[destination], length);
    LoadPromotedRow<T>(srcGm, sourceWide, sourceRaw, source, length,
                       events.load[slot]);
    Compare(equalMask, sourceWide, finalWide, CMPMODE::EQ,
            static_cast<int32_t>(length));
    Duplicate(candidate, element, static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    Select(candidate, equalMask, candidate, static_cast<int32_t>(-1),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, static_cast<int32_t>(length));
    SetFlag<HardEvent::V_MTE3>(events.compute[slot]);
    WaitFlag<HardEvent::V_MTE3>(events.compute[slot]);
    SetAtomicMax<int32_t>();
    DataCopy(argGm[destination], candidate, length);
    DisableDmaAtomic();
    SetFlag<HardEvent::MTE3_MTE2>(events.done[slot]);
    events.issued[slot] = true;
}

template <typename T>
__aicore__ inline void UpdateMinMaxArgRow(
    GlobalTensor<T>& srcGm, GlobalTensor<float>& valuesGm,
    GlobalTensor<int32_t>& argGm, LocalTensor<float> storage,
    DoubleBufferEvents& events, ScatterRow rowInfo,
    ScatterTilingData tiling, int32_t element, uint64_t& sequence)
{
    for (uint64_t inner = 0; inner < tiling.afterDim;
         inner += PROMOTE_TILE_ELEMENTS, ++sequence) {
        uint32_t slot = static_cast<uint32_t>(sequence & 1U);
        uint32_t length = TileLength(
            tiling.afterDim, inner, PROMOTE_TILE_ELEMENTS);
        UpdateMinMaxArgTile<T>(
            srcGm, valuesGm, argGm, storage, events,
            rowInfo.sourceBase + inner, rowInfo.outputBase + inner,
            element, length, slot);
    }
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void
scatter_vector_minmax_arg_kernel(GM_ADDR src, GM_ADDR index, GM_ADDR values,
                                 GM_ADDR argOut, ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0 || tiling.afterDim == 0) {
        return;
    }
    MinMaxArgGlobals<T> globals;
    globals.Init(src, values, argOut, tiling);
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuffer;
    pipe.InitBuffer(dataBuffer, 2 * MINMAX_ARG_SLOT_FLOATS * sizeof(float));
    LocalTensor<float> storage = dataBuffer.Get<float>();
    DoubleBufferEvents events;
    events.Init();
    uint64_t sequence = 0;
    uint64_t rows = tiling.beforeDim * tiling.dimLength;
    for (uint64_t row = block; row < rows; row += blocks) {
        ScatterRow rowInfo = ResolveScatterRow(
            row, (__gm__ int64_t*)index, tiling);
        bool isHot = tiling.hotTarget >= 0 &&
                     rowInfo.target == tiling.hotTarget;
        if (rowInfo.valid && !isHot) {
            int32_t element = static_cast<int32_t>(
                row - rowInfo.before * tiling.dimLength);
            UpdateMinMaxArgRow<T>(
                globals.source, globals.values, globals.arguments, storage,
                events, rowInfo, tiling, element, sequence);
        }
    }
    events.Finish();
}

struct HotMinMaxArgBuffers {
    LocalTensor<float> sourceWide;
    LocalTensor<float> finalWide;
    LocalTensor<int32_t> candidate;
    LocalTensor<int32_t> winner;
    LocalTensor<half> sourceRaw;
    LocalTensor<uint8_t> equalMask;

    __aicore__ inline void Init(LocalTensor<float> storage)
    {
        sourceWide = storage;
        finalWide = storage[PROMOTE_TILE_ELEMENTS];
        candidate = storage[2 * PROMOTE_TILE_ELEMENTS].ReinterpretCast<int32_t>();
        winner = storage[3 * PROMOTE_TILE_ELEMENTS].ReinterpretCast<int32_t>();
        sourceRaw = storage[4 * PROMOTE_TILE_ELEMENTS].ReinterpretCast<half>();
        equalMask = storage[4 * PROMOTE_TILE_ELEMENTS +
                            PROMOTE_TILE_ELEMENTS / 2].ReinterpretCast<uint8_t>();
    }
};

template <typename T>
__aicore__ inline void AccumulateHotMinMaxArg(
    GlobalTensor<T>& srcGm, HotMinMaxArgBuffers& buffers,
    uint64_t sourceBase, int32_t element, uint32_t length,
    event_t loadReady)
{
    LoadPromotedRow<T>(srcGm, buffers.sourceWide, buffers.sourceRaw,
                       sourceBase, length, loadReady);
    Compare(buffers.equalMask, buffers.sourceWide, buffers.finalWide, CMPMODE::EQ,
            static_cast<int32_t>(length));
    Duplicate(buffers.candidate, element, static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    Select(buffers.candidate, buffers.equalMask, buffers.candidate,
           static_cast<int32_t>(-1),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    Max(buffers.winner, buffers.winner, buffers.candidate,
        static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void StoreHotMinMaxArg(
    GlobalTensor<int32_t>& argGm, LocalTensor<int32_t> winner,
    uint64_t destination, uint32_t length, SingleBufferEvents& events)
{
    SetFlag<HardEvent::V_MTE3>(events.store);
    WaitFlag<HardEvent::V_MTE3>(events.store);
    SetAtomicMax<int32_t>();
    DataCopy(argGm[destination], winner, length);
    DisableDmaAtomic();
    SetFlag<HardEvent::MTE3_MTE2>(events.done);
    WaitFlag<HardEvent::MTE3_MTE2>(events.done);
}

template <typename T>
__aicore__ inline void AccumulateHotMinMaxArgRows(
    GlobalTensor<T>& srcGm, __gm__ int64_t* index,
    HotMinMaxArgBuffers& buffers, uint32_t length, uint64_t block,
    uint64_t blocks, ScatterTilingData tiling, SingleBufferEvents& events)
{
    uint64_t rows = tiling.beforeDim * tiling.dimLength;
    for (uint64_t row = block; row < rows; row += blocks) {
        ScatterRow rowInfo = ResolveScatterRow(row, index, tiling);
        if (!rowInfo.valid || rowInfo.target != tiling.hotTarget) {
            continue;
        }
        int32_t element = static_cast<int32_t>(
            row - rowInfo.before * tiling.dimLength);
        AccumulateHotMinMaxArg<T>(srcGm, buffers, rowInfo.sourceBase,
                                  element, length, events.load);
    }
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void
scatter_vector_hot_minmax_arg_kernel(GM_ADDR src, GM_ADDR index,
                                     GM_ADDR values, GM_ADDR argOut,
                                     ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0 || tiling.afterDim == 0 ||
        tiling.afterDim > PROMOTE_TILE_ELEMENTS || tiling.hotTarget < 0) {
        return;
    }
    MinMaxArgGlobals<T> globals;
    globals.Init(src, values, argOut, tiling);
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuffer;
    constexpr uint32_t maskFloats = PROMOTE_TILE_ELEMENTS / 8;
    constexpr uint32_t storageFloats =
        4 * PROMOTE_TILE_ELEMENTS + PROMOTE_TILE_ELEMENTS / 2 + maskFloats;
    pipe.InitBuffer(dataBuffer, storageFloats * sizeof(float));
    LocalTensor<float> storage = dataBuffer.Get<float>();
    HotMinMaxArgBuffers buffers;
    buffers.Init(storage);
    const uint32_t length = static_cast<uint32_t>(tiling.afterDim);

    SingleBufferEvents events;
    events.Init();
    uint64_t hotBase =
        static_cast<uint64_t>(tiling.hotTarget) * tiling.afterDim;
    DataCopy(buffers.finalWide, globals.values[hotBase], length);
    SetFlag<HardEvent::MTE2_V>(events.load);
    WaitFlag<HardEvent::MTE2_V>(events.load);
    Duplicate(buffers.winner, static_cast<int32_t>(-1),
              static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    AccumulateHotMinMaxArgRows<T>(
        globals.source, (__gm__ int64_t*)index, buffers, length, block,
        blocks, tiling, events);
    StoreHotMinMaxArg(globals.arguments, buffers.winner, hotBase, length,
                      events);
    events.Finish();
}

template <typename T>
__aicore__ inline void FinalizeMinMaxTile(
    GlobalTensor<float>& valuesGm, GlobalTensor<T>& outGm,
    GlobalTensor<int32_t>& argGm, LocalTensor<float> storage,
    DoubleBufferEvents& events, uint64_t offset, uint32_t length,
    uint32_t slot, uint64_t dimLength)
{
    events.WaitSlot(slot);
    uint32_t base = slot * MINMAX_FINAL_SLOT_FLOATS;
    LocalTensor<float> valueLocal = storage[base];
    LocalTensor<int32_t> argLocal =
        storage[base + 2048].ReinterpretCast<int32_t>();
    LocalTensor<float> emptyLocal = storage[base + 4096];
    LocalTensor<half> narrow = storage[base + 6144].ReinterpretCast<half>();
    LocalTensor<uint8_t> emptyMask =
        storage[base + 7168].ReinterpretCast<uint8_t>();
    DataCopy(valueLocal, valuesGm[offset], length);
    DataCopy(argLocal, argGm[offset], length);
    SetFlag<HardEvent::MTE2_V>(events.load[slot]);
    WaitFlag<HardEvent::MTE2_V>(events.load[slot]);
    CompareScalar(emptyMask, argLocal, static_cast<int32_t>(-1),
                  CMPMODE::EQ, static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    Duplicate(emptyLocal, 0.0F, static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    Select(valueLocal, emptyMask, emptyLocal, valueLocal,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> emptyArg = emptyLocal.ReinterpretCast<int32_t>();
    Duplicate(emptyArg, static_cast<int32_t>(dimLength),
              static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    Select(argLocal, emptyMask, emptyArg, argLocal,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<int32_t>(length));
    if constexpr (IsSameType<T, half>::value) {
        PipeBarrier<PIPE_V>();
        Cast(narrow, valueLocal, RoundMode::CAST_NONE,
             static_cast<int32_t>(length));
    }
    SetFlag<HardEvent::V_MTE3>(events.compute[slot]);
    WaitFlag<HardEvent::V_MTE3>(events.compute[slot]);
    if constexpr (IsSameType<T, half>::value) {
        DataCopy(outGm[offset], narrow, length);
    } else {
        DataCopy(outGm[offset], valueLocal, length);
    }
    DataCopy(argGm[offset], argLocal, length);
    SetFlag<HardEvent::MTE3_MTE2>(events.done[slot]);
    events.issued[slot] = true;
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void
scatter_vector_minmax_finalize_kernel(GM_ADDR values, GM_ADDR out,
                                      GM_ADDR argOut,
                                      ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0) {
        return;
    }
    GlobalTensor<float> valuesGm;
    GlobalTensor<T> outGm;
    GlobalTensor<int32_t> argGm;
    valuesGm.SetGlobalBuffer((__gm__ float*)values, tiling.outNumel);
    outGm.SetGlobalBuffer((__gm__ T*)out, tiling.outNumel);
    argGm.SetGlobalBuffer((__gm__ int32_t*)argOut, tiling.outNumel);
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuffer;
    pipe.InitBuffer(dataBuffer, 2 * MINMAX_FINAL_SLOT_FLOATS * sizeof(float));
    LocalTensor<float> storage = dataBuffer.Get<float>();
    DoubleBufferEvents events;
    events.Init();
    constexpr uint32_t tileElements = 2048;
    uint64_t start = tiling.outNumel * block / blocks;
    uint64_t end = tiling.outNumel * (block + 1) / blocks;
    uint64_t sequence = 0;
    for (uint64_t offset = start; offset < end;
         offset += tileElements, ++sequence) {
        uint32_t slot = static_cast<uint32_t>(sequence & 1U);
        uint32_t length = TileLength(end, offset, tileElements);
        FinalizeMinMaxTile<T>(valuesGm, outGm, argGm, storage, events,
                              offset, length, slot, tiling.dimLength);
    }
    events.Finish();
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void
scatter_vector_hot_minmax_kernel(GM_ADDR src, GM_ADDR index, GM_ADDR values,
                                 ScatterTilingData tiling)
{
    RunHotPromotedKernel<T>(src, index, values, tiling, tiling.reduce);
}
