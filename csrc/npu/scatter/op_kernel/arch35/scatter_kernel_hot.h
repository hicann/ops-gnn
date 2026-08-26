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

__aicore__ inline uint32_t TileLength(uint64_t total, uint64_t offset,
                                      uint32_t capacity)
{
    uint64_t remaining = total - offset;
    return static_cast<uint32_t>(
        remaining > static_cast<uint64_t>(capacity) ? capacity : remaining);
}

template <typename T>
__aicore__ inline void LoadPromotedRow(
    GlobalTensor<T>& srcGm, LocalTensor<float> wide, LocalTensor<half> raw,
    uint64_t sourceBase, uint32_t length, event_t loadReady)
{
    if constexpr (IsSameType<T, half>::value) {
        DataCopy(raw, srcGm[sourceBase], length);
    } else {
        DataCopy(wide, srcGm[sourceBase], length);
    }
    SetFlag<HardEvent::MTE2_V>(loadReady);
    WaitFlag<HardEvent::MTE2_V>(loadReady);
    if constexpr (IsSameType<T, half>::value) {
        Cast(wide, raw, RoundMode::CAST_NONE, static_cast<int32_t>(length));
        PipeBarrier<PIPE_V>();
    }
}

struct DoubleBufferEvents {
    event_t load[2];
    event_t compute[2];
    event_t done[2];
    bool issued[2];

    __aicore__ inline void Init()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            load[slot] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            compute[slot] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            done[slot] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
            issued[slot] = false;
        }
    }

    __aicore__ inline void WaitSlot(uint32_t slot)
    {
        if (issued[slot]) {
            WaitFlag<HardEvent::MTE3_MTE2>(done[slot]);
        }
    }

    __aicore__ inline void Finish()
    {
        WaitSlot(0);
        WaitSlot(1);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(load[slot]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(compute[slot]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(done[slot]);
        }
    }
};

struct CopyBufferEvents {
    event_t ready[2];
    event_t complete[2];
    bool active[2];

    __aicore__ inline void Init()
    {
        ready[0] = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        ready[1] = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        complete[0] = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        complete[1] = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        active[0] = false;
        active[1] = false;
    }

    __aicore__ inline void WaitSlot(uint32_t slot)
    {
        if (active[slot]) {
            WaitFlag<HardEvent::MTE3_MTE2>(complete[slot]);
        }
    }

    __aicore__ inline void Finish()
    {
        WaitSlot(0);
        WaitSlot(1);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(ready[slot]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(complete[slot]);
        }
    }
};

struct SingleBufferEvents {
    event_t load;
    event_t store;
    event_t done;

    __aicore__ inline void Init()
    {
        load = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        store = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        done = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    }

    __aicore__ inline void Finish()
    {
        PipeBarrier<PIPE_ALL>();
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(load);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(store);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(done);
    }
};

struct ScatterRow {
    uint64_t before;
    uint64_t sourceBase;
    uint64_t outputBase;
    int64_t target;
    bool valid;
};

__aicore__ inline ScatterRow ResolveScatterRow(
    uint64_t row, __gm__ int64_t* index, ScatterTilingData tiling)
{
    ScatterRow result{};
    result.before = row / tiling.dimLength;
    uint64_t element = row - result.before * tiling.dimLength;
    uint64_t indexOffset = tiling.indexDimLength == 1 ? 0 : element;
    result.target = index[indexOffset];
    result.valid = result.target >= 0 &&
        static_cast<uint64_t>(result.target) < tiling.outputDim;
    result.sourceBase = row * tiling.afterDim;
    if (result.valid) {
        result.outputBase =
            (result.before * tiling.outputDim +
             static_cast<uint64_t>(result.target)) * tiling.afterDim;
    }
    return result;
}

template <typename T>
struct MinMaxArgGlobals {
    GlobalTensor<T> source;
    GlobalTensor<float> values;
    GlobalTensor<int32_t> arguments;

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR valueBuffer,
                                GM_ADDR argBuffer, ScatterTilingData tiling)
    {
        source.SetGlobalBuffer((__gm__ T*)src, tiling.srcNumel);
        values.SetGlobalBuffer((__gm__ float*)valueBuffer, tiling.outNumel);
        arguments.SetGlobalBuffer((__gm__ int32_t*)argBuffer, tiling.outNumel);
    }
};

__aicore__ inline float PromotedInitialValue(uint32_t reduce)
{
    if (reduce == SCATTER_MIN) {
        return 3.402823466e38F;
    }
    if (reduce == SCATTER_MAX) {
        return -3.402823466e38F;
    }
    return 0.0F;
}

__aicore__ inline void ReducePromotedRow(
    LocalTensor<float> accumulator, LocalTensor<float> value,
    uint32_t length, uint32_t reduce)
{
    if (reduce == SCATTER_SUM) {
        Add(accumulator, accumulator, value, static_cast<int32_t>(length));
    } else if (reduce == SCATTER_MIN) {
        Min(accumulator, accumulator, value, static_cast<int32_t>(length));
    } else {
        Max(accumulator, accumulator, value, static_cast<int32_t>(length));
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void AtomicStorePromoted(
    GlobalTensor<float>& output, LocalTensor<float> value,
    uint64_t destination, uint32_t length, uint32_t reduce,
    SingleBufferEvents& events)
{
    SetFlag<HardEvent::V_MTE3>(events.store);
    WaitFlag<HardEvent::V_MTE3>(events.store);
    if (reduce == SCATTER_SUM) {
        SetAtomicAdd<float>();
    } else if (reduce == SCATTER_MIN) {
        SetAtomicMin<float>();
    } else {
        SetAtomicMax<float>();
    }
    DataCopy(output[destination], value, length);
    DisableDmaAtomic();
    SetFlag<HardEvent::MTE3_MTE2>(events.done);
    WaitFlag<HardEvent::MTE3_MTE2>(events.done);
}

template <typename T>
__aicore__ inline void AccumulatePromotedRows(
    GlobalTensor<T>& source, GlobalTensor<float>& output,
    __gm__ int64_t* index, LocalTensor<float> wide,
    LocalTensor<float> accumulator, LocalTensor<half> raw, uint32_t length,
    uint64_t block, uint64_t blocks, ScatterTilingData tiling,
    uint32_t reduce, SingleBufferEvents& events)
{
    uint64_t rows = tiling.beforeDim * tiling.dimLength;
    for (uint64_t row = block; row < rows; row += blocks) {
        ScatterRow rowInfo = ResolveScatterRow(row, index, tiling);
        if (!rowInfo.valid) {
            continue;
        }
        LoadPromotedRow<T>(source, wide, raw, rowInfo.sourceBase, length,
                           events.load);
        if (rowInfo.target == tiling.hotTarget) {
            ReducePromotedRow(accumulator, wide, length, reduce);
        } else {
            AtomicStorePromoted(output, wide, rowInfo.outputBase, length,
                                reduce, events);
        }
    }
}

template <typename T>
__aicore__ inline void RunHotPromotedKernel(
    GM_ADDR src, GM_ADDR index, GM_ADDR output, ScatterTilingData tiling,
    uint32_t reduce)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0 || tiling.afterDim == 0 ||
        tiling.afterDim > PROMOTE_TILE_ELEMENTS || tiling.hotTarget < 0) {
        return;
    }
    GlobalTensor<T> source;
    GlobalTensor<float> destination;
    source.SetGlobalBuffer((__gm__ T*)src, tiling.srcNumel);
    destination.SetGlobalBuffer((__gm__ float*)output, tiling.outNumel);
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuffer;
    constexpr uint32_t storageFloats =
        2 * PROMOTE_TILE_ELEMENTS + PROMOTE_TILE_ELEMENTS / 2;
    pipe.InitBuffer(dataBuffer, storageFloats * sizeof(float));
    LocalTensor<float> storage = dataBuffer.Get<float>();
    LocalTensor<float> wide = storage;
    LocalTensor<float> accumulator = storage[PROMOTE_TILE_ELEMENTS];
    LocalTensor<half> raw =
        storage[2 * PROMOTE_TILE_ELEMENTS].ReinterpretCast<half>();
    uint32_t length = static_cast<uint32_t>(tiling.afterDim);
    SingleBufferEvents events;
    events.Init();
    Duplicate(accumulator, PromotedInitialValue(reduce),
              static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    AccumulatePromotedRows<T>(source, destination, (__gm__ int64_t*)index,
                              wide, accumulator, raw, length, block, blocks,
                              tiling, reduce, events);
    uint64_t hotBase =
        static_cast<uint64_t>(tiling.hotTarget) * tiling.afterDim;
    AtomicStorePromoted(destination, accumulator, hotBase, length, reduce,
                        events);
    events.Finish();
}
