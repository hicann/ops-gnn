/*
 * Copyright (c) 2026 Starlink_. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"

// Packed FP16 extrema and first-index tie handling.
namespace ops_gnn {
namespace half_extrema {
namespace R = AscendC::Reg;

constexpr R::CastTrait kIndexWiden = {
    R::RegLayout::ZERO, R::SatMode::UNKNOWN, R::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

template <int OP>
__simd_callee__ inline void MergeCandidates(
    R::RegTensor<half>& value, R::RegTensor<uint16_t>& index, R::RegTensor<half>& candidate,
    R::RegTensor<uint16_t>& candidateIndex, R::MaskReg mask)
{
    R::MaskReg better, equal, earlier;
    if constexpr (OP == 1) {
        R::Compare<half, AscendC::CMPMODE::LT>(better, candidate, value, mask);
    } else {
        R::Compare<half, AscendC::CMPMODE::GT>(better, candidate, value, mask);
    }
    R::Compare<half, AscendC::CMPMODE::EQ>(equal, candidate, value, mask);
    R::Compare<uint16_t, AscendC::CMPMODE::LT>(earlier, candidateIndex, index, mask);
    R::And(equal, equal, earlier, mask);
    R::Or(better, better, equal, mask);
    R::Select(value, candidate, value, better);
    R::Select(index, candidateIndex, index, better);
}

template <R::HighLowPart PART>
__simd_callee__ inline void StoreIndices(
    __ubuf__ int64_t* arg, R::RegTensor<uint16_t>& index, uint32_t base, uint32_t count)
{
    R::RegTensor<uint32_t> index32;
    R::RegTensor<int64_t, R::RegTraitNumTwo> index64;
    auto mask = R::UpdateMask<uint32_t>(count);
    R::UnPack<uint32_t, uint16_t, PART>(index32, index);
    R::Adds(index32, index32, base, mask);
    R::Cast<int64_t, int32_t, kIndexWiden>(index64, (R::RegTensor<int32_t>&)index32, mask);
    R::StoreAlign(arg, index64, mask);
}

template <uint32_t PACKED_ROWS, int OP>
__simd_callee__ inline void MergePackedRows(
    R::RegTensor<half>& accum, R::RegTensor<uint16_t>& index,
    R::RegTensor<half>& candidate, R::RegTensor<uint16_t>& candidateIndex, R::MaskReg fullMask)
{
    if constexpr (PACKED_ROWS > 1) {
        R::RegTensor<uint16_t> permutation;
        R::Arange((R::RegTensor<int16_t>&)permutation, int16_t(0));
        R::Adds(permutation, permutation, uint16_t(64), fullMask);
        R::Gather<half, uint16_t>(candidate, accum, permutation);
        R::Gather<uint16_t, uint16_t>(candidateIndex, index, permutation);
        uint32_t count = 64;
        auto mergeMask = R::UpdateMask<half>(count);
        MergeCandidates<OP>(accum, index, candidate, candidateIndex, mergeMask);
        if constexpr (PACKED_ROWS == 4) {
            R::Arange((R::RegTensor<int16_t>&)permutation, int16_t(0));
            R::Adds(permutation, permutation, uint16_t(32), fullMask);
            R::Gather<half, uint16_t>(candidate, accum, permutation);
            R::Gather<uint16_t, uint16_t>(candidateIndex, index, permutation);
            count = 32;
            mergeMask = R::UpdateMask<half>(count);
            MergeCandidates<OP>(accum, index, candidate, candidateIndex, mergeMask);
        }
    }
}

// A register holds 128 channels, or several adjacent narrow rows. The caller
// supplies complete segments with a row count divisible by the packed count.
template <uint32_t CHANNELS, int OP>
__simd_vf__ inline void ReduceHalfPackedTile(
    __ubuf__ half* src, __ubuf__ half* out, __ubuf__ int64_t* arg, uint32_t segments, uint32_t rows, uint32_t rowBase)
{
    static_assert(CHANNELS == 32 || CHANNELS == 64 || CHANNELS == 128 || CHANNELS == 256, "Unsupported channel count");
    static_assert(OP == 1 || OP == 2, "Only extrema are supported");
    constexpr uint32_t PACKED_ROWS = CHANNELS < 128 ? 128 / CHANNELS : 1;
    constexpr uint32_t OUTPUT_LANES = CHANNELS < 128 ? CHANNELS : 128;
    auto fullMask = R::CreateMask<half, R::MaskPattern::ALL>();
    R::RegTensor<uint16_t> initialIndex;
    if constexpr (CHANNELS == 32 || CHANNELS == 64) {
        R::Arange((R::RegTensor<int16_t>&)initialIndex, int16_t(0));
        R::ShiftRights(initialIndex, initialIndex, int16_t(CHANNELS == 32 ? 5 : 6), fullMask);
    } else {
        R::Duplicate(initialIndex, uint16_t(0), fullMask);
    }

    for (uint32_t segment = 0; segment < segments; ++segment) {
        const uint32_t inputBase = segment * rows * CHANNELS;
        const uint32_t absoluteBase = rowBase + segment * rows;
        for (uint32_t channel = 0; channel < CHANNELS; channel += 128) {
            R::RegTensor<half> accum, candidate;
            R::RegTensor<uint16_t> index = initialIndex;
            R::RegTensor<uint16_t> candidateIndex = initialIndex;
            R::LoadAlign(accum, src + inputBase + channel);

            for (uint32_t row = PACKED_ROWS; row < rows; row += PACKED_ROWS) {
                R::LoadAlign(candidate, src + inputBase + row * CHANNELS + channel);
                R::Adds(candidateIndex, candidateIndex, uint16_t(PACKED_ROWS), fullMask);
                R::MaskReg better;
                if constexpr (OP == 1) {
                    R::Compare<half, AscendC::CMPMODE::LT>(better, candidate, accum, fullMask);
                } else {
                    R::Compare<half, AscendC::CMPMODE::GT>(better, candidate, accum, fullMask);
                }
                R::Select(accum, candidate, accum, better);
                R::Select(index, candidateIndex, index, better);
            }

            MergePackedRows<PACKED_ROWS, OP>(accum, index, candidate, candidateIndex, fullMask);

            const uint32_t outputBase = segment * CHANNELS + channel;
            uint32_t outputCount = OUTPUT_LANES;
            auto outputMask = R::UpdateMask<half>(outputCount);
            R::StoreAlign(out + outputBase, accum, outputMask);
            StoreIndices<R::HighLowPart::LOWEST>(
                arg + outputBase, index, absoluteBase, OUTPUT_LANES < 64 ? OUTPUT_LANES : 64);
            if constexpr (OUTPUT_LANES == 128) {
                StoreIndices<R::HighLowPart::HIGHEST>(arg + outputBase + 64, index, absoluteBase, 64);
            }
        }
    }
}

} // namespace half_extrema
} // namespace ops_gnn

// Signed INT64 extrema compare the high word, then unsigned low word.
namespace ops_gnn {

template <int OP>
__simd_callee__ inline void MergeInt64Candidate(
    AscendC::Reg::RegTensor<int32_t>& low, AscendC::Reg::RegTensor<int32_t>& high,
    AscendC::Reg::RegTensor<int32_t>& index, AscendC::Reg::RegTensor<int32_t>& nextLow,
    AscendC::Reg::RegTensor<int32_t>& nextHigh, uint32_t row, AscendC::Reg::MaskReg mask)
{
    namespace R64 = AscendC::Reg;
    using UnsignedWord = R64::RegTensor<uint32_t>;
    R64::RegTensor<int32_t> nextIndex;
    R64::MaskReg equalHigh, betterHigh, betterLow, better;
    R64::Compare<int32_t, AscendC::CMPMODE::EQ>(equalHigh, nextHigh, high, mask);
    if constexpr (OP == 1) {
        R64::Compare<int32_t, AscendC::CMPMODE::LT>(betterHigh, nextHigh, high, mask);
        R64::Compare<uint32_t, AscendC::CMPMODE::LT>(
            betterLow, reinterpret_cast<UnsignedWord&>(nextLow), reinterpret_cast<UnsignedWord&>(low),
            mask);
    } else {
        R64::Compare<int32_t, AscendC::CMPMODE::GT>(betterHigh, nextHigh, high, mask);
        R64::Compare<uint32_t, AscendC::CMPMODE::GT>(
            betterLow, reinterpret_cast<UnsignedWord&>(nextLow), reinterpret_cast<UnsignedWord&>(low),
            mask);
    }
    R64::Select(better, betterLow, betterHigh, equalHigh);
    R64::Duplicate(nextIndex, static_cast<int32_t>(row), mask);
    // Strict comparison and increasing row order retain the first
    // index without any equality/index merge instructions.
    R64::Select(low, nextLow, low, better);
    R64::Select(high, nextHigh, high, better);
    R64::Select(index, nextIndex, index, better);
}

// Nonempty uniform segments or a continuation of one segment. Input storage
// needs the same 512-byte readable tail padding as the normal int64 load path.
template <int OP>
__simd_vf__ inline void ReduceInt64Tile(
    __ubuf__ int64_t* src, __ubuf__ int64_t* out, __ubuf__ int64_t* arg, uint32_t segments, uint32_t rows,
    uint32_t channels, uint32_t rowBase, __ubuf__ int64_t* saved = nullptr, __ubuf__ int32_t* savedIndex = nullptr,
    bool first = true, bool last = true)
{
    static_assert(OP == 1 || OP == 2, "int64 extrema require min or max");
    namespace R64 = AscendC::Reg;
    using Word = R64::RegTensor<int32_t>;
    using UnsignedWord = R64::RegTensor<uint32_t>;

    for (uint32_t segment = 0; segment < segments; ++segment) {
        const uint32_t firstRow = rowBase + segment * rows;
        for (uint32_t channel = 0; channel < channels; channel += 64) {
            uint32_t count = channels - channel < 64 ? channels - channel : 64;
            auto mask = R64::UpdateMask<int32_t>(count);
            Word low, high, index;
            const uint32_t sourceOffset = segment * rows * channels + channel;
            const uint32_t destination = segment * channels + channel;
            uint32_t begin = 0;
            if (first) {
                R64::LoadAlign<int32_t, R64::LoadDist::DIST_DINTLV_B32>(
                    low, high, reinterpret_cast<__ubuf__ int32_t*>(src + sourceOffset));
                R64::Duplicate(index, static_cast<int32_t>(firstRow), mask);
                begin = 1;
            } else {
                R64::LoadAlign<int32_t, R64::LoadDist::DIST_DINTLV_B32>(
                    low, high, reinterpret_cast<__ubuf__ int32_t*>(saved + destination));
                R64::LoadAlign(index, savedIndex + destination);
            }

            for (uint32_t row = begin; row < rows; ++row) {
                Word nextLow, nextHigh;
                R64::LoadAlign<int32_t, R64::LoadDist::DIST_DINTLV_B32>(
                    nextLow, nextHigh, reinterpret_cast<__ubuf__ int32_t*>(src + sourceOffset + row * channels));
                MergeInt64Candidate<OP>(low, high, index, nextLow, nextHigh, firstRow + row, mask);
            }

            if (last) {
                R64::StoreAlign<int32_t, R64::StoreDist::DIST_INTLV_B32>(
                    reinterpret_cast<__ubuf__ int32_t*>(out + destination), low, high, mask);
                Word zero;
                R64::Duplicate(zero, 0, mask);
                R64::StoreAlign<int32_t, R64::StoreDist::DIST_INTLV_B32>(
                    reinterpret_cast<__ubuf__ int32_t*>(arg + destination), index, zero, mask);
            } else {
                R64::StoreAlign<int32_t, R64::StoreDist::DIST_INTLV_B32>(
                    reinterpret_cast<__ubuf__ int32_t*>(saved + destination), low, high, mask);
                R64::StoreAlign(savedIndex + destination, index, mask);
            }
        }
    }
}

} // namespace ops_gnn
