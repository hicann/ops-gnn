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

// Integer mean divides wrapped sums with truncation toward zero.
namespace ops_gnn {
namespace integer_mean {

template <typename T, typename V>
__simd_callee__ __attribute__((always_inline)) inline void Divide(
    V& value, uint32_t count, int16_t shift, AscendC::Reg::MaskReg mask)
{
    namespace R = AscendC::Reg;
    V denominator;
    if (shift >= 0) {
        // Bias negative values before arithmetic shift for truncation to zero.
        V sign, bias;
        R::ShiftRights(sign, value, int16_t(sizeof(T) * 8 - 1), mask);
        R::Duplicate(denominator, static_cast<T>(count - 1), mask);
        R::And(bias, sign, denominator, mask);
        R::Add(value, value, bias, mask);
        R::ShiftRights(value, value, shift, mask);
    } else {
        R::Duplicate(denominator, static_cast<T>(count), mask);
        R::Div(value, value, denominator, mask);
    }
}

} // namespace integer_mean
} // namespace ops_gnn

// Short uniform sum and mean tiles.
namespace ops_gnn {
namespace short_sum {
namespace R = AscendC::Reg;

constexpr R::CastTrait kWiden = {
    R::RegLayout::ZERO, R::SatMode::UNKNOWN, R::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr R::CastTrait kToFloat = {
    R::RegLayout::ZERO, R::SatMode::UNKNOWN, R::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
constexpr R::CastTrait kToHalf = {
    R::RegLayout::ZERO, R::SatMode::NO_SAT, R::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
constexpr R::CastTrait kToInteger = {
    R::RegLayout::ZERO, R::SatMode::NO_SAT, R::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};

template <typename T>
struct SumRegister {
    using Type = R::RegTensor<T>;
};
template <>
struct SumRegister<half> {
    using Type = R::RegTensor<float>;
};
template <>
struct SumRegister<int64_t> {
    using Type = R::RegTensor<int64_t, R::RegTraitNumTwo>;
};

template <typename T>
__simd_callee__ __attribute__((always_inline)) inline void Load(
    typename SumRegister<T>::Type& value, __ubuf__ T* src, R::MaskReg mask)
{
    if constexpr (AscendC::Std::is_same_v<T, half>) {
        R::RegTensor<half> packed;
        R::LoadAlign<half, R::LoadDist::DIST_UNPACK_B16>(packed, src);
        R::Cast<float, half, kWiden>(value, packed, mask);
    } else {
        R::LoadAlign(value, src);
    }
}

template <typename T, uint32_t ROWS, bool MEAN>
__simd_callee__ inline void StoreShortSum(
    typename SumRegister<T>::Type& sum, __ubuf__ T* out, R::MaskReg mask)
{
    if constexpr (MEAN) {
        if constexpr (AscendC::Std::is_same_v<T, int32_t> || AscendC::Std::is_same_v<T, int64_t>) {
            constexpr int16_t shift = ROWS == 2 ? 1 : ROWS == 4 ? 2 : 3;
            integer_mean::Divide<T>(sum, ROWS, shift, mask);
        } else {
            R::RegTensor<float> numerator;
            if constexpr (AscendC::Std::is_same_v<T, half>) {
                R::RegTensor<half> roundedSum;
                R::Cast<half, float, kToHalf>(roundedSum, sum, mask);
                R::Cast<float, half, kWiden>(numerator, roundedSum, mask);
            } else if constexpr (AscendC::Std::is_same_v<T, float>) {
                numerator = sum;
            }
            constexpr float reciprocal = ROWS == 2 ? 0.5f : ROWS == 4 ? 0.25f : 0.125f;
            R::Muls(numerator, numerator, reciprocal, mask);
            if constexpr (AscendC::Std::is_same_v<T, half> || AscendC::Std::is_same_v<T, float>) {
                sum = numerator;
            }
        }
    }

    if constexpr (AscendC::Std::is_same_v<T, half>) {
        R::RegTensor<half> packed;
        R::Cast<half, float, kToHalf>(packed, sum, mask);
        R::StoreAlign<half, R::StoreDist::DIST_PACK_B32>(out, packed, mask);
    } else {
        R::StoreAlign(out, sum, mask);
    }
}

// Every iteration produces 64 contiguous outputs. Row and channel dimensions
// are powers of two, so the source address requires only shifts and masks.
template <typename T, uint32_t ROWS, uint32_t CHANNELS, bool MEAN>
__simd_vf__ inline void ReduceShortSumTile(__ubuf__ T* src, __ubuf__ T* out, uint32_t segments)
{
    static_assert(ROWS == 2 || ROWS == 4 || ROWS == 8, "Only two, four or eight rows are supported");
    static_assert(CHANNELS == 64 || CHANNELS == 128 || CHANNELS == 256, "Unsupported channel count");
    static_assert(
        AscendC::Std::is_same_v<T, half> || AscendC::Std::is_same_v<T, float> || AscendC::Std::is_same_v<T, int32_t> ||
            AscendC::Std::is_same_v<T, int64_t>,
        "Unsupported dtype");
    using V = typename SumRegister<T>::Type;
    auto mask = R::CreateMask<int32_t, R::MaskPattern::ALL>();
    for (uint32_t outputOffset = 0; outputOffset < segments * CHANNELS; outputOffset += 64) {
        const uint32_t segment = outputOffset / CHANNELS;
        const uint32_t channel = outputOffset % CHANNELS;
        const uint32_t inputOffset = segment * ROWS * CHANNELS + channel;
        V sum, value;
        Load<T>(sum, src + inputOffset, mask);
        Load<T>(value, src + inputOffset + CHANNELS, mask);
        R::Add(sum, sum, value, mask);
        if constexpr (ROWS >= 4) {
            V secondPair, last;
            Load<T>(secondPair, src + inputOffset + 2 * CHANNELS, mask);
            Load<T>(last, src + inputOffset + 3 * CHANNELS, mask);
            R::Add(secondPair, secondPair, last, mask);
            R::Add(sum, sum, secondPair, mask);
        }
        if constexpr (ROWS == 8) {
            V thirdPair, fourthPair, fifth, seventh;
            Load<T>(thirdPair, src + inputOffset + 4 * CHANNELS, mask);
            Load<T>(fifth, src + inputOffset + 5 * CHANNELS, mask);
            Load<T>(fourthPair, src + inputOffset + 6 * CHANNELS, mask);
            Load<T>(seventh, src + inputOffset + 7 * CHANNELS, mask);
            R::Add(thirdPair, thirdPair, fifth, mask);
            R::Add(fourthPair, fourthPair, seventh, mask);
            R::Add(thirdPair, thirdPair, fourthPair, mask);
            R::Add(sum, sum, thirdPair, mask);
        }

        StoreShortSum<T, ROWS, MEAN>(sum, out + outputOffset, mask);
    }
}

} // namespace short_sum
} // namespace ops_gnn

// Narrow FP16 sum and mean tiles.
namespace ops_gnn {
namespace narrow_half_sum {
namespace R = AscendC::Reg;

constexpr R::CastTrait kWiden = {
    R::RegLayout::ZERO, R::SatMode::UNKNOWN, R::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr R::CastTrait kToHalf = {
    R::RegLayout::ZERO, R::SatMode::NO_SAT, R::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

__simd_callee__ __attribute__((always_inline)) inline void Load(
    R::RegTensor<float>& value, __ubuf__ half* src, R::MaskReg mask)
{
    R::RegTensor<half> packed;
    R::LoadAlign<half, R::LoadDist::DIST_UNPACK_B16>(packed, src);
    R::Cast<float, half, kWiden>(value, packed, mask);
}

template <uint32_t OFFSET, uint32_t PACKED_ROWS>
__simd_callee__ __attribute__((always_inline)) inline void AccumulateRemaining(
    R::RegTensor<float>& a0, R::RegTensor<float>& a1, R::RegTensor<float>& a2, R::RegTensor<float>& a3,
    __ubuf__ half* src, R::MaskReg mask)
{
    if constexpr (OFFSET < PACKED_ROWS) {
        R::RegTensor<float> v0, v1, v2, v3;
        Load(v0, src + OFFSET * 64, mask);
        Load(v1, src + (OFFSET + 1) * 64, mask);
        Load(v2, src + (OFFSET + 2) * 64, mask);
        Load(v3, src + (OFFSET + 3) * 64, mask);
        R::Add(a0, a0, v0, mask);
        R::Add(a1, a1, v1, mask);
        R::Add(a2, a2, v2, mask);
        R::Add(a3, a3, v3, mask);
        AccumulateRemaining<OFFSET + 4, PACKED_ROWS>(a0, a1, a2, a3, src, mask);
    }
}

// Two 32-channel rows occupy each float register after widening from half.
template <uint32_t ROWS, bool MEAN>
__simd_vf__ inline void ReduceNarrowHalfSum(__ubuf__ half* src, __ubuf__ half* out, uint32_t segments)
{
    static_assert(ROWS == 16 || ROWS == 32 || ROWS == 64, "Only 16, 32 or 64 rows are supported");
    auto fullMask = R::CreateMask<float, R::MaskPattern::ALL>();
    uint32_t outputCount = 32;
    auto outputMask = R::UpdateMask<float>(outputCount);
    R::RegTensor<uint32_t> permutation, flip;
    R::Arange((R::RegTensor<int32_t>&)permutation, int32_t(0));
    R::Duplicate(flip, uint32_t(32), fullMask);
    R::Xor(permutation, permutation, flip, fullMask);

    for (uint32_t segment = 0; segment < segments; ++segment) {
        auto segmentSrc = src + segment * ROWS * 32;
        R::RegTensor<float> a0, a1, a2, a3, paired;
        Load(a0, segmentSrc, fullMask);
        Load(a1, segmentSrc + 64, fullMask);
        Load(a2, segmentSrc + 128, fullMask);
        Load(a3, segmentSrc + 192, fullMask);
        AccumulateRemaining<4, ROWS / 2>(a0, a1, a2, a3, segmentSrc, fullMask);
        R::Add(a0, a0, a1, fullMask);
        R::Add(a2, a2, a3, fullMask);
        R::Add(a0, a0, a2, fullMask);
        R::Gather<float, uint32_t>(paired, a0, permutation);
        R::Add(a0, a0, paired, outputMask);

        R::RegTensor<half> packed;
        if constexpr (MEAN) {
            R::Cast<half, float, kToHalf>(packed, a0, outputMask);
            R::Cast<float, half, kWiden>(a0, packed, outputMask);
            R::Muls(a0, a0, 1.0f / static_cast<float>(ROWS), outputMask);
        }
        R::Cast<half, float, kToHalf>(packed, a0, outputMask);
        R::StoreAlign<half, R::StoreDist::DIST_PACK_B32>(out + segment * 32, packed, outputMask);
    }
}

} // namespace narrow_half_sum
} // namespace ops_gnn
