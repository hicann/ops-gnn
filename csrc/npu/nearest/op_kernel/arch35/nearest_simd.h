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

#include "kernel_operator.h"

namespace nearest_simd {
using namespace AscendC;
constexpr uint32_t TILE_X = 128;
constexpr uint32_t TILE_Y = 8192;

__simd_callee__ inline void LoadBest(Reg::RegTensor<float>& minimum,
                                    Reg::RegTensor<int32_t>& winner,
                                    __ubuf__ float* best, __ubuf__ int32_t* index,
                                    uint32_t row, uint32_t begin)
{
    if (begin == 0) {
        Reg::Duplicate(minimum, __builtin_inff());
        Reg::Duplicate(winner, 0);
    } else {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(minimum, best + row);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_BRC_B32>(winner, index + row);
    }
}

__simd_callee__ inline void UpdateBest(Reg::RegTensor<float>& minimum,
                                      Reg::RegTensor<int32_t>& winner,
                                      Reg::RegTensor<float>& distance,
                                      Reg::MaskReg& valid, uint32_t begin)
{
    Reg::RegTensor<int32_t> candidates;
    Reg::MaskReg update;
    Reg::Arange(candidates, (int32_t)begin);
    Reg::Compare<float, CMPMODE::LT>(update, distance, minimum, valid);
    Reg::Select(minimum, distance, minimum, update);
    Reg::Select(winner, candidates, winner, update);
}

__simd_callee__ inline void StoreBest(Reg::RegTensor<float>& minimum,
                                     Reg::RegTensor<int32_t>& winner,
                                     __ubuf__ float* best, __ubuf__ int32_t* index, uint32_t row)
{
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg one = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    Reg::RegTensor<float> reduced;
    Reg::RegTensor<int32_t> invalid, final_index;
    Reg::MaskReg equal;
    Reg::Duplicate(invalid, 0x7fffffff);
    Reg::Reduce<Reg::ReduceType::MIN>(reduced, minimum, all);
    Reg::Duplicate(reduced, reduced, all);
    Reg::Compare<float, CMPMODE::EQ>(equal, minimum, reduced, all);
    Reg::Select(winner, winner, invalid, equal);
    Reg::Reduce<Reg::ReduceType::MIN>(final_index, winner, all);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(best + row, reduced, one);
    Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(index + row, final_index, one);
}

__aicore__ inline void StoreIndices(GlobalTensor<int64_t> target, LocalTensor<int64_t> output,
                                   LocalTensor<int32_t> index, uint32_t count)
{
    Cast(output, index, RoundMode::CAST_NONE, count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(target, output, DataCopyExtParams{1, count * 8, 0, 0, 0});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

__aicore__ inline void Compute3(LocalTensor<float> x, LocalTensor<float> y,
                               LocalTensor<float> best, LocalTensor<int32_t> index,
                               uint32_t nx, uint32_t ny, uint32_t begin)
{
    __ubuf__ float* xp = (__ubuf__ float*)x.GetPhyAddr();
    __ubuf__ float* yp = (__ubuf__ float*)y.GetPhyAddr();
    __ubuf__ float* bp = (__ubuf__ float*)best.GetPhyAddr();
    __ubuf__ int32_t* ip = (__ubuf__ int32_t*)index.GetPhyAddr();
    uint16_t rounds = (ny + 63) / 64;
    __VEC_SCOPE__ {
        Reg::RegTensor<float> x0, x1, x2, y0, y1, y2, diff, distance, part;
        Reg::RegTensor<float> minimum;
        Reg::RegTensor<int32_t> winner;
        for (uint16_t i = 0; i < (uint16_t)nx; ++i) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(x0, xp + i);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(x1, xp + TILE_X + i);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(x2, xp + 2 * TILE_X + i);
            LoadBest(minimum, winner, bp, ip, i, begin);
            uint32_t remaining = ny;
            for (uint16_t j = 0; j < rounds; ++j) {
                Reg::MaskReg valid = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign(y0, yp + j * 64);
                Reg::LoadAlign(y1, yp + TILE_Y + j * 64);
                Reg::LoadAlign(y2, yp + 2 * TILE_Y + j * 64);
                Reg::Sub(diff, y0, x0, valid);
                Reg::Mul(distance, diff, diff, valid);
                Reg::Sub(diff, y1, x1, valid);
                Reg::Mul(part, diff, diff, valid);
                Reg::Add(distance, distance, part, valid);
                Reg::Sub(diff, y2, x2, valid);
                Reg::Mul(part, diff, diff, valid);
                Reg::Add(distance, distance, part, valid);
                UpdateBest(minimum, winner, distance, valid, begin + j * 64);
            }
            StoreBest(minimum, winner, bp, ip, i);
        }
    }
}

__attribute__((aiv)) __global__ __aicore__ void Search3(
    __gm__ float* work, __gm__ int64_t* out, uint32_t n, uint32_t m)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> xb, yb, bb, ib, ob;
    pipe.InitBuffer(xb, 3 * TILE_X * sizeof(float));
    pipe.InitBuffer(yb, 3 * TILE_Y * sizeof(float));
    pipe.InitBuffer(bb, TILE_X * sizeof(float));
    pipe.InitBuffer(ib, TILE_X * sizeof(int32_t));
    pipe.InitBuffer(ob, TILE_X * sizeof(int64_t));
    auto xl = xb.Get<float>();
    auto yl = yb.Get<float>();
    auto best = bb.Get<float>();
    auto index = ib.Get<int32_t>();
    auto output = ob.Get<int64_t>();
    GlobalTensor<float> xg, yg;
    GlobalTensor<int64_t> og;
    xg.SetGlobalBuffer(work);
    yg.SetGlobalBuffer(work + (uint64_t)n * 3);
    og.SetGlobalBuffer(out);
    for (uint32_t start = GetBlockIdx() * TILE_X; start < n;
         start += GetBlockNum() * TILE_X) {
        uint32_t nx = n - start < TILE_X ? n - start : TILE_X;
        for (uint32_t d = 0; d < 3; ++d) {
            DataCopyPad(xl[d * TILE_X], xg[(uint64_t)d * n + start],
                        DataCopyExtParams{1, nx * 4, 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        for (uint32_t begin = 0; begin < m; begin += TILE_Y) {
            uint32_t ny = m - begin < TILE_Y ? m - begin : TILE_Y;
            for (uint32_t d = 0; d < 3; ++d) {
                DataCopyPad(yl[d * TILE_Y], yg[(uint64_t)d * m + begin],
                            DataCopyExtParams{1, ny * 4, 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Compute3(xl, yl, best, index, nx, ny, begin);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        StoreIndices(og[start], output, index, nx);
    }
}

__simd_callee__ inline void HighDistance(Reg::RegTensor<float>& distance,
                                        Reg::RegTensor<float>& factor, Reg::MaskReg& valid,
                                        __ubuf__ float* xp, __ubuf__ float* yp,
                                        uint32_t features, uint32_t tx, uint32_t ty)
{
    Reg::RegTensor<float> xv, yv, dot;
    uint16_t panels = features > 448 ? 2 : 1;
    uint16_t panel_size = features > 448 ? (features + 1) / 2 : features;
    Reg::Duplicate(distance, 0.0f);
    // scipy's pinned OpenBLAS SkylakeX reference uses Q=448,
    // with a balanced split for the final two panels.
    for (uint16_t panel = 0; panel < panels; ++panel) {
        uint16_t begin_k = panel * panel_size;
        uint16_t count = panel == 0 ? panel_size : features - begin_k;
        Reg::Duplicate(dot, 0.0f);
        for (uint16_t k = 0; k < count; ++k) {
            uint16_t d = begin_k + k;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(xv, xp + d * tx);
            Reg::LoadAlign(yv, yp + d * ty);
            Reg::MulAddDst(dot, xv, yv, valid);
        }
        Reg::MulAddDst(distance, dot, factor, valid);
    }
}

__aicore__ inline void ComputeHigh(LocalTensor<float> x, LocalTensor<float> y,
                                  LocalTensor<float> best, LocalTensor<int32_t> index,
                                  uint32_t nx, uint32_t ny, uint32_t begin,
                                  uint32_t features, uint32_t tx, uint32_t ty)
{
    __ubuf__ float* xp = (__ubuf__ float*)x.GetPhyAddr();
    __ubuf__ float* yp = (__ubuf__ float*)y.GetPhyAddr();
    __ubuf__ float* bp = (__ubuf__ float*)best.GetPhyAddr();
    __ubuf__ int32_t* ip = (__ubuf__ int32_t*)index.GetPhyAddr();
    uint16_t rounds = (ny + 63) / 64;
    __VEC_SCOPE__ {
        Reg::RegTensor<float> distance, minimum, xnorm, ynorm, factor;
        Reg::RegTensor<int32_t> winner;
        Reg::Duplicate(factor, -2.0f);
        for (uint16_t i = 0; i < (uint16_t)nx; ++i) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(xnorm, xp + features * tx + i);
            LoadBest(minimum, winner, bp, ip, i, begin);
            uint32_t remaining = ny;
            for (uint16_t j = 0; j < rounds; ++j) {
                Reg::MaskReg valid = Reg::UpdateMask<float>(remaining);
                HighDistance(distance, factor, valid, xp + i, yp + j * 64, features, tx, ty);
                Reg::LoadAlign(ynorm, yp + features * ty + j * 64);
                Reg::Add(distance, distance, xnorm, valid);
                Reg::Add(distance, distance, ynorm, valid);
                UpdateBest(minimum, winner, distance, valid, begin + j * 64);
            }
            StoreBest(minimum, winner, bp, ip, i);
        }
    }
}

__attribute__((aiv)) __global__ __aicore__ void SearchHigh(
    __gm__ float* work, __gm__ int64_t* out, NearestTiling t)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> xb, yb, bb, ib, ob;
    uint32_t tx = t.tile_x;
    uint32_t ty = t.tile_y;
    pipe.InitBuffer(xb, (t.features + 1) * tx * sizeof(float));
    pipe.InitBuffer(yb, (t.features + 1) * ty * sizeof(float));
    pipe.InitBuffer(bb, tx * sizeof(float));
    pipe.InitBuffer(ib, tx * sizeof(int32_t));
    pipe.InitBuffer(ob, tx * sizeof(int64_t));
    auto xl = xb.Get<float>();
    auto yl = yb.Get<float>();
    auto best = bb.Get<float>();
    auto index = ib.Get<int32_t>();
    auto output = ob.Get<int64_t>();
    GlobalTensor<float> xg, yg, norms;
    GlobalTensor<int64_t> og;
    xg.SetGlobalBuffer(work);
    yg.SetGlobalBuffer(work + (uint64_t)t.n * t.features);
    norms.SetGlobalBuffer(work + (uint64_t)(t.n + t.m) * t.features);
    og.SetGlobalBuffer(out);
    for (uint32_t start = GetBlockIdx() * tx; start < t.n; start += GetBlockNum() * tx) {
        uint32_t nx = t.n - start < tx ? t.n - start : tx;
        DataCopyPad(xl, xg[start],
                    DataCopyExtParams{(uint16_t)t.features, nx * 4, (t.n - nx) * 4,
                                      (tx - (nx + 7) / 8 * 8) / 8, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(xl[t.features * tx], norms[start], DataCopyExtParams{1, nx * 4, 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        for (uint32_t begin = 0; begin < t.m; begin += ty) {
            uint32_t ny = t.m - begin < ty ? t.m - begin : ty;
            DataCopyPad(yl, yg[begin],
                        DataCopyExtParams{(uint16_t)t.features, ny * 4, (t.m - ny) * 4,
                                          (ty - (ny + 7) / 8 * 8) / 8, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
            DataCopyPad(yl[t.features * ty], norms[t.n + begin],
                        DataCopyExtParams{1, ny * 4, 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            ComputeHigh(xl, yl, best, index, nx, ny, begin, t.features, tx, ty);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        StoreIndices(og[start], output, index, nx);
    }
}
} // namespace nearest_simd
