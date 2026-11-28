/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file csr_row_wise_topk_kernel.h
 * @brief Multi-core AIV kernel for CSR row-wise topk on Ascend NPU.
 *
 * Per row, the CSR window [off, off+deg) of weights (gathered through the
 * eid mapping when the matrix carries a data array) is bulk-copied GM -> UB
 * and the head k (descending) or tail k (ascending) entries are selected with
 * a sentinel-marked scan, then gathered back to (row, col, eid) triples and
 * copied out through the VECOUT queue. select-all rows are emitted in CSR
 * order without selection.
 *
 * Weight ordering semantics: only float32 weights are consumed; non-f32
 * weights are normalized to f32 on the host before launch.
 */

#include "csr_row_wise_topk/op_kernel/arch22/csr_row_wise_topk_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

template <typename IdT>
class KernelCsrRowWiseTopk {
 public:
  __aicore__ inline KernelCsrRowWiseTopk() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
      GM_ADDR weight, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
      GM_ADDR row_split, GM_ADDR out_starts, GM_ADDR tiling_ptr, TPipe* pipe) {
    const __gm__ CsrRowWiseTopkTiling* tiling =
        (const __gm__ CsrRowWiseTopkTiling*)tiling_ptr;
    num_rows_ = tiling->num_rows;
    k_ = tiling->k;
    select_all_ = tiling->select_all;
    ascending_ = tiling->ascending;
    has_data_ = tiling->has_data;
    num_total_rows_ = tiling->num_total_rows;
    const uint32_t ub_available = tiling->ub_available;

    indptr_gm_.SetGlobalBuffer((__gm__ IdT*)indptr, num_total_rows_ + 1);
    indices_gm_.SetGlobalBuffer((__gm__ IdT*)indices);
    if (has_data_) data_gm_.SetGlobalBuffer((__gm__ IdT*)data);
    rows_gm_.SetGlobalBuffer((__gm__ IdT*)rows, num_rows_);
    weight_gm_.SetGlobalBuffer((__gm__ float*)weight);
    out_rows_gm_.SetGlobalBuffer((__gm__ IdT*)out_rows);
    out_cols_gm_.SetGlobalBuffer((__gm__ IdT*)out_cols);
    out_idxs_gm_.SetGlobalBuffer((__gm__ IdT*)out_idxs);

    const uint32_t block_idx = AscendC::GetBlockIdx();
    block_idx_ = block_idx;
    const __gm__ uint32_t* row_split_p = (const __gm__ uint32_t*)row_split;
    const __gm__ uint32_t* out_starts_p = (const __gm__ uint32_t*)out_starts;
    row_begin_ = row_split_p[block_idx];
    row_end_ = row_split_p[block_idx + 1];
    out_start_ = out_starts_p[block_idx];

    const uint32_t bytes_per_elem = 48 + 7 * static_cast<uint32_t>(sizeof(IdT));
    window_elems_ = ub_available / bytes_per_elem;
    if (window_elems_ > kVerifiedWindowElemsTopk) {
      window_elems_ = kVerifiedWindowElemsTopk;
    }
    window_elems_ = window_elems_ / kSortElemsPerRepeat * kSortElemsPerRepeat;
    pipe->InitBuffer(win_w_q_, kQueueDepth, window_elems_ * sizeof(float));
    pipe->InitBuffer(win_w_buf_, window_elems_ * sizeof(float));
    pipe->InitBuffer(win_e_q_, kQueueDepth, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_r_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_c_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_e_buf_, window_elems_ * sizeof(IdT));
  }

  __aicore__ inline void Process() {
    if (row_begin_ >= row_end_) return;
    uint32_t offset = 0;
    for (uint32_t i = row_begin_; i < row_end_; ++i) {
      IdT rid = rows_gm_.GetValue(i);
      if (rid < 0 || rid >= static_cast<IdT>(num_total_rows_)) {
        continue;  // defense in depth: drop invalid seed rows
      }
      IdT off = indptr_gm_.GetValue(static_cast<uint32_t>(rid));
      IdT end = indptr_gm_.GetValue(static_cast<uint32_t>(rid) + 1);
      const uint32_t deg = static_cast<uint32_t>(end - off);
      const uint32_t num_picks = select_all_ ? deg : (deg < k_ ? deg : k_);
      if (num_picks == 0) continue;
      if (select_all_ && num_picks == deg) {
        offset += EmitRowDirect(out_start_ + offset, rid, off, deg);
        continue;
      }
      if (deg <= window_elems_) {
        offset +=
            TopkRowThroughUb(out_start_ + offset, rid, off, deg, num_picks);
      }
    }
  }

 private:
  // Selection-based top-k: repeatedly scan the row window for the current
  // best (max for descending, min for ascending) weight, mark it with a
  // sentinel so it is not picked again, and emit it. O(deg * num_picks)
  // scalar reads inside UB, which keeps small-fanout sampling cheap and
  // avoids the hardware sort network (its full-sort path faults on the
  // current runtime for some shapes). select_all is handled earlier by
  // EmitRowDirect.
  __aicore__ inline void LoadWindowWeights(
      IdT off, uint32_t deg, LocalTensor<float>& win_w, LocalTensor<IdT>& win_e) {
    if (has_data_) {
      win_w = win_w_buf_.Get<float>();
      win_e = win_e_q_.AllocTensor<IdT>();
      DataCopyPad(
          win_e, data_gm_[off],
          DataCopyExtParams{1, deg * sizeof(IdT), 0, 0, 0},
          DataCopyPadExtParams<IdT>{false, 0, 0, 0});
      win_e_q_.EnQue(win_e);
      win_e = win_e_q_.DeQue<IdT>();
      for (uint32_t j = 0; j < deg; ++j) {
        win_w.SetValue(
            j, weight_gm_.GetValue(static_cast<uint32_t>(win_e.GetValue(j))));
      }
      return;
    }
    win_w = win_w_q_.AllocTensor<float>();
    DataCopyPad(
        win_w, weight_gm_[off],
        DataCopyExtParams{
            1, deg * static_cast<uint32_t>(sizeof(float)), 0, 0, 0},
        DataCopyPadExtParams<float>{false, 0, 0, 0});
    win_w_q_.EnQue(win_w);
    win_w = win_w_q_.DeQue<float>();
  }

  __aicore__ inline void EmitTopkPicks(
      uint32_t out_pos, IdT rid, IdT off, uint32_t deg, uint32_t num_picks,
      LocalTensor<float>& win_w, LocalTensor<IdT>& win_e) {
    LocalTensor<IdT> out_r = out_r_buf_.Get<IdT>();
    LocalTensor<IdT> out_c = out_c_buf_.Get<IdT>();
    LocalTensor<IdT> out_e = out_e_buf_.Get<IdT>();
    for (uint32_t j = 0; j < num_picks; ++j) {
      float best = ascending_ ? kPosInf : kNegInf;
      uint32_t best_local = 0;
      for (uint32_t t = 0; t < deg; ++t) {
        const float v = win_w.GetValue(t);
        if (ascending_ ? (v < best) : (v > best)) {
          best = v;
          best_local = t;
        }
      }
      win_w.SetValue(best_local, ascending_ ? kPosInf : kNegInf);
      const IdT picked = static_cast<IdT>(off) + static_cast<IdT>(best_local);
      out_r.SetValue(j, rid);
      out_c.SetValue(j, indices_gm_.GetValue(static_cast<uint64_t>(picked)));
      out_e.SetValue(j, has_data_ ? data_gm_.GetValue(static_cast<uint64_t>(picked)) : picked);
    }
    CopyOutStaged(out_r, out_rows_gm_[out_pos], num_picks);
    CopyOutStaged(out_c, out_cols_gm_[out_pos], num_picks);
    CopyOutStaged(out_e, out_idxs_gm_[out_pos], num_picks);
  }

  __aicore__ inline uint32_t TopkRowThroughUb(
      uint32_t out_pos, IdT rid, IdT off, uint32_t deg, uint32_t num_picks) {
    LocalTensor<float> win_w;
    LocalTensor<IdT> win_e;
    LoadWindowWeights(off, deg, win_w, win_e);
    EmitTopkPicks(out_pos, rid, off, deg, num_picks, win_w, win_e);
    if (has_data_) {
      win_e_q_.FreeTensor(win_e);
    } else {
      win_w_q_.FreeTensor(win_w);
    }
    return num_picks;
  }

  __aicore__ inline uint32_t EmitRowDirect(
      uint32_t out_pos, IdT rid, IdT off, uint32_t deg) {
    LocalTensor<IdT> out_r = out_r_buf_.Get<IdT>();
    LocalTensor<IdT> out_c = out_c_buf_.Get<IdT>();
    LocalTensor<IdT> out_e = out_e_buf_.Get<IdT>();
    for (uint32_t j = 0; j < deg; ++j) {
      const IdT picked = off + static_cast<IdT>(j);
      out_r.SetValue(j, rid);
      out_c.SetValue(j, indices_gm_.GetValue(picked));
      out_e.SetValue(j, has_data_ ? data_gm_.GetValue(picked) : picked);
    }
    CopyOutStaged(out_r, out_rows_gm_[out_pos], deg);
    CopyOutStaged(out_c, out_cols_gm_[out_pos], deg);
    CopyOutStaged(out_e, out_idxs_gm_[out_pos], deg);
    return deg;
  }

  static constexpr uint32_t kQueueDepth = 2;
  static constexpr uint32_t kSortElemsPerRepeat = 32;
  static constexpr float kNegInf = -3.402823466e38F;
  static constexpr float kPosInf = 3.402823466e38F;

  __aicore__ inline void CopyOutStaged(
      LocalTensor<IdT>& staging, GlobalTensor<IdT> dst, uint32_t count) {
    event_t eid =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eid);
    WaitFlag<HardEvent::V_MTE3>(eid);
    DataCopyExtParams cp{
        1, static_cast<uint32_t>(count * sizeof(IdT)), 0, 0, 0};
    DataCopyPad(dst, staging, cp);
    // DataCopyPad is asynchronous; drain MTE3 before the next row reuses the
    // same VECCALC staging buffer.
    event_t mte3v =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(mte3v);
    WaitFlag<HardEvent::MTE3_V>(mte3v);
  }

  GlobalTensor<IdT> indptr_gm_, indices_gm_, data_gm_, rows_gm_;
  GlobalTensor<float> weight_gm_;
  GlobalTensor<IdT> out_rows_gm_, out_cols_gm_, out_idxs_gm_;
  TQue<TPosition::VECIN, kQueueDepth> win_w_q_, win_e_q_;
  TBuf<TPosition::VECCALC> win_w_buf_;
  TBuf<TPosition::VECCALC> out_r_buf_, out_c_buf_, out_e_buf_;
  uint32_t num_rows_ = 0, k_ = 0, select_all_ = 0, ascending_ = 0;
  uint32_t has_data_ = 0, num_total_rows_ = 0;
  uint32_t out_start_ = 0, row_begin_ = 0, row_end_ = 0, window_elems_ = 0;
  uint32_t block_idx_ = 0;
};

extern "C" __global__ __aicore__ void csr_row_wise_topk_int32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows, GM_ADDR weight,
    GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs, GM_ADDR row_split,
    GM_ADDR out_starts, GM_ADDR tiling_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KernelCsrRowWiseTopk<int32_t> op;
  TPipe pipe;
  op.Init(indptr, indices, data, rows, weight, out_rows, out_cols, out_idxs,
          row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_row_wise_topk_int64(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows, GM_ADDR weight,
    GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs, GM_ADDR row_split,
    GM_ADDR out_starts, GM_ADDR tiling_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KernelCsrRowWiseTopk<int64_t> op;
  TPipe pipe;
  op.Init(indptr, indices, data, rows, weight, out_rows, out_cols, out_idxs,
          row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}
