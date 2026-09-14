/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "spmm_max_kernel.h"
#include <cstdint>
#include "kernel_operator.h"
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_RESERVED = 2048;
constexpr uint32_t ALIGN_BYTES = 32;

class SpmmMax {
public:
    __aicore__ inline void Init(
        GM_ADDR featureData,
        GM_ADDR outputData,
        GM_ADDR indptrData,
        GM_ADDR indicesData,
        GM_ADDR vectorRowSplitData,
        uint32_t numDstRows,
        uint32_t numSrcRows,
        uint32_t featureDim,
        uint32_t nonZeroCount, uint32_t ubBytes, uint32_t hasNan,
        AscendC::TPipe *pipe)
    {
        this->M = numDstRows;
        this->K = numSrcRows;
        this->N = featureDim;
        this->nnz = nonZeroCount;
        this->hasNan = hasNan;
        this->localRowCount = 0;
        // Defense in depth before alignment arithmetic or queue allocation.
        if (ubBytes <= UB_RESERVED || N == 0 ||
            N > ((ubBytes - UB_RESERVED) / 4 / ALIGN_BYTES * ALIGN_BYTES) / sizeof(half) ||
            K > 0xffffffffU / N || M > 0xffffffffU / N || M == 0xffffffffU) {
            return;
        }

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        this->startRow = 0;
        this->localRowCount = 0;

        rowSplitGm.SetGlobalBuffer((__gm__ uint32_t *)vectorRowSplitData, blockNum + 1);
        this->startRow = rowSplitGm.GetValue(blockIdx);
        uint32_t endRow = rowSplitGm.GetValue(blockIdx + 1);
        if (endRow < startRow || endRow > M) return;
        this->localRowCount = endRow - startRow;

        featureGm.SetGlobalBuffer((__gm__ half *)featureData, K * N);
        featureBitsGm.SetGlobalBuffer((__gm__ uint16_t *)featureData, K * N);
        outputGm.SetGlobalBuffer((__gm__ half *)outputData, M * N);
        indptrGm.SetGlobalBuffer((__gm__ uint32_t *)indptrData, M + 1);
        indicesGm.SetGlobalBuffer((__gm__ uint32_t *)indicesData, nnz);

        this->rowBytes = N * sizeof(half);
        this->rowAlignedBytes = (this->rowBytes + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        this->rowAlignedElems = this->rowAlignedBytes / sizeof(half);
        this->rightPadding = this->rowAlignedElems - N;

        uint32_t accumBytes = BUFFER_NUM * this->rowAlignedBytes;
        uint32_t available = ubBytes > UB_RESERVED ? ubBytes - UB_RESERVED : 0;
        uint32_t remainingUb = available > accumBytes ? available - accumBytes : 0;
        this->batchSize = remainingUb / (BUFFER_NUM * this->rowAlignedBytes);
        if (this->batchSize == 0) {
            this->localRowCount = 0;
            return; // Host rejects this configuration.
        }
        uint32_t batchBufferSize = this->batchSize * this->rowAlignedBytes;
        pipe->InitBuffer(accumQueue, BUFFER_NUM, this->rowAlignedBytes);
        pipe->InitBuffer(featureQueue, BUFFER_NUM, batchBufferSize);
    }

    __aicore__ inline void Process()
    {
        if (this->localRowCount == 0) {
            return;
        }
        uint32_t rowEnd = this->startRow + this->localRowCount;
        for (uint32_t row = this->startRow; row < rowEnd; ++row) {
            uint32_t rowStartPtr = indptrGm.GetValue(row);
            uint32_t rowEndPtr = indptrGm.GetValue(row + 1);
            AscendC::LocalTensor<half> accumUb = accumQueue.AllocTensor<half>();
            InitRow(accumUb, rowStartPtr, rowEndPtr);
            RepairNan(accumUb, rowStartPtr, rowEndPtr);
            accumQueue.EnQue(accumUb);
            CopyOut(row);
        }
    }

private:

    __aicore__ inline void InitRow(AscendC::LocalTensor<half> &accumUb,
                                   uint32_t rowStartPtr, uint32_t rowEndPtr)
    {
        uint32_t rowNonZeroCount = rowEndPtr - rowStartPtr;
        if (rowNonZeroCount == 0) {
            // Vector instructions operate on the 32-byte aligned UB row.
            // CopyOut still writes only N valid feature elements.
            AscendC::Duplicate<half>(accumUb, half(0.0f), this->rowAlignedElems);
            return;
        }
        AscendC::Duplicate<half>(accumUb, half(-__builtin_inff()), this->rowAlignedElems);
        uint32_t batchCount = rowNonZeroCount / this->batchSize + (rowNonZeroCount % this->batchSize != 0);
        for (uint32_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
            uint32_t batchStartPtr = rowStartPtr + batchIndex * this->batchSize;
            uint32_t currentBatchSize = (this->batchSize > rowEndPtr - batchStartPtr)
                                        ? (rowEndPtr - batchStartPtr)
                                        : this->batchSize;
            CopyInBatch(batchStartPtr, currentBatchSize);
            ComputeBatchMax(accumUb, currentBatchSize);
        }
    }

    __aicore__ inline void RepairNan(AscendC::LocalTensor<half> &accumUb,
                                     uint32_t rowStartPtr, uint32_t rowEndPtr)
    {
        // AscendC Max NaN rules differ across devices. Repair only when x
        // contains NaNs, preserving PyTorch amax propagation for each column.
        uint32_t rowNonZeroCount = rowEndPtr - rowStartPtr;
        if (!hasNan || rowNonZeroCount == 0) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t e = rowStartPtr; e < rowEndPtr; ++e) {
            uint32_t src = indicesGm.GetValue(e);
            for (uint32_t f = 0; f < N; ++f) {
                uint32_t offset = src * N + f;
                uint16_t bits = featureBitsGm.GetValue(offset);
                // IEEE FP16 NaN: all exponent bits set and a nonzero mantissa.
                if ((bits & 0x7c00U) == 0x7c00U && (bits & 0x03ffU) != 0) {
                    accumUb.SetValue(f, featureGm.GetValue(offset));
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInBatch(uint32_t batchStart, uint32_t batchNnz)
    {
        AscendC::LocalTensor<half> featureBatch = featureQueue.AllocTensor<half>();
        AscendC::DataCopyExtParams copyParams = {1, this->rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> padParams = {true, 0, (uint8_t)this->rightPadding, half(0.0f)};
        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t neighborIndex = indicesGm.GetValue(batchStart + i);
            AscendC::DataCopyPad<half>(featureBatch[i * this->rowAlignedElems], featureGm[neighborIndex * this->N], copyParams, padParams);
        }
        featureQueue.EnQue(featureBatch);
    }

    __aicore__ inline void ComputeBatchMax(AscendC::LocalTensor<half> &accumBlock, uint32_t batchNnz)
    {
        AscendC::LocalTensor<half> computeBatch = featureQueue.DeQue<half>();
        for (uint32_t i = 0; i < batchNnz; ++i) {
            AscendC::Max(accumBlock, accumBlock, computeBatch[i * this->rowAlignedElems],
                         this->rowAlignedElems);
        }
        featureQueue.FreeTensor(computeBatch);
    }

    __aicore__ inline void CopyOut(uint32_t row)
    {
        AscendC::LocalTensor<half> accumBlock = accumQueue.DeQue<half>();
        AscendC::DataCopyExtParams copyParams = {1, this->rowBytes, 0, 0, 0};
        AscendC::DataCopyPad<half>(outputGm[row * this->N], accumBlock, copyParams);
        accumQueue.FreeTensor(accumBlock);
    }

    uint32_t M, K, N, nnz, hasNan;
    uint32_t batchSize;
    uint32_t rowBytes, rowAlignedBytes, rowAlignedElems, rightPadding;
    uint32_t startRow, localRowCount;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> accumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> featureQueue;
    AscendC::GlobalTensor<half> featureGm;
    AscendC::GlobalTensor<uint16_t> featureBitsGm;
    AscendC::GlobalTensor<half> outputGm;
    AscendC::GlobalTensor<uint32_t> indptrGm, indicesGm, rowSplitGm;
};

extern "C" __global__ __aicore__ void spmm_max(
    GM_ADDR featureData,
    GM_ADDR outputData,
    GM_ADDR indptrData,
    GM_ADDR indicesData,
    GM_ADDR vectorRowSplitData,
    uint32_t numDstRows,
    uint32_t numSrcRows,
    uint32_t featureDim,
    uint32_t nonZeroCount, uint32_t ubBytes, uint32_t hasNan)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    SpmmMax vectorProcessor;
    vectorProcessor.Init(
        featureData,
        outputData,
        indptrData,
        indicesData,
        vectorRowSplitData,
        numDstRows,
        numSrcRows,
        featureDim,
        nonZeroCount, ubBytes, hasNan,
        &pipe);
    vectorProcessor.Process();
}

void LaunchSpmmMax(uint32_t blocks, aclrtStream stream, void* x, void* out,
                   void* ptr, void* idx, void* split, uint32_t m, uint32_t k,
                   uint32_t n, uint32_t nnz, uint32_t ubBytes, uint32_t hasNan)
{
    spmm_max<<<blocks, nullptr, stream>>>(static_cast<uint8_t*>(x),
        static_cast<uint8_t*>(out), static_cast<uint8_t*>(ptr),
        static_cast<uint8_t*>(idx), static_cast<uint8_t*>(split),
        m, k, n, nnz, ubBytes, hasNan);
}
