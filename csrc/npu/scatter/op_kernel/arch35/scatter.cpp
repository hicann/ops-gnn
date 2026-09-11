/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "scatter.h"
#include "scatter_kernel.h"

#include "tiling/platform/platform_ascendc.h"

using namespace ScatterNpu;

template <typename T>
static bool LaunchVectorMinMax(void* src, int64_t* index, void* out,
                               int32_t* count, void* argOut,
                               const ScatterTilingData& tiling,
                               aclrtStream stream)
{
    if constexpr (IsSameType<T, half>::value || IsSameType<T, float>::value) {
        if (tiling.path != SCATTER_VECTOR_MINMAX) {
            return false;
        }
        uint8_t* values = IsSameType<T, half>::value
                              ? reinterpret_cast<uint8_t*>(count)
                              : static_cast<uint8_t*>(out);
        scatter_minmax_init_kernel<<<tiling.blockNum, nullptr, stream>>>(
            values, static_cast<uint8_t*>(argOut), tiling);
        if (tiling.hotTarget >= 0) {
            scatter_vector_hot_minmax_kernel<T>
                <<<tiling.blockNum, nullptr, stream>>>(
                    static_cast<uint8_t*>(src),
                    reinterpret_cast<uint8_t*>(index),
                    values, tiling);
        } else {
            scatter_vector_minmax_atomic_kernel<T>
                <<<tiling.blockNum, nullptr, stream>>>(
                    static_cast<uint8_t*>(src),
                    reinterpret_cast<uint8_t*>(index),
                    values, tiling);
        }
        scatter_vector_minmax_arg_kernel<T>
            <<<tiling.blockNum, nullptr, stream>>>(
                static_cast<uint8_t*>(src),
                reinterpret_cast<uint8_t*>(index),
                values, static_cast<uint8_t*>(argOut), tiling);
        if (tiling.hotTarget >= 0) {
            scatter_vector_hot_minmax_arg_kernel<T>
                <<<tiling.blockNum, nullptr, stream>>>(
                    static_cast<uint8_t*>(src),
                    reinterpret_cast<uint8_t*>(index),
                    values, static_cast<uint8_t*>(argOut), tiling);
        }
        scatter_vector_minmax_finalize_kernel<T>
            <<<tiling.blockNum, nullptr, stream>>>(
                values, static_cast<uint8_t*>(out),
                static_cast<uint8_t*>(argOut), tiling);
        return true;
    }
    return false;
}

static bool LaunchF16FloatAtomic(void* src, int64_t* index, void* out,
                                 int32_t* count, void* argOut,
                                 const ScatterTilingData& tiling,
                                 aclrtStream stream)
{
    if (tiling.path != SCATTER_F16_FLOAT_ATOMIC) {
        return false;
    }
    scatter_vector_zero_kernel<float>
        <<<tiling.blockNum, nullptr, stream>>>(
            static_cast<uint8_t*>(argOut),
            reinterpret_cast<uint8_t*>(count), tiling);
    if (tiling.hotTarget >= 0) {
        scatter_vector_hot_atomic_kernel<half>
            <<<tiling.blockNum, nullptr, stream>>>(
                static_cast<uint8_t*>(src),
                reinterpret_cast<uint8_t*>(index),
                static_cast<uint8_t*>(argOut), tiling);
    } else {
        scatter_f16_float_atomic_kernel
            <<<tiling.blockNum, nullptr, stream>>>(
                static_cast<uint8_t*>(src),
                reinterpret_cast<uint8_t*>(index),
                static_cast<uint8_t*>(argOut), tiling);
    }
    if (tiling.reduce == SCATTER_MEAN) {
        scatter_compact_count_kernel
            <<<tiling.blockNum, nullptr, stream>>>(
                reinterpret_cast<uint8_t*>(index),
                reinterpret_cast<uint8_t*>(count), tiling);
        if (tiling.hotTarget >= 0) {
            scatter_compact_hot_count_kernel
                <<<tiling.blockNum, nullptr, stream>>>(
                    reinterpret_cast<uint8_t*>(index),
                    reinterpret_cast<uint8_t*>(count), tiling);
        }
    }
    scatter_f16_float_finalize_kernel
        <<<tiling.blockNum, nullptr, stream>>>(
            static_cast<uint8_t*>(argOut), static_cast<uint8_t*>(out),
            reinterpret_cast<uint8_t*>(count), tiling);
    return true;
}

template <typename T>
static void LaunchVectorMean(int64_t* index, void* out, int32_t* count,
                             const ScatterTilingData& tiling,
                             aclrtStream stream)
{
    if constexpr (IsSameType<T, half>::value || IsSameType<T, float>::value) {
        if (tiling.reduce != SCATTER_MEAN) {
            return;
        }
        scatter_compact_count_kernel<<<tiling.blockNum, nullptr, stream>>>(
            reinterpret_cast<uint8_t*>(index),
            reinterpret_cast<uint8_t*>(count), tiling);
        if (tiling.hotTarget >= 0) {
            scatter_compact_hot_count_kernel
                <<<tiling.blockNum, nullptr, stream>>>(
                    reinterpret_cast<uint8_t*>(index),
                    reinterpret_cast<uint8_t*>(count), tiling);
        }
        scatter_compact_mean_finalize_kernel<T>
            <<<tiling.blockNum, nullptr, stream>>>(
                static_cast<uint8_t*>(out),
                reinterpret_cast<uint8_t*>(count), tiling);
    }
}

template <typename T>
static bool LaunchVectorAtomic(void* src, int64_t* index, void* out,
                               int32_t* count,
                               const ScatterTilingData& tiling,
                               aclrtStream stream)
{
    if constexpr (IsSameType<T, half>::value ||
                  IsSameType<T, bfloat16_t>::value ||
                  IsSameType<T, float>::value ||
                  IsSameType<T, int32_t>::value) {
        if (tiling.path != SCATTER_VECTOR_ATOMIC) {
            return false;
        }
        if (tiling.hasOut == 0 || tiling.reduce == SCATTER_MEAN) {
            scatter_vector_zero_kernel<T>
                <<<tiling.blockNum, nullptr, stream>>>(
                    static_cast<uint8_t*>(out),
                    reinterpret_cast<uint8_t*>(count), tiling);
        }
        if constexpr (IsSameType<T, half>::value ||
                      IsSameType<T, float>::value) {
            if (tiling.hotTarget >= 0) {
                scatter_vector_hot_atomic_kernel<T>
                    <<<tiling.blockNum, nullptr, stream>>>(
                        static_cast<uint8_t*>(src),
                        reinterpret_cast<uint8_t*>(index),
                        static_cast<uint8_t*>(out), tiling);
            } else {
                scatter_vector_atomic_kernel<T>
                    <<<tiling.blockNum, nullptr, stream>>>(
                        static_cast<uint8_t*>(src),
                        reinterpret_cast<uint8_t*>(index),
                        static_cast<uint8_t*>(out), tiling);
            }
        } else {
            scatter_vector_atomic_kernel<T>
                <<<tiling.blockNum, nullptr, stream>>>(
                    static_cast<uint8_t*>(src),
                    reinterpret_cast<uint8_t*>(index),
                    static_cast<uint8_t*>(out), tiling);
        }
        LaunchVectorMean<T>(index, out, count, tiling, stream);
        return true;
    }
    return false;
}

template <typename T>
static void LaunchTyped(void* src, int64_t* index, void* out, int32_t* count,
                        void* argOut, const ScatterTilingData& tiling,
                        aclrtStream stream)
{
    if (LaunchVectorMinMax<T>(src, index, out, count, argOut, tiling, stream)) {
        return;
    }
    if constexpr (IsSameType<T, half>::value) {
        if (LaunchF16FloatAtomic(src, index, out, count, argOut, tiling, stream)) {
            return;
        }
    }
    if (LaunchVectorAtomic<T>(src, index, out, count, tiling, stream)) {
        return;
    }
    scatter_kernel<T><<<tiling.blockNum, nullptr, stream>>>(
        static_cast<uint8_t*>(src), reinterpret_cast<uint8_t*>(index),
        static_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(count),
        static_cast<uint8_t*>(argOut), tiling);
    if (tiling.path == SCATTER_ATOMIC && tiling.reduce == SCATTER_MEAN) {
        scatter_mean_finalize_kernel<T><<<tiling.blockNum, nullptr, stream>>>(
            static_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(count),
            tiling);
    }
}

void Scatter(void* src, int64_t* index, void* out, int32_t* count,
                         void* argOut, ScatterDType dtype,
                         const ScatterTilingData& tiling, aclrtStream stream)
{
    switch (dtype) {
        case SCATTER_FLOAT16:
            LaunchTyped<half>(src, index, out, count, argOut, tiling, stream);
            break;
        case SCATTER_BFLOAT16:
            LaunchTyped<bfloat16_t>(src, index, out, count, argOut, tiling, stream);
            break;
        case SCATTER_FLOAT32:
            LaunchTyped<float>(src, index, out, count, argOut, tiling, stream);
            break;
        case SCATTER_INT8:
            LaunchTyped<int8_t>(src, index, out, count, argOut, tiling, stream);
            break;
        case SCATTER_INT16:
            LaunchTyped<int16_t>(src, index, out, count, argOut, tiling, stream);
            break;
        case SCATTER_INT32:
            LaunchTyped<int32_t>(src, index, out, count, argOut, tiling, stream);
            break;
        case SCATTER_UINT8:
            LaunchTyped<uint8_t>(src, index, out, count, argOut, tiling, stream);
            break;
    }
}
