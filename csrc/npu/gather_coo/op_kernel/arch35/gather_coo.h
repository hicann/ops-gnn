/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0.
 * Please refer to the License for details.
 */

#pragma once

#include <acl/acl.h>
#include <cstdint>

#include "gather_coo_tiling.h"

template <typename T>
void GatherCoo(
    T* src,
    int64_t* index,
    T* out,
    const GatherCooTilingData& tiling,
    aclrtStream stream);
