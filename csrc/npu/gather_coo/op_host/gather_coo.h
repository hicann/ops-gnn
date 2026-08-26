/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0.
 * Please refer to the License for details.
 */

#pragma once

#include <c10/util/Optional.h>
#include <torch/extension.h>

torch::Tensor gather_coo(
    torch::Tensor src,
    torch::Tensor index,
    c10::optional<torch::Tensor> optional_out = c10::nullopt);
