# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from itertools import product
import logging
import math
import sys

import torch

try:
    import torch_scatter
except ImportError:
    torch_scatter = None


LOGGER = logging.getLogger(__name__)

CASES = [
    ([1, 2, 3, 4], [0, 0, 1, 1, 1, 3], [0, 2, 5, 5, 6], [1, 1, 2, 2, 2, 4]),
    (
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        [0, 0, 1, 1, 1, 3],
        [0, 2, 5, 5, 6],
        [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4], [7, 8]],
    ),
    (
        [[1, 3, 5, 7], [2, 4, 6, 8]],
        [[0, 0, 1, 1, 1, 3], [0, 0, 0, 1, 1, 2]],
        [[0, 2, 5, 5, 6], [0, 3, 5, 6, 6]],
        [[1, 1, 3, 3, 3, 7], [2, 2, 2, 4, 4, 6]],
    ),
    (
        [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],
        [[0, 0, 1], [0, 2, 2]],
        [[0, 2, 3, 3], [0, 1, 1, 3]],
        [[[1, 2], [1, 2], [3, 4]], [[7, 9], [12, 13], [12, 13]]],
    ),
    ([[1], [2]], [[0, 0], [0, 0]], [[0, 2], [0, 2]], [[1, 1], [2, 2]]),
    (
        [[[1, 1]], [[2, 2]]],
        [[0, 0], [0, 0]],
        [[0, 2], [0, 2]],
        [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
    ),
]

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
]


def gather_csr_cpu(src, indptr):
    dim = indptr.dim() - 1
    if src.numel() == 0:
        shape = list(src.shape)
        shape[dim] = 0
        return torch.zeros(shape, dtype=src.dtype)

    ptr_shape = list(indptr.shape)
    for axis in range(dim):
        ptr_shape[axis] = src.size(axis)
    ptr = indptr.expand(ptr_shape).contiguous()
    output_rows = int(ptr.flatten()[-1])
    shape = list(src.shape)
    shape[dim] = output_rows
    out = torch.zeros(shape, dtype=src.dtype)

    batch_count = math.prod(src.shape[:dim])
    segment_count = src.size(dim)
    feature_count = math.prod(src.shape[dim + 1:])
    src_view = src.contiguous().view(batch_count, segment_count, feature_count)
    ptr_view = ptr.view(batch_count, segment_count + 1)
    out_view = out.view(batch_count, output_rows, feature_count)
    for batch in range(batch_count):
        for segment in range(segment_count):
            start = int(ptr_view[batch, segment])
            end = int(ptr_view[batch, segment + 1])
            out_view[batch, start:end] = src_view[batch, segment]
    return out


def gather_csr_coo_cpu(src, indptr):
    dim = indptr.dim() - 1
    ptr_shape = list(indptr.shape)
    for axis in range(dim):
        ptr_shape[axis] = src.size(axis)
    ptr = indptr.expand(ptr_shape).contiguous()
    output_rows = int(ptr.flatten()[-1])
    batch_count = math.prod(src.shape[:dim])
    segment_count = src.size(dim)
    feature_count = math.prod(src.shape[dim + 1:])
    src_view = src.contiguous().view(batch_count, segment_count, feature_count)
    ptr_view = ptr.view(batch_count, segment_count + 1)
    rows = []
    for batch in range(batch_count):
        lengths = ptr_view[batch, 1:] - ptr_view[batch, :-1]
        coo = torch.arange(segment_count).repeat_interleave(lengths)
        rows.append(src_view[batch].index_select(0, coo))
    shape = list(src.shape)
    shape[dim] = output_rows
    return torch.stack(rows).view(shape)


def main():
    if torch_scatter is None:
        raise RuntimeError("torch_scatter is required to run this reference script")
    if tuple(int(part) for part in torch_scatter.__version__.split(".")[:3]) < (2, 1, 0):
        raise RuntimeError(f"torch_scatter >= 2.1.0 is required, got {torch_scatter.__version__}")

    passed = 0
    for (src_data, index_data, indptr_data, expected_data), dtype in product(CASES, DTYPES):
        src = torch.tensor(src_data, dtype=dtype)
        index = torch.tensor(index_data, dtype=torch.int64)
        indptr = torch.tensor(indptr_data, dtype=torch.int64)
        expected = torch.tensor(expected_data, dtype=dtype)
        csr = torch_scatter.gather_csr(src, indptr)
        coo = torch_scatter.gather_coo(src, index)
        if not torch.equal(csr, expected) or not torch.equal(coo, expected):
            raise RuntimeError(f"reference mismatch for dtype={dtype}, src_shape={tuple(src.shape)}")
        passed += 1

    LOGGER.info("torch_scatter=%s", torch_scatter.__version__)
    LOGGER.info("official gather_csr/gather_coo CPU reference: %d/%d passed", passed, passed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()
