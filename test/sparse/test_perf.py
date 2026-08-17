# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Lightweight performance smoke tests for ind2ptr/ptr2ind.

Full matrix benchmark: ``python test/sparse/bench_convert.py``.
Triggered by the repo-wide test entry (``./scripts/build.sh test`` / ``pytest test/``).
"""

import logging
import time

import pytest
import torch

import ops_gnn
from ops_gnn import _pybind

LOG = logging.getLogger(__name__)


def _sync():
    torch.npu.synchronize()


def _bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return sum(ts) / len(ts) * 1e3


@pytest.mark.parametrize('num_edges,num_rows', [
    (10_000, 1_024),
    (100_000, 8_192),
    (1_000_000, 32_768),
])
def test_ind2ptr_perf_smoke(num_edges, num_rows):
    torch.npu.set_device(0)
    ind = torch.sort(torch.randint(0, num_rows, (num_edges,), dtype=torch.long, device='npu')).values
    ind = ind.contiguous().clone()
    _sync()
    mean_ms = _bench(lambda: _pybind.ind2ptr(ind, num_rows))
    if mean_ms >= 5_000.0:
        raise AssertionError(f'ind2ptr too slow: {mean_ms:.2f} ms for num_edges={num_edges}')
    LOG.info('ind2ptr num_edges=%s num_rows=%s: %.3f ms', num_edges, num_rows, mean_ms)


@pytest.mark.parametrize('num_edges,num_rows', [
    (10_000, 1_024),
    (100_000, 8_192),
    (1_000_000, 32_768),
])
def test_ptr2ind_perf_smoke(num_edges, num_rows):
    torch.npu.set_device(0)
    ind = torch.sort(torch.randint(0, num_rows, (num_edges,), dtype=torch.long, device='npu')).values
    ptr = ops_gnn.ind2ptr(ind, num_rows).contiguous().clone()
    _sync()
    mean_ms = _bench(lambda: _pybind.ptr2ind(ptr, num_edges))
    if mean_ms >= 5_000.0:
        raise AssertionError(f'ptr2ind too slow: {mean_ms:.2f} ms for num_edges={num_edges}')
    LOG.info('ptr2ind num_edges=%s num_rows=%s: %.3f ms', num_edges, num_rows, mean_ms)
