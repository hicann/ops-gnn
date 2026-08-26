# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import statistics
import sys
import time

import pytest
import torch
import ops_gnn

_SPEC = spec_from_file_location("graclus_cluster_golden", Path(__file__).with_name("golden.py"))
_GOLDEN = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GOLDEN
_SPEC.loader.exec_module(_GOLDEN)
npu_or_skip = _GOLDEN.npu_or_skip


LOGGER = logging.getLogger(__name__)
BASELINE_MS = {
    4: 0.400,
    8: 0.507,
    16: 0.705,
    32: 1.234,
    64: 2.825,
}


def _complete_graph(num_nodes):
    row = []
    col = []
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src != dst:
                row.append(src)
                col.append(dst)
    return torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long)


def _bench(fn, warmup=5, repeat=30):
    for _ in range(warmup):
        torch.manual_seed(2026)
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(repeat):
        torch.manual_seed(2026)
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def _weighted_perf_runner(row, col, num_nodes):
    row = row.to("npu")
    col = col.to("npu")
    weight = torch.ones(row.numel(), dtype=torch.float32, device="npu")

    def fn():
        return ops_gnn.graclus_cluster(row, col, weight, num_nodes=num_nodes)

    torch.manual_seed(2026)
    out = fn()
    torch.npu.synchronize()
    if out.device.type != "npu":
        pytest.fail(f"expected NPU output, but got {out.device.type}")
    if out.dtype != torch.long:
        pytest.fail(f"expected torch.long output, but got {out.dtype}")
    if out.numel() != num_nodes:
        pytest.fail(f"expected {num_nodes} output elements, but got {out.numel()}")
    return row, fn


@pytest.mark.parametrize("num_nodes", [4, 8, 16, 32, 64])
def test_graclus_cluster_perf(num_nodes):
    npu_or_skip()
    row, col = _complete_graph(num_nodes)
    row, fn = _weighted_perf_runner(row, col, num_nodes)
    median_ms = _bench(fn)
    threshold_ms = BASELINE_MS[num_nodes] / 0.6
    LOGGER.info(
        "graclus_perf V=%s E=%s median_ms=%.3f threshold_ms=%.3f",
        num_nodes, row.numel(), median_ms, threshold_ms,
    )
    if median_ms > threshold_ms:
        pytest.fail(f"median_ms {median_ms:.3f} exceeds threshold_ms {threshold_ms:.3f}")


def _random_graph(num_nodes, num_edges, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    row = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long, generator=generator)
    col = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long, generator=generator)
    return row, col


@pytest.mark.parametrize("num_nodes,num_edges", [(128, 2048), (256, 8192)])
def test_graclus_cluster_extended_sparse_perf_profile(num_nodes, num_edges):
    npu_or_skip()
    row, col = _random_graph(num_nodes, num_edges, seed=num_nodes + num_edges)
    _, fn = _weighted_perf_runner(row, col, num_nodes)
    median_ms = _bench(fn, warmup=3, repeat=10)
    LOGGER.info("graclus_extended_sparse_perf V=%s E=%s median_ms=%.3f", num_nodes, num_edges, median_ms)
    if median_ms <= 0.0:
        pytest.fail(f"median_ms must be positive, but got {median_ms:.3f}")


@pytest.mark.parametrize("num_nodes", [128, 256])
def test_graclus_cluster_extended_complete_perf_profile(num_nodes):
    npu_or_skip()
    row, col = _complete_graph(num_nodes)
    row, fn = _weighted_perf_runner(row, col, num_nodes)
    median_ms = _bench(fn, warmup=3, repeat=10)
    LOGGER.info("graclus_extended_complete_perf V=%s E=%s median_ms=%.3f", num_nodes, row.numel(), median_ms)
    if median_ms <= 0.0:
        pytest.fail(f"median_ms must be positive, but got {median_ms:.3f}")
