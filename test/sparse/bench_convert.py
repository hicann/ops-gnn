# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Performance benchmark for ops_gnn.ind2ptr / ops_gnn.ptr2ind on Ascend NPU.

Reports:
  - end-to-end API latency (ops_gnn.*, includes clone/sync in Python wrapper)
  - kernel/host pybind latency (pre-materialized contiguous input)
  - CPU golden baseline on the same host
  - optional PyTorch bucketize reference (CPU)

Usage:
  python test/sparse/bench_convert.py
  python test/sparse/bench_convert.py --device 0 --out test/sparse/reports/sparse_perf.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict

import torch

import ops_gnn
from ops_gnn import _pybind

LOG = logging.getLogger(__name__)

BENCH_SHAPES = [
    (1_000, 128, 'E1k_M128'),
    (10_000, 1_024, 'E10k_M1k'),
    (50_000, 4_096, 'E50k_M4k'),
    (100_000, 8_192, 'E100k_M8k'),
    (500_000, 16_384, 'E500k_M16k'),
    (1_000_000, 32_768, 'E1M_M32k'),
    (2_000_000, 65_536, 'E2M_M64k'),
]


@dataclass
class BenchRunConfig:
    """Runtime knobs for a single benchmark case."""

    device: int
    warmup: int
    iters: int


@dataclass
class CaseData:
    """Prepared tensors and shape for one benchmark case."""

    ind_cpu: torch.Tensor
    ptr_cpu: torch.Tensor
    ind_npu: torch.Tensor
    ptr_npu: torch.Tensor
    num_edges: int
    num_rows: int


def make_sorted_ind(num_edges: int, num_rows: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if num_edges == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.sort(
        torch.randint(0, num_rows, (num_edges,), generator=generator, dtype=torch.long)
    ).values


def ind2ptr_cpu(ind: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Fast CPU golden via bucketize (equivalent to torch_sparse ind2ptr)."""
    if ind.numel() == 0:
        return torch.zeros(num_rows + 1, dtype=torch.long)
    return torch.bucketize(torch.arange(num_rows + 1, dtype=torch.long), ind)


def ptr2ind_cpu(ptr: torch.Tensor, num_edges: int) -> torch.Tensor:
    """CPU golden for ptr2ind."""
    out = torch.empty(num_edges, dtype=torch.long)
    if num_edges == 0:
        return out
    ptr_list = ptr.tolist()
    for i in range(ptr.numel() - 1):
        start, end = int(ptr_list[i]), int(ptr_list[i + 1])
        if end > start:
            out[start:end] = i
    return out


def ind2ptr_cpu_naive(ind: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Naive list golden (slow; only for small-edge timing baseline)."""
    out = torch.empty(num_rows + 1, dtype=torch.long)
    numel = ind.numel()
    if numel == 0:
        return out.zero_()
    ind_list = ind.tolist()
    out_list = [0] * (num_rows + 1)
    for i in range(ind_list[0] + 1):
        out_list[i] = 0
    for i in range(numel - 1):
        for k in range(ind_list[i], ind_list[i + 1]):
            out_list[k + 1] = i + 1
    for i in range(ind_list[numel - 1] + 1, num_rows + 1):
        out_list[i] = numel
    return torch.tensor(out_list, dtype=torch.long)


def synchronize():
    if hasattr(torch, 'npu'):
        torch.npu.synchronize()


def bench_npu(fn: Callable[[], None], warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    synchronize()
    ts = []
    for _ in range(iters):
        synchronize()
        t0 = time.perf_counter()
        fn()
        synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {
        'mean_ms': statistics.mean(ts),
        'median_ms': statistics.median(ts),
        'min_ms': min(ts),
        'max_ms': max(ts),
        'stdev_ms': statistics.stdev(ts) if len(ts) > 1 else 0.0,
        'iters': iters,
    }


def bench_cpu(fn: Callable[[], None], warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {
        'mean_ms': statistics.mean(ts),
        'median_ms': statistics.median(ts),
        'min_ms': min(ts),
        'max_ms': max(ts),
        'stdev_ms': statistics.stdev(ts) if len(ts) > 1 else 0.0,
        'iters': iters,
    }


def throughput_melems(num_edges: int, mean_ms: float) -> float:
    if mean_ms <= 0:
        return 0.0
    return (num_edges / (mean_ms * 1e-3)) / 1e6


def _prepare_case_tensors(
    num_edges: int, num_rows: int, tag: str, device: int
) -> CaseData:
    torch.npu.set_device(device)
    ind_cpu = make_sorted_ind(num_edges, num_rows, seed=0)
    ptr_cpu = ind2ptr_cpu(ind_cpu, num_rows)
    ind_npu = ind_cpu.to('npu').contiguous().clone()
    synchronize()
    ptr_npu = _pybind.ind2ptr(ind_npu, num_rows).contiguous().clone()
    synchronize()
    if not torch.equal(_pybind.ind2ptr(ind_npu, num_rows).cpu(), ptr_cpu):
        raise RuntimeError(f'ind2ptr correctness failed for {tag}')
    if not torch.equal(_pybind.ptr2ind(ptr_npu, num_edges).cpu(), ind_cpu):
        raise RuntimeError(f'ptr2ind correctness failed for {tag}')
    return CaseData(
        ind_cpu=ind_cpu,
        ptr_cpu=ptr_cpu,
        ind_npu=ind_npu,
        ptr_npu=ptr_npu,
        num_edges=num_edges,
        num_rows=num_rows,
    )


def _fill_latency_metrics(row: Dict, case: CaseData, config: BenchRunConfig) -> None:
    row['ind2ptr_api'] = bench_npu(
        lambda: ops_gnn.ind2ptr(case.ind_npu, case.num_rows),
        config.warmup,
        config.iters,
    )
    row['ptr2ind_api'] = bench_npu(
        lambda: ops_gnn.ptr2ind(case.ptr_npu, case.num_edges),
        config.warmup,
        config.iters,
    )
    row['ind2ptr_kernel'] = bench_npu(
        lambda: _pybind.ind2ptr(case.ind_npu, case.num_rows),
        config.warmup,
        config.iters,
    )
    row['ptr2ind_kernel'] = bench_npu(
        lambda: _pybind.ptr2ind(case.ptr_npu, case.num_edges),
        config.warmup,
        config.iters,
    )
    row['ind2ptr_cpu_bucketize'] = bench_cpu(
        lambda: ind2ptr_cpu(case.ind_cpu, case.num_rows),
        config.warmup,
        config.iters,
    )
    if case.num_edges <= 100_000:
        row['ptr2ind_cpu_golden'] = bench_cpu(
            lambda: ptr2ind_cpu(case.ptr_cpu, case.num_edges),
            max(1, config.warmup // 5),
            max(3, config.iters // 5),
        )
        row['ind2ptr_cpu_naive'] = bench_cpu(
            lambda: ind2ptr_cpu_naive(case.ind_cpu, case.num_rows),
            max(1, config.warmup // 10),
            max(2, config.iters // 10),
        )
    else:
        row['ptr2ind_cpu_golden'] = None
        row['ind2ptr_cpu_naive'] = None


def _attach_throughput(row: Dict, num_edges: int) -> None:
    for key in ('ind2ptr_api', 'ind2ptr_kernel', 'ptr2ind_api', 'ptr2ind_kernel'):
        metrics = row.get(key)
        if isinstance(metrics, dict) and 'mean_ms' in metrics:
            metrics['throughput_Medges_s'] = throughput_melems(
                num_edges, metrics['mean_ms']
            )


def run_one(
    num_edges: int,
    num_rows: int,
    tag: str,
    config: BenchRunConfig,
) -> Dict:
    case = _prepare_case_tensors(num_edges, num_rows, tag, config.device)
    row = {
        'tag': tag,
        'num_edges': num_edges,
        'num_rows': num_rows,
        'bytes_ind': num_edges * 8,
        'bytes_ptr': (num_rows + 1) * 8,
    }
    _fill_latency_metrics(row, case, config)
    _attach_throughput(row, num_edges)
    return row


def estimate_a100_memory_bound_us(
    num_edges: int, num_rows: int, bw_tb_s: float = 1.555
) -> Dict[str, float]:
    """Rough A100 HBM lower-bound estimate for torch_sparse CUDA kernels."""
    bytes_ind2ptr = num_edges * 8 + (num_rows + 1) * 8
    bytes_ptr2ind = (num_rows + 1) * 8 + num_edges * 8
    bandwidth = bw_tb_s * 1e12
    ind2ptr_lb_us = bytes_ind2ptr / bandwidth * 1e6
    ptr2ind_lb_us = bytes_ptr2ind / bandwidth * 1e6
    return {
        'a100_hbm_tb_s': bw_tb_s,
        'ind2ptr_memcpy_lb_us': ind2ptr_lb_us,
        'ptr2ind_memcpy_lb_us': ptr2ind_lb_us,
        'ind2ptr_est_us': ind2ptr_lb_us * 5.0 + 5.0,
        'ptr2ind_est_us': ptr2ind_lb_us * 5.0 + 5.0,
        'note': 'Estimate from A100 HBM bandwidth + parallel fill factor; not measured on A100.',
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--out', type=str, default='test/sparse/reports/sparse_perf.json')
    return parser.parse_args()


def _build_results_shell(args, device_name: str) -> Dict:
    return {
        'meta': {
            'device_id': args.device,
            'device_name': device_name,
            'warmup': args.warmup,
            'iters': args.iters,
            'torch': torch.__version__,
            'ops_gnn': getattr(ops_gnn, '__version__', 'unknown'),
            'impl_note': (
                'Current Ascend950 kernel is serial single-SIMT-thread (correctness-first), '
                'aligned with torch_sparse CPU algorithm; A100 torch_sparse uses parallel CUDA kernels.'
            ),
        },
        'cases': [],
    }


def _log_case_row(tag: str, num_edges: int, num_rows: int, row: Dict, a100: Dict) -> None:
    LOG.info(
        '%s %10d %8d | %10.3f %10.3f %10.3f %10.3f | %9.2f %10.2f',
        f'{tag:<14}',
        num_edges,
        num_rows,
        row['ind2ptr_kernel']['mean_ms'],
        row['ind2ptr_api']['mean_ms'],
        row['ptr2ind_kernel']['mean_ms'],
        row['ptr2ind_api']['mean_ms'],
        row['ind2ptr_kernel']['throughput_Medges_s'],
        a100['ind2ptr_est_us'],
    )


def _run_benchmark_suite(config: BenchRunConfig, results: Dict) -> None:
    for num_edges, num_rows, tag in BENCH_SHAPES:
        row = run_one(num_edges, num_rows, tag, config)
        a100 = estimate_a100_memory_bound_us(num_edges, num_rows)
        row['a100_estimate'] = a100
        results['cases'].append(row)
        _log_case_row(tag, num_edges, num_rows, row, a100)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = _parse_args()
    torch.npu.set_device(args.device)
    device_name = torch.npu.get_device_name(args.device)
    config = BenchRunConfig(device=args.device, warmup=args.warmup, iters=args.iters)
    results = _build_results_shell(args, device_name)

    LOG.info('Device: %s (npu:%s)', device_name, args.device)
    LOG.info('warmup=%s iters=%s', args.warmup, args.iters)
    LOG.info('-' * 96)
    LOG.info(
        f'{"tag":<14} {"edges":>10} {"rows":>8} | '
        f'{"i2p_ker_ms":>10} {"i2p_api_ms":>10} {"p2i_ker_ms":>10} {"p2i_api_ms":>10} | '
        f'{"i2p_Meps":>9} {"A100est_us":>10}'
    )
    LOG.info('-' * 96)
    _run_benchmark_suite(config, results)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as out_file:
        json.dump(results, out_file, indent=2)
    LOG.info('-' * 96)
    LOG.info('Wrote %s', args.out)


if __name__ == '__main__':
    main()
