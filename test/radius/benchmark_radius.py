# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Official task-book performance gate for the NPU radius operator.

Runs the five 3D random-coordinate shapes from the August 2026 radius task
book with float32 and float16.  Timing follows the official
``benchmark/run_benchmark.py`` mean-caliber method (warmup, then total
elapsed / iterations), and each row is compared against the published A100
baseline with the required >= 0.6x budget.  The script exits nonzero when
any row misses the budget.

Usage:
    source /usr/local/Ascend/cann-9.1.0-beta.3/bin/setenv.bash
    PYTHONPATH=python python test/radius/benchmark_radius.py

Use ``--quick`` for a smoke run (N=8K only) and ``--json-output`` to emit
machine-readable rows for the self-test report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from ops_gnn import radius


LOGGER = logging.getLogger(__name__)


# (n, r, A100 fp32 ms, A100 fp16 ms) from the community task book.
CASES = [
    (8192, 0.8, 2.118, 2.269),
    (16384, 0.5, 3.916, 4.424),
    (32768, 0.3, 8.446, 10.082),
    (65536, 0.2, 20.830, 28.418),
    (65536, 0.3, 20.902, 28.470),
]

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class _CaseSpec:
    n: int
    r: float
    dtype_name: str
    device: str
    warmup: int
    iters: int
    a100_ms: float
    budget_ms: float


def bench_one(x: torch.Tensor, r: float, max_num_neighbors: int,
              warmup: int, iters: int) -> float:
    """Return mean wall-clock latency in ms, matching the official caliber."""
    for _ in range(warmup):
        radius(x, x, r, max_num_neighbors=max_num_neighbors)
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iters):
        radius(x, x, r, max_num_neighbors=max_num_neighbors)
    torch.npu.synchronize()
    return (time.perf_counter_ns() - start) / iters / 1e6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=f"npu:{os.environ.get('NPU_DEVICE_ID', '0')}")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dtypes", default="float32,float16")
    parser.add_argument("--quick", action="store_true",
                        help="run only the N=8K smoke row")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)


def _parse_dtype_names(spec: str):
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = set(names) - set(DTYPES)
    if unknown:
        LOGGER.error("unsupported dtypes: %s", sorted(unknown))
        return None
    return names


def _run_case(case: _CaseSpec) -> dict:
    torch.manual_seed(0)
    x = torch.randn(case.n, 3, dtype=DTYPES[case.dtype_name],
                    device=case.device)
    mean_ms = bench_one(x, case.r, 32, case.warmup, case.iters)
    passed = mean_ms <= case.budget_ms
    return {
        "n": case.n,
        "r": case.r,
        "dtype": case.dtype_name,
        "mean_ms": mean_ms,
        "a100_ms": case.a100_ms,
        "budget_ms": case.budget_ms,
        "ratio_to_a100": case.a100_ms / mean_ms,
        "pass": passed,
    }


def _format_row(record: dict) -> str:
    return (f"{record['n']:>7} {record['r']:>5} {record['dtype']:>8} "
            f"{record['mean_ms']:>9.3f} {record['a100_ms']:>9.3f} "
            f"{record['budget_ms']:>9.3f} "
            f"{record['a100_ms'] / record['mean_ms']:>7.2f} "
            f"{'PASS' if record['pass'] else 'FAIL':>8}")


def _log_header(device: str, warmup: int, iters: int, threads: int) -> None:
    LOGGER.info("=" * 78)
    LOGGER.info("radius official benchmark | %s | warmup=%d iters=%d "
                "threads=%d", device, warmup, iters, threads)
    header = (f"{'N':>7} {'r':>5} {'dtype':>8} {'mean_ms':>9} "
              f"{'a100_ms':>9} {'budget_ms':>9} "
              f"{'ratio':>7} {'verdict':>8}")
    LOGGER.info(header)
    LOGGER.info("=" * 78)


def _write_json(path: Path, results: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    _setup_logging()
    if args.warmup < 0 or args.iters <= 0:
        LOGGER.error("--warmup must be >= 0 and --iters must be > 0")
        return 2
    dtype_names = _parse_dtype_names(args.dtypes)
    if dtype_names is None:
        return 2

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.npu.set_device(args.device)

    rows = CASES[:1] if args.quick else CASES
    _log_header(args.device, args.warmup, args.iters, args.threads)
    results = []
    for n, r, a100_fp32, a100_fp16 in rows:
        for dtype_name in dtype_names:
            a100_ms = a100_fp32 if dtype_name == "float32" else a100_fp16
            case = _CaseSpec(n, r, dtype_name, args.device, args.warmup,
                             args.iters, a100_ms, a100_ms / 0.6)
            record = _run_case(case)
            results.append(record)
            LOGGER.info("%s", _format_row(record))

    failed = sum(1 for record in results if not record["pass"])
    LOGGER.info("=" * 78)
    LOGGER.info("RESULT: %d/%d rows within budget",
                len(results) - failed, len(results))
    if args.json_output:
        _write_json(args.json_output, results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
