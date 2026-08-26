# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import csv
import json
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from ops_gnn import scatter


LOGGER = logging.getLogger(__name__)


SHAPES = [(16384, 512), (32768, 512), (65536, 512), (65536, 1024), (65536, 2048)]
A100_MS = {
    "sum": {
        "float32": [0.062, 0.124, 0.254, 0.496, 1.053],
        "float16": [0.077, 0.181, 0.331, 0.707, 1.455],
    },
    "mean": {
        "float32": [0.096, 0.169, 0.319, 0.633, 1.252],
        "float16": [0.112, 0.226, 0.440, 0.820, 1.643],
    },
    "min": {
        "float32": [0.281, 0.534, 0.982, 1.875, 3.755],
        "float16": [0.304, 0.579, 0.955, 1.940, 3.885],
    },
    "max": {
        "float32": [0.281, 0.534, 0.927, 1.873, 3.753],
        "float16": [0.305, 0.483, 0.955, 1.940, 3.886],
    },
}


def synchronize():
    torch.npu.synchronize()


def percentile(samples, fraction):
    ordered = sorted(samples)
    rank = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[rank]


@dataclass(frozen=True)
class MeasurementConfig:
    groups: int
    reduce: str
    warmup: int
    repeat: int
    timing: str


@dataclass(frozen=True)
class BenchmarkCase:
    reduce: str
    dtype_name: str
    n: int
    c: int


def measure(src, index, config):
    for _ in range(config.warmup):
        scatter(
            src, index, dim=0, dim_size=config.groups, reduce=config.reduce
        )
    synchronize()
    samples = []
    for _ in range(config.repeat):
        if config.timing == "event":
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            result = scatter(
                src, index, dim=0, dim_size=config.groups, reduce=config.reduce
            )
            end.record()
            synchronize()
            # Keep the output alive until the timed work has completed.
            del result
            samples.append(float(start.elapsed_time(end)))
        else:
            start_ns = time.perf_counter_ns()
            result = scatter(
                src, index, dim=0, dim_size=config.groups, reduce=config.reduce
            )
            synchronize()
            del result
            samples.append((time.perf_counter_ns() - start_ns) / 1e6)
    return statistics.median(samples), percentile(samples, 0.90), samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reductions", nargs="+", default=["sum", "mean", "min", "max"])
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float16"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument(
        "--timing",
        choices=("event", "host"),
        default="event",
        help="NPU event timing matches device-side A100 latency; host includes synchronize overhead.",
    )
    parser.add_argument("--groups", type=int, default=None,
                        help="group count; default N//4. Keep identical to the benchmark used for comparison.")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def validate_args(args):
    device_id = int(os.environ.get("NPU_DEVICE_ID", "0"))
    if torch_npu is not None and hasattr(torch, "npu"):
        torch.npu.set_device(device_id)
    if (
        torch_npu is None
        or not hasattr(torch, "npu")
        or not torch.npu.is_available()
    ):
        raise RuntimeError("Ascend NPU is not available")
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    if args.threshold <= 0:
        raise ValueError("threshold must be positive")
    for reduce in args.reductions:
        if reduce not in A100_MS:
            raise ValueError(f"unsupported performance reduction: {reduce}")
    for dtype_name in args.dtypes:
        if dtype_name not in ("float32", "float16"):
            raise ValueError(f"unsupported performance dtype: {dtype_name}")


def run_benchmarks(args):
    rows = []
    for reduce in args.reductions:
        for dtype_name in args.dtypes:
            rows.extend(run_dtype_cases(args, reduce, dtype_name))
    return rows


def run_dtype_cases(args, reduce, dtype_name):
    rows = []
    dtype = getattr(torch, dtype_name)
    for shape_index, (n, c) in enumerate(SHAPES):
        case = BenchmarkCase(reduce, dtype_name, n, c)
        groups = args.groups or max(1, n // 4)
        src = torch.randn((n, c), dtype=dtype, device="npu")
        index = torch.randint(0, groups, (n,), dtype=torch.int64, device="npu")
        config = MeasurementConfig(
            groups, reduce, args.warmup, args.repeat, args.timing
        )
        p50, p90, samples = measure(src, index, config)
        baseline = A100_MS[reduce][dtype_name][shape_index]
        row = build_result_row(
            args, case, config, baseline, (p50, p90, samples)
        )
        rows.append(row)
        LOGGER.info("%s", row)
    return rows


def build_result_row(args, case, config, baseline, measurements):
    p50, p90, samples = measurements
    ratio_p50 = baseline / p50
    return {
        "reduce": case.reduce,
        "dtype": case.dtype_name,
        "N": case.n,
        "C": case.c,
        "groups": config.groups,
        "timing": args.timing,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "npu_p50_ms": p50,
        "npu_p90_ms": p90,
        "npu_min_ms": min(samples),
        "npu_max_ms": max(samples),
        "a100_ms": baseline,
        "ratio_p50": ratio_p50,
        "ratio_p90": baseline / p90,
        "threshold": args.threshold,
        "status": "PASS" if ratio_p50 >= args.threshold else "FAIL",
    }


def write_csv(path, rows):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, args, rows):
    if path is None:
        return
    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
        "device": torch.npu.current_device(),
        "device_name": (
            torch.npu.get_device_name(torch.npu.current_device())
            if hasattr(torch.npu, "get_device_name")
            else "unknown"
        ),
        "seed": args.seed,
        "timing": args.timing,
        "threshold": args.threshold,
        "total": len(rows),
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failed": sum(row["status"] != "PASS" for row in rows),
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)
    if hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(args.seed)

    rows = run_benchmarks(args)
    write_csv(args.csv, rows)
    write_json(args.json, args, rows)

    failed = [row for row in rows if row["status"] != "PASS"]
    LOGGER.info(
        "summary: total=%d passed=%d failed=%d",
        len(rows), len(rows) - len(failed), len(failed)
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
