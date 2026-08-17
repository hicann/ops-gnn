# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import csv
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401

from ops_gnn import gather_csr


LOGGER = logging.getLogger(__name__)

CASES = [
    (262144, 512, 0.100, 0.060),
    (524288, 1024, 0.176, 0.096),
    (1048576, 1024, 0.330, 0.174),
    (1048576, 2048, 0.333, 0.175),
    (2097152, 4096, 0.638, 0.326),
    (4194304, 4096, 1.246, 0.627),
]

FIELDNAMES = [
    "rows",
    "segments",
    "dtype",
    "median_ms",
    "p90_ms",
    "output_GBps",
    "a100_ms",
    "ratio",
    "status",
]


@dataclass(frozen=True)
class BenchmarkResult:
    rows: int
    segments: int
    dtype: torch.dtype
    measurements: tuple
    baseline_ms: float | None = None
    passed: bool = True


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def benchmark(rows, segments, dtype, warmup, iterations):
    rows_per_segment = rows // segments
    seed = rows + segments + torch.tensor([], dtype=dtype).element_size()
    generator = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        src_values = torch.randn((segments, 128), generator=generator, dtype=torch.float32).to(dtype)
    elif dtype == torch.uint8:
        src_values = torch.randint(0, 97, (segments, 128), generator=generator, dtype=torch.int64).to(dtype)
    else:
        src_values = torch.randint(-48, 49, (segments, 128), generator=generator, dtype=torch.int64).to(dtype)
    src = src_values.to("npu")
    indptr = torch.arange(segments + 1, dtype=torch.int64, device="npu") * rows_per_segment
    for _ in range(warmup):
        result = gather_csr(src, indptr)
    torch.npu.synchronize()

    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = gather_csr(src, indptr)
        torch.npu.synchronize()
        elapsed.append((time.perf_counter_ns() - start) / 1e6)

    probe = torch.tensor(
        [0, rows_per_segment - 1, rows_per_segment, rows // 2, rows - 1],
        dtype=torch.int64,
        device="npu",
    )
    source_probe = torch.div(probe, rows_per_segment, rounding_mode="floor")
    if not torch.equal(result.index_select(0, probe).cpu(), src.index_select(0, source_probe).cpu()):
        raise AssertionError("performance case failed correctness probes")

    median_ms = statistics.median(elapsed)
    output_gb = rows * 128 * torch.tensor([], dtype=dtype).element_size() / 1e9
    return median_ms, percentile(elapsed, 0.9), output_gb / (median_ms / 1e3)


def parse_args():
    parser = argparse.ArgumentParser(description="Gather CSR 950PR acceptance benchmark")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=101)
    parser.add_argument("--all-l1", action="store_true", help="benchmark L1 dtypes without A100 baselines")
    parser.add_argument("--csv", type=Path, help="optionally write measured cases to a CSV file")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 3:
        parser.error("warmup must be >= 1 and iterations must be >= 3")
    return args


def make_record(result):
    median_ms, p90_ms, bandwidth = result.measurements
    ratio = "" if result.baseline_ms is None else f"{result.baseline_ms / median_ms:.3f}"
    return {
        "rows": str(result.rows),
        "segments": str(result.segments),
        "dtype": str(result.dtype).split(".")[-1],
        "median_ms": f"{median_ms:.6f}",
        "p90_ms": f"{p90_ms:.6f}",
        "output_GBps": f"{bandwidth:.3f}",
        "a100_ms": "" if result.baseline_ms is None else f"{result.baseline_ms:.3f}",
        "ratio": ratio,
        "status": "PASS" if result.passed else "FAIL",
    }


def log_record(record):
    LOGGER.info(",".join(record[field] for field in FIELDNAMES))


def run_official_cases(args):
    LOGGER.info(",".join(FIELDNAMES))
    records = []
    all_passed = True
    for rows, segments, fp32_ms, fp16_ms in CASES:
        for dtype, baseline_ms in ((torch.float16, fp16_ms), (torch.float32, fp32_ms)):
            measurements = benchmark(rows, segments, dtype, args.warmup, args.iterations)
            passed = baseline_ms / measurements[0] >= 0.6
            all_passed = all_passed and passed
            result = BenchmarkResult(rows, segments, dtype, measurements, baseline_ms, passed)
            record = make_record(result)
            records.append(record)
            log_record(record)
    if not all_passed:
        raise RuntimeError("one or more cases did not reach the required 0.6x ratio")
    return records


def run_l1_cases(args):
    LOGGER.info("")
    LOGGER.info("rows,segments,dtype,median_ms,p90_ms,output_GBps,correct")
    records = []
    regression_dtypes = (torch.bfloat16, torch.int8, torch.int16, torch.int32, torch.uint8)
    for rows, segments, _, _ in CASES:
        for dtype in regression_dtypes:
            measurements = benchmark(rows, segments, dtype, args.warmup, args.iterations)
            record = make_record(BenchmarkResult(rows, segments, dtype, measurements))
            records.append(record)
            LOGGER.info(",".join(record[field] for field in FIELDNAMES[:6]) + ",PASS")
    return records


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    LOGGER.info("csv=%s", path.resolve())


def main():
    args = parse_args()
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))
    records = run_official_cases(args)
    if args.all_l1:
        records.extend(run_l1_cases(args))
    if args.csv:
        write_csv(args.csv, records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()
