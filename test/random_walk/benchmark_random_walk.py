# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reproducible Ascend 950 performance benchmark for random_walk."""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
import torch_npu  # noqa: F401

import ops_gnn


LOGGER = logging.getLogger(__name__)

CASES = [
    {"edges": 64 * 1024, "walk_length": 128, "starts": 8 * 1024, "golden_time_ms": 0.576},
    {"edges": 256 * 1024, "walk_length": 128, "starts": 16 * 1024, "golden_time_ms": 0.709},
    {"edges": 512 * 1024, "walk_length": 128, "starts": 32 * 1024, "golden_time_ms": 1.139},
    {"edges": 512 * 1024, "walk_length": 256, "starts": 32 * 1024, "golden_time_ms": 1.867},
]


@dataclass(frozen=True)
class BenchmarkInputs:
    row: torch.Tensor
    col: torch.Tensor
    start: torch.Tensor
    walk_length: int
    num_nodes: int


def make_graph(case: Dict[str, float], device: str):
    edges = int(case["edges"])
    num_nodes = max(edges // 8, 2)
    edge_index = torch.arange(edges, dtype=torch.int64)
    row = edge_index.remainder(num_nodes)
    col = (edge_index * 1103515245 + 12345).remainder(num_nodes)
    permutation = torch.argsort(row * num_nodes + col)
    row = row[permutation].to(device)
    col = col[permutation].to(device)
    start_count = int(case["starts"])
    start = (torch.arange(start_count, dtype=torch.int64) * 2654435761).remainder(num_nodes).to(device)
    return row, col, start, num_nodes


def measure(call, warmup: int, repeats: int) -> Dict[str, float]:
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    samples: List[float] = []
    for _ in range(repeats):
        begin = time.perf_counter_ns()
        call()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - begin) / 1_000_000.0)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=f"npu:{os.environ.get('NPU_DEVICE_ID', '0')}")
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "10")))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "50")))
    parser.add_argument("--case", type=int, choices=range(len(CASES)), action="append")
    parser.add_argument("--tsv", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative and repeats must be positive")
    return args


def _run_walk(inputs: BenchmarkInputs, coalesced: bool):
    return ops_gnn.random_walk(
        inputs.row, inputs.col, inputs.start, inputs.walk_length,
        coalesced=coalesced, num_nodes=inputs.num_nodes,
    )


def _run_case(case_index: int, args) -> dict:
    case = CASES[case_index]
    row, col, start, num_nodes = make_graph(case, args.device)
    inputs = BenchmarkInputs(row, col, start, int(case["walk_length"]), num_nodes)
    end_stats = measure(partial(_run_walk, inputs, True), args.warmup, args.repeats)
    presorted_stats = measure(partial(_run_walk, inputs, False), args.warmup, args.repeats)
    limit_ms = float(case["golden_time_ms"]) / 0.6
    return {
        "case": case_index,
        **case,
        "num_nodes": num_nodes,
        "limit_ms": limit_ms,
        "end_to_end": end_stats,
        "presorted": presorted_stats,
        "speed_ratio_vs_golden": float(case["golden_time_ms"]) / end_stats["median_ms"],
        "pass": end_stats["median_ms"] <= limit_ms,
    }


def _write_json(path: Path, results: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, results: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "case\tedges\twalk_length\tstarts\tgolden_time_ms\tlimit_ms\t"
        "end_to_end_median_ms\tpresorted_median_ms\tspeed_ratio_vs_golden\tpass\n"
    )
    rows = [
        "{case}\t{edges}\t{walk_length}\t{starts}\t{golden_time_ms:.6f}\t{limit_ms:.6f}\t"
        "{end_ms:.6f}\t{presorted_ms:.6f}\t{ratio:.6f}\t{passed}\n".format(
            **item,
            end_ms=item["end_to_end"]["median_ms"],
            presorted_ms=item["presorted"]["median_ms"],
            ratio=item["speed_ratio_vs_golden"],
            passed="PASS" if item["pass"] else "FAIL",
        )
        for item in results
    ]
    path.write_text(header + "".join(rows), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    torch.npu.set_device(args.device)
    selected = args.case if args.case is not None else list(range(len(CASES)))
    results = [_run_case(case_index, args) for case_index in selected]
    for result in results:
        LOGGER.info("%s", json.dumps(result, sort_keys=True))
    if args.json:
        _write_json(args.json, results)
    if args.tsv:
        _write_tsv(args.tsv, results)
    if args.enforce and not all(item["pass"] for item in results):
        raise SystemExit("one or more end-to-end cases missed the 0.6x performance gate")


if __name__ == "__main__":
    main()
