# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it
# under the terms and conditions of CANN Open Software License Agreement
# Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except
# in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text.

"""
Performance gate for the five Gather COO task shapes.

The command measures the public Python API with NPU events.  Allocation and
the optional ``out`` copy are therefore part of the reported end-to-end API
time, while input construction and index sorting remain outside the timed
region.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import runpy
import sys

import pytest
import torch

_GOLDEN = runpy.run_path(Path(__file__).with_name("golden.py"))
NPU_READY = _GOLDEN["NPU_READY"]
ops_gnn = _GOLDEN["ops_gnn"]


pytestmark = pytest.mark.skipif(
    not NPU_READY, reason="requires an available NPU"
)


LOGGER = logging.getLogger(__name__)
CASES = (
    (16384, 65536),
    (65536, 65536),
    (65536, 262144),
    (262144, 524288),
    (262144, 1048576),
)
THRESHOLDS_MS = {
    torch.float16: (0.1117, 0.1233, 0.4917, 0.8600, 1.7150),
    torch.float32: (0.1283, 0.1383, 0.5250, 0.9033, 1.7983),
}
A100_MS = {
    torch.float16: (0.067, 0.074, 0.295, 0.516, 1.029),
    torch.float32: (0.077, 0.083, 0.315, 0.542, 1.079),
}


@pytest.fixture(scope="module", autouse=True)
def _select_npu_device():
    device_id = int(os.environ.get("NPU_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)


def _dtype_from_name(name):
    names = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return names[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "dtype must be float16/fp16 or float32/fp32"
        ) from exc


def _measure(call, warmup, repeats):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()

    try:
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repeats
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "NPU Event timing is required; do not replace it with an "
            "implicit host synchronization"
        ) from exc


def _run(dtype, warmup, repeats, case_ids=None):
    device = torch.device("npu")
    thresholds = THRESHOLDS_MS[dtype]
    records = []
    selected_case_ids = case_ids or range(1, len(CASES) + 1)
    for case_id in selected_case_ids:
        source_rows, index_rows = CASES[case_id - 1]
        threshold = thresholds[case_id - 1]
        a100_ms = A100_MS[dtype][case_id - 1]
        src = torch.randn((source_rows, 128), dtype=dtype, device=device)
        index = (
            torch.randint(
                0, source_rows, (index_rows,), dtype=torch.int64, device=device
            )
            .sort()
            .values
        )
        elapsed_ms = _measure(
            lambda: ops_gnn.gather_coo(src, index),
            warmup=warmup,
            repeats=repeats,
        )
        record = {
            "case": case_id,
            "source_shape": [source_rows, 128],
            "index_shape": [index_rows],
            "dtype": str(dtype).replace("torch.", ""),
            "elapsed_ms": elapsed_ms,
            "a100_ms": a100_ms,
            "ratio_a100_over_npu": a100_ms / elapsed_ms,
            "threshold_ms": threshold,
            # The published task limits are rounded to four decimals; use
            # those limits for the executable gate and retain the unrounded
            # ratio as report data.
            "pass": elapsed_ms <= threshold,
        }
        records.append(record)
        LOGGER.info("%s", json.dumps(record, sort_keys=True))
    return records


@pytest.mark.parametrize(
    ("dtype", "case_id"),
    [
        pytest.param(
            dtype,
            case_id,
            id=f"{str(dtype).replace('torch.', '')}-case{case_id}",
        )
        for dtype in (torch.float16, torch.float32)
        for case_id in range(1, len(CASES) + 1)
    ],
)
def test_gather_coo_performance(dtype, case_id):
    """Make the ten task performance points visible to default pytest."""
    torch.manual_seed(20260804)
    record = _run(dtype, warmup=20, repeats=100, case_ids=(case_id,))[0]
    if not record["pass"]:
        pytest.fail(
            f"case {case_id} {record['dtype']} took "
            f"{record['elapsed_ms']:.6f} ms, exceeding "
            f"{record['threshold_ms']:.6f} ms"
        )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        type=_dtype_from_name,
        default=None,
        help="one dtype; omit to run fp16 and fp32",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--case",
        type=int,
        action="append",
        dest="case_ids",
        help="run only the selected case number; repeat for multiple cases",
    )
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def _validate_args(args):
    if not hasattr(torch, "npu"):
        raise RuntimeError(
            "an available NPU is required for Gather COO performance tests"
        )
    device_id = int(os.environ.get("NPU_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    if not torch.npu.is_available():
        raise RuntimeError(
            "an available NPU is required for Gather COO performance tests"
        )
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be > 0")
    if args.case_ids and any(
        case_id < 1 or case_id > len(CASES) for case_id in args.case_ids
    ):
        raise ValueError(f"case must be in [1, {len(CASES)}]")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout
    )
    args = _parse_args()
    _validate_args(args)
    torch.manual_seed(20260804)
    dtypes = (
        (args.dtype,)
        if args.dtype is not None
        else (torch.float16, torch.float32)
    )
    records = []
    for dtype in dtypes:
        records.extend(_run(dtype, args.warmup, args.repeats, args.case_ids))

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2, sort_keys=True)

    failed = [record for record in records if not record["pass"]]
    if failed:
        LOGGER.error(
            "Gather COO performance gate failed for %d case(s)", len(failed)
        )
        return 1
    LOGGER.info("Gather COO performance gate passed: %d case(s)", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
