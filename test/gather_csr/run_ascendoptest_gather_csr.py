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
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch_npu  # noqa: F401
import torch_scatter

from ops_gnn import gather_csr


LOGGER = logging.getLogger(__name__)

DTYPES = [
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
    ("float", torch.float32),
    ("int8", torch.int8),
    ("int16", torch.int16),
    ("int32", torch.int32),
    ("uint8", torch.uint8),
    ("float64", torch.float64),
    ("int64", torch.int64),
]


@dataclass(frozen=True)
class CaseTensors:
    src: torch.Tensor
    indptr: torch.Tensor
    actual: torch.Tensor
    expected: torch.Tensor


def version_tuple(version):
    parts = version.split("+", 1)[0].split(".")
    return tuple(int(part) for part in parts[:3])


def write_tensor(tensor, path):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor.view(torch.uint16).numpy().tofile(path)
    else:
        tensor.numpy().tofile(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ascendoptest-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("build/ascendoptest_gather_csr"))
    args = parser.parse_args()
    compare_entry = args.ascendoptest_root.resolve() / "compare" / "data_compare.py"
    if not compare_entry.is_file():
        parser.error(f"AscendOpTest compare entry not found: {compare_entry}")
    if version_tuple(torch_scatter.__version__) < (2, 1, 0):
        parser.error(
            "torch_scatter >= 2.1.0 is required as the CPU reference, got "
            f"{torch_scatter.__version__}"
        )
    return args.work_dir.resolve(), compare_entry


def source_values(dtype):
    values = torch.arange(1, 4 * 17 + 1, dtype=torch.int64).reshape(4, 17)
    if dtype == torch.int64:
        values = values + (1 << 40)
    return values.to(dtype)


def write_case_tensors(work_dir, case_name, tensors):
    input_path = work_dir / f"{case_name}_src.bin"
    ptr_path = work_dir / f"{case_name}_indptr.bin"
    output_path = work_dir / f"{case_name}_output.bin"
    golden_path = work_dir / f"{case_name}_golden.bin"
    for tensor, path in (
        (tensors.src, input_path),
        (tensors.indptr, ptr_path),
        (tensors.actual, output_path),
        (tensors.expected, golden_path),
    ):
        write_tensor(tensor, path)
    return input_path, ptr_path, output_path, golden_path


def build_case_description(case_name, dtype_name, paths):
    input_path, ptr_path, output_path, golden_path = paths
    return {
        "case_name": case_name,
        "op_name": "GatherCsr",
        "case_path": "",
        "input_desc": [
            {
                "name": "src",
                "format": "ND",
                "data_type": dtype_name,
                "param_type": "required",
                "shape": [4, 17],
                "data_path": str(input_path),
            },
            {
                "name": "indptr",
                "format": "ND",
                "data_type": "int64",
                "param_type": "required",
                "shape": [5],
                "data_path": str(ptr_path),
            },
        ],
        "output_desc": [
            {
                "name": "out",
                "format": "ND",
                "data_type": dtype_name,
                "param_type": "required",
                "shape": [9, 17],
                "data_path": str(output_path),
                "golden_path": str(golden_path),
                "err_threshold": [0.0, 0.0],
            }
        ],
        "attr_desc": [],
    }


def create_case(work_dir, dtype_name, dtype, indptr_cpu):
    src_cpu = source_values(dtype)
    actual = gather_csr(src_cpu.to("npu"), indptr_cpu.to("npu"))
    expected = torch_scatter.gather_csr(src_cpu, indptr_cpu)
    torch.npu.synchronize()
    case_name = f"GatherCsr_{dtype_name}"
    tensors = CaseTensors(src_cpu, indptr_cpu, actual, expected)
    paths = write_case_tensors(work_dir, case_name, tensors)
    return build_case_description(case_name, dtype_name, paths)


def create_cases(work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    indptr_cpu = torch.tensor([0, 2, 5, 5, 9], dtype=torch.int64)
    return [
        create_case(work_dir, dtype_name, dtype, indptr_cpu)
        for dtype_name, dtype in DTYPES
    ]


def write_configuration(work_dir, cases):
    ir_path = work_dir / "gather_csr_ir.json"
    case_path = work_dir / "gather_csr_cases.json"
    result_path = work_dir / "result.csv"
    ir_data = [
        {
            "op": "GatherCsr",
            "input_desc": [
                {
                    "name": "src",
                    "param_type": "required",
                    "format": ["ND"] * len(DTYPES),
                    "type": [name for name, _ in DTYPES],
                },
                {
                    "name": "indptr",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["int64"],
                },
            ],
            "output_desc": [
                {
                    "name": "out",
                    "param_type": "required",
                    "format": ["ND"] * len(DTYPES),
                    "type": [name for name, _ in DTYPES],
                }
            ],
        }
    ]
    ir_path.write_text(json.dumps(ir_data, indent=2), encoding="utf-8")
    case_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    result_path.write_text("case_name,name,data_path,golden_path,compare_result\n", encoding="utf-8")
    return ir_path, case_path, result_path


def run_comparison(compare_entry, ir_path, case_path, result_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(compare_entry),
            "-i",
            str(ir_path),
            "-c",
            str(case_path),
            "-r",
            str(result_path),
        ],
        cwd=compare_entry.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return
    if completed.stdout:
        LOGGER.error("%s", completed.stdout.rstrip())
    if completed.stderr:
        LOGGER.error("%s", completed.stderr.rstrip())
    raise subprocess.CalledProcessError(completed.returncode, completed.args)


def validate_results(result_path):
    with result_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(DTYPES) or any(row["compare_result"].strip() != "pass" for row in rows):
        raise RuntimeError(f"AscendOpTest comparison failed; inspect {result_path}")
    for row in rows:
        LOGGER.info("%s: %s", row["case_name"], row["compare_result"].strip())
    LOGGER.info("AscendOpTest: %d/%d cases passed; result=%s", len(rows), len(DTYPES), result_path)


def main():
    work_dir, compare_entry = parse_args()
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))
    cases = create_cases(work_dir)
    ir_path, case_path, result_path = write_configuration(work_dir, cases)
    run_comparison(compare_entry, ir_path, case_path, result_path)
    validate_results(result_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()
