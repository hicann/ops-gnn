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

"""Segment CSR NPU latency benchmark.

Usage: python3 benchmark_segment_csr.py [--reduce sum] [--iter 100] [--warmup 20]
"""

import argparse
import logging
import os
import sys
import time
from itertools import product

import torch
import ops_gnn


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def emit_status(*parts):
    logging.getLogger(__name__).info("%s", " ".join(map(str, parts)))


def benchmark_shapes():
    shapes = []
    base_configs = ((256, 16), (512, 32), (1024, 32), (2048, 64), (4096, 64))
    for (n_k, seg_k), channels in product(base_configs, (32, 64, 128, 256)):
        if n_k * channels <= 4096 * 128:
            shapes.append((n_k * 1024, channels, seg_k * 1024))
    for n_k, seg_k in product((256, 1024, 4096), (8, 16, 64, 128)):
        if (n_k, seg_k) not in ((256, 16), (4096, 64)):
            shapes.append((n_k * 1024, 128, seg_k * 1024))
    return shapes


def make_inputs(n, channels, segment_count, dtype, device):
    if dtype in (torch.int32, torch.int64):
        src = torch.randint(-100, 100, (n, channels), dtype=dtype, device=device)
    else:
        src = torch.randn(n, channels, dtype=dtype, device=device)
    segment_size = n // segment_count
    indptr = torch.arange(0, n + 1, segment_size, device=device).long()
    if indptr.size(0) - 1 < segment_count:
        indptr = torch.linspace(0, n, segment_count + 1, device=device).long()
    elif indptr.size(0) - 1 > segment_count:
        indptr = indptr[: segment_count + 1]
        indptr[-1] = n
    return [src, indptr, None]


SHAPES = benchmark_shapes()
SHAPES = [
    (f" N={n // 1024:>4}K C={c:>3} seg={s // 1024:>3}K", n, c, s) for n, c, s in SHAPES
]
DTYPES = (torch.float32, torch.float16, torch.int32, torch.int64)


def setup_device(name):
    device = torch.device(name)
    if device.type == "npu":
        torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


@torch.no_grad()
def bench(tensor_args, warmup, iters, device):
    for _ in range(warmup):
        ops_gnn.segment_csr(*tensor_args)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        ops_gnn.segment_csr(*tensor_args)
    synchronize(device)
    return (time.perf_counter() - start) / iters * 1000


def main():
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--reduce", default="sum", choices=("sum", "mean", "min", "max")
    )
    args = parser.parse_args()
    device = setup_device(args.device)

    emit_status(
        f"device={device}  warmup={args.warmup}  iters={args.iter}  reduce={args.reduce}\n\nsegment_csr"
    )
    emit_status(
        f"{'shape':>35s}"
        + "".join((f" {str(dtype).removeprefix('torch.'):>10s}" for dtype in DTYPES))
    )
    emit_status("-" * 83)
    for label, n, channels, segment_count in SHAPES:
        row = f"{label:>35s}"
        for dtype in DTYPES:
            tensor_args = [
                *make_inputs(n, channels, segment_count, dtype, device),
                args.reduce,
            ]
            latency = bench(tensor_args, args.warmup, args.iter, device)
            row += f" {latency:>8.3f}ms"
        emit_status(row)


if __name__ == "__main__":
    main()
