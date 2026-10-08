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

"""Segment COO NPU latency benchmark.

Usage: python3 benchmark_segment_coo.py [--reduce sum] [--iter 100] [--warmup 20]
"""

import argparse
import logging
import os
import time
from typing import NamedTuple

import torch
import ops_gnn


_BASE_CONFIGS = ((256, 16), (512, 32), (1024, 32), (2048, 64), (4096, 64))
_MAX_ELEMENTS = 4096 * 128


def benchmark_shapes():
    shapes = []
    for n_k, seg_k in _BASE_CONFIGS:
        for channels in (32, 64, 128, 256):
            if n_k * channels <= _MAX_ELEMENTS:
                label = f" N={n_k:>4}K C={channels:>3} seg={seg_k:>3}K"
                shapes.append((label, n_k * 1024, channels, seg_k * 1024))
    for n_k in (256, 1024, 4096):
        for seg_k in (8, 16, 64, 128):
            if (n_k, seg_k) not in ((256, 16), (4096, 64)):
                label = f" N={n_k:>4}K C=128 seg={seg_k:>3}K"
                shapes.append((label, n_k * 1024, 128, seg_k * 1024))
    return shapes


SHAPES = benchmark_shapes()
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


class InputShape(NamedTuple):
    rows: int
    channels: int
    segments: int


def make_inputs(shape, reduce, dtype, device):
    n, channels, segment_count = shape
    if dtype in (torch.int32, torch.int64):
        src = torch.randint(-100, 100, (n, channels), dtype=dtype, device=device)
    else:
        src = torch.randn(n, channels, dtype=dtype, device=device)
    segment_size = n // segment_count
    index = torch.arange(segment_count, device=device).repeat_interleave(segment_size)
    if index.size(0) < n:
        tail = torch.full((n - index.size(0),), segment_count - 1, device=device)
        index = torch.cat((index, tail))
    return [src, index, None, segment_count, reduce]


@torch.no_grad()
def bench(tensor_args, warmup, iters, device):
    for _ in range(warmup):
        ops_gnn.segment_coo(*tensor_args)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        ops_gnn.segment_coo(*tensor_args)
    synchronize(device)
    return (time.perf_counter() - start) / iters * 1000


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--reduce", default="sum", choices=("sum", "mean", "min", "max")
    )
    args = parser.parse_args()
    device = setup_device(args.device)

    logging.info(
        f"device={device}  warmup={args.warmup}  iters={args.iter}  reduce={args.reduce}\n\nsegment_coo"
    )
    logging.info(
        f"{'shape':>35s}"
        + "".join(f" {str(dtype).removeprefix('torch.'):>10s}" for dtype in DTYPES)
    )
    logging.info("-" * 83)
    for label, n, channels, segment_count in SHAPES:
        row = f"{label:>35s}"
        for dtype in DTYPES:
            tensor_args = make_inputs(
                InputShape(n, channels, segment_count), args.reduce, dtype, device
            )
            latency = bench(tensor_args, args.warmup, args.iter, device)
            row += f" {latency:>8.3f}ms"
        logging.info("%s", row)


if __name__ == "__main__":
    main()
