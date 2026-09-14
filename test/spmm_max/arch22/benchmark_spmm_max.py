# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run from the repository root with PYTHONPATH=python; timings include validation."""
import argparse
import json
import logging
import os
import platform
import statistics
import time

import torch
import torch_npu
from ops_gnn import spmm_max_csr
from golden import spmm_max_reference

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--sources", type=int, default=4096)
    parser.add_argument("--edges", type=int, default=65536)
    parser.add_argument("--features", type=int, default=33)
    parser.add_argument("--distribution", choices=["uniform", "powerlaw", "hub", "empty"], default="uniform")
    parser.add_argument("--index-dtype", choices=["int32", "int64"], default="int64")
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    if min(args.rows, args.sources, args.features, args.repeat) <= 0 or args.edges < 0:
        parser.error("rows, sources, features, repeat must be positive; edges nonnegative")
    torch.manual_seed(42)
    weights = torch.ones(args.rows)
    if args.distribution == "powerlaw":
        weights = torch.arange(1, args.rows + 1).float().pow(-1.5)
    if args.distribution == "hub":
        weights[0] = args.rows * 100
    if args.distribution == "empty":
        weights[1:] = 0
    rows = torch.multinomial(weights, args.edges, replacement=True) if args.edges else torch.empty(0, dtype=torch.long)
    dtype = getattr(torch, args.index_dtype)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(rows, minlength=args.rows).cumsum(0)]).to(dtype)
    idx = torch.randint(args.sources, (args.edges,), dtype=dtype)
    x = torch.randn(args.sources, args.features, dtype=torch.float16)
    expected = spmm_max_reference(ptr, idx, x)
    ptr, idx, x = ptr.npu(), idx.npu(), x.npu()

    def timed():
        torch.npu.synchronize()
        start = time.perf_counter()
        result = spmm_max_csr(ptr, idx, x)
        torch.npu.synchronize()
        return (time.perf_counter() - start) * 1000, result

    cold, got = timed()
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    for _ in range(10):
        timed()
    times = [timed()[0] for _ in range(args.repeat)]
    _LOGGER.info(json.dumps({"parameters": vars(args), "python": platform.python_version(),
                             "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                             "device": torch.npu.get_device_name(), "NPU_ARCH": os.getenv("TARGET_NPU_ARCH"),
                             "cold_ms": cold, "median_ms": statistics.median(times),
                             "min_ms": min(times), "max_ms": max(times),
                             "mean_ms": statistics.mean(times), "stdev_ms": statistics.pstdev(times)},
                            indent=2))


if __name__ == "__main__":
    main()
