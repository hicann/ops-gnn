# Public interface signatures are adapted from torch_scatter 2.1.2 (MIT).
# Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
# The upstream permission notice is retained in the root LICENSE third-party notices.
# NPU bindings and task-specific semantics are CANN additions under the license below.
# Copyright (c) 2026.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Sorted COO reductions. index must be sorted along its final dimension."""

from typing import Optional, Tuple
import torch
from . import _pybind


def segment_sum_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Sum each sorted segment; a supplied output is overwritten."""
    return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_SUM)[0]


def segment_add_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Alias for segment_sum_coo with the same output-buffer semantics."""
    return segment_sum_coo(src, index, out, dim_size)


def segment_mean_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute segment means; integer division truncates toward zero."""
    return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_MEAN)[0]


def segment_min_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return segment minima and int64 positions, choosing the later tied index."""
    return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_MIN)


def segment_max_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return segment maxima and int64 positions, choosing the later tied index."""
    return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_MAX)


def segment_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Reduce along index.dim() - 1; return values only for every reduction.

    Index must contain sorted, in-range int64 segment identifiers. See the
    API reference in docs/zh/api_reference.md for broadcasting and output buffers.
    """
    if reduce in ("sum", "add"):
        return segment_sum_coo(src, index, out, dim_size)
    if reduce == "mean":
        return segment_mean_coo(src, index, out, dim_size)
    if reduce == "min":
        return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_MIN, False)[0]
    if reduce == "max":
        return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_MAX, False)[0]
    raise ValueError("segment_coo supports sum/add/mean/min/max, not " + str(reduce))
