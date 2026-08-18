# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Extra functional coverage for :func:`ops_gnn.radius` / :func:`ops_gnn.radius_graph`.

Covers batch isolation, neighbor truncation, ignore_same_index, the grid
kernel paths, dtype matrix and the float64 CPU fallback. Assertions use set
equivalence (except explicit exact-order checks) per the task-book precision
standard for the random-truncation path.
"""

from __future__ import annotations

import os
import warnings

import pytest
import torch

import ops_gnn
from ops_gnn import _pybind


pytest.importorskip("torch_npu")
ops_gnn = pytest.importorskip("ops_gnn")
if not hasattr(ops_gnn, "radius"):
    pytest.skip("ops_gnn.radius is not built", allow_module_level=True)
radius = ops_gnn.radius
radius_graph = ops_gnn.radius_graph


@pytest.fixture(scope="module")
def device():
    """Select the NPU card from ``NPU_DEVICE_ID`` (default 0)."""
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    return f"npu:{int(os.environ.get('NPU_DEVICE_ID', '0'))}"


def _full_neighbors(x: torch.Tensor, y: torch.Tensor, r: float,
                    ignore_same_index: bool, batches: tuple | None = None):
    """Brute-force oracle: every in-radius x index per query, ascending order.

    ``batches`` is an optional ``(batch_x, batch_y)`` pair of Long tensors.
    """
    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    n, m = x.shape[0], y.shape[0]
    if batches is None:
        bx = torch.zeros(n, dtype=torch.long)
        by = torch.zeros(m, dtype=torch.long)
    else:
        bx, by = batches
    full = [[] for _ in range(m)]
    r2 = float(r) * float(r)
    for q in range(m):
        idx = torch.arange(n, dtype=torch.long)
        mask = bx == by[q].item()
        if ignore_same_index:
            mask = mask & (idx != q)
        if mask.sum().item() == 0:
            continue
        valid_idx = torch.nonzero(mask).flatten()
        diff = x[valid_idx].double() - y[q].double()
        dist2 = (diff * diff).sum(dim=-1)
        full[q] = valid_idx[dist2 <= r2].tolist()
    return full


class _QuerySpec:
    """Packed validation options for the edge oracle (radius, K, batches)."""

    def __init__(self, r: float, max_num_neighbors: int,
                 batches: tuple | None = None, ignore_same_index: bool = False):
        self.r = r
        self.max_num_neighbors = max_num_neighbors
        self.batches = batches
        self.ignore_same_index = ignore_same_index

    def full_neighbors(self, x: torch.Tensor, y: torch.Tensor):
        return _full_neighbors(x, y, self.r, self.ignore_same_index, self.batches)


def _assert_edges(edge: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                  spec: _QuerySpec):
    """Set-equivalence check against the brute-force oracle."""
    edge = edge.cpu()
    assert edge.dtype == torch.long
    assert edge.shape[0] == 2
    x = x.cpu()
    y = y.cpu()
    if spec.batches is not None:
        spec = _QuerySpec(
            spec.r, spec.max_num_neighbors,
            (spec.batches[0].cpu(), spec.batches[1].cpu()),
            spec.ignore_same_index,
        )
    full = spec.full_neighbors(x, y)
    accepted = {}
    for q, i in zip(edge[0].tolist(), edge[1].tolist()):
        accepted.setdefault(q, []).append(i)
    m = y.shape[0]
    for q in range(m):
        got = sorted(set(accepted.get(q, [])))
        assert len(got) == len(accepted.get(q, []))
        assert len(got) <= spec.max_num_neighbors
        assert len(got) == min(len(full[q]), spec.max_num_neighbors)
        if len(full[q]) <= spec.max_num_neighbors:
            assert set(got) == set(full[q])
        else:
            assert set(got).issubset(set(full[q]))


def _small_tensor(device, dtype=torch.float32):
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
        device=device, dtype=dtype,
    )


def test_basic_radius(device):
    x = _small_tensor(device)
    edge = radius(x, x.clone(), 1.5, max_num_neighbors=4)
    _assert_edges(edge, x, x, _QuerySpec(1.5, 4))


def test_batch_radius(device):
    x = _small_tensor(device)
    y = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], device=device)
    batch_x = torch.tensor([0, 0, 1, 1], device=device)
    batch_y = torch.tensor([0, 1], device=device)
    edge = radius(x, y, 1.5, batch_x, batch_y, max_num_neighbors=4)
    _assert_edges(edge, x, y, _QuerySpec(1.5, 4, (batch_x, batch_y)))


def test_max_num_neighbors_truncation(device):
    torch.manual_seed(7)
    x = torch.rand(128, 3, device=device)
    y = torch.tensor([[0.5, 0.5, 0.5]], device=device)
    edge = radius(x, y, 0.6, max_num_neighbors=5)
    _assert_edges(edge, x, y, _QuerySpec(0.6, 5))
    # Exact-order assertion matching the official test_radius.py (torch.equal):
    # the kernel must emit neighbors in x-index ascending order.
    full = _full_neighbors(x.cpu(), y.cpu(), 0.6, False)
    exp_row = []
    exp_col = []
    for q, nb in enumerate(full):
        for i in nb[:5]:
            exp_row.append(q)
            exp_col.append(i)
    expected = torch.tensor([exp_row, exp_col], dtype=torch.long)
    assert torch.equal(edge.cpu(), expected), \
        "radius truncation output must be exact x-index ascending order"


def test_ignore_same_index(device):
    x = _small_tensor(device)
    edge = radius(x, x.clone(), 1.5, max_num_neighbors=4, ignore_same_index=True)
    _assert_edges(edge, x, x, _QuerySpec(1.5, 4, ignore_same_index=True))


def test_radius_graph_loop_flow(device):
    x = _small_tensor(device)
    full = _full_neighbors(x.cpu(), x.cpu(), 1.5, True)

    edge = radius_graph(x, 1.5, loop=False, max_num_neighbors=4,
                        flow="source_to_target")
    expected = set()
    for q, neighbors in enumerate(full):
        for i in neighbors:
            expected.add((i, q))  # source_to_target swaps radius rows
    assert set(zip(edge[0].tolist(), edge[1].tolist())) == expected

    edge_t = radius_graph(x, 1.5, loop=False, max_num_neighbors=4,
                          flow="target_to_source")
    expected_t = {(q, i) for q, neighbors in enumerate(full) for i in neighbors}
    assert set(zip(edge_t[0].tolist(), edge_t[1].tolist())) == expected_t

    full_loop = _full_neighbors(x.cpu(), x.cpu(), 1.5, False)
    expected_loop = set()
    for q, neighbors in enumerate(full_loop):
        for i in neighbors:
            expected_loop.add((i, q))
    edge_loop = radius_graph(x, 1.5, loop=True, max_num_neighbors=4,
                             flow="source_to_target")
    assert set(zip(edge_loop[0].tolist(), edge_loop[1].tolist())) == expected_loop


def test_empty_input(device):
    x = torch.empty(0, 3, device=device)
    edge = radius(x, x.clone(), 1.0)
    assert edge.shape == (2, 0)
    assert edge.dtype == torch.long
    assert edge.device.type == "npu"

    x2 = torch.rand(4, 3, device=device)
    edge2 = radius(x2, torch.empty(0, 3, device=device), 1.0)
    assert edge2.shape == (2, 0)


def test_1d_input(device):
    x = torch.tensor([0.0, 1.0, 10.0, 11.0], device=device)
    y = torch.tensor([0.0, 10.0], device=device)
    edge = radius(x, y, 1.5, max_num_neighbors=4)
    _assert_edges(edge, x.view(-1, 1), y.view(-1, 1), _QuerySpec(1.5, 4))


def test_grid_path_random(device):
    # n=m=64 -> the host builds the CSR grid (grid requires n>=8 and m>=8).
    torch.manual_seed(11)
    x = torch.rand(64, 3, device=device)
    y = torch.rand(64, 3, device=device)
    edge = radius(x, y, 0.35, max_num_neighbors=64)
    _assert_edges(edge, x, y, _QuerySpec(0.35, 64))


def test_grid_path_batched(device):
    # 3 sorted batches in the same coordinate range: batch-isolation break
    # would surface as cross-batch neighbors within r.
    torch.manual_seed(13)
    x = torch.rand(96, 3, device=device)
    y = torch.rand(96, 3, device=device)
    batch_x = torch.repeat_interleave(torch.arange(3, device=device), 32)
    batch_y = batch_x.clone()
    edge = radius(x, y, 0.4, batch_x, batch_y, max_num_neighbors=32)
    _assert_edges(edge, x, y, _QuerySpec(0.4, 32, (batch_x, batch_y)))


def test_grid_path_truncation(device):
    # Dense queries overflow K=7, firing the truncation path (subset equiv).
    torch.manual_seed(19)
    x = torch.rand(256, 3, device=device)
    y = torch.rand(64, 3, device=device) * 0.2 + 0.4
    edge = radius(x, y, 0.9, max_num_neighbors=7)
    _assert_edges(edge, x, y, _QuerySpec(0.9, 7))


def test_grid_path_ignore_same_index(device):
    torch.manual_seed(17)
    x = torch.rand(96, 3, device=device)
    edge = radius(x, x, 0.4, max_num_neighbors=96, ignore_same_index=True)
    _assert_edges(edge, x, x, _QuerySpec(0.4, 96, ignore_same_index=True))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtypes(device, dtype):
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [2.0, 2.0, 2.0],
        ],
        device=device, dtype=dtype,
    )
    edge = radius(x, x.clone(), 0.75, max_num_neighbors=10)
    _assert_edges(edge, x, x, _QuerySpec(0.75, 10))


def test_float64_cpu_fallback(device):
    x = _small_tensor(device, dtype=torch.float64)
    edge = radius(x, x.clone(), 1.5, max_num_neighbors=4)
    _assert_edges(edge, x, x, _QuerySpec(1.5, 4))


def test_non_contiguous_inputs(device):
    torch.manual_seed(5)
    base = torch.rand(16, 3, device=device)
    x = base[::2]
    y = base[1::2]
    assert not x.is_contiguous()
    assert not y.is_contiguous()
    stream = int(torch.npu.current_stream().npu_stream)
    expected = _pybind.radius(
        x.contiguous(), y.contiguous(), None, None, 0.4, 8, 1, False, stream)
    result = _pybind.radius(
        x, y, None, None, 0.4, 8, 1, False, stream)
    assert torch.equal(result.cpu(), expected.cpu())


def test_cpu_ptr_rejected(device):
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
         [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
        device=device,
    )
    y = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], device=device)
    ptr_x = torch.tensor([0, 4], dtype=torch.long)
    ptr_y = torch.tensor([0, 2], dtype=torch.long)
    stream = int(torch.npu.current_stream().npu_stream)
    with pytest.raises(RuntimeError, match="same device"):
        _pybind.radius(
            x, y, ptr_x, ptr_y, 1.5, 4, 1, False, stream)


@pytest.mark.parametrize(
    "ptr_x,ptr_y,match",
    [
        ([1, 4], [0, 2], "must start with 0"),
        ([0, 5], [0, 2], "must end with the data length"),
        ([0, 3, 2, 4], [0, 0, 1, 2], "must be non-decreasing"),
        ([0, 5, 4], [0, 0, 2], "value out of range"),
    ],
)
def test_invalid_batch_ptr_rejected(device, ptr_x, ptr_y, match):
    x = torch.rand(4, 3, device=device)
    y = torch.rand(2, 3, device=device)
    stream = int(torch.npu.current_stream().npu_stream)
    with pytest.raises(RuntimeError, match=match):
        _pybind.radius(
            x, y, torch.tensor(ptr_x, dtype=torch.long, device=device),
            torch.tensor(ptr_y, dtype=torch.long, device=device),
            1.0, 4, 1, False, stream)


def test_grid_path_no_internal_format_warning(device):
    torch.manual_seed(11)
    x = torch.rand(64, 3, device=device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        radius(x, x, 0.35, max_num_neighbors=64)
    internal_warnings = [
        w for w in caught if "internal format" in str(w.message).lower()
    ]
    assert not internal_warnings


def test_non_finite_coordinates_no_crash(device):
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0],
         [1.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]],
        device=device,
    )
    edge = radius(x, x, 1.0, max_num_neighbors=4)
    assert edge.shape[0] == 2
    assert edge.device.type == "npu"
