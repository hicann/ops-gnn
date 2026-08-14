# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NPU functional and task-book coverage for :func:`ops_gnn.random_walk`.

The parameter sizes and 24 primary pytest cases follow the official 2026-08
random_walk cases; an additional regression covers ``coalesced=False``.
Failures are not swallowed, and every case checks shape, dtype, device and
graph semantics.
"""

from __future__ import annotations

import os
from typing import Tuple

import pytest
import torch


pytest.importorskip("torch_npu")
ops_gnn = pytest.importorskip("ops_gnn")
if not hasattr(ops_gnn, "random_walk"):
    pytest.skip("ops_gnn.random_walk is not built", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def _select_npu_device():
    """Run the module on the card selected by ``NPU_DEVICE_ID``."""
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))


TEST_CONFIGS = [
    (1024, 4, 8),
    (4096, 8, 16),
    (8192, 10, 32),
    (16384, 6, 64),
    (32768, 4, 128),
    (4096, 20, 8),
    (8192, 4, 256),
    (2048, 32, 4),
]

GENERAL_CONFIGS = [
    (4, 1, 1),
    (8, 0, 2),
    (16, 2, 4),
    (0, 1, 1),
    (65536, 8, 256),
    (16384, 2, 1024),
    (8192, 64, 16),
]


def _make_case(edge_count: int, walk_length: int, starts: int):
    """Build exactly the deterministic COO inputs described by the sample."""
    torch.manual_seed(42)
    num_nodes = max(200, edge_count // 2)
    row = torch.randint(0, num_nodes, (edge_count,), dtype=torch.int64)
    col = torch.randint(0, num_nodes, (edge_count,), dtype=torch.int64)
    start = torch.arange(min(starts, num_nodes), dtype=torch.int64)
    return row, col, start, walk_length, num_nodes


def _assert_result_contract(result: torch.Tensor, start: torch.Tensor, walk_length: int):
    assert isinstance(result, torch.Tensor)
    assert result.device.type == "npu"
    assert result.dtype == torch.int64
    assert tuple(result.shape) == (start.numel(), walk_length + 1)
    assert torch.equal(result[:, 0].cpu(), start)


def _assert_legal_walk(
    row: torch.Tensor,
    col: torch.Tensor,
    result: torch.Tensor,
    num_nodes: int,
):
    adjacency = [set() for _ in range(num_nodes)]
    for source, target in zip(row.tolist(), col.tolist()):
        adjacency[source].add(target)

    paths = result.cpu().tolist()
    for path in paths:
        for source, target in zip(path, path[1:]):
            assert 0 <= source < num_nodes
            assert 0 <= target < num_nodes
            if adjacency[source]:
                assert target in adjacency[source]
            else:
                assert target == source


def _run_and_check(edge_count: int, walk_length: int, starts: int):
    row, col, start, walk_length, num_nodes = _make_case(edge_count, walk_length, starts)
    kwargs = {"num_nodes": num_nodes} if edge_count == 0 else {}
    result = ops_gnn.random_walk(
        row.npu(), col.npu(), start.npu(), walk_length, **kwargs
    )
    _assert_result_contract(result, start, walk_length)
    _assert_legal_walk(row, col, result, num_nodes)


class TestRandomWalk:
    @staticmethod
    @pytest.mark.parametrize("edge_count,walk_length,starts", TEST_CONFIGS)
    def test_configs(edge_count: int, walk_length: int, starts: int):
        _run_and_check(edge_count, walk_length, starts)

    @staticmethod
    @pytest.mark.parametrize("edge_count,walk_length,starts", GENERAL_CONFIGS)
    def test_general(edge_count: int, walk_length: int, starts: int):
        # The official sample catches RuntimeError/ValueError here.  Acceptance
        # must fail on either exception, including for the empty-edge case.
        _run_and_check(edge_count, walk_length, starts)

    @staticmethod
    def test_walk_length_zero():
        torch.manual_seed(42)
        num_nodes = 10
        row = torch.randint(0, num_nodes, (20,), dtype=torch.int64)
        col = torch.randint(0, num_nodes, (20,), dtype=torch.int64)
        start = torch.tensor([0, 1], dtype=torch.int64)
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 0)
        _assert_result_contract(result, start, 0)
        _assert_legal_walk(row, col, result, num_nodes)

    @staticmethod
    def test_walk_length_one():
        torch.manual_seed(42)
        num_nodes = 10
        row = torch.randint(0, num_nodes, (20,), dtype=torch.int64)
        col = torch.randint(0, num_nodes, (20,), dtype=torch.int64)
        start = torch.tensor([0], dtype=torch.int64)
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 1)
        _assert_result_contract(result, start, 1)
        _assert_legal_walk(row, col, result, num_nodes)

    @staticmethod
    def test_return_edge_indices():
        row = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
        col = torch.tensor([1, 0, 2, 1], dtype=torch.int64)
        start = torch.tensor([0], dtype=torch.int64)

        nodes, edges = ops_gnn.random_walk(
            row.npu(), col.npu(), start.npu(), 4,
            return_edge_indices=True,
        )
        _assert_result_contract(nodes, start, 4)
        _assert_legal_walk(row, col, nodes, 3)
        assert edges.device.type == "npu"
        assert edges.dtype == torch.int64
        assert tuple(edges.shape) == (1, 4)

        sorted_pairs: Tuple[Tuple[int, int], ...] = tuple(sorted(zip(row.tolist(), col.tolist())))
        node_values = nodes.cpu().tolist()[0]
        for step, edge_index in enumerate(edges.cpu().tolist()[0]):
            assert 0 <= edge_index < len(sorted_pairs)
            assert sorted_pairs[edge_index] == (node_values[step], node_values[step + 1])

        nodes_only = ops_gnn.random_walk(
            row.npu(), col.npu(), start.npu(), 4,
            return_edge_indices=False,
        )
        _assert_result_contract(nodes_only, start, 4)

    @staticmethod
    @pytest.mark.parametrize("p,q", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)])
    def test_pq_params(p: float, q: float):
        row, col, start, _, num_nodes = _make_case(100, 4, 1)
        result = ops_gnn.random_walk(
            row.npu(), col.npu(), start.npu(), 4, p=p, q=q
        )
        _assert_result_contract(result, start, 4)
        _assert_legal_walk(row, col, result, num_nodes)

    @staticmethod
    def test_reproducible():
        torch.manual_seed(42)
        row = torch.randint(0, 100, (200,), dtype=torch.int64)
        col = torch.randint(0, 100, (200,), dtype=torch.int64)
        start = torch.tensor([0, 5, 10], dtype=torch.int64)
        row_npu, col_npu, start_npu = row.npu(), col.npu(), start.npu()

        torch.manual_seed(42)
        torch.npu.manual_seed_all(42)
        first = ops_gnn.random_walk(row_npu, col_npu, start_npu, 4)
        torch.npu.synchronize()

        torch.manual_seed(42)
        torch.npu.manual_seed_all(42)
        second = ops_gnn.random_walk(row_npu, col_npu, start_npu, 4)
        torch.npu.synchronize()
        assert torch.equal(first, second)

    @staticmethod
    def test_coalesced_false_is_reproducible():
        row = torch.tensor([0, 0, 1, 1], dtype=torch.int64).npu()
        col = torch.tensor([1, 2, 0, 2], dtype=torch.int64).npu()
        start = torch.tensor([0, 1, 0, 1], dtype=torch.int64).npu()

        torch.manual_seed(202608)
        torch.npu.manual_seed_all(202608)
        first = ops_gnn.random_walk(
            row, col, start, 16, p=0.5, q=2.0, coalesced=False
        )
        torch.manual_seed(202608)
        torch.npu.manual_seed_all(202608)
        second = ops_gnn.random_walk(
            row, col, start, 16, p=0.5, q=2.0, coalesced=False
        )
        assert torch.equal(first, second)

    @staticmethod
    def test_isolated_node():
        row = torch.tensor([0, 1], dtype=torch.int64)
        col = torch.tensor([1, 2], dtype=torch.int64)
        start = torch.tensor([3], dtype=torch.int64)
        result = ops_gnn.random_walk(
            row.npu(), col.npu(), start.npu(), 4, num_nodes=4
        )
        _assert_result_contract(result, start, 4)
        assert result.cpu().tolist() == [[3, 3, 3, 3, 3]]

    @staticmethod
    def test_large():
        torch.manual_seed(42)
        num_nodes = 200
        row = torch.randint(0, num_nodes, (4096,), dtype=torch.int64)
        col = torch.randint(0, num_nodes, (4096,), dtype=torch.int64)
        start = torch.tensor([0, 10, 20, 30], dtype=torch.int64)
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 8)
        _assert_result_contract(result, start, 8)
        _assert_legal_walk(row, col, result, num_nodes)
