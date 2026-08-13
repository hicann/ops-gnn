# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import ops_gnn


@dataclass
class _ReferenceGraph:
    rowptr_list: list
    col_list: list
    weight_list: Optional[list]
    cluster: list
    node_count: int


def _maybe_num_nodes(row: torch.Tensor, col: torch.Tensor, num_nodes: Optional[int]) -> int:
    if num_nodes is not None:
        return int(num_nodes)
    if row.numel() == 0 and col.numel() == 0:
        return 0
    return int(torch.maximum(row.max(), col.max()).item()) + 1


def _reference_preprocess_edges(row, col, weight, node_count):
    mask = row != col
    row = row[mask]
    col = col[mask]
    weight = weight[mask] if weight is not None else None
    if row.numel() == 0:
        return torch.zeros(node_count + 1, dtype=torch.long), col, weight
    row, col, weight = _reference_shuffle(row, col, weight)
    row, col, weight = _reference_sort(row, col, weight)
    rowptr = torch.bucketize(torch.arange(node_count + 1, dtype=torch.long), row)
    return rowptr, col, weight


def _reference_shuffle(row, col, weight):
    if weight is not None:
        return row, col, weight
    edge_perm = torch.randperm(row.numel(), device=torch.device("cpu"))
    return row[edge_perm], col[edge_perm], weight


def _reference_sort(row, col, weight):
    is_sorted = row.numel() <= 1 or bool((row[1:] >= row[:-1]).all().item())
    if is_sorted:
        return row, col, weight
    perm = torch.argsort(row, stable=True)
    weight = weight[perm] if weight is not None else None
    return row[perm], col[perm], weight


def _reference_available(graph: _ReferenceGraph, neighbor):
    return 0 <= neighbor < graph.node_count and graph.cluster[neighbor] < 0


def _reference_assign(graph: _ReferenceGraph, node, neighbor):
    cluster_value = min(node, neighbor)
    graph.cluster[node] = cluster_value
    graph.cluster[neighbor] = cluster_value


def _reference_match_unweighted(graph: _ReferenceGraph, node):
    graph.cluster[node] = node
    start = graph.rowptr_list[node]
    end = graph.rowptr_list[node + 1]
    for edge_idx in range(start, end):
        neighbor = graph.col_list[edge_idx]
        if _reference_available(graph, neighbor):
            _reference_assign(graph, node, neighbor)
            return


def _reference_best_weighted_neighbor(graph: _ReferenceGraph, node):
    best_neighbor = node
    best_weight = 0.0
    start = graph.rowptr_list[node]
    end = graph.rowptr_list[node + 1]
    for edge_idx in range(start, end):
        neighbor = graph.col_list[edge_idx]
        if not _reference_available(graph, neighbor):
            continue
        edge_weight = float(graph.weight_list[edge_idx])
        if edge_weight >= best_weight:
            best_weight = edge_weight
            best_neighbor = neighbor
    return best_neighbor


def _reference_process_node(graph: _ReferenceGraph, node):
    if graph.cluster[node] >= 0:
        return
    if graph.weight_list is None:
        _reference_match_unweighted(graph, node)
        return
    _reference_assign(graph, node, _reference_best_weighted_neighbor(graph, node))


def _make_reference_graph(rowptr, col, weight, node_count):
    weight_list = weight.tolist() if weight is not None else None
    return _ReferenceGraph(rowptr.tolist(), col.tolist(), weight_list, [-1] * node_count, node_count)


def _reference_greedy(rowptr, col, weight, node_count):
    graph = _make_reference_graph(rowptr, col, weight, node_count)
    node_perm = torch.randperm(node_count, device=torch.device("cpu")).tolist()
    for node in node_perm:
        _reference_process_node(graph, node)
    return torch.tensor(graph.cluster, dtype=torch.long)


def reference_graclus_cluster(
    row: torch.Tensor,
    col: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    row = row.cpu().contiguous()
    col = col.cpu().contiguous()
    weight = weight.cpu().contiguous() if weight is not None else None
    node_count = _maybe_num_nodes(row, col, num_nodes)
    if node_count == 0:
        return torch.empty(0, dtype=torch.long)
    rowptr, csr_col, csr_weight = _reference_preprocess_edges(row, col, weight, node_count)
    return _reference_greedy(rowptr, csr_col, csr_weight, node_count)


def _npu_or_skip():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pytest.skip("torch_npu is not installed")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU device is not available")
    try:
        torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    except RuntimeError as exc:
        pytest.skip(f"NPU runtime is not usable: {exc}")


def _run_reference_and_npu(row, col, weight=None, num_nodes=None, seed=2026):
    torch.manual_seed(seed)
    expected = reference_graclus_cluster(row, col, weight, num_nodes)
    _npu_or_skip()
    torch.manual_seed(seed)
    actual = ops_gnn.graclus_cluster(
        row.to("npu"),
        col.to("npu"),
        weight.to("npu") if weight is not None else None,
        num_nodes,
    )
    if actual.device.type != "npu":
        pytest.fail(f"expected NPU output, but got {actual.device.type}")
    if actual.dtype != torch.long:
        pytest.fail(f"expected torch.long output, but got {actual.dtype}")
    if not torch.equal(actual.cpu(), expected):
        pytest.fail("NPU result is not bit-wise equal to CPU reference")



def make_random_edges(num_nodes, num_edges, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    row = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long, generator=generator)
    col = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long, generator=generator)
    return row, col
