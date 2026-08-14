# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Offline correctness tests for the deterministic random_walk CPU golden."""

import math

import pytest
import torch

from .golden import (
    PhiloxStream,
    _is_neighbor,
    assert_walk_consistent,
    philox4x32_10,
    prepare_csr,
    random_walk_golden,
)


def test_philox_known_vector():
    assert philox4x32_10((0, 0, 0, 0), (0, 0)) == (
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    )


def test_philox_stream_is_repeatable_and_partitioned():
    first = PhiloxStream(seed=42, offset=8, walk_index=3)
    second = PhiloxStream(seed=42, offset=8, walk_index=3)
    other_walk = PhiloxStream(seed=42, offset=8, walk_index=4)
    values = [first.next_u32() for _ in range(12)]
    assert values == [second.next_u32() for _ in range(12)]
    assert values != [other_walk.next_u32() for _ in range(12)]


def test_uniform_degree_one_and_isolated_nodes_are_exact():
    row = torch.tensor([0, 1], dtype=torch.int64)
    col = torch.tensor([1, 0], dtype=torch.int64)
    start = torch.tensor([0, 1, 2], dtype=torch.int64)
    node, edge = random_walk_golden(
        row,
        col,
        start,
        4,
        num_nodes=3,
        return_edge_indices=True,
        seed=99,
    )
    assert node.tolist() == [
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [2, 2, 2, 2, 2],
    ]
    assert edge.tolist() == [[0, 1, 0, 1], [1, 0, 1, 0], [-1, -1, -1, -1]]


@pytest.mark.parametrize("p,q", [(1.0, 1.0), (0.25, 4.0), (4.0, 0.25)])
def test_paths_are_graph_consistent(p, q):
    row = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.int64)
    col = torch.tensor([1, 0, 2, 3, 1, 4, 1, 4, 2, 3], dtype=torch.int64)
    start = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    rowptr, sorted_col = prepare_csr(row, col, start)
    node, edge = random_walk_golden(
        row,
        col,
        start,
        32,
        p=p,
        q=q,
        return_edge_indices=True,
        seed=202608,
    )
    assert_walk_consistent(rowptr, sorted_col, node, edge)


def test_coalesced_sort_and_edge_index_semantics():
    row = torch.tensor([1, 0, 1], dtype=torch.int64)
    col = torch.tensor([2, 1, 0], dtype=torch.int64)
    start = torch.tensor([0], dtype=torch.int64)
    node, edge = random_walk_golden(
        row,
        col,
        start,
        1,
        coalesced=True,
        return_edge_indices=True,
        seed=0,
    )
    assert node.tolist() == [[0, 1]]
    assert edge.tolist() == [[0]]


def test_uncoalesced_large_unsorted_neighbor_lookup():
    targets = list(range(1, 21))
    targets[0], targets[-1] = targets[-1], targets[0]
    rowptr = torch.tensor([0, 20] + [20] * 20, dtype=torch.int64)
    col = torch.tensor(targets, dtype=torch.int64)
    assert _is_neighbor(rowptr, col, 0, 20, neighbors_sorted=False)
    assert not _is_neighbor(rowptr, col, 0, 99, neighbors_sorted=False)


def test_walk_length_zero():
    row = torch.tensor([0], dtype=torch.int64)
    col = torch.tensor([0], dtype=torch.int64)
    start = torch.tensor([0, 0], dtype=torch.int64)
    node, edge = random_walk_golden(
        row,
        col,
        start,
        0,
        p=2.0,
        q=0.5,
        return_edge_indices=True,
    )
    assert node.shape == (2, 1)
    assert edge.shape == (2, 0)
    assert node[:, 0].tolist() == [0, 0]


@pytest.mark.parametrize("p,q", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (math.inf, 1.0)])
def test_invalid_bias_is_rejected(p, q):
    value = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(ValueError):
        random_walk_golden(value, value, value, 1, p=p, q=q)


def test_extreme_finite_bias_does_not_overflow_normalization():
    value = torch.tensor([0], dtype=torch.int64)
    result = random_walk_golden(value, value, value, 2, p=5e-324, q=1e308)
    assert result.tolist() == [[0, 0, 0]]
