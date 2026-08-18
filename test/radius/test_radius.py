# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional coverage for :func:`ops_gnn.radius` (official 25 cases).

Assertions follow the official self-test (torch.equal exact ordering); the
NPU device is selected via ``NPU_DEVICE_ID``.
"""

from __future__ import annotations

import os

import pytest
import torch


pytest.importorskip("torch_npu")
ops_gnn = pytest.importorskip("ops_gnn")
if not hasattr(ops_gnn, "radius"):
    pytest.skip("ops_gnn.radius is not built", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def _select_npu_device():
    """Run the module on the card selected by ``NPU_DEVICE_ID``."""
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))


SUPPORTED_DTYPES = [torch.float16, torch.float32]

TEST_SHAPES = [(128, 3), (1024, 64), (256, 3), (2048, 32), (512, 128)]

GENERAL_SHAPES = [(1, 3), (8, 3), (4, 64), (10000, 3)]


def _radius_cpu(x: torch.Tensor, y: torch.Tensor, r: float,
                max_num_neighbors: int = 32,
                ignore_same_index: bool = False) -> torch.Tensor:
    xc = x.cpu().float()
    yc = y.cpu().float()
    rows = []
    cols = []
    for i in range(yc.size(0)):
        diff = xc - yc[i].unsqueeze(0)
        dist = diff.pow(2).sum(1)
        neighbors = (dist <= r * r).nonzero(as_tuple=False).flatten()
        if ignore_same_index:
            neighbors = neighbors[neighbors != i]
        count = 0
        for n in neighbors.tolist():
            if count >= max_num_neighbors:
                break
            rows.append(i)
            cols.append(int(n))
            count += 1
    if not rows:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def _radius_op(x: torch.Tensor, y: torch.Tensor, r: float,
               max_num_neighbors: int = 32,
               ignore_same_index: bool = False) -> torch.Tensor:
    return ops_gnn.radius(
        x.npu(), y.npu(), r=r,
        max_num_neighbors=max_num_neighbors,
        ignore_same_index=ignore_same_index,
    )


class TestRadius:
    @staticmethod
    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_shapes(shape, dtype):
        torch.manual_seed(42)
        x = torch.randn(shape).to(dtype)
        expected = _radius_cpu(x, x, r=1.0)
        result = _radius_op(x, x, r=1.0)
        assert result.device.type == "npu"
        assert result.shape == expected.shape
        assert torch.equal(result.cpu(), expected.cpu())

    @staticmethod
    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_general_shapes(shape, dtype):
        torch.manual_seed(42)
        x = torch.randn(shape).to(dtype)
        expected = _radius_cpu(x, x, r=1.0)
        result = _radius_op(x, x, r=1.0)
        assert result.device.type == "npu"
        assert result.shape == expected.shape
        assert torch.equal(result.cpu(), expected.cpu())

    @staticmethod
    def test_small_radius_empty():
        torch.manual_seed(42)
        x = torch.randn(128, 3, device="cpu")
        expected = _radius_cpu(x, x, r=1e-9)
        result = _radius_op(x, x, r=1e-9)
        assert result.device.type == "npu"
        assert result.size(1) == expected.size(1)
        assert torch.equal(result.cpu(), expected.cpu())

    @staticmethod
    def test_large_radius_all():
        torch.manual_seed(42)
        x = torch.randn(10, 3, device="cpu")
        expected = _radius_cpu(x, x, r=1e9)
        result = _radius_op(x, x, r=1e9)
        assert result.device.type == "npu"
        assert result.size(1) == expected.size(1)
        assert torch.equal(result.cpu(), expected.cpu())

    @staticmethod
    def test_self_connections():
        x = torch.randn(10, 3, device="cpu")
        expected = _radius_cpu(x, x, r=10.0, ignore_same_index=True)
        result = _radius_op(x, x, r=10.0, ignore_same_index=True)
        assert result.device.type == "npu"
        assert torch.equal(result.cpu(), expected.cpu())

    @staticmethod
    def test_max_neighbors():
        torch.manual_seed(42)
        x = torch.randn(100, 3, device="cpu")
        expected = _radius_cpu(x, x, r=100.0, max_num_neighbors=5)
        result = _radius_op(x, x, r=100.0, max_num_neighbors=5)
        assert result.device.type == "npu"
        assert result.size(1) == expected.size(1)
        assert torch.equal(result.cpu(), expected.cpu())

    @staticmethod
    def test_deterministic():
        torch.manual_seed(42)
        x = torch.randn(128, 3, device="cpu")
        r1 = _radius_cpu(x, x, r=1.0)
        r2 = _radius_cpu(x, x, r=1.0)
        assert torch.equal(r1, r2)
        r3 = _radius_op(x, x, r=1.0)
        r4 = _radius_op(x, x, r=1.0)
        assert torch.equal(r3.cpu(), r4.cpu())

    @staticmethod
    def test_empty_input():
        x = torch.empty(0, 3, device="cpu")
        expected = _radius_cpu(x, x, r=1.0)
        result = _radius_op(x, x, r=1.0)
        assert result.device.type == "npu"
        assert result.numel() == expected.numel()
        assert torch.equal(result.cpu(), expected.cpu())

    @staticmethod
    def test_known_case():
        torch.manual_seed(42)
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        expected = _radius_cpu(x, y, r=1.1)
        assert expected.size(1) == 6
        assert expected.size(0) == 2
        result = _radius_op(x, y, r=1.1)
        assert result.device.type == "npu"
        assert result.shape == expected.shape
        assert torch.equal(result.cpu(), expected.cpu())
