"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import os
import pytest
import torch
import ops_gnn


def segment_max_csr_cpu(src, indptr):
    """CPU reference implementation of segment_max_csr (matching torch_scatter semantics)"""
    indptr_dim = indptr.dim()
    n_segments = indptr.shape[-1] - 1
    
    result_shape = list(src.shape)
    result_shape[indptr_dim - 1] = n_segments
    result = torch.empty(result_shape, dtype=src.dtype)
    
    src_cpu = src.cpu()
    indptr_cpu = indptr.cpu()
    src_shape = src.shape
    
    for batch_idx in range(src_shape[0]):
        for seg in range(n_segments):
            start = int(indptr_cpu[0, seg]) if indptr_dim > 1 else int(indptr_cpu[seg])
            end = int(indptr_cpu[0, seg+1]) if indptr_dim > 1 else int(indptr_cpu[seg+1])
            
            if start >= end:
                fill_value = float('-inf') if src.dtype == torch.float32 else (
                    -65504 if src.dtype == torch.float16 else (
                        -2147483648 if src.dtype == torch.int32 else -32768
                    )
                )
                result[batch_idx, seg] = fill_value
            else:
                if indptr_dim == 1:
                    result[seg] = src_cpu[start:end].max(dim=0).values
                else:
                    result[batch_idx, seg] = src_cpu[batch_idx, start:end].max(dim=0).values
    
    return result


def test_segment_max_csr_basic():
    """测试基本功能 - 1D indptr，沿 dim 0 分段"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([[3, 4], [7, 8]], dtype=torch.float32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.equal(result, expected)


def test_segment_max_csr_1d_simple():
    """测试简单的1D场景"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    
    src = torch.tensor([1, 3, 2, 5, 4], dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 2, 5], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([3., 5.], dtype=torch.float32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2,)
    assert torch.equal(result, expected)


def test_segment_max_csr_float16():
    """测试float16数据类型"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float16, device='npu')
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([[3.0, 4.0], [7.0, 8.0]], dtype=torch.float16, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


def test_segment_max_csr_int32():
    """测试int32数据类型"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32, device='npu')
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([[3, 4], [7, 8]], dtype=torch.int32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.equal(result, expected)


def test_segment_max_csr_empty_segment():
    """测试空segment"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 0, 2, 4], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    assert result.device.type == 'npu'
    assert result.shape == (3, 2)
    
    expected = torch.tensor([[-float('inf'), -float('inf')], [3, 4], [7, 8]], dtype=torch.float32, device='npu')
    assert torch.allclose(result[1:], expected[1:])
    assert result[0][0] < -1e30 and result[0][1] < -1e30


def test_segment_max_csr_with_optional_out():
    """测试带有optional_out参数"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
    optional_out = torch.tensor([[10, 20], [30, 40]], dtype=torch.float32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr, optional_out)
    
    expected = torch.tensor([[10, 20], [30, 40]], dtype=torch.float32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.equal(result, expected)


def test_segment_max_csr_2d_indptr():
    """测试2D indptr（沿dim 1分段，每个batch有不同的分段）"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
    indptr = torch.tensor([[0, 2, 4], [0, 1, 3]], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([[2, 4], [5, 7]], dtype=torch.float32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.equal(result, expected)


def test_segment_max_csr_broadcast():
    """测试indptr广播（1D indptr广播到多个batch）"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu').view(1, -1)
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([[2, 4], [6, 8]], dtype=torch.float32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.equal(result, expected)


def test_segment_max_csr_large():
    """测试较大规模数据"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.randn(128, 64, dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = segment_max_csr_cpu(src, indptr).to('npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (4, 64)
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


def test_segment_max_csr_complex_shape():
    """测试复杂shape（3D src, 2D indptr）"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.randn(3, 8, 16, dtype=torch.float32, device='npu')
    indptr = torch.tensor([[0, 4, 8]], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = segment_max_csr_cpu(src, indptr).to('npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (3, 2, 16)
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
