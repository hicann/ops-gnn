# ops-gnn API Reference

This document provides detailed API interface documentation for the ops-gnn library, including operator signatures, parameter descriptions, return values, and usage examples.

## 1. Type Definitions

### 1.1 Tensor

```python
import torch
Tensor = torch.Tensor
```

Tensor on NPU device, memory managed by PyTorch.

### 1.2 OptTensor

```python
from typing import Optional
OptTensor = Optional[torch.Tensor]
```

Optional Tensor type for parameters that may have default values. When `None` is passed, the Python layer automatically converts it to an empty Tensor for the underlying C++ implementation.

---

## 2. Core Operator APIs

### 2.1 add_sample — Element-wise Addition

**Function Signature:**

```python
def add_sample(
    src1: Tensor,
    src2: Tensor,
) -> Tensor:
```

**Parameters:**

| Parameter | Type | I/O | Description |
|-----------|------|-----|-------------|
| src1 | Tensor | Input | First input Tensor, must be on NPU device |
| src2 | Tensor | Input | Second input Tensor, must be on NPU device, same shape as `src1` |

**Return Value:**

Returns a new Tensor, the element-wise sum of `src1` and `src2`, located on NPU device, with the same shape and dtype as inputs.

**Description:**

Performs element-wise addition (`src1[i] + src2[i]`) on two Tensors, using AscendC SIMT mode for parallel computation on NPU. Supports uint8 type.

**Implementation Architecture:**

- **Kernel Mode**: SIMT (`__simt_vf__` + `VF_CALL`), single-file implementation
- **Data Movement**: Direct GM read/write, no Tiling
- **Use Case**: Element-wise operations, simple element-wise operators

**Notes:**

- Both input Tensors must have the same shape
- Inputs must be on NPU device (`device='npu'`)
- Currently only supports uint8 (`torch.uint8`) type
- Computation runs directly on NPU, no CPU fallback

**Usage Example:**

```python
import torch
import ops_gnn

# Set NPU device
device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
torch.npu.set_device(device_id)
torch.manual_seed(42)

# Create input Tensors (NPU device)
src1 = torch.randint(0, 128, (1024, 1024), dtype=torch.uint8, device='npu')
src2 = torch.randint(0, 128, (1024, 1024), dtype=torch.uint8, device='npu')

# Call operator
result = ops_gnn.add_sample(src1, src2)

# Verify results
expected = src1 + src2
assert result.device.type == 'npu'
assert result.shape == (1024, 1024)
assert torch.equal(result, expected)
```

---

### 2.2 segment_max_csr — CSR Segmented Max

**Function Signature:**

```python
def segment_max_csr(
    src: Tensor,
    indptr: Tensor,
    optional_out: Optional[Tensor] = None,
) -> Tensor:
```

**Parameters:**

| Parameter | Type | I/O | Description |
|-----------|------|-----|-------------|
| src | Tensor | Input | Input data Tensor, must be on NPU device |
| indptr | Tensor | Input | CSR format index pointer Tensor, dtype must be `torch.int32`, must be on NPU device. Last dimension length - 1 determines the number of segments |
| optional_out | Optional[Tensor] | Input | Optional output Tensor, on NPU device. If provided, the first element of each segment is compared with the corresponding value in `optional_out` via max (i.e., `max(src[start:end], optional_out)`), rather than taking max only from src |

**Return Value:**

Returns a new Tensor, same dtype as `src`. Shape is the same as `src` but the dimension corresponding to the last dim of `indptr` is reduced to `nSegments = indptr_last_dim - 1`.

**Description:**

Performs segmented max reduction on `src` along the dimension specified by CSR format `indptr`. For each segment `[indptr[seg], indptr[seg+1])`, computes the maximum of all elements within that segment. Supports multi-dimensional broadcasting: when `indptr` has fewer dimensions than `src`, indptr is broadcast along corresponding dimensions.

**Implementation Architecture:**

- **Kernel Mode**: Kernel class mode (TPipe + Buffer + Event), 4-file implementation
- **Data Movement**: DataCopy + double-buffer pipeline (MTE2 ↔ VECCALC ↔ MTE3)
- **Tiling**: `SegmentMaxCsrTilingData` struct, includes block parameters (coreDataNum, KloopTime, ALIGN_NUM, etc.)
- **Multi-AIV Parallelism**: Evenly distributes `E_1` (broadcast dimension) across AIV cores, each core processes independently

**Call Chain:**

```text
ops_gnn.segment_max_csr(src, indptr, optional_out)
    → _pybind.segment_max_csr()           # PyTorch binding
        → segment_max_csr()                # Host: dimension parsing, Tiling fill, dtype dispatch
            → LaunchSegmentMaxCsrKernel<T>()  # Kernel Launch: get AIV cores, <<<>>> launch
                → segment_max_csr_kernel<T>   # Device: Init → Process → Compute pipeline
```

**Supported dtypes:**

| dtype | Fill value (empty segment) |
|-------|---------------------------|
| `torch.float32` | `-3.4028235e+38` (-FLT_MAX) |
| `torch.float16` | `-65504` (-HALF_MAX) |
| `torch.int32` | `-2147483648` (INT32_MIN) |
| `torch.int16` | `-32768` (INT16_MIN) |

**Notes:**

- `indptr` dtype must be `torch.int32`
- `indptr` values must be non-decreasing (i.e., `indptr[seg] <= indptr[seg+1]`)
- Empty segment (`indptr[seg] == indptr[seg+1]`): if no `optional_out`, fills with the corresponding dtype minimum; if `optional_out` exists, directly copies the value from `optional_out`
- Supports broadcasting: 1D indptr can broadcast to multiple batches (`indptr.view(1, -1)`)
- Inputs must be on NPU device (`device='npu'`)

**Usage Examples:**

#### Basic Usage — 1D indptr

```python
import torch
import ops_gnn

device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
torch.npu.set_device(device_id)
torch.manual_seed(42)

# Create input data
# src shape: (4, 2), 4 "rows", 2 elements each
src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
# indptr: [0, 2, 4] means segment 0 takes src[0:2], segment 1 takes src[2:4]
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[3, 4], [7, 8]]  — shape: (2, 2)
#           seg0: max(src[0:2])  seg1: max(src[2:4])
```

#### Different dtypes

```python
# float16
src = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16, device='npu')
indptr = torch.tensor([0, 2], dtype=torch.int32, device='npu')
result = ops_gnn.segment_max_csr(src, indptr)

# int32
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device='npu')
indptr = torch.tensor([0, 2], dtype=torch.int32, device='npu')
result = ops_gnn.segment_max_csr(src, indptr)
```

#### Empty Segment

```python
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='npu')
# indptr: [0, 0, 2] — segment 0 is empty, segment 1 takes src[0:2]
indptr = torch.tensor([0, 0, 2], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result[0] = [-inf, -inf]  — empty segment filled with minimum
# result[1] = [3, 4]        — max of src[0:2]
```

#### With optional_out

```python
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='npu')
indptr = torch.tensor([0, 2], dtype=torch.int32, device='npu')
optional_out = torch.tensor([[10, 20]], dtype=torch.float32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr, optional_out)
# result = [[10, 20]]  — max(src[0:2], optional_out) = max([[1,2],[3,4]], [[10,20]])
```

#### 2D indptr (Different Segments per Batch)

```python
src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
# 2D indptr: batch 0 uses [0, 2, 4], batch 1 uses [0, 1, 3]
indptr = torch.tensor([[0, 2, 4], [0, 1, 3]], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[2, 4],    — batch 0: seg0=src[0:2], seg1=src[2:4]
#           [5, 7]]    — batch 1: seg0=src[0:1], seg1=src[1:3]]
```

#### indptr Broadcasting

```python
src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
# 1D indptr, viewed as (1, -1) to broadcast across two batches
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu').view(1, -1)

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[2, 4],    — batch 0: seg0=src[0:2], seg1=src[2:4]
#           [6, 8]]    — batch 1: same segments
```

#### Complex Shapes (3D src + 2D indptr)

```python
src = torch.randn(3, 8, 16, dtype=torch.float32, device='npu')
indptr = torch.tensor([[0, 4, 8]], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result shape: (3, 2, 16) — indptr last dim reduced from 3→2, other dims unchanged
```

---

### 2.3 graclus_cluster - Greedy Graph Clustering

**Function Signature:**

```python
def graclus_cluster(
    row: Tensor,
    col: Tensor,
    weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
```

**Parameters:**

| Parameter | Type | I/O | Description |
|-----------|------|-----|-------------|
| row | Tensor | Input | COO source node indices, dtype must be `torch.long` |
| col | Tensor | Input | COO target node indices, dtype must be `torch.long` |
| weight | Optional[Tensor] | Input | Optional edge weights. float16, bfloat16, and float32 use the NPU path; float64 uses CPU fallback semantics |
| num_nodes | Optional[int] | Input | Number of nodes. If omitted, it is inferred from `row` and `col` |

**Return Value:**

Returns a `torch.long` Tensor with shape `[num_nodes]`. Each element is the cluster ID of the corresponding node.

**Description:**

Implements the greedy graph clustering semantics of `torch_cluster.graclus_cluster`. The Python layer infers `num_nodes`, removes self-loops, shuffles edges when `weight` is absent, sorts edges by row, and converts COO to CSR. The NPU path launches an Ascend C kernel to visit nodes in random order and greedily match each unmarked node with an unmarked neighbor. When `weight` is provided, the neighbor with maximum weight is selected; otherwise the first unmarked neighbor is selected.

**Implementation Architecture:**

- **Kernel Mode**: Ascend C kernel implementation with host-side launch wrapper
- **Data Layout**: COO inputs are preprocessed to CSR row pointer and sorted column tensors
- **Parallelism**: The NPU path processes the CSR adjacency and writes one cluster ID per node

**Notes:**

- `row` and `col` must be 1D `torch.long` tensors
- `row`, `col`, and `weight` must be on the same device
- The algorithm contains randomness; set `torch.manual_seed` before calling the operator when reproducible output is required
- Self-loops are removed before clustering
- float64 weights follow the task-book L2 CPU fallback semantics

**Usage Example:**

```python
import torch
import ops_gnn

torch.npu.set_device(4)
torch.manual_seed(42)

row = torch.tensor([0, 1, 1, 2], dtype=torch.long, device="npu")
col = torch.tensor([1, 0, 2, 1], dtype=torch.long, device="npu")
weight = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32, device="npu")

cluster = ops_gnn.graclus_cluster(row, col, weight, num_nodes=3)
assert cluster.device.type == "npu"
assert cluster.shape == (3,)
```

---

## 3. Testing Guide

### 3.1 Running Tests

```sh
# Run all tests
pytest test/ -v

# Run single operator test
pytest test/test_example.py -v
pytest test/test_segment_max_csr.py -v
pytest test/graclus_cluster/test_graclus_functional.py -v

# Run single test case
pytest test/test_segment_max_csr.py::test_segment_max_csr_basic -v
```

### 3.2 Test Writing Template

```python
import pytest
import torch
import ops_gnn

def test_my_operator():
    """Test basic functionality"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)          # 1. Set NPU device
    torch.manual_seed(42)            # 2. Set random seed

    # 3. Create NPU Tensor
    src = torch.tensor([...], dtype=torch.float32, device='npu')

    # 4. Call operator
    result = ops_gnn.my_op(src)

    # 5. Verify: device, shape, values
    assert result.device.type == 'npu'
    assert result.shape == expected_shape
    assert torch.allclose(result, expected)
```

---

## 4. Back to Main

- **[Back to README](../../README_EN.md)**
- **[Development Guide](development_guide.md)**
