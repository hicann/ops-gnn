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

Optional Tensor type for parameters that may have default values. `None` is represented explicitly at the pybind boundary; a valid empty Tensor remains a real `out` argument.

---

## 2. Core Operator APIs

### 2.1 segment_max_csr — CSR Segmented Max

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

### 2.2 radius / radius_graph — Radius Neighbor Search

NPU implementation interface-compatible with `torch_cluster.radius` /
`radius_graph` (>= 1.6.0), Ascend 950PR. For each query point in `y`, finds all
neighbors in `x` with Euclidean distance `dist² <= r²` (within the same batch);
when the count exceeds `max_num_neighbors`, keeps the first K in scan order
(deterministic). Outputs `edge_index [2, E]` (int64).

```python
def radius(
    x, y, r,
    batch_x=None, batch_y=None,
    max_num_neighbors=32, num_workers=1,
    batch_size=None, ignore_same_index=False,
) -> torch.Tensor          # [2, E] int64

def radius_graph(
    x, r,
    batch=None, loop=False,
    max_num_neighbors=32,
    flow='source_to_target', num_workers=1, batch_size=None,
) -> torch.Tensor          # [2, E] int64
```

| Parameter | Type | In/Out | Description |
|-----------|------|--------|-------------|
| `x` | Tensor [N,F] | in | Neighbor candidate set (float16/bf16/float32; float64 via CPU fallback) |
| `y` | Tensor [M,F] | in | Query point set |
| `r` | float | in | Search radius (>0) |
| `batch_x` / `batch_y` | Tensor | in | Batch membership (must be sorted) |
| `max_num_neighbors` | int | in | Max neighbors per query (default 32) |
| `ignore_same_index` | bool | in | Skip `i == q` self-loops |
| `loop` / `flow` | bool/str | in | radius_graph self-loop and direction semantics |

```python
x = torch.randn(1000, 3, device='npu')
edge = ops_gnn.radius(x, x, 0.8)          # neighbors within radius 0.8
edge_g = ops_gnn.radius_graph(x, 0.8)     # build K-NN graph (loop=False by default)
```

**Implementation architecture:**

- **Kernel mode**: Ascend C SIMT (`__simt_vf__` + `VF_CALL`), spatial-grid pruning + sorted-array top-K + early-break
- **Grid build**: executed on device (min/max, cell, sort, index_select, offsets), avoiding CPU sort bottleneck
- **Output compaction**: device-side `repeat_interleave` + `masked_select` into `[2, E]`
- **Use case**: 3D point-cloud / GNN neighborhood construction (PointNet++, DGCNN, etc.)

**Notes:**

- `x` / `y` must be tensors on the same NPU device, with matching feature dim `F`; non-contiguous inputs are made contiguous internally
- `batch_x` / `batch_y` must be sorted; search is confined within each batch
- Exceeding `max_num_neighbors` keeps first K in scan order (deterministic, consistent with CPU reference)
- `radius_graph` `loop` / `flow` semantics match `torch_cluster`
- L1: float16 / bfloat16 / float32 (NPU); float64 via CPU fallback (bit-wise, not performance-tested)
- Empty input returns `[2, 0]` LongTensor without entering the kernel

### 2.3 graclus_cluster — Greedy Graph Clustering

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

**Notes:**

- `row` and `col` must be 1D `torch.long` tensors
- `row`, `col`, and `weight` must be on the same device
- The algorithm contains randomness; set `torch.manual_seed` before calling the operator when reproducible output is required
- Self-loops are removed before clustering

### 2.4 gather_coo — COO Row Expansion

**Function signature:**

```python
def gather_coo(
    src: Tensor,
    index: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
```

Let `dim = index.dim() - 1`. The prefix shape of `index` must match the first `dim` dimensions of `src`. The output has the same shape as `src`, except `src.size(dim)` is replaced by `index.size(-1)`, and:

```text
out[..., e, ...] = src[..., index[..., e], ...]
```

`index` must be an NPU `torch.int64` tensor and satisfy the task's sorted, in-range preconditions. The operator is forward-only and performs a raw bit copy; it supports ranks 1–8, non-contiguous inputs/outputs, empty tensors, and FP16/BF16/FP32, INT8/16/32, UINT8, FP64, and INT64.

When `out` is supplied, it must have the exact inferred shape, matching dtype, and matching device. The returned tensor shares storage with `out`. `None` is distinct from an explicitly supplied empty `out` tensor.

**Example:**

```python
import torch
import ops_gnn

device = torch.device("npu")  # use the process's current NPU; no fixed ordinal
src = torch.arange(20, dtype=torch.float32, device=device).reshape(5, 4)
index = torch.tensor([0, 1, 1, 4], dtype=torch.int64, device=device)
out = ops_gnn.gather_coo(src, index)
# out.shape == (4, 4); repeated index=1 copies the same source row

provided = torch.empty_like(out)
returned = ops_gnn.gather_coo(src, index, out=provided)
assert returned.data_ptr() == provided.data_ptr()
```

**Implementation and performance notes:**

- The Host flattens inputs to `B × N × K`, `B × E`, and `B × E × K`, using 64-bit lengths and offsets. The kernel reuses PyTorch's current NPU stream and never creates or synchronizes a private ACL stream.
- Non-contiguous `src/index` tensors are made contiguous on the current stream. Explicit `out` uses a contiguous temporary result before copy-back, covering non-contiguous and aliasing cases.
- Consecutive equal sorted indices reuse a source row within one UB tile. Dispatch does not depend on test-case IDs or fixed-shape whitelists.

---

### 2.5 ind2ptr — Sorted Row Indices to CSR Row Pointer

**Function signature:**

```python
def ind2ptr(
    ind: Tensor,
    num_rows: int,
) -> Tensor:
```

**Parameters:**

| Parameter | Type | I/O | Description |
|-----------|------|-----|-------------|
| ind | Tensor | Input | 1-D `torch.long` non-decreasing row indices on NPU |
| num_rows | int | Input | Number of rows (historically `M` in torch_sparse); output length is `num_rows + 1` |

**Returns:**

A 1-D `torch.long` CSR row-pointer tensor of shape `[num_rows + 1]` on the same device as `ind`.

**Description:**

Converts sorted row indices to a CSR row pointer. Drop-in replacement for `torch.ops.torch_sparse.ind2ptr(ind, M)`. Runs asynchronously on the current NPU stream; synchronize with `torch.npu.synchronize()` before reading results on host if needed.

**Notes:**

- `ind` must be non-decreasing; an empty input yields an all-zero pointer of length `num_rows + 1`
- dtype must be `torch.long` (int64)
- Non-contiguous inputs are made contiguous before the call

**Example:**

```python
import torch
import ops_gnn

row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
rowptr = ops_gnn.ind2ptr(row, 8)
# tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], device='npu:0')
```

---

### 2.6 ptr2ind — CSR Row Pointer to Row Indices

**Function signature:**

```python
def ptr2ind(
    ptr: Tensor,
    num_edges: int,
) -> Tensor:
```

**Parameters:**

| Parameter | Type | I/O | Description |
|-----------|------|-----|-------------|
| ptr | Tensor | Input | 1-D `torch.long` CSR row pointer of length `num_rows + 1` on NPU |
| num_edges | int | Input | Number of edges / non-zeros (historically `E` in torch_sparse); output length is `num_edges` |

**Returns:**

A 1-D `torch.long` row-index tensor of shape `[num_edges]` on the same device as `ptr`.

**Description:**

Converts a CSR row pointer to row indices. Drop-in replacement for `torch.ops.torch_sparse.ptr2ind(ptr, E)`. Runs asynchronously on the current NPU stream; synchronize with `torch.npu.synchronize()` before reading results on host if needed.

**Notes:**

- dtype must be `torch.long` (int64)
- `num_edges == 0` returns an empty index tensor
- Non-contiguous inputs are made contiguous before the call

**Example:**

```python
import torch
import ops_gnn

rowptr = torch.tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], dtype=torch.long, device='npu')
row = ops_gnn.ptr2ind(rowptr, 6)
# tensor([2, 2, 4, 5, 5, 6], device='npu:0')
```

---

### 2.7 random_walk — NPU Random Walk

**Function Signature:**

```python
def random_walk(
    row: Tensor,
    col: Tensor,
    start: Tensor,
    walk_length: int,
    p: float = 1,
    q: float = 1,
    coalesced: bool = True,
    num_nodes: Optional[int] = None,
    return_edge_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
```

**Parameters:**

| Parameter | Type | I/O | Description |
|-----------|------|-----|-------------|
| row | Tensor | Input | COO source nodes, a 1D `torch.int64` NPU Tensor |
| col | Tensor | Input | COO destination nodes, a 1D `torch.int64` NPU Tensor with the same length as `row` |
| start | Tensor | Input | Walk starting nodes, a 1D `torch.int64` NPU Tensor |
| walk_length | int | Input | Number of walk steps; must be non-negative |
| p | float | Input | node2vec return parameter; must be finite and greater than zero |
| q | float | Input | node2vec BFS/DFS parameter; must be finite and greater than zero |
| coalesced | bool | Input | Sorts edges by `(row, col)` when True; when False, edges must already be grouped by source node |
| num_nodes | Optional[int] | Input | Number of nodes; inferred from the maxima of `row`, `col`, and `start` when None |
| return_edge_indices | bool | Input | Whether to return the edge position selected at each step |

**Return Value:**

Returns `node_seq`, a `torch.int64` Tensor with shape `[S, walk_length + 1]`.
When `return_edge_indices=True`, returns `(node_seq, edge_seq)`, where `edge_seq`
is a `torch.int64` Tensor with shape `[S, walk_length]`. An isolated node remains
unchanged and its corresponding `edge_seq` value is `-1`.

**Description:**

Runs uniform or node2vec-biased random walks from multiple starting nodes on a
COO graph. The Python layer sorts the COO edges and builds CSR data on NPU. An
Ascend 950 SIMT kernel performs uniform sampling when `p=1, q=1`; otherwise it
uses rejection sampling with node2vec transition probabilities.

**Implementation Architecture:**

- **Python preprocessing**: argument validation, COO sorting, degree counting, and CSR `rowptr` construction
- **Host layer**: obtains the current PyTorch NPU stream and Philox seed/offset, then calculates integer node2vec thresholds
- **Kernel mode**: Ascend C SIMT, with one thread processing one walk
- **Output path**: skips edge-sequence writes when `return_edge_indices=False`

**Notes:**

- `row`, `col`, and `start` must be on the same NPU device, with `0 <= index < num_nodes`
- `coalesced=False` does not reorder edges; input edges must already be grouped by source node
- With sorting enabled, edge indices refer to positions in the sorted COO/CSR representation
- The total edge count and the degree of each node must not exceed `2^32-1`
- Only forward computation is supported; use `torch.manual_seed(seed)` to seed the default NPU Generator

**Usage Example:**

```python
import torch
import ops_gnn

device = "npu:0"
row = torch.tensor([0, 1, 1, 2], dtype=torch.int64, device=device)
col = torch.tensor([1, 0, 2, 1], dtype=torch.int64, device=device)
start = torch.tensor([0, 2], dtype=torch.int64, device=device)

torch.manual_seed(202608)
nodes, edges = ops_gnn.random_walk(
    row, col, start, walk_length=8, p=0.5, q=2.0,
    return_edge_indices=True,
)
assert nodes.shape == (2, 9)
assert edges.shape == (2, 8)
```

---

### 2.8 gather_csr - CSR Segment Expansion

**Signature:**

```python
def gather_csr(
    src: Tensor,
    indptr: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
```

`gather_csr` expands segment features and is the inverse operation of
`segment_csr`. Let `dim = indptr.dim() - 1`. For every segment `i`:

```text
out[..., indptr[i]:indptr[i + 1], ...] = src[..., i, ...]
```

The interface contains only `src`, `indptr`, and optional `out`; it has no
`reduce`, `dim`, or `dim_size` argument.

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | Tensor | NPU tensor with a supported L1 or L2 dtype |
| `indptr` | Tensor | Non-decreasing int64 CSR pointers on the same NPU |
| `out` | Optional[Tensor] | Optional output matching the inferred dtype, device, rank, and shape |

The operator supports prefix broadcasting, empty segments and tensors, and
non-contiguous `src`, `indptr`, and `out`. When provided, the same `out` object
is updated and returned. Only forward execution is supported.

**Supported inputs:**

| Item | Support |
|------|---------|
| Hardware | Ascend 950PR (`dav-3510`) |
| `src/out` dtype | float16, bfloat16, float32, int8, int16, int32, uint8, float64, int64 |
| `indptr` dtype | int64 |
| Rank | `1 <= indptr.dim() <= src.dim()` |
| Layout | Contiguous and non-contiguous tensors |
| Special cases | Batch broadcasting, empty segments/tensors, optional `out` |
| Direction | Forward only |

Every dtype is copied as raw bytes without arithmetic or type conversion.
Float64 and int64 are L2 functional paths and are excluded from performance
acceptance.

**Constraints:**

- `src`, `indptr`, and optional `out` must be on the same NPU.
- `indptr.dim() <= src.dim()`, and its prefix dimensions must broadcast to `src`.
- If `indptr.size(-1) > 0`, `src.size(dim) == indptr.size(-1) - 1`. If the last dimension of
  `indptr` is empty, `src.size(dim)` must be `0`.
- Every `indptr` row must be non-decreasing, stay in `[0, endpoint]`, and have
  the same endpoint after broadcasting.
- The output length at `dim` is the endpoint, or zero for an empty `indptr`.
- Optional `out` must match the inferred dtype, device, rank, and shape.

Empty `src` inputs still use the endpoint to infer the output Shape and validate
broadcasting, monotonicity, and range. An empty `indptr` uses an endpoint of zero.

**Implementation:**

The Host layer validates arguments, materializes broadcast pointers, handles
non-contiguous tensors, allocates output, and prepares tiling. The Ascend C
Kernel selects `SegmentMajor` or `OutputMajor`. The former assigns
`(batch, segment)` jobs to AIV cores; the latter partitions output rows when
there are too few segments or one segment is heavily skewed. Aligned features
use a 64 KB UB repeat buffer for batched writes, while other features use 16 KB
tiles. The Kernel runs on the current PyTorch NPU stream.

**Example:**

```python
import os
import torch
from ops_gnn import gather_csr

device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
torch.npu.set_device(device_id)
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device="npu")
indptr = torch.tensor([0, 2, 5], dtype=torch.int64, device="npu")
out = gather_csr(src, indptr)
# [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
```

**Build and test:**

```bash
# Install torch_scatter >= 2.1.0 for the CPU reference.
cmake -S . -B build/cmake_release \
  -DNPU_ARCH=dav-3510 -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake_release -j4
export PYTHONPATH=$PWD/python
export NPU_DEVICE_ID=<device_id>
pytest -q test/gather_csr/arch35/test_gather_csr.py
python test/gather_csr/arch35/golden.py
python test/gather_csr/arch35/benchmark_gather_csr.py --ascendoptest \
  --ascendoptest-root /path/to/AscendOpTest
python test/gather_csr/arch35/benchmark_gather_csr.py --warmup 20 --iterations 101
```

---

### 2.9 scatter — Scatter Reductions

The Scatter family follows the forward semantics of `torch_scatter` 2.1.2:

```python
ops_gnn.scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum")
ops_gnn.scatter_sum(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_add(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_mul(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_mean(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_min(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_max(src, index, dim=-1, out=None, dim_size=None)
```

`scatter_min` and `scatter_max` return `(out, arg_out)`; the other functions
return `out`. `scatter_add` and `scatter_sum` have identical reduction
semantics.

#### Arguments

- `src`: input Tensor with rank 1 through 8.
- `index`: INT64 Tensor broadcastable to `src` under `torch_scatter` rules.
- `dim`: reduction dimension; negative dimensions are supported. Default: `-1`.
- `out`: optional output Tensor, updated in place while preserving identity.
- `dim_size`: optional output size on `dim`; inferred from `index` when omitted.
- `reduce`: `sum`, `add`, `mul`, `mean`, `min`, or `max`.

#### Dtypes and execution paths

| Level | `src/out` dtype | Execution path | Reductions |
|---|---|---|---|
| L1 | float16, bfloat16, float32, int8, int16, int32, uint8 | Ascend C NPU kernel | all six |
| L2 | float64, int64 | synchronous CPU fallback copied back to the original device | all six |

Integer `mean` uses floor division. For equal extrema, `min/max` selects the
later writer. Unwritten output positions contain zero and their `arg_out`
value is `src.size(dim)`. The APIs support explicit `dim_size`, empty and
non-contiguous Tensors, unordered or duplicate indices, and high-contention
indices. Only forward computation is provided.

#### Example

```python
import os
import torch
import ops_gnn

torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))
src = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
index = torch.tensor([0, 1, 0, 1], dtype=torch.long, device="npu")

out = ops_gnn.scatter_sum(src, index)
# tensor([4., 6.], device='npu:0')

values, arg = ops_gnn.scatter_max(src, index)
# values: tensor([3., 4.], device='npu:0')
# arg:    tensor([2, 3], device='npu:0')
```

### 2.10 spmm_max_csr — CSR Sparse Matrix-Vector Max Aggregation

```python
ops_gnn.spmm_max_csr(indptr, indices, x, out=None) -> Tensor
```

Forward-only `copy_lhs + max` over CSR **rows representing destination nodes**.
`indices[e]` is the source node of edge `e`. A conventional source-row CSR
aggregates in the opposite direction and must be transposed before use.

- `x`: `[K, N]`, float16, NPU; `N > 0`.
- `indptr`: `[M+1]`; `indices`: `[nnz]`, matching int32 or int64, same NPU as `x`.
- CSR starts at zero, ends at nnz, is non-decreasing, and source indices lie in `[0,K)`.
- Returns `[M,N]` float16. Empty rows are zero; infinities are preserved and any
  contributing NaN propagates to that output feature.
- Noncontiguous inputs are made contiguous. `out`, if supplied, must be contiguous,
  match shape/dtype/device, and share no storage with any input.
- Autograd is unsupported; `x` and `out` requiring gradients are rejected.
- No CPU fallback, other reductions, batched features or other feature dtypes.

```python
import torch
import torch_npu
from ops_gnn import spmm_max_csr

# Destination 0 receives sources 0 and 2; destination 1 is empty.
ptr = torch.tensor([0, 2, 2], dtype=torch.int64, device='npu')
idx = torch.tensor([0, 2], dtype=torch.int64, device='npu')
x = torch.tensor([[-2, 3], [7, 8], [-1, 2]], dtype=torch.float16, device='npu')
y = spmm_max_csr(ptr, idx, x)  # [[-1, 3], [0, 0]]
```

The kernel runs on the current PyTorch NPU stream. Scalar CSR validation and NaN
presence checks synchronize; the implementation does not download CSR arrays.
NaN-containing input uses an additional device scalar scan to guarantee propagation,
which may be slower. End-to-end benchmarks include these costs and int64 conversion.

With device UB capacity `U` bytes, the maximum feature dimension is
`16 * floor((U - 2048) / 128)`: two accumulation and two feature buffers must fit.
Dimensions and address byte spans must fit uint32; a larger input raises an error.
The actual limit is included in the exception. Features are not tiled.

Source routing uses `op_kernel/arch22`; compilation still uses `NPU_ARCH` for the
actual device. Hardware support requires successful validation on that target.

## 3. Testing Guide

### 3.1 Running Tests

```sh
# Run all tests for the local chip model (950→arch35, A2/A3→arch22; `pytest test/` only collects the matching directory automatically)
pytest test/ -v

# Run single operator test (arch35 on 950, arch22 on A2/A3)
pytest test/gather_csr/arch35/test_gather_csr.py -v
pytest test/segment_max_csr/arch35/test_segment_max_csr.py -v
pytest test/graclus_cluster/arch35/test_graclus_cluster.py -v
python -m pytest test/gather_coo/arch35/test_gather_coo.py -v
pytest test/random_walk/arch35/test_random_walk.py -v
pytest test/radius/arch35/test_radius.py -v
pytest test/sparse/arch35/test_sparse.py -v

# Run the random_walk performance benchmark
NPU_DEVICE_ID=<device_id> python test/random_walk/arch35/benchmark_random_walk.py

# Run the radius official-baseline performance benchmark
python test/radius/arch35/benchmark_radius.py

# Run single test case
pytest test/segment_max_csr/arch35/test_segment_max_csr.py::test_segment_max_csr_basic -v
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
    src = torch.tensor([...], dtype=torch.float32, device=device)

    # 4. Call operator
    result = ops_gnn.my_op(src)

    # 5. Verify: device, shape, values
    assert result.device.type == 'npu'
    assert result.shape == expected_shape
    assert torch.allclose(result, expected)
```

---

## 4. Back to Main

- **[Back to README](../../README_en.md)**
- **[Development Guide](development_guide.md)**
