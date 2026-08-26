# ops-gnn

English | [简体中文](README.md)

ops-gnn is an operator library for Graph Neural Networks (GNNs) in the Ascend ecosystem, designed to accelerate GNN computation on NPUs.

## Project Structure

```text
ops-gnn/
├── csrc/                       # C++/AscendC source code directory
│   ├── pybind.cpp              # PyTorch binding code
│   └── npu/                    # NPU-related code (organized by operator)
├── docs/                       # Docs (API reference: docs/*/api_reference.md)
├── python/                     # Python source directory
│   └── ops_gnn/                # Python package directory
├── test/                       # Test directory
├── scripts/                    # Build scripts directory
│   └── build.sh                # Unified build script
├── cmake/                      # CMake configuration
│   └── OpsGNNConfig.cmake.in
├── CMakeLists.txt              # CMake build configuration
├── setup.py                    # Python installation script (using PyTorch cpp_extension)
├── setup.cfg                   # setuptools configuration
├── pyproject.toml              # Modern Python project configuration
├── MANIFEST.in                 # Packaging manifest
└── LICENSE                     # CANN License
```

Each operator directory under `csrc/npu` contains `op_host` and `op_kernel/<arch>`. Each operator directory under `test` contains `golden.py`, a functional test file, and a performance test file. Here, `<arch>` is the directory name for the target architecture.

## Requirements

- Python 3.9+
- CMake 3.18+
- PyTorch 2.7+
- A torch_npu build matched to PyTorch and CANN (Gather COO was verified with PyTorch 2.7.1 / torch_npu 2.7.1.post4)
- CANN Toolkit (AscendC compiler)
- C++17 or later compiler
- CANN 9.1.0 or later + HDK (driver/firmware) 25.7.rc1 or later
- Supported platform: Ascend 950 series; other platforms are not supported yet

### CANN Environment Setup

```bash
# Activate CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

If CANN is installed in a custom path, run:

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

## Installation

### Method 1: Install via pip

```bash
# Install in development mode
python3 -m pip install --no-build-isolation --no-deps -e .
```

### Method 2: Using the build script

```bash
cd scripts

# Build Python package
./build.sh python
```

### Method 3: Using CMake (Linux)

```bash
# Use the CANN setup script that matches PyTorch/torch_npu on this machine.
export CANN_SETENV=/path/to/cann/bin/setenv.bash
source "$CANN_SETENV"

mkdir -p build_cmake
cd build_cmake
cmake ..
cmake --build .
```

## Functional Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Quick start: run a single or a few test groups first
pytest test/test_import.py -v            # Import tests
pytest test/test_example.py -v           # A single NPU operator test (add_sample)

# Run all functional tests
pytest test/ -v
```

Run the functional tests for one operator:

```bash
NPU_DEVICE_ID=<device_id> python3 -m pytest test/<op>/test_<op>.py -v
```

Here, `<op>` is the operator name, and `<device_id>` is the device ID.

## Performance Tests

Run the performance tests for one operator:

```bash
NPU_DEVICE_ID=<device_id> python3 test/<op>/benchmark_<op>.py
```

Here, `<op>` is the operator name, and `<device_id>` is the device ID.

## Usage Examples

```python
import torch
import ops_gnn

# NPU tensor addition
src1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint8, device='npu')
src2 = torch.tensor([5, 4, 3, 2, 1], dtype=torch.uint8, device='npu')
result = ops_gnn.add_sample(src1, src2)
print(result)  # output: tensor([6, 6, 6, 6, 6], device='npu:0', dtype=torch.uint8)

# CSR segment max operation
src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
result = ops_gnn.segment_max_csr(src, indptr)
print(result)  # output: tensor([[3, 4], [7, 8]], device='npu:0')

# torch_sparse compatibility: row indices <-> CSR rowptr
row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
rowptr = ops_gnn.ind2ptr(row, 8)          # replaces torch.ops.torch_sparse.ind2ptr(row, 8)
row2 = ops_gnn.ptr2ind(rowptr, 6)         # replaces torch.ops.torch_sparse.ptr2ind(rowptr, 6)

# COO row expansion; index must be sorted int64
src = torch.arange(20, dtype=torch.float32, device='npu').reshape(5, 4)
index = torch.tensor([0, 1, 1, 4], dtype=torch.int64, device='npu')
result = ops_gnn.gather_coo(src, index)
print(result.shape)  # output: torch.Size([4, 4])

# Expand segment features using CSR pointers
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16, device='npu')
indptr = torch.tensor([0, 2, 5], dtype=torch.int64, device='npu')
result = ops_gnn.gather_csr(src, indptr)
print(result.shape)  # torch.Size([5, 2])

# torch_scatter-compatible indexed reduction
src = torch.randn(10, 6, 64, dtype=torch.float32, device='npu')
index = torch.randint(0, 4, (10,), dtype=torch.long, device='npu')
result = ops_gnn.scatter(src, index, dim=0, dim_size=4, reduce='sum')
print(result.shape)  # torch.Size([4, 6, 64])
```

## Operator List

| Operator | Description | Device Support |
|----------|-------------|----------------|
| `add_sample` | Element-wise addition of two tensors | NPU |
| `gather_coo` | Expand source rows by sorted COO indices | NPU |
| [`gather_csr`](docs/en/api_reference.md) | Expands segment features using CSR pointers | NPU |
| `random_walk` | Uniform or node2vec-biased random walks on COO graphs | NPU |
| `segment_max_csr` | Segmented max reduction on CSR format | NPU |
| `radius` / `radius_graph` | Radius neighbor search (torch_cluster compatible, Ascend 950PR) | NPU |
| `graclus_cluster` | Greedy graph clustering | NPU / CPU fallback for float64 |
| `ind2ptr` | Sorted row indices to CSR row pointer (torch_sparse-aligned) | NPU |
| `ptr2ind` | CSR row pointer to row indices (torch_sparse-aligned) | NPU |
| `scatter` / `scatter_*` | torch_scatter-compatible indexed reductions | NPU / CPU fallback for float64 and int64 |

See the [API reference](docs/en/api_reference.md#210-scatter--scatter-reductions) for the complete Scatter API, dtype tiers, and examples.

## Development Guide

### Adding a New NPU Operator

1. Create AscendC kernel files (`.cpp`) and headers (`.h`) in `csrc/npu/<op>/op_kernel/<arch>/`
2. Create operator interface implementation (`.cpp`) and header (`.h`) in `csrc/npu/<op>/op_host/`
3. Add PyTorch bindings in `csrc/pybind.cpp`
4. Create Python interface declaration files in `python/ops_gnn/`
5. Update `python/ops_gnn/__init__.py` to export the new function
6. Add `golden.py`, `test_<op>.py`, and `benchmark_<op>.py` under `test/<op>/`

Here, `<op>` is the operator name.

## License

CANN Open Software License Agreement Version 2.0
