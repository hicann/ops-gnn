# ops-gnn

English | [简体中文](README.md)

ops-gnn is an operator library for Graph Neural Networks (GNNs) in the Ascend ecosystem, designed to accelerate GNN computation on NPUs.

## Project Structure

```text
ops-gnn/
├── csrc/                       # C++/AscendC source code
│   ├── pybind.cpp              # PyTorch binding code
│   └── npu/                    # NPU-related code
│       ├── host/               # Host-side code (organized by operator)
│       ├── kernel/             # AscendC kernel implementation (organized by operator)
│       └── sparse/             # sparse operators (one subdirectory per op)
│           ├── ind2ptr/
│           │   ├── op_host/   # Host dispatch
│           │   └── op_kernel/
│           │       └── arch35/  # Ascend950 Kernel
│           └── ptr2ind/
│               ├── op_host/
│               └── op_kernel/
│                   └── arch35/
├── docs/                       # Docs (API reference: docs/*/api_reference.md)
├── python/                     # Python source code
│   └── ops_gnn/                # Python package
│       ├── __init__.py         # Package initialization
│       ├── add_sample.py       # Python interface declaration
│       ├── gather_coo.py       # Python interface declaration
│       ├── gather_csr.py       # Gather CSR Python interface
│       ├── ind2ptr.py          # ind2ptr Python interface
│       ├── ptr2ind.py          # ptr2ind Python interface
│       ├── random_walk.py      # Random-walk Python interface
│       ├── segment_max_csr.py  # Python interface declaration
│       └── typing.py           # Type definitions
├── test/                       # Test directory
├── scripts/                    # Build scripts
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

## Requirements

- Python 3.9+
- CMake 3.18+
- PyTorch 2.7+
- A torch_npu build matched to PyTorch and CANN (Gather COO was verified with PyTorch 2.7.1 / torch_npu 2.7.1.post4)
- CANN Toolkit (AscendC compiler)
- C++17 or later compiler

### CANN Environment Setup

```bash
export CANN_SETENV=/path/to/cann/bin/setenv.bash
source "$CANN_SETENV"
```

## Installation

### Method 1: Install via pip

```bash
# Activate CANN environment
export CANN_SETENV=/path/to/cann/bin/setenv.bash
source "$CANN_SETENV"

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

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest test/ -v
```

Gather COO tests are under `test/gather_coo/`:

```bash
OPSGNN_REQUIRE_TORCH_SCATTER=1 \
python3 -m pytest test/gather_coo/test_gather_coo_functional.py -v
python3 test/gather_coo/test_gather_coo_performance.py --warmup 20 --repeats 100
```

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
```

## Operator List

| Operator | Description | Device Support |
|----------|-------------|----------------|
| `add_sample` | Element-wise addition of two tensors | NPU |
| `gather_coo` | Expand source rows by sorted COO indices | NPU |
| [`gather_csr`](docs/en/api_reference.md) | Expands segment features using CSR pointers | NPU |
| `random_walk` | Uniform or node2vec-biased random walks on COO graphs | NPU |
| `segment_max_csr` | Segmented max reduction on CSR format | NPU |
| `graclus_cluster` | Greedy graph clustering | NPU / CPU fallback for float64 |
| `ind2ptr` | Sorted row indices to CSR row pointer (torch_sparse-aligned) | NPU |
| `ptr2ind` | CSR row pointer to row indices (torch_sparse-aligned) | NPU |

## Development Guide

### Adding a New NPU Operator

1. Create AscendC kernel files (`.cpp`) and headers (`.h`) in `csrc/npu/kernel/<operator_name>/`
2. Create operator interface implementation (`.cpp`) and header (`.h`) in `csrc/npu/host/<operator_name>/`
3. Add PyTorch bindings in `csrc/pybind.cpp`
4. Create Python interface declaration files in `python/ops_gnn/`
5. Update `python/ops_gnn/__init__.py` to export the new function
6. Add test files in `test/`

## License

CANN Open Software License Agreement Version 2.0
