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
│       └── kernel/             # AscendC kernel implementation (organized by operator)
├── docs/                       # Documentation directory
├── python/                     # Python source code
│   └── ops_gnn/                # Python package
│       ├── __init__.py         # Package initialization
│       ├── add_sample.py       # Python interface declaration
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
- torch_npu 26.0.0+ (PyTorch NPU extension)
- CANN Toolkit (AscendC compiler)
- C++17 or later compiler

### CANN Environment Setup

```bash
source ${ASCEND_HOME_PATH}/bin/setenv.bash
```

## Installation

### Method 1: Install via pip

```bash
# Activate CANN environment
source ${ASCEND_HOME_PATH}/bin/setenv.bash

# Install in development mode
pip install --no-build-isolation -e .
```

### Method 2: Using the build script

```bash
cd scripts

# Build Python package
./build.sh python
```

### Method 3: Using CMake (Linux)

```bash
source /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash

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
```

## Operator List

| Operator | Description | Device Support |
|----------|-------------|----------------|
| `add_sample` | Element-wise addition of two tensors | NPU |
| `segment_max_csr` | Segmented max reduction on CSR format | NPU |

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
