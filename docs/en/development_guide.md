# ops-gnn Development Guide

This document provides guidance on development environment setup, operator development, and testing for the ops-gnn library.

## 1. Environment Setup

### 1.1 Install CANN

#### Pre-installation

For offline installation, click [here](https://www.hiascend.com/developer/download/community/result?module=cann) to download the CANN package and upload it to any path on the installation environment.

#### Install CANN

CANN 9.1.0-beta.1 or later is required. Other versions are not currently supported.

```sh
chmod +x Ascend-cann-toolkit_${VERSION}_linux-$(arch).run
./Ascend-cann-toolkit_${VERSION}_linux-$(arch).run --install
```

Where `${VERSION}` is the CANN version (e.g., 9.1.0) and `$(arch)` is the CPU architecture.

#### Post-installation Configuration

```sh
source /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash
```

If CANN is installed in a different path, replace with the actual path.

### 1.2 Detailed CANN Installation Guide

Visit [Ascend Documentation](https://www.hiascend.com/document) → CANN Community Edition → Software Installation for detailed installation steps based on your environment, OS, and use case.

### 1.3 Dependencies

ops-gnn depends on the following components:

| Component | Version | Description |
|-----------|---------|-------------|
| Python | 3.9+ | |
| CMake | 3.18+ | C++ build system |
| GCC | 7.0+ | C++17 compiler |
| PyTorch | 2.7+ | Ascend-adapted version from Ascend Community |
| torch_npu | 26.0.0 | PyTorch NPU device backend |
| CANN Toolkit | 9.1.0-beta.1+ | AscendC compiler, Bisheng compiler, runtime libraries |

`torch` and `torch_npu` must be obtained from the [Ascend Community Download Page](https://www.hiascend.com/developer/download/community) for Ascend hardware-adapted versions.

```sh
# System toolchain
sudo apt install build-essential cmake

# Test dependencies (optional)
pip3 install pytest pytest-cov
```

### 1.4 Install ops-gnn

```sh
# Activate CANN environment
source /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash

# Method 1: pip development mode
cd /path/to/ops-gnn
pip install -e .

# Method 2: Build script
cd scripts
./build.sh python

# Method 3: CMake manual build
mkdir build_cmake && cd build_cmake
cmake ..
cmake --build .
```

## 2. Project Architecture

### 2.1 Layered Architecture

ops-gnn adopts a layered architecture of PyTorch extension + AscendC kernel:

```
+---------------------------------------------+
|              Python Interface Layer           |
|   python/ops_gnn/<op>.py                    |
|   (Type annotations, parameter preprocessing,|
|    documentation)                            |
+---------------------------------------------+
|              PyTorch Binding Layer            |
|   csrc/pybind.cpp                           |
|   (PYBIND11_MODULE registration, param map)  |
+---------------------------------------------+
|              Host-side Operator Layer         |
|   csrc/npu/host/<op>/<op>.cpp/.h            |
|   (Argument parsing, Tiling calc,            |
|    Kernel launch, stream management)         |
+---------------------------------------------+
|              AscendC Kernel Layer             |
|   csrc/npu/kernel/<op>/<op>_kernel.cpp/.h   |
|   (AscendC SIMT/VF programming,              |
|    vector computation, data movement)        |
+---------------------------------------------+
```

**Architecture Description:**

- **Python Interface Layer**: Provides user-friendly Python API with type annotations, docstrings, and usage examples
- **PyTorch Binding Layer**: Exposes C++ functions to Python via pybind11, handles Tensor parameter passing
- **Host-side Operator Layer**: Runs on host CPU, responsible for parameter validation, Tiling data computation, AscendC Kernel launch, and Stream synchronization
- **AscendC Kernel Layer**: Runs on NPU device (AIV cores), implements core computation logic

### 2.2 Directory Structure

```
ops-gnn
├── csrc/                           # C++/AscendC source code
│   ├── pybind.cpp                  # PyTorch binding code
│   └── npu/                        # NPU-related code
│       ├── host/                   # Host-side code (organized by operator)
│       │   ├── add_sample/
│       │   │   ├── add_sample.h    # Host interface declaration
│       │   │   └── add_sample.cpp  # Host implementation (Tiling + Launch)
│       │   └── segment_max_csr/
│       │       ├── segment_max_csr.h
│       │       └── segment_max_csr.cpp
│       └── kernel/                 # AscendC kernel implementation (by operator)
│           ├── add_sample/
│           │   ├── add_sample_kernel.h   # Kernel Launch interface
│           │   └── add_sample_kernel.cpp # Kernel implementation (AscendC SIMT)
│           └── segment_max_csr/
│               ├── segment_max_csr_kernel.h       # Kernel Launch interface
│               ├── segment_max_csr_kernel.cpp     # Kernel entry + template instantiation
│               ├── segment_max_csr_kernel_impl.h  # Kernel core implementation class
│               └── segment_max_csr_tiling.h       # Tiling data structure
├── docs/                           # Documentation
├── python/                         # Python source code
│   └── ops_gnn/                    # Python package
│       ├── __init__.py             # Package init, export list
│       ├── add_sample.py           # add_sample Python interface
│       ├── segment_max_csr.py      # segment_max_csr Python interface
│       └── typing.py               # Type alias definitions
├── test/                           # Test directory
│   ├── test_import.py              # Import verification test
│   ├── test_example.py             # add_sample operator test
│   └── test_segment_max_csr.py     # segment_max_csr operator test
├── scripts/                        # Build scripts
│   └── build.sh                    # Unified build script (python/cpp/all)
├── cmake/                          # CMake configuration
│   └── OpsGNNConfig.cmake.in       # CMake package config template
├── CMakeLists.txt                  # CMake build configuration
├── setup.py                        # Python install script (setuptools + cmake)
├── setup.cfg                       # setuptools configuration
├── pyproject.toml                  # Modern Python project config
├── MANIFEST.in                     # Packaging manifest
└── LICENSE                         # CANN License
```

### 2.3 Core File Descriptions

| File | Description |
|------|-------------|
| `csrc/pybind.cpp` | PyTorch binding entry, registers all C++ operators to Python via `PYBIND11_MODULE` |
| `csrc/npu/host/<op>/<op>.h` | Host-side operator interface declaration |
| `csrc/npu/host/<op>/<op>.cpp` | Host-side operator implementation: Tensor dimension parsing, Tiling calc, dtype dispatch, Stream management |
| `csrc/npu/kernel/<op>/<op>_kernel.h` | Kernel Launch function declaration (Host-side call entry) |
| `csrc/npu/kernel/<op>/<op>_kernel.cpp` | Kernel Launch implementation + explicit template instantiation + `<<<>>>` launch syntax |
| `csrc/npu/kernel/<op>/<op>_kernel_impl.h` | AscendC Kernel core class implementation (Init → Process → Compute pipeline) |
| `csrc/npu/kernel/<op>/<op>_tiling.h` | Tiling data structure definition (parameter struct passed to Device side) |
| `python/ops_gnn/<op>.py` | Python interface: type annotations, default value handling, calls `_pybind.<op>` |
| `python/ops_gnn/__init__.py` | Package entry, imports from modules and registers to `__all__` |
| `python/ops_gnn/typing.py` | Type aliases (`Tensor`, `OptTensor`) |

## 3. Coding Standards

### 3.1 Naming Conventions

**C++ Layer:**
- File names: lowercase underscore-separated, e.g., `segment_max_csr_kernel.h`
- Function names: PascalCase, e.g., `LaunchSegmentMaxCsrKernel`
- Struct names: PascalCase, e.g., `SegmentMaxCsrTilingData`
- Class names: PascalCase, e.g., `SegmentMaxCsrKernel`
- Template parameters: PascalCase or single uppercase letter, e.g., `typename T`

**Python Layer:**
- File names: lowercase underscore-separated, e.g., `segment_max_csr.py`
- Function names: lowercase underscore-separated, e.g., `segment_max_csr`
- Consistent with PyTorch style

### 3.2 Code Style

- C++ follows C++17 standard, uses `#pragma once` for header guards
- AscendC Kernel uses `__aicore__`, `__global__`, `__gm__` modifiers
- Use APIs under namespace `AscendC`
- Python interfaces use type annotations (`Tensor`, `OptTensor`, `Optional[Tensor]`)
- Each file includes CANN Open Software License copyright header

### 3.3 File Organization

Each operator strictly follows the following file split:

```
csrc/npu/
├── host/<op>/
│   ├── <op>.h          # Host interface declaration
│   └── <op>.cpp        # Host implementation
└── kernel/<op>/
    ├── <op>_kernel.h        # Kernel Launch declaration
    ├── <op>_kernel.cpp      # Kernel Launch implementation + template instantiation
    ├── <op>_kernel_impl.h   # Kernel core class (simple ops can merge into kernel.cpp)
    └── <op>_tiling.h        # Tiling struct
```

## 4. Build System

### 4.1 CMake Build

CMakeLists.txt compiles AscendC Kernels and PyTorch binding library:

1. Uses Bisheng compiler to compile Kernel source files into `libopsgnn_npu_kernel.so`
2. Compiles `csrc/pybind.cpp` + host-side code into `_pybind.so`
3. Outputs artifacts to `output/kernel/`

Key CMake variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NPU_ARCH` | NPU architecture | `dav-3510` (950) / `dav-2201` (910B) |
| `WITH_PYTHON` | Enable Python binding build | `ON` |
| `ASCEND_HOME_PATH` | CANN installation path | Read from env |
| `Python3_ROOT_DIR` | Python root directory | Auto-exported by build.sh |

### 4.2 setuptools Build

`setup.py` invokes CMake to compile and generate a wheel package. `build_with_cmake()` cleans old `CMakeCache.txt` before building to avoid path inconsistencies from pip temp directories.

## 5. Testing Guide

### 5.1 Running Tests

```sh
pytest test/ -v                                    # Run all tests
pytest test/test_segment_max_csr.py -v             # Single operator test
pytest test/test_segment_max_csr.py::test_func -v  # Single test case
```

### 5.2 Test Writing Pattern

1. `torch.npu.set_device(4)` to specify NPU device
2. `torch.manual_seed(42)` for reproducibility
3. Use `.npu()` method to create NPU Tensor (e.g., `torch.randint(...).npu()`)
4. Call `ops_gnn.<op>(...)`
5. Verify results: device type, shape, numerical correctness

### 5.3 Test Coverage Requirements

| Scenario | Description |
|----------|-------------|
| Basic functionality | Typical input |
| Different dtypes | float32, float16, int32, int16 |
| Edge cases | Empty segment, single element |
| Optional parameters | optional_out |
| Multi-dimensional input | 2D/3D Tensor |
| Broadcast scenarios | indptr broadcasting |

## 6. Operator Development Walkthrough

This section uses `segment_max_csr` as an example to illustrate the operator development process.

### 6.1 Call Chain Analysis

```
User Code
    └─→ ops_gnn.segment_max_csr(src, indptr, optional_out)
            │  python/ops_gnn/segment_max_csr.py
            │  (default value handling: None → empty Tensor)
            └─→ _pybind.segment_max_csr(src, indptr, optional_out)
                    │  csrc/pybind.cpp
                    │  (PYBIND11_MODULE registration)
                    └─→ segment_max_csr()  [Host]
                            │  csrc/npu/host/segment_max_csr/segment_max_csr.cpp
                            │  (dimension parsing, Tiling fill, dtype dispatch, Stream management)
                            └─→ LaunchSegmentMaxCsrKernel<T>()
                                    │  csrc/npu/kernel/segment_max_csr/segment_max_csr_kernel.cpp
                                    │  (get AIV core count, <<<>>> launch)
                                    └─→ segment_max_csr_kernel<T>  [Device]
                                            │  segment_max_csr_kernel_impl.h
                                            │  (Init → Process → Compute pipeline)
```

Each layer's responsibilities:

- **Python** (`python/ops_gnn/<op>.py`): Type annotations, optional param defaults, calls `_pybind.<op>`
- **PyBind** (`csrc/pybind.cpp`): `m.def("<op>", &func, py::arg(...), ...)` registers C++ function
- **Host** (`csrc/npu/host/<op>/`): Tensor dimension parsing → Tiling calc → `torch::empty` create output → `aclrtCreateStream` → dtype dispatch launch → `aclrtSynchronizeStream` → destroy Stream
- **Kernel Launch** (`csrc/npu/kernel/<op>/<op>_kernel.cpp`): `GetCoreNumAiv()` get core count → `<<<coreNum, nullptr, stream>>>` launch → explicit template instantiation
- **Kernel Impl** (`<op>_kernel_impl.h`): `Init()` parse Tiling + allocate Buffer/Event → `Process()` assign work range per Block → `Compute()` double-buffer pipeline (DataCopy → Max/Add → DataCopy)

### 6.2 New Operator File Checklist

Using `segment_max_csr` as a template, each new operator needs:

```
csrc/npu/host/<op>/
├── <op>.h                   # torch::Tensor <op>(torch::Tensor ...);
└── <op>.cpp                 # Host implementation

csrc/npu/kernel/<op>/
├── <op>_kernel.h            # template<typename T> void Launch<Op>Kernel(...);
├── <op>_kernel.cpp          # Launch impl + template instantiation
├── <op>_kernel_impl.h       # Kernel core class (Init/Process/Compute)
└── <op>_tiling.h            # struct <Op>TilingData { uint32_t ... };

python/ops_gnn/
└── <op>.py                  # Python interface

test/
└── test_<op>.py             # Unit test
```

Additionally, two files must be modified:
- `csrc/pybind.cpp`: `#include "host/<op>/<op>.h"` + `m.def("<op>", ...)`
- `python/ops_gnn/__init__.py`: `from .<op> import <op>` + add to `__all__`

Simple operators can merge files: omit `_tiling.h` when no Tiling, merge `_kernel_impl.h` into `_kernel.cpp` when logic is simple (see `add_sample`).

### 6.3 Two Kernel Modes

| Feature | add_sample | segment_max_csr |
|---------|-----------|----------------|
| File count | 1 kernel.cpp | kernel + impl + tiling (4 files) |
| Programming model | SIMT (`__simt_vf__` + `VF_CALL`) | Kernel class (TPipe + Buffer + Event) |
| Data movement | Direct GM read/write | DataCopy + double-buffer pipeline |
| Tiling | None (scalar params direct) | Tiling struct |
| Applicable | Element-wise ops | Reductions, segmentation, multi-stage pipelines |

### 6.4 Development Process Summary

1. Create directory structure and files following `segment_max_csr`
2. Define Tiling struct (only `uint32_t` types, no pointers)
3. Implement Kernel core class (Init → Process → Compute), use TPipe + Event for double-buffer pipeline
4. Implement Launch function, wrap `<<<>>>` with `#pragma GCC diagnostic`
5. Host side: dimension parsing → Tiling fill → `torch::empty` → `aclrtCreateStream` → dtype dispatch launch → sync → destroy Stream
6. Register with `m.def(...)` in `pybind.cpp`
7. Write Python interface (type annotations + docstring + default handling)
8. Update `__init__.py` exports
9. Write tests (at minimum: basic functionality, multiple dtypes, edge cases)

## 7. More Resources

- **[Back to README](../../README_EN.md)**
- **[AscendC Programming Guide](https://www.hiascend.com/document)**
- **[PyTorch C++ Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)**
- **[CANN Community Edition Docs](https://www.hiascend.com/document)**
