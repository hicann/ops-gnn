# ops-gnn

[English](README_EN.md) | 简体中文

ops-gnn 是昇腾生态下针对图神经网络（GNN）推出的一款算子库，支持基于NPU对图神经网络进行加速。

## 项目结构

```text
ops-gnn/
├── csrc/                       # C++/AscendC源码目录
│   ├── pybind.cpp              # PyTorch绑定代码
│   └── npu/                    # NPU相关代码
│       ├── host/               # Host端代码（按算子分类）
│       ├── kernel/             # AscendC内核实现（按算子分类）
│       └── sparse/             # sparse 算子（按算子分子目录）
│           ├── ind2ptr/
│           │   ├── op_host/   # Host 调度
│           │   └── op_kernel/
│           │       └── arch35/  # Ascend950 Kernel
│           └── ptr2ind/
│               ├── op_host/
│               └── op_kernel/
│                   └── arch35/
├── docs/                       # 文档目录（API 说明见 docs/*/api_reference.md）
├── python/                     # Python源码目录
│   └── ops_gnn/                # Python包目录
│       ├── __init__.py         # 包初始化文件
│       ├── add_sample.py       # Python接口声明
│       ├── gather_coo.py       # Python接口声明
│       ├── gather_csr.py       # Gather CSR Python接口
│       ├── ind2ptr.py          # ind2ptr Python 接口
│       ├── ptr2ind.py          # ptr2ind Python 接口
│       ├── random_walk.py      # 随机游走 Python 接口
│       ├── graclus_cluster.py  # Graclus 聚类 Python 接口
│       ├── segment_max_csr.py  # Python接口声明
│       └── typing.py           # 类型定义
├── test/                       # 测试目录
├── scripts/                    # 构建脚本目录
│   └── build.sh                # 统一构建脚本
├── cmake/                      # CMake配置
│   └── OpsGNNConfig.cmake.in
├── CMakeLists.txt              # CMake构建配置
├── setup.py                    # Python安装脚本 (使用PyTorch cpp_extension)
├── setup.cfg                   # setuptools配置
├── pyproject.toml              # 现代Python项目配置
├── MANIFEST.in                 # 打包清单
└── LICENSE                     # CANN 许可证
```

## 环境要求

- Python 3.9+
- CMake 3.18+
- PyTorch 2.7+
- 与 PyTorch、CANN 匹配的 torch_npu（Gather COO 实测为 PyTorch 2.7.1 / torch_npu 2.7.1.post4）
- CANN Toolkit (AscendC 编译器)
- C++17 或更高版本编译器

### CANN 环境配置

```bash
export CANN_SETENV=/path/to/cann/bin/setenv.bash
source "$CANN_SETENV"
```

## 安装方法

### 方法1：使用pip直接安装

```bash
# 激活 CANN 环境
export CANN_SETENV=/path/to/cann/bin/setenv.bash
source "$CANN_SETENV"

# 安装开发模式
python3 -m pip install --no-build-isolation --no-deps -e .
```

### 方法2：使用构建脚本

```bash
cd scripts

# 构建 Python 包
./build.sh python
```

### 方法3：使用CMake（Linux）

```bash
# 使用当前机器上与 PyTorch/torch_npu 匹配的 CANN 初始化脚本
export CANN_SETENV=/path/to/cann/bin/setenv.bash
source "$CANN_SETENV"

mkdir -p build_cmake
cd build_cmake
cmake ..
cmake --build .
```

## 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-cov

# 运行所有测试
pytest test/ -v
```

Gather COO 测试位于 `test/gather_coo/`：

```bash
OPSGNN_REQUIRE_TORCH_SCATTER=1 \
python3 -m pytest test/gather_coo/test_gather_coo_functional.py -v
python3 test/gather_coo/test_gather_coo_performance.py --warmup 20 --repeats 100
```

## 使用示例

```python
import torch
import ops_gnn

# NPU tensor 加法运算
src1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint8, device='npu')
src2 = torch.tensor([5, 4, 3, 2, 1], dtype=torch.uint8, device='npu')
result = ops_gnn.add_sample(src1, src2)
print(result)  # 输出: tensor([6, 6, 6, 6, 6], device='npu:0', dtype=torch.uint8)

# CSR 格式的分段最大值运算
src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
result = ops_gnn.segment_max_csr(src, indptr)
print(result)  # 输出: tensor([[3, 4], [7, 8]], device='npu:0')

# torch_sparse 兼容：row indices <-> CSR rowptr
row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
rowptr = ops_gnn.ind2ptr(row, 8)          # 替换 torch.ops.torch_sparse.ind2ptr(row, 8)
row2 = ops_gnn.ptr2ind(rowptr, 6)         # 替换 torch.ops.torch_sparse.ptr2ind(rowptr, 6)

# COO 行扩展；index 必须为有序 int64
src = torch.arange(20, dtype=torch.float32, device='npu').reshape(5, 4)
index = torch.tensor([0, 1, 1, 4], dtype=torch.int64, device='npu')
result = ops_gnn.gather_coo(src, index)
print(result.shape)  # 输出: torch.Size([4, 4])

# 按 CSR 指针展开 segment 特征
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16, device='npu')
indptr = torch.tensor([0, 2, 5], dtype=torch.int64, device='npu')
result = ops_gnn.gather_csr(src, indptr)
print(result.shape)  # torch.Size([5, 2])
```

## 算子列表

| 算子 | 功能 | 设备支持 |
|------|------|----------|
| `add_sample` | 两个 tensor 逐元素相加 | NPU |
| `gather_coo` | 按有序 COO 索引扩展源行 | NPU |
| [`gather_csr`](docs/zh/api_reference.md) | 按 CSR 指针展开 segment 特征 | NPU |
| `random_walk` | COO 图上的均匀或 node2vec 偏置随机游走 | NPU |
| `segment_max_csr` | CSR 格式的分段最大值运算 | NPU |
| `graclus_cluster` | 图贪心聚类 | NPU / CPU float64 回退 |
| `ind2ptr` | 有序行索引转 CSR 行指针（对齐 torch_sparse） | NPU |
| `ptr2ind` | CSR 行指针转行索引（对齐 torch_sparse） | NPU |

## 开发指南

### 添加新的 NPU 算子

1. 在 `csrc/npu/kernel/<算子名>/` 中创建 AscendC 内核文件（`.cpp`）和头文件（`.h`）
2. 在 `csrc/npu/host/<算子名>/` 中创建算子接口实现（`.cpp`）和头文件（`.h`）
3. 在 `csrc/pybind.cpp` 中添加 PyTorch 绑定
4. 在 `python/ops_gnn/` 中创建 Python 接口声明文件
5. 更新 `python/ops_gnn/__init__.py` 导出新函数
6. 在 `test/` 中添加测试文件

## 许可证

CANN Open Software License Agreement Version 2.0
