# ops-gnn

[English](README_en.md) | 简体中文

ops-gnn 是昇腾生态下针对图神经网络（GNN）推出的一款算子库，支持基于NPU对图神经网络进行加速。

## 项目结构

```text
ops-gnn/
├── csrc/                       # C++/AscendC源码目录
│   ├── pybind.cpp              # PyTorch绑定代码
│   └── npu/                    # NPU相关代码（按算子分类）
├── docs/                       # 文档目录（API 说明见 docs/*/api_reference.md）
├── python/                     # Python源码目录
│   └── ops_gnn/                # Python包目录
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

`csrc/npu` 下各算子目录包含 `op_host` 和 `op_kernel/<arch>`；`test` 下各算子目录包含 `golden.py`、功能测试文件和性能测试文件。其中，`<arch>` 表示目标架构对应的目录名。

## 环境要求

- Python 3.9+
- CMake 3.18+
- CANN 9.1.0 及以上 + HDK（驱动/固件）25.7.rc1 及以上
- PyTorch 2.7+
- 与 PyTorch、CANN 匹配的 torch_npu
- CANN Toolkit (AscendC 编译器)
- C++17 或更高版本编译器
- 支持平台：Ascend 950系列；暂不支持其他平台

### CANN 环境配置

```bash
# 激活 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

如果 CANN 安装在自定义路径，请执行：

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

## 安装方法

### 方法1：使用pip直接安装

```bash
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

mkdir -p build_cmake
cd build_cmake
cmake ..
cmake --build .
```

## 功能测试

```bash
# 安装测试依赖
pip install pytest pytest-cov

# 快速开始：先跑单个/少量组别用例
pytest test/test_import.py -v            # 导入用例

# 运行所有功能测试
pytest test/ -v
```

运行单个算子的功能测试：

```bash
NPU_DEVICE_ID=<device_id> python3 -m pytest test/<op>/test_<op>.py -v
```

其中，`<op>` 表示算子名，`<device_id>` 表示设备 ID。

## 性能测试

运行单个算子的性能测试：

```bash
NPU_DEVICE_ID=<device_id> python3 test/<op>/benchmark_<op>.py
```

其中，`<op>` 表示算子名，`<device_id>` 表示设备 ID。

## 使用示例

```python
import torch
import ops_gnn

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

# torch_scatter 兼容的 Scatter 归约
src = torch.randn(10, 6, 64, dtype=torch.float32, device='npu')
index = torch.randint(0, 4, (10,), dtype=torch.long, device='npu')
result = ops_gnn.scatter(src, index, dim=0, dim_size=4, reduce='sum')
print(result.shape)  # torch.Size([4, 6, 64])
```

## 算子列表

| 算子 | 功能 | 设备支持 | API 文档 |
|------|------|----------|----------|
| `gather_coo` | 按有序 COO 索引扩展源行 | NPU | [gather_coo — COO 行扩展](docs/zh/api_reference.md#25-gather_coo--coo-行扩展) |
| `gather_csr` | 按 CSR 指针展开 segment 特征 | NPU | [gather_csr - CSR 分段展开](docs/zh/api_reference.md#29-gather_csr---csr-分段展开) |
| `random_walk` | COO 图上的均匀或 node2vec 偏置随机游走 | NPU | [random_walk — NPU 随机游走](docs/zh/api_reference.md#28-random_walk--npu-随机游走) |
| `segment_max_csr` | CSR 格式的分段最大值运算 | NPU | [segment_max_csr — CSR 分段最大值](docs/zh/api_reference.md#22-segment_max_csr--csr-分段最大值) |
| `radius` / `radius_graph` | 半径内邻居搜索（torch_cluster 兼容，Ascend 950PR） | NPU | [radius / radius_graph — 半径内邻居搜索](docs/zh/api_reference.md#23-radius--radius_graph--半径内邻居搜索) |
| `graclus_cluster` | 图贪心聚类 | NPU / CPU float64 回退 | [graclus_cluster — 图贪心聚类](docs/zh/api_reference.md#24-graclus_cluster--图贪心聚类) |
| `ind2ptr` | 有序行索引转 CSR 行指针（对齐 torch_sparse） | NPU | [ind2ptr — 行索引转 CSR 行指针](docs/zh/api_reference.md#26-ind2ptr--行索引转-csr-行指针) |
| `ptr2ind` | CSR 行指针转行索引（对齐 torch_sparse） | NPU | [ptr2ind — CSR 行指针转行索引](docs/zh/api_reference.md#27-ptr2ind--csr-行指针转行索引) |
| `scatter` / `scatter_*` | 与 torch_scatter 对齐的索引分组归约 | NPU / CPU float64、int64 回退 | [scatter — Scatter 系列归约](docs/zh/api_reference.md#210-scatter--scatter-系列归约) |

## 开发指南

### 添加新的 NPU 算子

1. 在 `csrc/npu/<op>/op_kernel/<arch>/` 中创建 AscendC 内核文件（`.cpp`）和头文件（`.h`）
2. 在 `csrc/npu/<op>/op_host/` 中创建算子接口实现（`.cpp`）和头文件（`.h`）
3. 在 `csrc/pybind.cpp` 中添加 PyTorch 绑定
4. 在 `python/ops_gnn/` 中创建 Python 接口声明文件
5. 更新 `python/ops_gnn/__init__.py` 导出新函数
6. 在 `test/<op>/` 中添加 `golden.py`、`test_<op>.py` 和 `benchmark_<op>.py`

其中，`<op>` 表示算子名。

## 许可证

CANN Open Software License Agreement Version 2.0
