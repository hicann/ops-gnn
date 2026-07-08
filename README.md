# ops-gnn

ops-gnn 是昇腾生态下图神经网络（GNN）算子和特性的集合，支持基于 AscendC 的 NPU 加速。

该项目参考了 [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) 的框架结构，主要运行在 Linux 系统上。

## 项目结构

```
ops-gnn/
├── csrc/                       # C++/AscendC源码目录
│   ├── pybind.cpp              # PyTorch绑定代码
│   └── npu/                    # NPU相关代码
│       ├── host/               # Host端代码（按算子分类）
│       └── kernel/             # AscendC内核实现（按算子分类）
├── python/                     # Python源码目录
│   └── ops_gnn/                # Python包目录
│       ├── __init__.py         # 包初始化文件
│       ├── add_sample.py       # Python接口声明
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
└── LICENSE                     # MIT许可证
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- torch_npu (PyTorch NPU 扩展)
- CANN Toolkit (AscendC 编译器)
- C++17 或更高版本编译器

### CANN 环境配置

```bash
source /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash
```

## 安装方法

### 方法1：使用pip直接安装

```bash
# 激活 CANN 环境
source /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash

# 安装开发模式
pip install -e .
```

### 方法2：使用构建脚本

```bash
cd scripts

# 构建 Python 包
./build.sh python
```

### 方法3：使用CMake（Linux）

```bash
source /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash

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
```

## 算子列表

| 算子 | 功能 | 设备支持 |
|------|------|----------|
| `add_sample` | 两个 tensor 逐元素相加 | NPU |
| `segment_max_csr` | CSR 格式的分段最大值运算 | NPU |

## 开发指南

### 添加新的 NPU 算子

1. 在 `csrc/npu/kernel/<算子名>/` 中创建 AscendC 内核文件（`.cpp`/`.asc`）和头文件（`.h`）
2. 在 `csrc/npu/host/<算子名>/` 中创建算子接口实现（`.cpp`）和头文件（`.h`）
3. 在 `csrc/pybind.cpp` 中添加 PyTorch 绑定
4. 在 `python/ops_gnn/` 中创建 Python 接口声明文件
5. 更新 `python/ops_gnn/__init__.py` 导出新函数
6. 在 `test/` 中添加测试文件

## 参考

- [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster)
- [PyTorch C++ Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [AscendC Programming Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/developerguide/ascendcdev/ascendcdev_0001.html)

## 许可证

MIT License