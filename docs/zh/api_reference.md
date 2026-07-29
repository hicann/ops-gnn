# ops-gnn API文档和使用示例

本文档提供 ops-gnn 库的详细 API 接口说明，包括算子签名、参数说明、返回值和使用示例。

## 一、类型定义

### 1.1 Tensor

```python
import torch
Tensor = torch.Tensor
```

NPU 设备上的 Tensor，由 PyTorch 管理内存。

### 1.2 OptTensor

```python
from typing import Optional
OptTensor = Optional[torch.Tensor]
```

可选 Tensor 类型，用于可能有默认值的可选参数。当传入 `None` 时，Python 层会自动转换为空 Tensor 传给底层 C++ 实现。

---

## 二、核心算子API

### 2.1 add_sample — 逐元素加法

**函数签名：**

```python
def add_sample(
    src1: Tensor,
    src2: Tensor,
) -> Tensor:
```

**参数说明：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| src1 | Tensor | 输入 | 第一个输入 Tensor，必须位于 NPU 设备 |
| src2 | Tensor | 输入 | 第二个输入 Tensor，必须位于 NPU 设备，与 `src1` 形状相同 |

**返回值说明：**

返回一个新的 Tensor，为 `src1` 和 `src2` 的逐元素相加结果，位于 NPU 设备，形状和 dtype 与输入相同。

**功能说明：**

对两个 Tensor 执行逐元素加法运算（`src1[i] + src2[i]`），在 NPU 上使用 AscendC SIMT 模式并行计算，支持 uint8 类型。

**实现架构：**

- **Kernel 模式**：SIMT（`__simt_vf__` + `VF_CALL`），单文件实现
- **数据搬移**：直接 GM 读写，无 Tiling
- **适用场景**：逐元素操作，逻辑简单的 element-wise 算子

**注意事项：**

- 两个输入 Tensor 必须形状一致
- 输入必须位于 NPU 设备（`device='npu'`）
- 当前仅支持 uint8（`torch.uint8`）类型
- 运算直接在 NPU 上执行，无 CPU 回退路径

**使用示例：**

```python
import torch
import ops_gnn

# 设置 NPU 设备
torch.npu.set_device(4)
torch.manual_seed(42)

# 创建输入 Tensor（NPU 设备）
src1 = torch.randint(0, 128, (1024, 1024), dtype=torch.uint8, device='npu')
src2 = torch.randint(0, 128, (1024, 1024), dtype=torch.uint8, device='npu')

# 调用算子
result = ops_gnn.add_sample(src1, src2)

# 验证结果
expected = src1 + src2
assert result.device.type == 'npu'
assert result.shape == (1024, 1024)
assert torch.equal(result, expected)
```

---

### 2.2 segment_max_csr — CSR 分段最大值

**函数签名：**

```python
def segment_max_csr(
    src: Tensor,
    indptr: Tensor,
    optional_out: Optional[Tensor] = None,
) -> Tensor:
```

**参数说明：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| src | Tensor | 输入 | 输入数据 Tensor，必须位于 NPU 设备 |
| indptr | Tensor | 输入 | CSR 格式的索引指针 Tensor，dtype 必须为 `torch.int32`，必须位于 NPU 设备。最后一维的长度 - 1 决定分段数量 |
| optional_out | Optional[Tensor] | 输入 | 可选输出 Tensor，位于 NPU 设备。如果提供，每个 segment 的第一个元素会与 `optional_out` 的对应值做 max 比较（即 `max(src[start:end], optional_out)`），而非仅从 src 中取最大值 |

**返回值说明：**

返回一个新的 Tensor，dtype 与 `src` 相同。形状与 `src` 相同，但 `indptr` 最后一维对应的维度被缩减为 `nSegments = indptr_last_dim - 1`。

**功能说明：**

沿 CSR 格式的 `indptr` 指定的维度对 `src` 进行分段最大值规约。对于每个分段 `[indptr[seg], indptr[seg+1])`，计算该分段内所有元素的最大值。支持多维广播：当 `indptr` 的维度低于 `src` 时，沿对应的维度广播 indptr。

**实现架构：**

- **Kernel 模式**：Kernel 类模式（TPipe + Buffer + Event），4 文件实现
- **数据搬移**：DataCopy + 双缓冲流水线（MTE2 ↔ VECCALC ↔ MTE3）
- **Tiling**：`SegmentMaxCsrTilingData` 结构体，包含分块参数（coreDataNum、KloopTime、ALIGN_NUM 等）
- **多 AIV 核并行**：按 `E_1`（广播维度）均匀分配到各 AIV 核，每核独立处理

**调用链：**

```test
ops_gnn.segment_max_csr(src, indptr, optional_out)
    → _pybind.segment_max_csr()           # PyTorch 绑定
        → segment_max_csr()                # Host 端：维度解析、Tiling 填充、dtype 分发
            → LaunchSegmentMaxCsrKernel<T>()  # Kernel Launch：获取 AIV 核数、<<<>>> 启动
                → segment_max_csr_kernel<T>   # Device 端：Init → Process → Compute 流水线
```

**支持的 dtype：**

| dtype | 对应 fill 值（空 segment） |
|-------|--------------------------|
| `torch.float32` | `-3.4028235e+38`（-FLT_MAX） |
| `torch.float16` | `-65504`（-HALF_MAX） |
| `torch.int32` | `-2147483648`（INT32_MIN） |
| `torch.int16` | `-32768`（INT16_MIN） |

**注意事项：**

- `indptr` 的 dtype 必须为 `torch.int32`
- `indptr` 的值必须是非递减的（即 `indptr[seg] <= indptr[seg+1]`）
- 空 segment（`indptr[seg] == indptr[seg+1]`）：若无 `optional_out`，填充对应 dtype 的最小值；若有 `optional_out`，直接拷贝 `optional_out` 对应位置的值
- 支持广播：1D indptr 可广播到多个 batch（`indptr.view(1, -1)`）
- 输入必须位于 NPU 设备（`device='npu'`）

**使用示例：**

#### 基本用法 — 1D indptr

```python
import torch
import ops_gnn

torch.npu.set_device(4)
torch.manual_seed(42)

# 创建输入数据
# src shape: (4, 2)，4 个"行"，每行 2 个元素
src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
# indptr: [0, 2, 4] 表示第 0 个分段取 src[0:2]，第 1 个取 src[2:4]
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[3, 4], [7, 8]]  — shape: (2, 2)
#           分段0: max(src[0:2])  分段1: max(src[2:4])
```

#### 不同 dtype

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

#### 空 segment

```python
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='npu')
# indptr: [0, 0, 2] — 第 0 个分段是空的，第 1 个取 src[0:2]
indptr = torch.tensor([0, 0, 2], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result[0] = [-inf, -inf]  — 空分段填充最小值
# result[1] = [3, 4]        — src[0:2] 的最大值
```

#### 带 optional_out

```python
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='npu')
indptr = torch.tensor([0, 2], dtype=torch.int32, device='npu')
optional_out = torch.tensor([[10, 20]], dtype=torch.float32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr, optional_out)
# result = [[10, 20]]  — max(src[0:2], optional_out) = max([[1,2],[3,4]], [[10,20]])
```

#### 2D indptr（每 batch 不同分段）

```python
src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
# 2D indptr: batch 0 用 [0, 2, 4]，batch 1 用 [0, 1, 3]
indptr = torch.tensor([[0, 2, 4], [0, 1, 3]], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[2, 4],    — batch 0: seg0=src[0:2], seg1=src[2:4]
#           [5, 7]]    — batch 1: seg0=src[0:1], seg1=src[1:3]]
```

#### indptr 广播

```python
src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
# 1D indptr，view 成 (1, -1) 后广播到两个 batch
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu').view(1, -1)

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[2, 4],    — batch 0: seg0=src[0:2], seg1=src[2:4]
#           [6, 8]]    — batch 1: 同上分段
```

#### 复杂形状（3D src + 2D indptr）

```python
src = torch.randn(3, 8, 16, dtype=torch.float32, device='npu')
indptr = torch.tensor([[0, 4, 8]], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result shape: (3, 2, 16) — indptr 的最后一维从 3→2，其他维度不变
```

---

## 三、测试指南

### 3.1 运行测试

```sh
# 运行所有测试
pytest test/ -v

# 运行单个算子测试
pytest test/test_example.py -v
pytest test/test_segment_max_csr.py -v

# 运行单个测试用例
pytest test/test_segment_max_csr.py::test_segment_max_csr_basic -v
```

### 3.2 测试编写模板

```python
import pytest
import torch
import ops_gnn

def test_my_operator():
    """测试基本功能"""
    torch.npu.set_device(4)          # 1. 指定 NPU 设备
    torch.manual_seed(42)            # 2. 设置随机种子

    # 3. 创建 NPU Tensor
    src = torch.tensor([...], dtype=torch.float32, device='npu')

    # 4. 调用算子
    result = ops_gnn.my_op(src)

    # 5. 验证：设备、形状、数值
    assert result.device.type == 'npu'
    assert result.shape == expected_shape
    assert torch.allclose(result, expected)
```

---

## 四、返回主文档

- **[返回 README](../../README.md)**
- **[查看开发指导](development_guide.md)**
