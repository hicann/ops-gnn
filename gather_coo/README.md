# Gather COO

`ops_gnn.gather_coo` 在 Ascend NPU 上按照有序 COO `index` 从 `src` 复制行/切片：

```python
out[..., e, ...] = src[..., index[..., e], ...]
```

它是 `segment_coo` 的扩展方向操作，适用于把节点/分组特征广播到边或成员。

## 接口

```python
ops_gnn.gather_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

- `dim = index.dim() - 1`；
- `index` 必须是 INT64，并沿最后一维非降序；
- `index` 的前缀维必须与 `src` 对齐，且值位于 `[0, src.size(dim))`；
- 输出形状等于 `src.shape`，但第 `dim` 维替换为 `index.size(-1)`；
- 支持 rank 1～8、空 Tensor、非连续 `src/index/out`、多 batch 和显式 `out`；
- 本任务只实现前向 gather，不提供 `reduce`、`dim_size` 或反向验收。

支持的 `src/out` 类型：

| 级别 | dtype | 路径 |
| --- | --- | --- |
| L1 | FP16、BF16、FP32、INT8、INT16、INT32、UINT8 | Ascend C Kernel |
| L2 | FP64、INT64 | NPU 原始位宽搬运，不参与性能门禁 |

## 示例

```python
import torch
import ops_gnn

src = torch.arange(20, dtype=torch.float32, device="npu").reshape(5, 4)
index = torch.tensor([0, 1, 1, 4], dtype=torch.int64, device="npu")
out = ops_gnn.gather_coo(src, index)
print(out.shape)  # torch.Size([4, 4])
```

## 代码与测试路径

| 内容 | 路径 |
| --- | --- |
| Python 适配层 | `python/ops_gnn/gather_coo.py` |
| Host | `csrc/npu/host/gather_coo/` |
| Ascend C Kernel | `csrc/npu/kernel/gather_coo/` |
| TC-01～TC-13 | `test/gather_coo/test_gather_coo_functional.py` |
| 十点性能 | `test/gather_coo/test_gather_coo_performance.py` |
| 统一构建脚本 | `scripts/build.sh` |

## 构建与测试

先激活兼容的 CANN/Python/torch_npu 环境，然后在仓库根目录使用统一构建脚本：

```bash
bash scripts/build.sh python
```

功能、性能与回归测试直接调用 `test/` 下的入口：

```bash
OPSGNN_REQUIRE_TORCH_SCATTER=1 \
python3 -m pytest test/gather_coo/test_gather_coo_functional.py -v
python3 test/gather_coo/test_gather_coo_performance.py \
  --warmup 20 --repeats 100 \
  --json-output /tmp/gather_coo_perf.json
python3 -m pytest test/test_import.py -v
```
