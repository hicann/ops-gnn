# RandomWalk

## 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| Ascend 950PR | √ |

## 功能说明

`RandomWalk` 在 COO 图上从一组起点采样固定长度的随机游走，PyTorch 接口与
`torch_cluster.random_walk` 对齐。Python 层在 NPU 上完成 COO 排序和 CSR 构造，
Ascend C SIMT kernel 完成以下两种采样：

- `p=1, q=1`：均匀随机游走；
- `p!=1` 或 `q!=1`：按照 node2vec 转移概率进行拒绝采样。

公开接口为：

```python
random_walk(
    row, col, start, walk_length, p=1, q=1, coalesced=True,
    num_nodes=None, return_edge_indices=False,
)
```

返回 `node_seq: int64[S, walk_length+1]`。当 `return_edge_indices=True` 时返回
`(node_seq, edge_seq)`，其中 `edge_seq: int64[S, walk_length]`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| -- | -- | -- | -- | -- |
| row | 输入 | COO 源节点，shape 为 `[E]` | INT64 | ND |
| col | 输入 | COO 目标节点，shape 为 `[E]` | INT64 | ND |
| start | 输入 | 游走起始节点，shape 为 `[S]` | INT64 | ND |
| walk_length | 属性 | 游走步数，取值大于等于 0 | INT | - |
| p | 属性 | node2vec 的返回参数，有限且大于 0 | FLOAT | - |
| q | 属性 | node2vec 的 BFS/DFS 参数，有限且大于 0 | FLOAT | - |
| coalesced | 属性 | 为 True 时按 `row*num_nodes+col` 排序输入边 | BOOL | - |
| num_nodes | 可选属性 | 图节点数；为 None 时由输入最大节点编号推导 | OPTIONAL INT | - |
| return_edge_indices | 属性 | 是否同时返回每一步的边索引 | BOOL | - |
| node_seq | 输出 | 节点序列，shape 为 `[S, walk_length+1]` | INT64 | ND |
| edge_seq | 可选输出 | 边序列，shape 为 `[S, walk_length]` | INT64 | ND |

## 约束说明

- `row`、`col`、`start` 必须是一维 INT64 NPU Tensor，且位于同一设备；
- `row` 与 `col` 元素数相同，节点编号满足 `0 <= index < num_nodes`；
- `coalesced=False` 时不会重排输入，边必须已经按源节点分组；
- `return_edge_indices=True` 且开启排序时，边索引对应排序后的 COO/CSR 位置；
- 孤立节点后续保持当前节点，对应 `edge_seq` 为 `-1`；
- 总边数及单节点出度不超过 `2^32-1`；
- 仅支持前向计算，不进行 CPU 回退；
- 算子包含随机采样。可用 `torch.manual_seed(seed)` 固定 NPU 默认 Generator，验证同一环境下的可复现性。

## 实现说明

- Python 层：输入校验、COO 排序、degree 统计及 CSR `rowptr` 构造；
- Host 层：获取当前 PyTorch NPU stream 和 Philox seed/offset，计算 node2vec 整数门限；
- Kernel 层：每个 SIMT 线程负责一条 walk。均匀路径直接按出度采样；node2vec 路径根据
  上一节点、相邻关系和 `p/q` 门限执行拒绝采样；
- `return_edge_indices=False` 时不写边序列；需要边序列时输出与节点跳转一一对应的 CSR 边位置。

## 调用说明

完成仓库构建并加载 CANN 环境后，可通过 Python API 调用：

```python
import torch
from ops_gnn import random_walk

device = "npu:0"
row = torch.tensor([0, 1, 1, 2], dtype=torch.int64, device=device)
col = torch.tensor([1, 0, 2, 1], dtype=torch.int64, device=device)
start = torch.tensor([0, 2], dtype=torch.int64, device=device)

torch.manual_seed(202608)
nodes, edges = random_walk(
    row, col, start, walk_length=8, p=0.5, q=2.0,
    return_edge_indices=True,
)
print(nodes.shape)  # torch.Size([2, 9])
print(edges.shape)  # torch.Size([2, 8])
```

## 测试与复现

```bash
bash scripts/build.sh cpp --clean
NPU_DEVICE_ID=0 python3 -m pytest test/random_walk -v
NPU_DEVICE_ID=0 python3 test/random_walk/benchmark.py \
  --device npu:0 --warmup 10 --repeats 50
```

## 参考资源

- [torch_cluster random_walk](https://github.com/rusty1s/pytorch_cluster)
- [任务书](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202608/random_walk_task_doc.md)
