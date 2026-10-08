# Ascend 950 segment_coo 算子设计

本文件说明产品代码的接口、数据流与边界。社区任务的设计评审记录见
[设计 PR #1485](https://gitcode.com/cann/cann-ops-competitions/pull/1485)。
本次产品接入基于验收仓提交 `0cdb291cea227d6c5be03ed8b3cf85c4e9d4e9f9`；
本版本的测试结论以产品 PR 所附的同版日志为准。

## 接口与输入合同

提供 `segment_coo` 与 `segment_sum_coo`、`segment_add_coo`、
`segment_mean_coo`、`segment_min_coo`、`segment_max_coo`。
`index` 是沿最后一维排序、非负且不越界的 int64 segment ID。
归约维为 `index.dim() - 1`，src rank 为 1–8；index 前部维度可为 1
或对应 src 维度，尾部 feature 维共享 index。

支持 float16、float32、bfloat16、int8、uint8、int32、int64。
可选 `dim_size` 指定输出段数，提供 `out` 时以 out 的对应维度确定段数。
专用 min/max 返回 `(values, indices)`，通用接口只返回 values。
不支持的 reduce（包括 mul）由 Python 接口拒绝。

sum/add 覆盖给定 out；mean 使用 out 初值参与归约后除以段内计数；
min/max 与 out 初值比较。无 out 的空段值为零，有 out 时 mean/min/max
保留空段初值；极值平局选择较后的源坐标。上述 task-specific 行为与
CPU 参考库的非零 out 累加和首次极值规则不完全相同，测试分别检查。

## Host 数据流

1. 校验设备、dtype、rank、广播及 out 元数据，选择输入 dtype 分发编号。
2. 广播并连续化 index，连续化 src。连续视图可能带非零 storage offset，
   因而继续检查实际 src/out 地址的 32 字节对齐；不满足时克隆到对齐暂存。
3. 由 src/index 形状计算 batch、归约长度、尾部 channels、segments 和
   index batch 数，写入 `SegmentCooPlan`。
4. 按路径需要分配段边界指针暂存，复用 PyTorch 当前 NPU stream。
5. 用 `OpCommand::RunOpApi` 排入框架任务队列。lambda 持有 src/index/out/
   arg/ptr Tensor，确保连续化和对齐暂存活到启动处理完成。
6. 调用 Kernel 入口 `SegmentCoo`。若提供的 out 使用了暂存，则将结果
   写回原视图，并返回原 out，保留调用者的存储身份。

## Kernel 组织

`op_kernel/arch35/segment_coo.h` 定义 plan 与入口；
`segment_coo.cpp` 包含实际 index 边界查找、SIMT 归约和启动逻辑。
Host 对应文件位于 `op_host/segment_coo.h/.cpp`。Kernel 入口遵循同仓
`SegmentMaxCsr`、`GatherCoo` 等算子名命名方式。

每次调用都读取真实 index 并查找边界；没有依赖均匀分段输入，也不缓存
之前调用的 index 或输出。实现按 dtype、channels 和坐标范围选择标量或
packed 访存，以及长段 sum/mean 的协作归约。INT64 mean 的特定对齐路径
可直接读取 index，减少中间指针生成。2 的幂坐标分解仅替代可证明等价的
除余运算，其余形状保留精确的通用计算。

FP16/BF16 使用较宽浮点累加；整数归约保持整数路径。INT8/UINT8 mean
保留参考实现的窄化行为；当非空段计数窄化为零时，使用明确的数学均值，
不调用可能发生除零的 CPU 原生参考，也不将该边界标为参考库一致。

## 测试与性能口径

`test/segment_coo/arch35/` 只保留三个 Python 文件：

- `test_segment_coo.py`：接口、数值、边界、异步 stream 与地址对齐用例，
  以参数化区分 shape/dtype/reduce；原用例的断言与参数矩阵保留。两处
  非连续输入先传输连续底层数据，再在 NPU 上创建视图并断言非连续性，
  避免 CPU 转置视图直接转换引发内部格式告警；该告警按测试失败处理。
- `golden.py`：真实 CPU `torch_scatter` 参考入口与可比较语义范围。
- `benchmark_segment_coo.py`：保持原题形状、输入与计时逻辑的日志版本。
  正式验证另用附件中保留原字节的 `benchmark_supplied.py`。

完整性能分母为四种 reduce × 29 组 shape × 四种 dtype，即 464 项。
warmup=20、iteration=100，逐项使用任务书 GPU 基准/NPU 延迟 ≥ 0.45。
原脚本输出到 0.001 ms，外部报告检查使用耗时上端点 `printed_ms + 0.0005`
计算保守比值。原始日志、任务书基线与检查脚本随验证附件交付，不把
诊断脚本、自测报告或历史二进制放入产品代码树。

CPU 参考库的支持范围和 task-specific 语义由各自用例说明；未把二者
混为完全等价。Host 校验元数据，不遍历 NPU index 内容，调用者需满足
index 排序和范围前提。Ascend 950 为本版本目标，其他架构不导出本算子。

## AI 参与说明

原实现及证据整理使用过 ZCode/GLM-5.3 与 Codex。本次产品接入、测试合并
及地址对齐保护使用 GPT-6（Codex）辅助，提交人负责检查代码和真机结果。
