# Scatter 测试说明

该目录包含三类可复现验证：

- `test_scatter.py`：接口签名、广播、`out`、空张量、非连续输入、全部 dtype 与六种 reduce 的 CPU 语义回归；
- `test_scatter_official.py`：任务书同级 `self_test_case/ops-gnn` 的完整 Scatter 功能用例，并包含整数归约、min/max `arg_out` 等定向回归；
- `benchmark_scatter.py`：任务书规定的 5 个 shape、float32/float16、sum/mean/min/max 共 40 个性能点。

## 构建

```bash
source ${ASCEND_HOME_PATH}/bin/setenv.bash
pip install --no-build-isolation -e .
```

## 功能与精度

```bash
pytest test/scatter/test_scatter.py -v
pytest test/scatter/test_scatter_official.py -v
```

## 性能

```bash
python test/scatter/benchmark_scatter.py \
  --reductions sum mean min max \
  --dtypes float32 float16 \
  --warmup 50 --repeat 200 \
  --threshold 0.6 \
  --csv /tmp/scatter_benchmark.csv \
  --json /tmp/scatter_benchmark.json
```

脚本使用 NPU Event 计时，以任务书列出的 A100 延迟为标杆，逐项计算 `A100_ms / NPU_ms`；任何一项低于 `0.6` 时以非零状态退出。

官方功能矩阵中的所有 L1 dtype/reduce 组合均为必测项，不允许通过
`pytest.skip` 排除整数 `mean/min/max`。提交前应保存完整 pytest 与性能
日志作为验收报告。
