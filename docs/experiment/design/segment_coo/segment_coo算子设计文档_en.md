# Ascend 950 segment_coo design

This document describes the product interface, data flow and boundaries. The
community design review is [PR #1485](https://gitcode.com/cann/cann-ops-competitions/pull/1485).
The integration originates from acceptance-repository commit
`0cdb291cea227d6c5be03ed8b3cf85c4e9d4e9f9`. Validation results for the product
revision are provided in the product PR's corresponding evidence archive.

## Interface and input contract

The API provides `segment_coo`, `segment_sum_coo`, `segment_add_coo`,
`segment_mean_coo`, `segment_min_coo` and `segment_max_coo`.
`index` contains sorted, nonnegative, in-range int64 segment IDs along its final
dimension. The reduction dimension is `index.dim() - 1`; source rank is 1–8.
Leading index dimensions may be 1 or match the source dimensions. Trailing
feature dimensions share the index.

Supported types are float16, float32, bfloat16, int8, uint8, int32 and int64.
`dim_size` optionally specifies the number of output segments. When `out` is
provided, its corresponding dimension determines that number. Dedicated min/max
functions return `(values, indices)`; the generic interface returns values only.
Unsupported reductions, including mul, are rejected in Python.

Sum/add overwrite a supplied output. Mean includes its initial value before
dividing by the segment count. Min/max compare against the initial output.
Empty segments without `out` are zero; mean/min/max preserve supplied values in
empty segments. Extremum ties choose the later source position. These task
semantics differ from upstream CPU nonzero-output accumulation and first-tie
behavior; separate tests explicitly verify those differences.

## Host data flow

1. Validate device, dtype, rank, broadcasting and output metadata; select a dtype ID.
2. Broadcast and materialize the index, and make the source contiguous. Contiguous
   views can have storage offsets, so check actual source/output addresses for
   32-byte alignment and clone unaligned buffers.
3. Compute batches, reduction length, trailing channels, segments and index
   batches; populate `SegmentCooPlan`.
4. Allocate segment-boundary storage when the selected path needs it and use
   the current PyTorch NPU stream.
5. Enqueue through `OpCommand::RunOpApi`. The lambda captures source, index,
   output, argument and boundary tensors to retain temporary storage until launch.
6. Invoke `SegmentCoo`. Copy temporary output back into a supplied view and return
   the original output, preserving its storage identity.

## Kernel organization

`op_kernel/arch35/segment_coo.h` declares the plan and entry point;
`segment_coo.cpp` contains boundary searches, SIMT reductions and launch selection.
Host declarations and definitions are in `op_host/segment_coo.h/.cpp`.
The entry-point naming follows neighboring `SegmentMaxCsr` and `GatherCoo` operators.

Every call reads the actual index and searches its boundaries. No path assumes
uniform segments or caches previous indices/results. Dtype, channels and address
range select scalar or packed accesses and cooperative sum/mean for long segments.
The aligned INT64 mean path can search indices directly without intermediate
boundary construction. Power-of-two coordinates use equivalent unsigned shifts
and masks; all other shapes retain general coordinate calculations.

FP16/BF16 accumulate in wider floating point; integer reductions retain integer
arithmetic. INT8/UINT8 mean follows upstream narrowing. When a nonempty count
narrows to zero, the implementation defines a mathematical mean and avoids the
CPU reference's division by zero. That boundary is not claimed as CPU equivalence.

## Tests and performance

`test/segment_coo/arch35/` contains three Python files:

- `test_segment_coo.py`: parameterized interface, numerical, boundary, asynchronous
  stream and alignment coverage. Original parameter matrices and assertions are
  retained. Two strided-input cases transfer contiguous backing storage before
  constructing and checking NPU views; internal-format warnings fail those tests.
- `golden.py`: the actual CPU `torch_scatter` reference and its comparable domain.
- `benchmark_segment_coo.py`: a logging version preserving the supplied shapes,
  input construction and timing. Formal validation uses the byte-identical
  `benchmark_supplied.py` retained separately in the evidence archive.

The complete performance matrix has four reductions × 29 shapes × four dtypes,
or 464 cases, with 20 warmups and 100 iterations. Each case must reach the task's
GPU-baseline/NPU-latency ratio of 0.45. The supplied script prints 0.001 ms precision;
the external checker uses `printed_ms + 0.0005` to compute a conservative ratio.
Raw logs, task baselines and the checker are included in validation evidence;
diagnostic scripts, self-test reports and historical binaries stay outside the
product source tree.

CPU-reference coverage and task-specific behavior are documented separately.
Host validation checks metadata without traversing NPU indices; callers must
satisfy sortedness and range prerequisites. This revision targets Ascend 950 and
does not export these operators on other architectures.

## AI assistance

The original implementation and evidence preparation used ZCode/GLM-5.3 and
Codex. GPT-6 (Codex) assisted with this integration, test consolidation and
alignment protection. The contributor is responsible for reviewing the code
and device results.
