# segment_csr design (Ascend 950PR / arch35)

## 1. Requirements and acceptance

Implement `segment_csr` and the sum/add/mean/min/max helpers specified by the
task book. The generic entry returns values; min/max helpers also return int64
indices of the first extremum. Empty segments produce zero and use the input
reduction length M as the index. Inputs support float16, float32, bfloat16,
int8, uint8, int32, and int64; pointers are int64.

The performance matrix has 29 shapes, four performance dtypes
(float16/float32/int32/int64), and four reductions, totaling 464 cases. Each
case must pass correctness and achieve `baseline time / NPU time >= 0.45`.
BF16/int8/uint8 require functional validation. Reduction order is deterministic,
without atomic accumulation or multiple writers for an output.

This design describes register and generic paths. Acceptance results must refer
to the current build; historical correctness logs and partial timings are not
full acceptance evidence.

## 2. Host dimensions and output contract

```text
dim  = indptr.dim() - 1
E1   = prod(src.shape[:dim])
M    = src.shape[dim]
K    = prod(src.shape[dim + 1:])
nSeg = indptr.shape[-1] - 1
src[e, j, k] address = ((e * M) + j) * K + k
```

When all leading pointer dimensions are one, batches share a pointer sequence;
otherwise expand and materialize contiguous per-batch pointers. Noncontiguous
source and pointers are made contiguous on the current device. Output shape
replaces the reduction dimension with nSeg. A supplied output must match dtype,
device, and shape. Noncontiguous output is copied back from a contiguous buffer.
Empty source initializes an omitted output to zero or preserves a supplied
output. Min/max always return two tensors, including zero segments, batches,
and channels.

Validate dimension relationships before accessing shapes. Use the source
device's current NPU stream without creating temporary streams or host stream
synchronization. Importing the module does not launch kernels. The production
entry does not require repeated first launches. Contiguous inputs can reuse
storage; noncontiguous inputs, broadcast materialization, and output writeback
perform actual device copies.

## 3. Two execution paths

Route to AIV register kernels when all of the following hold:

- dtype is float32, float16, int32, or int64;
- E1 equals one with shared pointers; M is nonzero and at most INT32_MAX;
- row bytes are a multiple of 32 and K is at most 1024;
- int64 min/max additionally require row bytes to be a multiple of 256.

Other inputs use SIMT, including BF16/int8/uint8, batch broadcast, per-batch
pointers, irregular K, and shapes outside register routing limits. Host code
handles empty source. Routing limits do not require padding general input.

## 4. AIV tiling and pointer planning

Each core owns consecutive output segments. Obtain core count using
`GetCoreNumAiv()` and UB capacity using `GetCoreMemSize(UB)`:

```text
tileBytes = min(64 KiB, floor((UB - 80 KiB) / 2 / 256) * 256)
maxRows = tileBytes / (K * sizeof(T))
maxSegments = 2048 / K
```

Create TPipe at kernel entry and pass it to the computation class.

| Buffer | Purpose |
| --- | --- |
| Two input queue slots | tileBytes + 512B each for copy/compute overlap |
| saved | Cross-tile accumulator; float32 for half inputs |
| savedIndex | Absolute int32 row indices across tiles |
| result / argResult | Up to 2048 values and int64 indices |
| ptrCache | Up to 2048 int64 CSR boundaries |

Loads have additional readable tail space; stores always use valid lane masks.
Use DataCopyPad between GM and UB. TQue controls slot production, consumption,
and reuse. V-to-MTE3 events protect result writeback; MTE3-to-V events protect
output-buffer reuse.

### Detecting uniform segments

Validate the complete sequence as `base + index * length` only when all
boundaries assigned to the core fit in cache and are valid. Register checks
cover both 32-bit halves of each int64 boundary, including partial blocks of
fewer than 64 entries, and accumulate error flags.

After successful validation, planning uses the proven arithmetic sequence.
Otherwise load and clip actual boundaries individually and group only
consecutive equal-length segments. Neither endpoints nor shape alone prove
uniformity. Split long segments across tiles with first/last state. Empty
segments do not read input but write zero and index M. Segment groups and tiles
respect UB budgets and output capacity.

## 5. Register reductions

Process channel blocks with values in RegTensor rather than repeatedly moving
through UB. Long tiles use four fixed row chains and merge in a fixed order;
short tiles use smaller loops. Eligible complete K=32 segments can pack adjacent
rows into 64 float32/int32 lanes and merge channel halves in registers.

`segment_csr_short_sum.h` specializes complete segments of 2/4/8 rows for
K=64/128/256 and all four performance dtypes. Each iteration handles 64 output
elements with compile-time row expansion. Floating mean rounds to output dtype
before exact scaling by 0.5/0.25/0.125. Integer mean uses shifts and a negative
bias for exact truncation toward zero. Other lengths use generic reduction and
division.

`segment_csr_narrow_half_sum.h` handles complete FP16 K=32 segments with
16/32/64 rows. A float32 register holds two rows. Four accumulator chains start
from input values; loads and additions expand at compile time before channel
reordering and merging. Sum accumulates in float32. Mean rounds the sum to half
before the matching power-of-two scaling. Other shapes and continuation tiles
use the generic path.

Min/max merges carry values and original row indices. Strict improvements
replace the value; ties choose the smaller row index. The rule applies to
four-chain merges, packed channels, and cross-tile continuation.

### FP16 extrema

`segment_csr_half_extrema.h` handles complete, nonempty, packable segments.
K=32/64 pack 4/2 rows into 128 half lanes; K=128/256 use full vectors. Each chain
uses strict comparisons, and merging explicitly preserves the first row on
ties. Segment-relative indices are uint16, so routing limits length to 65535.
Output adds the absolute row base and widens to int64. The path only compares
and selects existing half values. Other shapes promote values to float32 for
generic comparisons; FP16 sums always accumulate in float32.

### INT64 extrema

`segment_csr_int64_extrema.h` uses one sequential chain. DIST_DINTLV_B32 splits
int64 values into high signed-int32 and low uint32 registers. Lexicographic
comparison preserves the full int64 range without floating conversion. Equal
values retain the earlier index. Continuation loads saved state, begins at row
zero of the new tile, and either saves intermediate state or writes interleaved
int64 values and indices. This avoids generic four-chain 64-bit merge overhead.

## 6. Numerical semantics and SIMT fallback

Floating mean follows the real CPU torch_scatter order: reduce; round the sum
to source dtype; convert segment length to source dtype; promote both to
float32; divide; and convert output with the matching rounding rule. Therefore
FP16 length 2049 rounds to 2048, and 65520 rounds to infinity.
Integer mean wraps sum and length in source dtype and divides exactly toward
zero without a float32 intermediate. Positive power-of-two lengths allow a
negative bias and arithmetic right shift. If an int8/uint8 divisor wraps to
zero, CPU division is undefined and the NPU implementation returns zero.

The supplied Python golden differs from the compiled CPU library for large
integer mean and empty-segment indices. Direct CPU-extension tests check these
semantics and reports preserve the differences.

A historical ATK comparison found a CPU torch_scatter 2.1.2 uint8 max boundary
case: src=[1,7,2,0], indptr=[0,0,3,4] yields arg=[4,1,1], placing the final index
outside its segment. The NPU first-index rule gives [4,1,3]. This was submitted
for design review. The supplementary comparison was 34/35; the remaining case
was not counted as passed. ATK uses source dtype for the CPU call to avoid
changing FP16/BF16 mean semantics through promotion.

SIMT assigns an output cell or 16B channel packet to each thread. Eligible
aligned types load 16B packets; other cases use scalar loads. FP16/BF16 promote
to float32; integers accumulate in int64. FP16 encoding handles subnormal,
infinity, overflow, and round-to-nearest-even. Floating mean rounds sum and
length to source dtype before float32 division, matching register paths.
Independent regressions cover numerical boundaries. NaN extrema propagation
is outside the current contract. Kernel clipping is not full validation of
invalid CSR parameters.

## 7. Validation and build boundaries

Load CANN and build the full repository with
`python3 setup.py build_ext --inplace`. CSR uses the main `_pybind.so` and
`libopsgnn_npu_kernel.so`; a standalone development library cannot replace
integration validation. See the [development guide](../../../en/development_guide.md)
for build and test commands.

Python calls `_pybind.segment_csr`. The product test for TC-11 compares NPU CSR
with an independent pure-PyTorch CPU COO reducer using equivalent indices, five
reductions, and empty-segment layouts. The frozen acceptance candidate was also
compared against compiled CPU `torch_scatter.segment_coo`; that per-case evidence
remains in the PR validation attachment and is not a product-test dependency.

For int32 pointers, public segment_max_csr forwards to the legacy kernel,
returns values only, and preserves initial-output maximum behavior. Int64
pointers use the new tuple-returning family. New interfaces keep task-book
signatures. The explicit segment_max_csr_legacy alias accepts historical
optional_out=; the new function retains int32 forwarding through a third
positional argument or out=. The development _C wrapper keeps the same contract.

Task-specific acceptance and profiler drivers referenced below are retained in
the PR evidence attachment under `validation_inputs/`; the product checkout uses
`test/segment_csr/arch35/{test_segment_csr.py,golden.py,benchmark_segment_csr.py}`.

Validation has four parts:

- Functional regressions: official and extended cases, vector paths, output
  contracts, all dtypes, broadcast, irregular/empty/long segments, first indices,
  determinism, internal pointer perturbations, numerical boundaries, and views.
- Real CPU reference: torch_scatter 2.1.2's actual loaded CSR CPU extension,
  checking values and indices and recording binary hashes.
- Task-book acceptance: all 464 cases through run_segment_csr_acceptance.py,
  checking complete values and indices and the 0.45 interface-time threshold.
- Profiler: warmup=5, active=5, custom and same-device composed NPU paths in
  independent processes with matching input and library hashes,
  aggregating Total Time(us) in op_statistic.csv to explain device time.

Reports record source and loaded binary hashes, device, software, timing
settings, and each case outcome. Only a complete matrix for the current build
supports a final validation claim.
