# FlashInfer-Gaudi native backend qualification

The native project targets Gaudi2 and SynapseAI 1.24.1. Triton development is
paused. A torch Tensor or an outer `torch.compile` is allowed; internal
PyTorch numerical decomposition does not count as a native operation.

## Current delivery, not a whole-library completion claim

- Whole-operation strict guards cover the current public GDN compute APIs,
  their private prologues, and enabled vLLM GDN adapter entries.
- `auto` preserves the established serving graph tactics. A library loading
  successfully and an environment override cannot promote a prototype.
- `silu_and_mul` has an AOT TPC-C kernel and public CustomOp registration.
  It has no torch activation/multiplication, input casts, or hidden copies.
  The final multiplication retains the established BF16 activation boundary;
  sigmoid evaluation uses FP32.
- Native SiLU supports nonempty contiguous rank-2 BF16 HPU input `[B,2*D]`,
  D divisible by 128, inference only, and `out=None`. Preallocated outputs,
  arbitrary strides, and other dtypes are not implemented natively yet.
- GDN's TPC prototype still has a torch prologue. Strict GDN calls reject it;
  the numerical prototype is retained for explicitly labeled research.
- `get_capabilities()['native_coverage']` separates implementation,
  correctness, performance and production status. Planned domains are not
  fake APIs. The inventory covers the current Python surface, not every
  upstream inference symbol.

No native tactic in this delivery is enabled automatically in vLLM.

## Experimental gated activation / row-FP8 quantization

`flashinfer_gaudi.silu_and_mul_quant` accepts contiguous inference BF16 HPU
input `[B,2D]`, B positive and at most INT32_MAX, with D divisible by 128 in
`[256,17408]`. It returns E4M3 values `[B,D]` and FP32 **dequantization** scales
`[B,1]`. This is a Gaudi-specific extension, not upstream FlashInfer's NVFP4
or block-quantization API.

The native recipe contains a logical split, Synapse `silu_fwd_bf16`, and an
AOT TPC kernel fusing multiplication, absmax, scale, reciprocal, normalization
and FP8 packing. It is not a monolithic custom SiLU kernel: vendor SiLU keeps
the established HPU unary approximation. There is no PyTorch numerical
prologue, nested launch, input conversion or output copy in the native path.

The contract preserves BF16 rounding boundaries in
`a = silu(gate) * up`, `s = (amax(abs(a)) + 1e-8) / 240`, and `1/s`, followed
by HPU E4M3 conversion with range +/-240. The scale is not rounded to a power
of two. Zero rows retain the positive epsilon scale. Explicit lane reduction
and a BF16-qualified reciprocal avoid LUT use, allowing the intermediate row
to stay in the compiler-approved VLM budget even at the maximum width.

Build both libraries with the commands in the next section, then load the
ABI-locked adapter **before** compiling:

```python
import torch
from flashinfer_gaudi import load_native_extensions, set_backend_policy, silu_and_mul_quant
from flashinfer_gaudi._bridge import load_bridge_adapter

load_native_extensions()
load_bridge_adapter()
set_backend_policy("native")
fn = torch.compile(silu_and_mul_quant, backend="hpu_backend", fullgraph=True, dynamic=False)
quantized, dequant_scale = fn(x)
```

`native` and `bridge` fail closed; `public` rejects the private-ABI adapter.
`auto` and `pytorch` keep the reference. The CPU reference is for contract
testing, not bitwise equivalence with HPU unary approximations. No serving
dispatch is changed. In particular, this does not replace preserved model
block-FP8 weight scales with channel scales or claim an end-to-end speedup
for a model that does not use this row-activation quantization path.

```bash
python tools/benchmark_flashinfer_native_quant.py \
  --output /tmp/native-silu-quant.json --trace
```

This compares the complete compiled candidate against both the ordinary HPU
chain and the `calculate_scale_for_cast` CGUID chain. Both gates must pass.
Reports include artifact hashes, shape reentry, scale/reconstruction checks,
native-only FX audit and the actual device kernels. Synapse may name vendor
SiLU as a generated `fused_kernel`; that is not evidence of an extra custom
activation or of a single-kernel recipe. Timing includes host submission gaps.
The candidate remains unqualified and is not automatically promoted.

The initial complete matrix passed correctness but not the joint performance
gate. At some larger shapes the ordinary compiled HPU chain is already one
fused TPC kernel; the candidate still uses two stages. Further work must
qualify single-kernel numerical equivalence and reduce inter-stage scheduling
cost, not infer a gain from kernel count or isolated profiler durations.

## Experimental mixed-engine plan

### Exact-scale block-weight linear

`flashinfer_gaudi.block_fp8_linear(x, weight, scale)` implements the preserved
block-weight route: native TPC weight dequantization followed by BF16 MME GEMM
in one Synapse recipe. This is **not FP8 MME** and adds no activation
quantization. The TPC kernel uses load-pipe 8-to-16 unpack before conversion
to avoid generic linear-conversion lane shuffles.

The v1 contract is contiguous inference BF16 `x[M,K]`, E4M3 `weight[N,K]` and
FP32 `scale[N/128,K/128]`, with positive dimensions and N/K divisible by 128.
Dimensions must fit INT32. Weight values follow the existing Gaudi2 loader's
encoding/scale adjustment; this API does not reinterpret an unadjusted CUDA
checkpoint. Weight and scale are each rounded to BF16 before BF16
multiplication, exactly as in the existing block-weight dequantization path.
The result is BF16 `[M,N]`; there is no bias, padding/unpadding, arbitrary
stride, autograd, output mutation or persistent BF16-weight cache in v1.

`block_fp8_dequant(weight, scale)` exposes the same TPC conversion separately
for verification. Use the complete linear chain for performance qualification,
not an isolated conversion timing. Both functions require explicit
`load_bridge_adapter()` before compiling their native path, use the same
private adapter/version/hash rules below, and never promote themselves in
`auto`. `native`/`bridge` fail closed; `public` rejects the private adapter.

```python
from flashinfer_gaudi import block_fp8_linear, set_backend_policy
from flashinfer_gaudi._bridge import load_bridge_adapter

load_bridge_adapter()
set_backend_policy("native")
linear = torch.compile(block_fp8_linear, backend="hpu_backend", fullgraph=True, dynamic=False)
y = linear(x, weight, scale)
```

```bash
python tools/benchmark_flashinfer_native_block_linear.py \
  --output /tmp/native-block-linear.json --trace
```

The default matrix covers MLP TP1/TP2 local weight shapes and decode/prefill
batch sizes; these are weight-free operator tests, not distributed or model
benchmarks. The strong reference is compiled existing vLLM block-weight
dequantization plus GEMM, excluding model metadata `.item()` overhead on both
sides. A predequantized BF16 GEMM is also measured as a diagnostic: it uses a
different persistent-memory contract and is not the promotion baseline.
Reports retain failed cases/workers, native FX/CPU/engine/recipe-launch audits,
and source/ELF hashes. No new vLLM linear dispatch is enabled by this delivery.

Offline qualification currently shows a limited win for the down-projection
shape, not the full matrix. Gate/up cases still regress. Keep dispatch
unchanged until matching real-weight, model integration and end-to-end
qualification; shape-only synthetic evidence does not authorize a global
native linear override.

### BF16 GEMM / gated activation

`flashinfer_gaudi.gemm.GemmSiluPlan` adds a BF16 MME GEMM followed by the
native TPC SiLU-multiply kernel **inside one Synapse graph**. Inputs are
contiguous BF16 `x[M,K]`, `weight[K,2*D]`, with positive dimensions and D
divisible by 128. The GEMM result is BF16, SiLU is rounded to BF16 before
the multiplication, and no quantization or scale semantics are changed.
This is a Gaudi-specific compound primitive, not a claim of upstream
FlashInfer GEMM API completeness or FP8 model acceleration.

The optional adapter uses private Bridge headers and is restricted to Bridge
`1.24.1.482`, Synapse `1.24.1` and eager mode. Build against the source and
generated dependency tree matching the installed Bridge:

```bash
python tools/build_flashinfer_gaudi.py
PT_HPU_LAZY_MODE=0 python tools/build_flashinfer_gaudi_bridge.py \
  --bridge-source "$GAUDI_PYTORCH_BRIDGE_SOURCE" \
  --bridge-build "$GAUDI_BRIDGE_BUILD"
```

The builder reuses Bridge's `_deps` checkouts and generated headers. It does
not modify or rebuild Bridge. This adapter is source-build-only; ordinary
wheels exclude its private-ABI binary and local manifest.
`bridge_artifact_v1.json` binds the adapter,
TPC kernel database, Synapse library and both linked Bridge libraries by
SHA256, together with the exact torch/Bridge/Synapse versions and C++ ABI.
The loader rejects mismatches before loading adapter code. Rebuilding the
TPC database or changing the runtime requires rebuilding the adapter
artifact and starting a fresh process. This is compatibility checking for
trusted local builds, not a signed-artifact security boundary.

```python
from flashinfer_gaudi.gemm import GemmSiluPlan

plan = GemmSiluPlan(m, k, d)
# Allocate x, weight and out on HPU before the hot path.
plan.run(x, weight, out=out)
# For integration in an allocating compiled graph:
fn = torch.compile(plan.functional_op, backend="hpu_backend",
                   fullgraph=True, dynamic=False)
result = fn(x, weight)
```

`out` is bound directly through Bridge's native out interface, with no
post-compute `copy_`. Invalid shapes, types, layouts, autograd inputs and
shared-storage aliases reject before submission. `run` is an eager native
recipe-replay interface; do not compile its mutable out variant. The
functional registered op is the supported fullgraph compilation boundary.

Plan construction verifies the adapter; the first run compiles the recipe.
Plans hold no tensors or mutable workspace. Independent output buffers can
be submitted on different streams; callers own cross-stream ordering and
must not reuse the same output concurrently. Internal temporaries and scratch
are owned by Synapse. Caller-provided workspaces, guaranteed SRAM placement,
general graph artifacts, and recipe serialization are not implemented yet.
`plan.artifact` serializes only this static graph specification and its
adapter identity, not a warmed recipe, tensor pointer, or model weights.

Measure both compiled functional and native out replay against the same
compiled allocating HPU reference:

```bash
python tools/benchmark_flashinfer_native_gemm.py \
  --output /tmp/native-gemm-silu.json --trace
```

The output mode is labeled explicitly: its preallocated output contract is
different from the allocating baseline. It must not be reported as isolated
MME/TPC compute acceleration. Neither mode is auto-promoted by a report.

The first mixed-chain qualification did not meet the performance gate.
Reference traces already contain MME GEMM plus a fused TPC epilogue; simply
replacing that epilogue does not remove a graph launch or an engine stage.
Prioritize activation/quant and residual/norm/quant fusion with unchanged
scale semantics, and separately reduce out-replay submission overhead.

## Experimental residual / RMSNorm / row-FP8 fusion

`flashinfer_gaudi.fused_add_rmsnorm_quant(x, residual, weight, eps=1e-6)`
is a functional Gaudi extension, not the upstream in-place or block-scale API.
It returns four independently allocated outputs: E4M3 values `[B,D]`, FP32 row
scales `[B,1]`, BF16 normalized values `[B,D]`, and BF16 residual sums `[B,D]`.
Inputs are read-only; input/residual aliases are allowed. Native support is
limited to contiguous inference BF16 inputs and weights, positive rows within
signed 32-bit index range, and widths from 256 to 17408 divisible by 128.
The epsilon must be a positive normal FP32 value.

The public CustomOp lowers to one AOT TPC kernel, without a numerical torch
prologue or the private mixed-graph Bridge adapter. Four independent input
accumulators and a balanced row reduction shorten dependency chains. Weighted
BF16 values are cached in VLM; direct BF16-to-FP8 conversion preserves the
BF16 product rounding while avoiding repeated FP32 conversion.
Widths through 8192 use the compiler's lookup rsqrt with a bounded 16-KiB
row cache. Wider rows use the lookup-free variant with a larger cache. The
host instantiation selects the ELF by width; both keep the same GUID and
scalar ABI. Host and hardware tests cover the resource boundary.

The normalization preserves the current HPU vendor's BF16 weight-product
boundary. Row scales use Gaudi2's finite E4M3 range of +/-240. The explicit
`scale_mode` selects `bf16_reciprocal` (BF16-rounded reciprocal of 240) or
`fp32_divide` (FP32 reciprocal), with BF16-rounded scale and inverse scale.
Vendor, ordinary compiled, and CGUID graphs can differ in these intermediate
precision boundaries, especially when embedded in a GEMM consumer. Do not
infer the correct mode from shape, or treat matching isolated outputs as
proof that a full projection or model is numerically equivalent.

Use `native` or `public` policy to request the native implementation, and load
the extension before compilation. `auto` and `pytorch` remain reference-only;
no model dispatch is changed by this candidate or by a benchmark report.

```bash
FLASHINFER_GAUDI_RUN_HARDWARE_TESTS=1 python -m pytest -q \
  tests/unit_tests/ops/test_flashinfer_norm_quant_hardware.py
python tools/benchmark_flashinfer_native_norm.py \
  --output /tmp/native-norm-quant.json --sessions 3 --trace
```

The hardware tests cover both scale modes, all output lanes, tails, shape
reentry, input updates, read-only aliases, output non-aliasing, non-default
streams, and the compiled reference's BF16 scale boundary. Qualification
requires a first-session device trace, audited native FX graphs in every
session, unchanged matching source/library hashes, and disabled eager
fallback. Omitting `--trace` permits screening but cannot qualify a shape.
Each shape must beat the ordinary vendor, CGUID, and formula baselines under
the existing paired performance gate. Full FP8-GEMM consumer, model-quality,
and end-to-end qualification remain separate requirements.

## Complete FP8 projections and model integration

An isolated norm/quant win is not a projection win. The projection harness
includes residual addition, normalization, row quantization and the public
`hpu.fp8_gemm_v2` consumer. Ordinary serving quantization uses the exact
`max(dim).values` and reciprocal expressions; CGUID and formula comparisons
remain separate numerical contracts. An incompatible baseline stays failed
in the report even when another baseline qualifies.

```bash
python tools/benchmark_flashinfer_native_projection.py \
  --output /tmp/native-projection.json --sessions 3 --trace
```

Per-shape fixtures are independent of traversal order. Qualification requires
three distinct process identities, matching source/library/input hashes,
unchanged inputs, shape reentry, native FX dataflow and device trace evidence.
The projection is not one kernel: the public MME consumer can also generate
an FP32 device kernel. All stages and recipe launches are reported.

The real-model validation tools provide three deliberately separate steps:

```bash
# Use the same channel-FP8, CGUID and grouped-compilation settings in both arms.
export VLLM_HPU_FORCE_CHANNEL_FP8=true
export VLLM_HPU_CGUID_DYNAMIC_QUANT=1
export VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS=32
export VLLM_HPU_QWEN3_COMPILE_LAYER_GROUP_SIZE=8
export VLLM_SKIP_WARMUP=true
python tools/capture_flashinfer_model_projection.py \
  --model MODEL --output /tmp/model-projection-inputs
python tools/replay_flashinfer_model_projection.py \
  --capture /tmp/model-projection-inputs \
  --output /tmp/model-projection-replay.json --sessions 3 --trace
python tools/benchmark_flashinfer_model_projection.py \
  --model MODEL --output /tmp/model-projection-aba --trace
```

Capture uses named worker-control RPCs, without enabling pickle RPCs. It
loads the actual checkpoint and captures postprocessed FP8 weights plus
residual/normalization/projection tensors. Hooks do not change model outputs
and are removed afterwards. Capture runs eager and is **not** a performance
benchmark. The requested decode buckets must actually be observed; request
concurrency alone does not establish the operator batch. Compiled replay
checks its baselines separately from the originating eager model output.
Differences between eager and compiled outputs are retained, not explained
away by changing tolerances.

`vllm_gaudi.ops.flashinfer_projection_fusion` is an experimental graph adapter,
not a new stable FlashInfer API. It has no automatic registration: an empty
shape request is a no-op. The trial tool explicitly requests only the qualified
CGUID B=8, K=5120, N=34816 pattern. Other shapes, ordinary quantization,
unknown layouts, extra consumers and externally observable mutation remain
unchanged. Identity views are accepted only with matching contiguous metadata.
Bridge-created `add_` reuse is replaced only after proving a graph-private
owner with no escaping aliases or future reads; the obsolete update is then
explicitly erased because FX DCE preserves impure nodes.
Only inserted nodes receive fresh, non-aliasing metadata. Do not rerun
whole-graph FakeTensor propagation after Bridge's HPU layout rewrites: some
resulting BMM graphs are valid for HPU lowering but no longer valid aten
programs for generic shape execution. Training/backward contexts are skipped.

The model A/B/A screen uses fresh processes and checks source stability,
actual native-kernel execution, token IDs and baseline-return drift. Profiling
is limited to five B=8 decode model forwards, outside request timings. Device
kernel activity records are not logical invocation counts. Match the existing
serving eager-fallback policy in both model arms (Bridge normally permits
metadata views); this is separate from the projection tests, which disable
eager fallback. A small warm-request screen is not general LLM acceptance,
and neither a report nor the adapter enables a production default.
The harness now requires at least two symmetric warmup rounds and rejects
timed rounds deviating more than ten percent from their worker median. Earlier
single-warmup screens exposed residual cold/recompile overhead; retain those
rounds as diagnostics, not as steady-state speedup evidence. Forced-length
generation ignores EOS for timing, so raw token disagreement must also be
examined before the first EOS; it is not itself a semantic quality score.

## Build and execute

```bash
python tools/build_flashinfer_gaudi.py
FLASHINFER_GAUDI_BACKEND=native python your_native_test.py
```

Load the library before compiling a call. Compile the public
`flashinfer_gaudi.silu_and_mul` with `backend="hpu_backend", fullgraph=True,
dynamic=False`; it lowers to `custom_op.flashinfer_gaudi_silu_and_mul`, then
`flashinfer_gaudi_silu_and_mul_bf16_gaudi2` in the TPC kernel database.
CUDA programmatic dependent launch is explicitly unsupported.

Run the opt-in hardware tests with `FLASHINFER_GAUDI_RUN_HARDWARE_TESTS=1`.
They cover compiled shape reentry, changing input contents, unsupported
contracts without output mutation, and submission on different HPU streams.
This is compiled-recipe replay, not the lazy-only `HPUGraph` capture API.

Serving policy must be fixed before compilation. Changing an already compiled
callable's policy or replacing registered libraries requires a fresh callable
or process; the one-shot native resolver is constant during compilation.

## Evidence and promotion

```bash
python tools/benchmark_flashinfer_native_activation.py \
  --output /tmp/native-activation.json --trace
```

The runner executes three fresh process sessions, each with 15 interleaved
reference/candidate waves per shape against the compiled HPU implementation.
It uses identical seeded inputs, checks ordinary and widened input ranges,
and revisits the first shape after the full sweep. A backend audit rejects
native FX graphs containing anything other than the registered native op.
Optional traces show the actual TPC GUID. Reports hash both the registration
library and the ELF-containing kernel database.

The offline gate requires complete-operation correctness, three sessions,
15 paired waves per session, median speedup >=1.05, and a session-clustered
paired bootstrap 95% lower bound above 1. Timings include queue submission
gaps and must not be described as isolated TPC instruction time. Failed
correctness or reentry suppresses qualification. Failed workers cannot be
omitted from an aggregate.

The runner never updates production dispatch. Model quality, integration,
ABI and concurrency qualification are separate requirements. Microbenchmarks
are not end-to-end model speedups. Existing graph tactics remain the baseline.

## Remaining implementation sequence

1. Complete the pinned upstream API inventory, execution reporting, ABI-bound
   AOT artifacts and general mixed-engine adapter. Extend the first static
   GEMM-SiLU plan to caller workspaces, recipe serialization and broader
   concurrency qualification.
2. Qualify residual/norm/quant inside complete models; finish gated
   activation/quant and QK/RoPE/KV fusion, native GDN decode/MTP/prefill,
   and KDA/Mamba/state contracts.
3. Add MME linear/GEMM and exact-scale block quantization pipelines; then
   paged/variable-length GQA, dense, MLA, sparse and cascade attention.
4. Add MoE dispatch/grouped compute/combine, device sampling and speculative
   state handling, followed by HCCL scheduling and overlap.

The first model matrix is Qwen3.8-27B-FP8; DeepSeek-V4 and weight-free
Llama/Mixtral-style matrices cover the remaining domains. Keep scale semantics
and FP32 recurrent state unchanged. FP4 conversion is not advertised as native
FP4 matrix hardware. Each domain requires reproducible acceleration; report
end-to-end model metrics without a fixed percentage completion target.
