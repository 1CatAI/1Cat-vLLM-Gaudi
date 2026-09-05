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

## Experimental mixed-engine plan

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
2. Implement residual/norm/quant, gated activation/quant, QK/RoPE/KV fusion;
   complete native GDN decode/MTP/prefill and KDA/Mamba/state contracts.
3. Add MME linear/GEMM and exact-scale block quantization pipelines; then
   paged/variable-length GQA, dense, MLA, sparse and cascade attention.
4. Add MoE dispatch/grouped compute/combine, device sampling and speculative
   state handling, followed by HCCL scheduling and overlap.

The first model matrix is Qwen3.8-27B-FP8; DeepSeek-V4 and weight-free
Llama/Mixtral-style matrices cover the remaining domains. Keep scale semantics
and FP32 recurrent state unchanged. FP4 conversion is not advertised as native
FP4 matrix hardware. Each domain requires reproducible acceleration; report
end-to-end model metrics without a fixed percentage completion target.
