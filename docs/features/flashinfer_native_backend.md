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
   AOT artifacts and mixed-engine adapter. Add plan/run workspaces,
   preallocated outputs and concurrency qualification.
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
