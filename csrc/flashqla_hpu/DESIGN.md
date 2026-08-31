# FlashQLA-style GDN prefill on Gaudi2

Status date: 2026-08-30

This document defines the native-kernel work needed to transfer FlashQLA's
useful ideas to Gaudi2. It does not claim that the CUDA/TileLang kernels can be
compiled for HPU, and it does not put any research kernel on the vLLM runtime
path by default.

## Scope and measured boundary

Target shape:

- Qwen3.8-27B-FP8, TP1, one Gaudi2.
- One sequence with 16,384 prompt tokens.
- 16 Q/K heads, 48 value heads, `DK=DV=128`, chunk size 64.
- Compiled execution with `PT_HPU_LAZY_MODE=0`, never model eager mode.

Accepted endpoint baseline:

- 6,322.428 input tokens/s.
- 2.591409 seconds TTFT.
- 48 GDN layers consume 1.943664 seconds, or 75.0% of endpoint TTFT.
- One sampled GDN layer consumes 40.618 ms.

The measured GDN layer is not one indivisible FlashQLA target:

| Stage | Time/layer | FlashQLA forward can replace it |
| --- | ---: | --- |
| Input norm and projections | 4.229 ms | No |
| Gating and causal convolution | 4.655 ms | No |
| Q/K normalization and preprocessing | 2.102 ms | Optional later fusion |
| Phase A | 5.015 ms | Yes |
| Phase B precompute | 4.647 ms | Yes |
| Phase B recurrent scan | 3.783 ms | Yes |
| State save, output norm/gate/projection | 2.162 ms | State save only |
| Post-attention norm and MLP | 14.026 ms | No |

The direct FlashQLA-equivalent region is therefore about 13.445 ms/layer.
Including Q/K normalization expands the upper bound to 15.547 ms/layer. This
distinction matters: GDN is 75% of TTFT, but the first native FlashQLA op does
not replace all 75%.

The trace attributes about 79.6% of GDN engine time to TPC, 18.7% to MME, and
1.7% to DMA. FP32 TPC kernels alone account for about 61.1% of GDN engine time.
The current path is not HBM-bandwidth-bound and is not FP8-MME-bound. It is
dominated by graph decomposition, FP32 elementwise work, recurrent state
traffic, and synchronization between many TPC and MME nodes.

## What FlashQLA actually does

The reviewed upstream source is QwenLM/FlashQLA commit
`7c7dfe16416ad21b1d03258189fc8d3b8460ae06`. Its forward API runs three major
pieces:

1. Chunk-local cumulative sum of the log gate.
2. A shape-specialized 64x64 KKT solve.
3. A fused persistent recurrent/output kernel.

The H200 result for the exact 27B TP1 16K shape is 0.671 ms for the complete
high-level forward call, versus 0.781 ms for FlashInfer and 1.311 ms for FLA.
That is a 1.95x improvement over FLA, not a 1.95x speedup of the complete
transformer layer.

### Gate-free KKT algebra

For one chunk, let `D = diag(exp(g))`. The gated unit-lower matrix can be
written as:

```text
Lg = D * L0 * D^-1
Lg^-1 = D * L0^-1 * D^-1
```

`L0` depends on `K @ K^T` and beta, but does not contain pairwise gate
exponentials. FlashQLA solves `L0` and applies the diagonal similarity
transform while consuming the result. This reduces SFU and scalar work inside
the solve and gives a regular 64x64 problem.

The KKT kernel is specialized to `chunk=64` and `DK=128`. It computes
`K @ K^T`, forms the strict-lower unit matrix, solves four 16x16 diagonal
blocks, then combines them through 32x32 and 64x64 block products. Matrix
products use Tensor Cores while scalar triangular work stays in fragments and
shared memory.

The first gate-free prototype rematerialized the complete pairwise decay after
the solve, so it captured little of FlashQLA's benefit. The completed graph
pushes `D` and `D^-1` into the two right-hand sides, factors the phase-B local
decay into centered, scaled Q/K inputs:

| Path | Phase A | Complete measured GDN core |
| --- | ---: | ---: |
| Current HPU graph | 4.806 ms | 12.487 ms |
| First gate-free prototype | 4.827 ms | 11.625 ms |
| Complete graph reformulation + deferred scan | 4.024 ms | 9.437 ms |

The completed portable path gains 16.3% in Phase A and 24.4% in the complete
core. It is useful, but the remaining NVIDIA advantage still comes from a
persistent execution schedule and recurrent-state residency.

### Persistent fused forward

FlashQLA launches work per value head and value tile. For each head, a
`128x128` FP32 recurrent state remains in register fragments across the entire
chunk loop. The kernel does not write and reread that state after each chunk.

It uses separate producer and consumer warpgroups:

- Producers asynchronously load Q, K, V, A, gate, and beta.
- One consumer owns and updates recurrent state.
- One consumer computes corrected values and state updates.
- One consumer computes query output and local causal interaction.
- Two shared-memory stages double-buffer the next chunk while the current
  chunk is on Tensor Cores and CUDA scalar/vector units.

For each chunk, the fused dataflow is approximately:

```text
U       = K @ state
W       = V - exp(g) * U
Vd      = Ag @ W
Vprime  = exp(g_last - g) * Vd
state   = exp(g_last) * state + K.T @ Vprime
output  = exp(g) * Q @ old_state + gated(Q @ K.T) @ Vd
```

The essential optimization is not merely fewer Python calls. It is that MME
matrix work, TPC gate work, state update, and asynchronous loads share one
persistent schedule while the recurrent state remains on chip.

### Intra-card context parallelism

FlashQLA can split one long sequence into independently scheduled segments and
reconstruct their initial states using the exponential gate. On Hopper, the
27B TP1 16K case (`H=48`, 256 chunks) satisfies its automatic CP threshold.
This increases CTA occupancy when the value-head count is too small for the
GPU's SM count.

Gaudi2 has 24 TPCs and this model has 48 value heads, so there are already two
head tasks per TPC before sequence splitting. Context parallelism is a later
experiment, not the first porting milestone. It should be enabled only after
the serial persistent kernel is correct and after a head-occupancy trace shows
idle TPCs or MME bubbles.

## Gaudi2 mapping

Gaudi2 has 24 TPCs, two MMEs, and 48 MB of shared on-die SRAM. The architecture
is explicitly designed to overlap MME and TPC execution. The complete FP32
state for this model is about 3 MiB:

```text
48 heads * 128 * 128 * 4 bytes = 3 MiB
```

One head's state is 64 KiB. A two-stage BF16 Q/K/V/A input buffer for one head
is roughly 112 KiB before temporary outputs. Capacity is therefore plausible;
banking, MME layout, lifetime analysis, and synchronization are the actual
problems.

The public TPC CustomOp API cannot implement the required operation. A custom
TPC kernel cannot issue MME matrix operations. This was measured directly:

| Prototype | Candidate | Existing compiled graph | Result |
| --- | ---: | ---: | --- |
| Persistent TPC recurrence | 589.863 ms | 11.545 ms | 51.1x slower |
| TPC exact 16x16 inverse | 4.337 ms | 1.442 ms | 3.01x slower |
| TPC pair transform | 2.248 ms | 1.239 ms | 1.81x slower |
| Parallel affine scan | 20.543 ms | 11.545 ms | 1.78x slower |
| Two-chunk blocked scan | 12.355 ms | 11.953 ms | 3.4% slower |
| Four-chunk blocked scan | 12.339 ms | 11.819 ms | 4.4% slower |

Replacing MME work with TPC arithmetic is a closed path.

## Native extension route

The installed `libhabana_pytorch_backend.so` exports `KernelRegistry`,
`OpBackend::BuildNode`, and `synapse_helpers::graph::add_node`. An out-of-tree
extension can therefore register an ordinary HPU backend and emit a mixed
Synapse subgraph containing both `batch_gemm` and TPC GUIDs.

`native_backend/` now contains a research probe that lowers one operation to:

```text
MME batch_gemm -> TPC add_fwd_bf16
```

The probe has been compiled against bridge tag `v1.24.1-482`, loaded against
`habana-torch-plugin==1.24.1.482`, and executed on Gaudi2. It registers both
HPU and Meta dispatcher kernels. User bundling and strict scheduling produced
the same device schedule as the ordinary graph, so it is not loaded by vLLM.

This route removes the immediate need for a complete bridge fork, but it is
ABI-private. The extension must hard-fail on a bridge/plugin version mismatch.
It also does not by itself guarantee SRAM placement or cross-engine fusion.

## Proposed operator contract

Use two native operations first, matching FlashQLA's proven split:

```text
hpu::flashqla_kkt64(
    k_bf16, beta_f32, cu_seqlens_i32?) -> a0_bf16

hpu::flashqla_gdn_fwd(
    q_bf16, k_bf16, v_bf16,
    a0_bf16, g_cumsum_f32, beta_f32,
    initial_state_f32?, scale_f32, cu_seqlens_i32?
) -> output_bf16, final_state_f32
```

Constraints for the first optimized specialization:

- Fixed `DK=DV=128` and `chunk=64`.
- `Hq=Hk=16`, `Hv=48`, with head mapping inside the op. Do not materialize
  repeated 48-head Q/K tensors.
- Fixed-length batch-one path first. Varlen and batched requests retain the
  existing PyTorch fallback.
- FP32 gate, KKT inverse accumulation, and recurrent state.
- BF16 Q/K/V/A and output.
- No eager fallback and no silent dtype downgrade.

The final state is optional for pure prompt benchmarking but must remain part
of the production contract because vLLM needs it for decode continuation.

## Implementation milestones

### M0: mixed-engine registration

Status: complete. Device smoke and trace passed, but compound wrapping did not
change the generated MME/TPC schedule.

Acceptance:

- One compiled recipe contains `BatchGemm` followed by `add_fwd_bf16` under a
  single custom op.
- BF16 output matches `torch.bmm(lhs, rhs) + bias`.
- Trace confirms MME and TPC engines, with no CPU fallback.

### M1: KKT64 lowering

Status: TPC-only candidate rejected. The custom KKT-form kernel took 2.443 ms
versus 2.087 ms for the compiled graph, despite 0.011% relative-L2 error.

Build `flashqla_kkt64` using compact 16-head `K @ K.T`, then map beta to 48
value heads. Start from the existing exact recursive base-16 solve, because it
is faster than every tested TPC inverse. Use the gate-free matrix and avoid
pairwise exponentials in the solve.

The first purpose of the native op is to give Synapse one explicit bundle and
stable tensor lifetimes, not to claim a new inversion algorithm. Compare user
bundling and strict scheduling against the current 4.003 ms solve-plus-two-RHS
microbenchmark. Continue only if the isolated path improves by at least 15%
without changing the FP32 solve order beyond the accepted tolerance.

### M2: persistent forward graph

Status: graph-level algebra and deferred output scheduling implemented. The
complete core is 24.4% faster and reaches 99% any-engine activity, but state is
still represented by ordinary tensors; SRAM residency is not guaranteed.

Implement `flashqla_gdn_fwd` as one native mixed-engine subgraph. Preserve one
logical state buffer per value head across 256 chunks. Use two chunk buffers
and explicit data dependencies. Apply user bundling/scheduling hints so MME
work for chunk `i` can overlap TPC preparation and DMA for chunk `i+1`.

Required trace evidence:

- The state is not copied through a persistent HBM tensor every chunk.
- Intermediate state lifetime is one recipe-local workspace allocation.
- MME and TPC intervals overlap inside each chunk pipeline.
- No `triangular_solve_cpu`, host callback, or graph break appears.

The public regular tensor API cannot force an ordinary intermediate into SRAM;
`MEMORY_ATTRIBUTE_SRAM` is marked unsupported. If graph-compiler placement
still spills the state, M2 cannot reproduce FlashQLA's main advantage through
public Synapse nodes alone.

### M3: compiler/GUID escalation if state spills

If M2 cannot keep state on chip, the remaining correct route is an Intel graph
compiler complex GUID or an Intel-provided fused GDN primitive that can:

- Own SRAM workspace across MME and TPC stages.
- Schedule MME operations from the compound implementation.
- Expose per-chunk barriers and double buffering.
- Keep the 128x128 FP32 state resident until the head's sequence is complete.

The installed headers mention GDN kernels such as
`chunk_scaled_dot_kkt_fwd_f32`, `chunk_gated_delta_rule_h_fwd_f32`, and
`fused_gdn_gating_fwd_f32`, but the installed public GUID registry does not
expose a complete runnable GDN path. Header parameter structs are not evidence
that an application can instantiate those kernels.

The existing `mamba_pscan_fwd` GUID is not a substitute. Its diagonal SSM scan
contract cannot represent GDN's 128x128 matrix state update and query-dependent
output.

### M4: optional context parallelism

Only after M2 or M3 is fast, test gate-based sequence splitting at 64K, 128K,
and 256K. The implementation must first compute warmup chunks, reconstruct
segment initial states, and fall back when gate decay is insufficient. A
simple independent split changes semantics and is forbidden.

## Performance gates

The current directly replaceable Phase A/B region is 13.445 ms/layer. Use the
following isolated targets:

| Native QLA time/layer | Projected TTFT | Projected input throughput |
| ---: | ---: | ---: |
| 8 ms | 2.330 s | 7.0K token/s |
| 6 ms | 2.234 s | 7.3K token/s |
| 4 ms | 2.138 s | 7.7K token/s |
| 3 ms | 2.090 s | 7.8K token/s |

If Q/K normalization is also fused, a 3 ms native region projects to about
1.989 seconds or 8.2K token/s. These projections hold all other measured
stages constant and should be treated as planning estimates, not benchmark
results.

The current 3.050 ms/layer gain projects the endpoint from 6.32K to about 6.70K
token/s. Even deleting all Q/K preprocessing, Phase A, Phase B precompute, and
recurrent scan leaves about 1.845 seconds, or 8.9K token/s. FlashQLA alone
cannot deliver the 9K endpoint target. MLP quantization/SwiGLU or full-attention
SDPA must also improve after the GDN port.

Go/no-go gates:

- `<= 8 ms/layer`: useful first native result.
- `<= 6 ms/layer`: minimum target for integration work.
- `<= 4 ms/layer`: strong Gaudi2 result and worth production hardening.
- `> 8 ms/layer`: keep as research; do not enable in vLLM.

## Correctness gates

Kernel development does not require repeated model endpoint runs. Use isolated
tests until the native path meets the 6 ms gate:

1. FP32 recurrent reference on small random tensors with nonzero initial state.
2. Exact Qwen head mapping, chunk padding, and final-state tests.
3. Relative L2 error below 2%, matching FlashQLA's upstream forward tolerance.
4. No NaN/Inf and bounded max absolute error for gate extremes.
5. Bitwise comparison where the operation order is intentionally unchanged.

Only an integration candidate that passes the performance gate proceeds to
the existing short prompts, exact-16K needle case, and output-hash checks. No
quality-changing approximation is enabled silently.

## Immediate next experiment

The next native milestone is a reusable recurrent scan or loop IR that avoids
unrolling one graph node chain per 64-token chunk. The optimized core sustains
1.85-1.87M core token/s at 64K and 128K, but the 128K graph takes about four
minutes and 5.9 GB host RSS to compile. A production implementation should:

1. Preserve the accepted factorized Phase A/B algebra.
2. Express the state loop without Python graph expansion.
3. Keep the current output/state error below the 2% gate.
4. Retain or improve the 99% any-engine activity trace.
5. Run model-level short, needle, and long-context quality checks before the
   experimental flag becomes a default.
