# Qwen3.8-27B-FP8 Gaudi2 prefill gap against the NVIDIA stack

Date: 2026-08-30

## Scope

This document compares the complete 16K, TP1 prefill path, rather than only
the Gated Delta Net (GDN) core. The target workload is
`Qwen/Qwen3.8-27B-FP8` on one Gaudi2 with compiled execution. Eager execution
is out of scope.

The accepted endpoint is:

- Input tokens: 16,384
- TTFT: 2.591409 seconds
- Input throughput: 6,322.428 tokens/s
- Decoder-layer time: 2.503781 seconds
- Time outside decoder layers: about 87.6 ms

The model has 64 decoder layers: 48 GDN layers and 16 full-attention layers.
The current 9K tokens/s target requires a TTFT of 1.820444 seconds, or about
771 ms less than the accepted endpoint.

## Measured Gaudi2 breakdown

The percentages below are shares of the measured decoder-layer time. They are
wall-time categories, not engine resource percentages.

| Region | Share | Approx. time | Current observation |
| --- | ---: | ---: | --- |
| Post-attention norm and MLP | 36.13% | 904.6 ms | FP8 MME GEMMs are strong; activation and dynamic quantization remain expensive |
| Full-attention SDPA and post-processing | 9.65% | 241.6 ms | Native causal FusedSDPA is already selected |
| GDN phase A | 9.59% | 240.1 ms | Many short BMM and elementwise graph nodes |
| GDN gating and causal convolution | 8.90% | 222.8 ms | Conv is reasonably optimized, but surrounding operations are separate |
| GDN phase-B precompute | 8.89% | 222.6 ms | Materialization and mixed TPC/MME/DMA work |
| GDN input norm and projections | 8.09% | 202.5 ms | Large projection GEMMs are not the first target |
| GDN recurrent scan | 7.23% | 181.0 ms | Dependency latency and state traffic dominate |
| GDN state save, output norm/gate, projection | 4.13% | 103.4 ms | RMSNorm plus gate has already been fused on the prompt path |
| GDN Q/K normalization and preprocessing | 4.02% | 100.7 ms | Separate passes over Q and K |
| Full-attention input path | 2.79% | 69.9 ms | Norm, QKV projection, Q/K norm, and RoPE |
| Full-attention output projection | 0.58% | 14.5 ms | Low priority |

The old whole-layer profile reported TPC 79.45%, MME 19.22%, and DMA 1.33%
as accumulated engine resource time. These values do not mean that TPC alone
occupies 79% of the wall clock, because engines overlap and a launch can use
many TPCs. The optimized isolated GDN core confirms this distinction: it has
99.34% any-engine activity, while only 1.559 ms of its 8.876 ms device span is
TPC-exclusive.

## What the NVIDIA path does differently

### 1. Input normalization, FP8 quantization, and dense projections

Current vLLM CUDA compilation has explicit fusion passes and native kernels
for RMSNorm-to-FP8 quantization and SiLU-times-gate-to-FP8 quantization. For
this checkpoint's 128x128 weight blocks, CUDA uses activation groups of 128.
The fused activation kernel computes the activation, multiplication, group
maximum, scale, and FP8 packing without materializing a BF16 activation
between those stages. Dense FP8 GEMM can then select and tune CUTLASS,
DeepGEMM, or FlashInfer-style implementations.

The HPU graph sends the two large MLP GEMMs to MME efficiently:

- Gate/up projection: about 849 TFLOP/s effective
- Down projection: about 802 TFLOP/s effective

The accepted HPU service sets `VLLM_HPU_FORCE_CHANNEL_FP8=true`. During model
loading, it dequantizes the checkpoint's block-scaled weight and requantizes it
with one scale per output channel. At runtime, activations use one dynamic
scale per token, not one scale per 128 features. This layout gives the current
fast HPU MME path, but it is not numerically or structurally identical to the
CUDA block-FP8 path.

The main HPU gap is therefore not another GEMM rewrite. It is the missing
native, numerically compatible activation-plus-current-quant and
norm-plus-current-quant boundary. The plugin enables the upstream fusion
flags, but it does not register HPU equivalents of CUDA's fused custom ops for
the HPU per-token/channelwise layout. Synapse graph fusion helps, but the
profile still attributes about 17.6% of decoder-layer time to SwiGLU and the
two dynamic quantization stages.

### 2. GDN projection, convolution, and post-convolution preparation

The CUDA path performs causal convolution and then invokes
`fused_post_conv_prep`. One Triton kernel replaces:

```text
split -> rearrange -> contiguous Q/K/V -> Q/K L2 norm -> gate and beta
```

It reads convolution output plus `a` and `b` once and writes Q, K, V, `g`, and
`beta` directly in the layouts consumed by the GDN kernel. The HPU path still
computes gating before convolution, transposes the convolution result,
rearranges Q/K/V, and normalizes Q/K in the GDN entry path.

The current convolution itself is not an obvious generic-library problem. At
the true shape, alternate Habana and grouped-convolution paths were no faster
than the current prompt implementation. The transferable NVIDIA idea is the
single post-convolution pass and target-layout write, not merely replacing the
convolution call.

HPU serving uses regional compilation at decoder-layer granularity. Therefore
placing the existing Python helpers inside one more Python function does not by
itself create NVIDIA's one-pass kernel. A diagnostic that compared artificial
multi-graph helpers against one compiled helper improved 3.019 ms to 2.478 ms,
but that 0.541 ms result is not counted in the endpoint budget: the production
layer is already one compile region. A real follow-up needs a native compound
operation or trace evidence that Synapse emitted fewer materializations.

There is also a grouped-value-head difference. NVIDIA GDN kernels consume the
compact 16 Q/K heads together with 48 value heads and map them internally. The
HPU graph expands Q/K from 16 to 48 heads before entering phase A. Merely
normalizing before expansion was previously measured at 2.843 ms versus
2.819 ms and did not help. The verified correction is to normalize and cast
the compact Q/K tensors to the BF16 GDN compute dtype before expansion. This
avoids a 3x enlarged FP32 intermediate. At the exact 16K shape, the compiled
Q/K path changed from 3.062 ms to 1.788 ms per GDN layer, a 1.71x speedup and a
projected 61.2 ms endpoint saving. Outputs are identical (`max_abs=0`,
`relative_l2=0`). A later native GDN kernel should also consume compact heads
directly and remove the expanded Q/K HBM tensor altogether.

### 3. GDN core

NVIDIA has several shape-specialized implementations in the same serving
stack: FlashInfer, FLA/Triton, and CuteDSL, with architecture-dependent
dispatch. FlashQLA is a useful best-in-class reference for the same dataflow,
but it is only this stage of prefill.

The key techniques are:

- Keep the recurrent 128x128 state in register/shared-memory fragments across
  chunks instead of crossing graph and HBM boundaries.
- Tile KKT work to tensor-core-friendly shapes.
- Use producer/consumer specialization and double buffering.
- Overlap asynchronous copies, matrix operations, and scalar correction work.
- Fuse the local transform, recurrent state update, and output production so
  intermediate matrices are not materialized.

For the exact 27B TP1, 1x16,384, 16-QK-head, 48-value-head shape, the public
H200 FlashQLA benchmark reports 0.671 ms, FlashInfer 0.781 ms, and FLA
1.311 ms. The optimized HPU graph currently takes 9.437 ms for one GDN core.
This is not a direct hardware-normalized comparison, but a 12-14x kernel gap
is much larger than the roughly 2x HBM-bandwidth difference between H200 and
Gaudi2. Kernel organization is the dominant explanation for this stage.

The current HPU graph reformulation reduced one GDN core from 12.487 ms to
9.437 ms, a 3.050 ms or 24.4% gain. Applied to 48 layers, this projects the
endpoint to about 2.445 seconds or 6.70K tokens/s. It is not yet the accepted
model-level default because output and final-state relative L2 differences are
0.752% and 0.640%, respectively.

### 4. Full causal attention

The NVIDIA reference is FlashAttention/FlashInfer rather than FlashQLA.
FlashAttention-3 uses tiled online softmax, warp specialization, asynchronous
copy, and overlap between matrix multiplication and scalar softmax work. The
important general lesson is the same: keep score tiles and normalization state
on chip and pipeline data movement with matrix execution.

The HPU path already uses native causal FusedSDPA. An explicit BF16 mask was
more than twice as slow, and FP8 attention experiments did not improve latency
while causing about 10.4% relative-L2 error. The next step is BF16 FusedSDPA
tile and engine profiling, not forcing FP8 attention.

For context, one full-attention layer performs about 3.299 TFLOP of logical
causal QK/PV work at 16K. A 15.138 ms measured SDPA/post time corresponds to
roughly 218 TFLOP/s before accounting for non-GEMM work. This leaves useful
headroom, but the H200 result cannot be copied directly because H200 has both
more BF16 compute and about twice the HBM bandwidth.

### 5. GDN output and state handling

Mainline CUDA uses a fused RMSNorm-gated kernel before the output projection.
The HPU prompt path now implements the same fusion concept. At the true
operation shape it reduced the measured RMSNorm/gate operation from 5.808 ms
to 2.258 ms, and the accepted endpoint improved by about 213.8 ms. This is an
example of an NVIDIA dataflow optimization that transfers well to Gaudi2.

State save and the recurrent dependency chain remain separate concerns. The
isolated recurrent span is 2.085 ms, with only 0.295 ms TPC-exclusive time;
the limit is ordered BMM, state movement, and synchronization rather than low
TPC lane occupancy.

## Hardware-normalized interpretation

Gaudi2 exposes 96 GB HBM2e at about 2.45 TB/s, 48 MB SRAM, 432 BF16 MME
TFLOP/s, and 865 FP8 MME TFLOP/s, with MME, TPC, and DMA able to operate
concurrently. H200 exposes 141 GB HBM3e at 4.8 TB/s and much higher advertised
BF16/FP8 tensor throughput. NVIDIA should win a fair raw hardware comparison,
especially for full attention and large GEMMs.

That hardware difference does not explain the whole software gap:

- HPU MLP GEMMs already exercise MME well, so more GEMM tuning has a small
  ceiling there.
- The optimized GDN core already has 99.34% device activity and balanced use
  of all 24 TPC lanes. Occupancy tuning alone has a small ceiling.
- The remaining loss is largely useful-work efficiency: extra FP32/BF16
  elementwise passes, short BMMs, intermediate writes, and dependency stalls.
- NVIDIA's strongest advantage is a mature collection of persistent,
  shape-specialized, cross-operation kernels, not merely CUDA syntax.

The correct porting target is therefore NVIDIA's schedule and dataflow. A
line-by-line CUDA or Triton translation would not map to Gaudi2's MME/TPC/DMA
execution model.

## Optimization priorities

### P0: fused HPU post-convolution preparation

Build one compiled compound/native operation that consumes convolution output,
`a`, `b`, `A_log`, and `dt_bias`, and emits final-layout Q/K/V/g/beta. Preserve
the accepted FP32 accumulation points for Q/K normalization and gating.

The gating/conv and Q/K preparation regions total about 323.5 ms. The
convolution itself accounts for about 104 ms across 48 layers, leaving roughly
219 ms of surrounding work. Recovering 30-50% of that surrounding work would
save about 66-110 ms at the endpoint.

The normalize-and-cast-before-expand change has already captured 61.2 ms of
projected saving from this area with exact output. This is separate from, and
compatible with, a future one-pass post-convolution operation.

### P0: native SiLU-mul plus current dynamic FP8 quantization

Implement the exact current per-token activation scale and FP8 packing
semantics in one HPU operation, and feed its output directly to the
down-projection MME GEMM. A previous raw TPC prototype was slower and packed
FP8 differently, so the next prototype must reuse the production quant
GUID/compound graph semantics. A native 128-group/block-weight HPU path is a
separate experiment and must be judged against both speed and model quality.

SwiGLU plus dynamic quantization represent about 440 ms of decoder-layer time.
A realistic first target is a 20-35% reduction in this region, or about
90-155 ms, rather than assuming that all of it can disappear.

### P1: native mixed-engine GDN core

The graph reformulation is useful, but reaching NVIDIA-like efficiency needs a
primitive that can retain recurrent state in SRAM while scheduling MME and TPC
work. A further 3-6 ms per GDN layer would save about 144-288 ms. This requires
either an Intel fused GUID/backend primitive or a lower-level mixed-engine API;
a TPC-only custom kernel cannot replace MME BMM efficiently.

### P1: BF16 FusedSDPA scheduling

Profile tile sizes, MME/TPC overlap, softmax kernels, and intermediate DMA for
the exact 24-head, head-dim-256, 16K causal shape. Reducing SDPA from about
15.1 ms to 10-12 ms per full-attention layer would save roughly 50-82 ms.

### P2: norm-to-quant boundaries and endpoint overhead

Inspect GDN/full-attention input RMSNorm-to-FP8 boundaries and the 87.6 ms
outside decoder layers. Projection GEMMs and the full-attention output
projection are low-priority unless a fused boundary removes a materialization.

## Target budget

After applying the already measured GDN graph gain, the projected TTFT is
about 2.445 seconds. Adding the exact compact-Q/K cast projection gives about
2.384 seconds or 6.87K tokens/s. Reaching 9K tokens/s still requires about
563 ms:

| Candidate | Plausible endpoint saving |
| --- | ---: |
| Normalize and cast compact Q/K before expansion (measured) | 61.2 ms |
| Remaining fused post-convolution preparation | 40-80 ms |
| Fused SiLU-mul plus current FP8 quantization | 90-155 ms |
| Further native/persistent GDN core work | 144-288 ms |
| Better BF16 full-attention schedule | 50-82 ms |
| Norm/quant boundaries and endpoint overhead | 20-60 ms |

These ranges overlap and are not additive guarantees. After the measured Q/K
change, the remaining candidate range is roughly 290-490 ms. The 563 ms target
is therefore aggressive and requires results near the upper end plus one more
source of savings. Small PyTorch rewrites alone are unlikely to reach 9K.

## Validation gates

Every candidate should pass the following before model-level benchmarking:

1. Exact 16K production shape under `torch.compile`, never eager.
2. Same-process alternating A/B timing with warmup and device traces.
3. Output, state, scale, and FP8-byte comparison at the operator boundary.
4. A projected endpoint saving large enough to matter, normally at least
   1 ms per affected GDN layer or about 50 ms per full model pass.
5. Only then, one model-level quality and TTFT confirmation.

## References

- vLLM Qwen GDN implementation: <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py>
- vLLM activation/quant fusion: <https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/act_quant_fusion.py>
- FlashInfer GDN prefill API: <https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_prefill.py>
- FlashQLA H200 benchmark: <https://github.com/QwenLM/FlashQLA/blob/main/benchmark/benchmark_results_H200.txt>
- FlashAttention-3 paper: <https://arxiv.org/abs/2407.08608>
- Intel Gaudi architecture: <https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html>
- Intel Gaudi2/Gaudi3 feature comparison: <https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf>
- NVIDIA H200 specifications: <https://www.nvidia.com/en-us/data-center/h200/>
