# FlashInfer-compatible Gaudi operators

## Native backend migration

The project is transitioning from compatible graph tactics to complete native
operations. See [Native backend qualification](flashinfer_native_backend.md).
`native`, `public`, and `bridge` are now whole-operation strict policies:
all GDN compute entry points reject their remaining PyTorch decomposition,
including the packed prototype's exponential/cast/contiguous prologue.
The prototype kernel remains available for explicitly labeled diagnostic
benchmarks; it is not a complete native implementation of the public API.
Existing `auto` graph tactics remain the serving default.
Environment promotion overrides no longer bypass the offline gates.

`vllm-gaudi` contains an independently namespaced `flashinfer_gaudi` package
for inference primitives whose public semantics match portable FlashInfer
operations while using Intel Gaudi execution paths.

The first supported domain is Qwen gated-delta-rule prefill and decode. The
public Python surface follows FlashInfer 0.6.18 for `chunk_gated_delta_rule`,
`gated_delta_rule_decode_pretranspose` (VK/K-last state),
`gated_delta_rule_decode` (KV/K-major state), and `gated_delta_rule_mtp`
(pooled multi-token decode). The implementation keeps a compile-friendly
reference for the portable contract and provides dispatch points for public
TPC custom kernels and an optional version-locked bridge backend. The public
surface also includes FlashInfer 0.6.18's `gdn_fused_decode_step` and its
support probe.

Enable the vLLM adapter with:

```bash
export VLLM_HPU_FLASHINFER_GDN=1
export FLASHINFER_GAUDI_BACKEND=auto
```

This also enables the shape-gated Qwen3.8-27B prefill tactic. Set
`VLLM_HPU_FLASHINFER_GDN_PREFILL=0` to keep only the decode path enabled.
The prefill dispatcher is deliberately shape-gated to BF16
`Hq=Hk=16`, `Hv=48` for TP1, or rank-local `Hq=Hk=8`, `Hv=24` for TP2,
with `K=V=128`, chunk size 128, and one uniform sequence. The TP2 shapes require
`VLLM_HPU_FLASHINFER_GDN_TP2=1` in addition to the parent GDN flag;
all other layouts retain the general HPU implementation.

The prefill tactic transfers the useful FlashQLA algebra to Gaudi's execution
model: it removes pairwise gate exponentials from the triangular solve,
reuses the causal-decay tensor in phase B, shares KKT products across repeated
key heads, keeps Q/K in their rank-local grouped-query form until a value-head
operation actually needs the three-way broadcast, replaces three full
triangular-band materialization passes with chunk-static causal masks, uses a
recursive 16x16 block inverse, fuses the phase-B state projections, and defers
output additions outside the recurrent dependency chain. The masked decay
also replaces invalid upper-triangle gate deltas with negative infinity before
the exponential, preserving overflow safety without a second triangular pass.
Keeping Q/K compact also means their L2 normalization no longer runs
redundantly over the three times larger set of value heads. The normalization remains outside the
compiled graph because compiling it is a measured Gaudi2 regression. The
promoted correctness-first tactic keeps the graph in FP32; BF16 bulk math
remains a research option until it passes model-level quality gates.
Matrix products still run on MME while `torch.compile` fuses the
surrounding tensor graph; this is not an eager PyTorch fallback and does not
replace MME work with a slower TPC-only kernel.

The public prefill entry point is available at both locations:

```python
from flashinfer_gaudi import chunk_gated_delta_rule
from flashinfer_gaudi.gdn_prefill import chunk_gated_delta_rule
```

It accepts FlashInfer's packed `[total_seq_len, heads, dim]` tensors and alpha
gate semantics. Context-parallel checkpoints and indexed prefill state pools
currently raise `NotImplementedError` rather than silently using a different
contract.

Enabling the adapter also selects Gaudi's fused scale-calculation CGUID for
decode-sized dynamic FP8 linear inputs. This removes the separate
absolute-value and maximum-reduction chain without changing decoded tokens.
The optimization is capped at 32 flattened rows by default, so prefill keeps
the existing numerical path. Set `VLLM_HPU_CGUID_DYNAMIC_QUANT=0` to disable
it independently, or tune the decode cap with
`VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS`.

`auto` uses the compiled PyTorch implementation only with vLLM's contiguous
group-major recurrent-state layout. Indexed state layouts fall through to
vLLM's existing decode implementation, because the extra gather/scatter is a
measured regression. A public native op is selected only after its offline
tactic is promoted. Native tactics remain unpromoted in the bundled offline
tactic manifest until they pass both model-level quality and performance
gates. Strict policies currently reject all GDN compute entry points before
recurrent state is modified, including reference-only prefill and MTP.

The same proven group-major contract is also applied to the Qwen decode
convolution cache. One-token decode consumes its native
`[batch, state_length, channels]` view directly, eliminating integer-index
gather/scatter, a transpose, and a temporary convolution window. This path is
selected only when every active request exactly fills the static decode bucket;
padding, prefix caching, and multi-token/speculative decode retain the general
indexed implementation.

The production `H=16`, `HV=48`, `K=V=128` direct-state route has a static
Gaudi recipe rather than deriving head dimensions inside the compiled graph.
It normalizes packed Q/K together with an explicit reciprocal-square-root
form, preserving the additive-epsilon contract `x * rsqrt(sum(x*x) + 1e-6)`,
and uses `addcmul` for the FP32 rank-one state update. Other shapes keep
the general FlashInfer-compatible implementation.

For Qwen3.8-27B TP1/TP2 decode batches 1, 2, 4, 8, 16, and 32, the fused direct-state
recipe additionally combines gating, the static width-4 convolution update,
and packed recurrence in one compile graph. Recurrent projections stay on MME,
and the recurrent state update retains the regional-graph-friendly
out-of-place ordering before copying into the owned contiguous cache view. Set
`VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE=0` to disable this recipe independently
while keeping the remaining FlashInfer-Gaudi route.

The default-off TP2 port requires `VLLM_HPU_FLASHINFER_GDN_TP2=1` and uses a static `H=8`, `HV=24`, packed-width-5120 recurrence in
this fused composition. Each rank retains its own head shard and state; the
existing tensor-parallel reductions are unchanged. TP2 has focused operator
checks and a batch-one serving screen. This does not extend the original TP1
qualification to every TP2 bucket or to speculative decoding.

The first native Gaudi2 GUID specializes Qwen3.8 decode (`H=16`, `HV=48`,
`K=V=128`) with packed BF16 Q/K/V and beta, BF16 output, FP32 recurrent state,
grouped Q/K reuse, negative-index padding, and in-place pooled-state updates.
It is intentionally opt-in until it beats the compiled graph across the full
decode bucket matrix; availability alone never changes the stable route.

The default state precision is FP32. BF16 recurrent state is not selected by
the production dispatcher until it passes end-to-end token and state-quality
validation.

## DFlash2 for Qwen3.8

The experimental HPU V1 DFlash2 path supports the official
`incoai/Qwen3.8-27B-DFlash2` checkpoint with a Qwen3.8/Qwen3.5 hybrid target.
It performs all seven drafts in one pass, runs the top-16 candidate selector
on HPU, and verifies an eight-token block with rollback-safe convolution and
GDN state checkpoints. A fused TPC recurrent kernel for the production Qwen
shape is available as a qualification candidate, replacing eight framework
recurrence steps per GDN layer when explicitly selected.

The initial qualification scope is greedy text generation, TP1/PP1/DP1,
`max_num_seqs <= 16`, compact GDN state, and prefix caching and LoRA disabled.
A target checkpoint may contain a vision tower, but requests carrying
multimodal inputs are rejected until M-RoPE context mapping is qualified.
Size the KV-cache budget so the startup-reported maximum concurrency is at
least the concurrency under test. DFlash2 adds a draft attention cache; an
undersized `--kv-cache-memory-bytes` budget silently queues otherwise
concurrent requests and produces a serialized throughput measurement.

```bash
export VLLM_HPU_FLASHINFER_GDN=1
export VLLM_HPU_FLASHINFER_DFLASH2=1
export VLLM_COMPACT_GDN=1

vllm serve /path/to/Qwen3.8-27B-FP8 \
  --max-num-seqs 16 \
  --speculative-config '{
    "method": "dflash",
    "model": "/path/to/Qwen3.8-27B-DFlash2",
    "num_speculative_tokens": 7
  }'
```

Warmup feeds one generated draft block back through every decode bucket, so
the target verification and rollback graphs are captured before serving.
Each verification token uses its own causal KV-page extent, including when a
block crosses a page boundary. Padding within a request writes only to the
reserved padding page, never to a live prefix-cache position.

`VLLM_HPU_DFLASH2_FULL_QUERY_CONV` optionally removes redundant convolution
masks and state reads for full verification blocks with compact request-owned
state and prefix caching disabled. CPU metadata must prove all token positions
are real; ragged batches and single-token transitions retain the general path.
The fast path preserves accumulation order, activation rounding, and accepted
history selection. It remains disabled pending full-model qualification.

`VLLM_HPU_DFLASH2_DIRECT_CHECKPOINTS` separately enables a contiguous write
destination for the prepared native Target verifier. Before device transfer,
CPU metadata must prove all active requests occupy compact base slots in
input order and every verification checkpoint belongs to the expected group.
The operator still produces all eight states and preserves accepted-state
selection. Reordered, padded, noncompact, or prefix-cached layouts retain
indexed writes. This opt-in does not promote the underlying native kernel.

Unsupported native shapes retain the exact FlashInfer-Gaudi PyTorch reference
under `auto`; forcing `FLASHINFER_GAUDI_BACKEND=public` fails early if the
native MTP kernel is unavailable. The MTP kernel has its own auto-promotion
gate, `FLASHINFER_GAUDI_ENABLE_MTP_AUTO`, and remains unpromoted until hardware
rollback-state parity, output-quality, acceptance, and latency gates pass.
Even after promotion, automatic selection retains the graph reference for B1
and compiled target models. The native kernel is considered only for eager
batch sizes 2 through 16, where its launch cost is amortized without blocking
cross-op graph fusion.
The seven dependent selector `argmax`/gather steps also have an exact-shape
Gaudi2 TPC qualification candidate. Its independent
`FLASHINFER_GAUDI_ENABLE_DFLASH2_SELECT_AUTO` gate remains off until the native
walk matches the FP32 framework path bit-for-bit and wins across batches 1, 8,
and 16, with intermediate buckets covered as well.
Vocabulary top-16 uses Intel's existing TopK CGUID rather than duplicating a
radix kernel in TPC-C. The optional
`FLASHINFER_GAUDI_ENABLE_DFLASH2_TOPK_AUTO` route removes the small stable
sort/gather canonicalization graph around that CGUID; it remains unpromoted
until candidate-set and downstream generated-token parity are qualified.
An additional TPC qualification candidate fuses edge scoring with the greedy
walk. It evaluates only the predecessor row selected by the preceding step,
instead of materializing all 16-by-16 transitions, and is guarded by
`FLASHINFER_GAUDI_ENABLE_DFLASH2_SCORE_SELECT_AUTO`. This route deliberately
stays independent from the standalone walk so numerical parity and its
different BF16 reduction order can be evaluated separately.

## Microbenchmarking

Compare the fused DFlash2 MTP kernel with the compiled eight-step reference,
including output and complete checkpoint-state validation, with:

```bash
python3 tools/benchmark_flashinfer_gaudi_dflash2.py \
  --batches 1,2,4,8,16 --warmups 20 --wave-iterations 100 --waves 7
```

Compare the single-launch native DFlash2 lattice walk with the compiled
seven-step framework selector, including exact token validation, with:

```bash
python3 tools/benchmark_flashinfer_gaudi_dflash2_select.py \
  --batches 1,2,4,8,16 --warmups 20 --wave-iterations 100 --waves 7
```

Compare direct vendor-CGUID top-16 with the canonical reference wrapper at
the production vocabulary shape with:

```bash
python3 tools/benchmark_flashinfer_gaudi_dflash2_topk.py \
  --batches 1,2,4,8,16 --warmups 10 --wave-iterations 20 --waves 7
```

Compare fused selected-row edge scoring and selection with the full-lattice
framework graph, requiring exact draft-token parity, with:

```bash
python3 tools/benchmark_flashinfer_gaudi_dflash2_score_select.py \
  --batches 1,2,4,8,16 --warmups 20 --wave-iterations 100 --waves 7
```

Compare the general HPU GDN prefill graph with the promoted FlashQLA graph
tactic at production Qwen3.8 shapes:

```bash
python3 tools/benchmark_flashinfer_gaudi_gdn_prefill.py \
  --tokens 2048,4096,16384 --iterations 10 --waves 7
```

Measure the incremental grouped-head optimization, including the production
Q/K normalization boundary, with:

```bash
python3 tools/benchmark_flashinfer_gaudi_gdn_prefill.py \
  --reference flashqla-expanded \
  --reference-qk-l2norm eager \
  --tokens 2048,4096,16384 --iterations 10 --waves 7
```

Measure the incremental static-triangular-mask optimization against the
otherwise identical compact-Q/K FlashQLA graph with:

```bash
python3 tools/benchmark_flashinfer_gaudi_gdn_prefill.py \
  --reference flashqla-compact \
  --candidate-masked-triangular-decay \
  --tokens 2048,4096,16384 --iterations 10 --waves 7
```

Run the production bucket matrix with both synchronized host latency and
continuous device-wave timing:

```bash
python3 tools/benchmark_flashinfer_gaudi_gdn.py \
  --batches 1,8,16,32 --timing-mode both --backend public
```

Compare the production direct recipe with the existing vLLM decode path with:

```bash
python3 tools/benchmark_flashinfer_gaudi_gdn.py \
  --batches 1,8,16,32 --timing-mode both \
  --reference-layout legacy --backend pytorch
```

Benchmark the indexed and group-major convolution-cache recipes with:

```bash
python3 tools/benchmark_flashinfer_gaudi_decode_conv.py \
  --batches 1,8,16,32 --warmups 100 \
  --wave-iterations 500 --waves 15
```

Compare fused decode candidates against the exact current Qwen3.8 chain with:

```bash
python3 tools/benchmark_flashinfer_gaudi_gdn_fused_decode.py \
  --batches 1,2,4,8,12,16,20,32 --warmups 50 \
  --wave-iterations 750 --waves 16
```

The default matrix deliberately includes the exponential strategy's B12 and
B20 decode buckets. They remain on the established fallback until separate
operator and end-to-end measurements qualify them for the fused recipe.

`device_speedup` uses HPU events around an interleaved multi-iteration wave
and is the promotion metric for kernel execution. `median_speedup` includes a
device synchronization after every invocation and is retained as the exposed
single-call latency diagnostic.

## Experimental graph-native MTP core

`FLASHINFER_GAUDI_ENABLE_MTP_PREPARED=1` opts compiled Qwen3.8 B1/T8
full-query verification into a separate native qualification path. Keep
`FLASHINFER_GAUDI_BACKEND=auto` when testing this switch; forced `pytorch`
continues to mean reference execution. Rebuild the native libraries first.

The compiled graph prepares normalized/scaled Q, normalized K, FP32 V and
gating, and selects the accepted state. The functional TPC core retains a
small FP32 state tile across all eight updates and returns every checkpoint.
The graph writes the checkpoints back to scheduler-owned slots. It never
discards intermediate rollback state. Partial queries and padded batches
remain on the established route.

The compact no-prefix-cache scheduler proves distinct token slots and opts
into a coalesced write. Other callers keep ordered writes unless they
explicitly provide the same distinct-slot guarantee; duplicate indices
must retain last-token-wins semantics.

The internal op uses the HPU CustomOp API's `custom_op` namespace so the
backend can retain it within a larger compiled graph. This is distinct from
having `torch.compile(fullgraph=True)`: that setting alone does not prevent
backend eager fallback. Native qualification tests also disable backend
eager fallback explicitly.

This switch is not a production promotion. Small FP32 reduction differences
remain possible, so operator tolerances and microbenchmarks alone do not
establish output-quality or model-latency qualification.

Q/K normalization adds `1e-6` to the squared norm, matching the FP32 GDN
contract in the [upstream FlashInfer decode kernel](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_kernels/gdn_decode_pretranspose.py).
The previous local reference incorrectly clamped the squared norm instead;
rebuild the native libraries and re-establish numerical/performance baselines
after updating. Small-Q/K and repeated-rollback tests cover this correction.
This does not establish equivalence to every intermediate BF16 rounding
boundary of a Transformers fallback. Model qualification must still use an
independent target-only reference and fixed-prefix numerical checks.

After rebuilding, run the operator qualification benchmark with
`python tools/benchmark_flashinfer_gaudi_mtp_prepared.py --batches 1,2`.
It includes normalization, accepted-state selection, every checkpoint write,
and disables backend eager fallback. `--pool-slots` can match a larger
serving state pool. It does not include projections, MLPs, or the full round.

## Compatibility namespace

Applications should import `flashinfer_gaudi` directly. An opt-in shim is
available for programs that hard-code `flashinfer.gdn_decode`:

```python
from flashinfer_gaudi.compat import install_flashinfer_shim

install_flashinfer_shim()
```

The shim refuses to replace an installed official FlashInfer package.
