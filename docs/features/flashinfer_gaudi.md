# FlashInfer-compatible Gaudi operators

`vllm-gaudi` contains an independently namespaced `flashinfer_gaudi` package
for inference primitives whose public semantics match portable FlashInfer
operations while using Intel Gaudi execution paths.

The first supported domain is Qwen gated-delta-rule prefill and decode. The
public Python surface follows FlashInfer 0.6.18 for `chunk_gated_delta_rule`,
`gated_delta_rule_decode_pretranspose` (VK/K-last state),
`gated_delta_rule_decode` (KV/K-major state), and `gated_delta_rule_mtp`
(pooled multi-token decode). The implementation keeps a compile-friendly
reference for the portable contract and provides dispatch points for public
TPC custom kernels and an optional version-locked bridge backend.

Enable the vLLM adapter with:

```bash
export VLLM_HPU_FLASHINFER_GDN=1
export FLASHINFER_GAUDI_BACKEND=auto
```

This also enables the promoted Qwen3.8-27B TP1 prefill tactic. Set
`VLLM_HPU_FLASHINFER_GDN_PREFILL=0` to keep only the decode path enabled.
The prefill dispatcher is deliberately shape-gated to BF16
`Hq=Hk=16`, `Hv=48`, `K=V=128`, chunk size 128, and one uniform sequence;
all other layouts retain the general HPU implementation.

The prefill tactic transfers the useful FlashQLA algebra to Gaudi's execution
model: it removes pairwise gate exponentials from the triangular solve,
reuses the causal-decay tensor in phase B, shares KKT products across repeated
key heads, keeps Q/K in their 16-head grouped-query form until a value-head
operation actually needs the three-way broadcast, replaces three full
triangular-band materialization passes with chunk-static causal masks, uses a
recursive 16x16 block inverse, fuses the phase-B state projections, and defers
output additions outside the recurrent dependency chain. The masked decay
also replaces invalid upper-triangle gate deltas with negative infinity before
the exponential, preserving overflow safety without a second triangular pass.
Keeping Q/K compact also means their L2 normalization no longer runs
redundantly over 48 materialized heads. The normalization remains outside the
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
gates. `public` and `bridge` are strict policies: an unavailable implementation
raises before recurrent state is modified.

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
form and uses `addcmul` for the FP32 rank-one state update. Other shapes keep
the general FlashInfer-compatible implementation.

The first native Gaudi2 GUID specializes Qwen3.8 decode (`H=16`, `HV=48`,
`K=V=128`) with packed BF16 Q/K/V and beta, BF16 output, FP32 recurrent state,
grouped Q/K reuse, negative-index padding, and in-place pooled-state updates.
It is intentionally opt-in until it beats the compiled graph across the full
decode bucket matrix; availability alone never changes the stable route.

The default state precision is FP32. BF16 recurrent state is not selected by
the production dispatcher until it passes end-to-end token and state-quality
validation.

## Microbenchmarking

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

`device_speedup` uses HPU events around an interleaved multi-iteration wave
and is the promotion metric for kernel execution. `median_speedup` includes a
device synchronization after every invocation and is retained as the exposed
single-call latency diagnostic.

## Compatibility namespace

Applications should import `flashinfer_gaudi` directly. An opt-in shim is
available for programs that hard-code `flashinfer.gdn_decode`:

```python
from flashinfer_gaudi.compat import install_flashinfer_shim

install_flashinfer_shim()
```

The shim refuses to replace an installed official FlashInfer package.
