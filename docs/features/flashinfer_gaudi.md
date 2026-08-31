# FlashInfer-compatible Gaudi operators

`vllm-gaudi` contains an independently namespaced `flashinfer_gaudi` package
for inference primitives whose public semantics match portable FlashInfer
operations while using Intel Gaudi execution paths.

The first supported domain is Qwen gated-delta-rule decode. The public Python
surface follows FlashInfer 0.6.18 for `gated_delta_rule_decode_pretranspose`
(VK/K-last state), `gated_delta_rule_decode` (KV/K-major state), and
`gated_delta_rule_mtp` (pooled multi-token decode). The implementation keeps a
compile-friendly reference for the complete portable contract and provides
dispatch points for public TPC custom kernels and an optional version-locked
bridge backend.

Enable the vLLM adapter with:

```bash
export VLLM_HPU_FLASHINFER_GDN=1
export FLASHINFER_GAUDI_BACKEND=auto
```

`auto` uses the compiled PyTorch implementation only with vLLM's contiguous
group-major recurrent-state layout. Indexed state layouts fall through to
vLLM's existing decode implementation, because the extra gather/scatter is a
measured regression. A public native op is selected only after its offline
tactic is promoted. Native tactics remain unpromoted in the bundled offline
tactic manifest until they pass both model-level quality and performance
gates. `public` and `bridge` are strict policies: an unavailable implementation
raises before recurrent state is modified.

The first native Gaudi2 GUID specializes Qwen3.8 decode (`H=16`, `HV=48`,
`K=V=128`) with packed BF16 Q/K/V and beta, BF16 output, FP32 recurrent state,
grouped Q/K reuse, negative-index padding, and in-place pooled-state updates.
It is intentionally opt-in until it beats the compiled graph across the full
decode bucket matrix; availability alone never changes the stable route.

The default state precision is FP32. BF16 recurrent state is not selected by
the production dispatcher until it passes end-to-end token and state-quality
validation.

## Compatibility namespace

Applications should import `flashinfer_gaudi` directly. An opt-in shim is
available for programs that hard-code `flashinfer.gdn_decode`:

```python
from flashinfer_gaudi.compat import install_flashinfer_shim

install_flashinfer_shim()
```

The shim refuses to replace an installed official FlashInfer package.
