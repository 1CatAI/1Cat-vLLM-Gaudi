# FlashInfer-compatible Gaudi operators

`vllm-gaudi` contains an independently namespaced `flashinfer_gaudi` package
for inference primitives whose public semantics match portable FlashInfer
operations while using Intel Gaudi execution paths.

The first supported domain is Qwen gated-delta-rule decode with VK/K-last
recurrent state. The implementation keeps a PyTorch reference for unsupported
shapes and provides dispatch points for public TPC custom kernels and an
optional version-locked bridge backend.

Enable the vLLM adapter with:

```bash
export VLLM_HPU_FLASHINFER_GDN=1
export FLASHINFER_GAUDI_BACKEND=auto
```

`auto` uses a validated public native op when one is loaded and otherwise
falls back before execution. Native tactics remain unpromoted in the bundled
offline tactic manifest until they pass model-level gates. `public` and
`bridge` are strict policies: an
unavailable implementation raises before recurrent state is modified.

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
