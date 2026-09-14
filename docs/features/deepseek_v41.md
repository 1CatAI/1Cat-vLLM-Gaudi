# DeepSeek V4.1 Flash prepared execution

This experimental profile is under qualification. Enabling a switch is not
evidence of completed model, numerical, memory or performance validation.
The initial contract is Gaudi2, TP2×PP2, one request, context up to 512 tokens,
and greedy sampling. Unsupported sampling modifiers fail explicitly.

## Preparation

Start from a validated V4.1 engine and its matching native replay runtime.
The repository does not yet publish a complete installable V4.1 engine lock;
the upstream PR audit is evidence collection, not an engine installer or a
qualified compatibility lock. Preserve the actual engine revision, patches and
runtime binary fingerprints with the prepared manifest. See the
[engineering guide](../1cat_gaudi_guide.md#deepseek-v41) for prerequisites,
profile recording, and explicit device selection. The original Hugging Face checkpoint is the recovery source;
the four prepared files are immutable derived artifacts.

```bash
.venv/bin/python tools/audit_deepseek_v41_upstream.py --help
.venv/bin/python tools/sync_deepseek_v41_checkpoint.py MODEL_DIR --evidence CHECKPOINT_AUDIT
.venv/bin/python tools/prepare_deepseek_v41_shards.py MODEL_DIR PREPARED_DIR \
  --checkpoint-audit CHECKPOINT_AUDIT/complete.json --upstream-lock UPSTREAM_LOCK
.venv/bin/python tools/build_deepseek_v41.py --build-root BUILD_DIR
```

Build the native TP2 bridge using `tools/communication/build_tp2_fused_ar_norm_bridge.py`
and the exact Bridge, Synapse and HCL source/build paths for the replay runtime.
Its ABI manifest remains mandatory. See the native replay build documentation.

Preparation creates `pp0-tp0.safetensors`, `pp0-tp1.safetensors`,
`pp1-tp0.safetensors`, `pp1-tp1.safetensors`, the topology/quantization manifest,
and two Engram host-shard manifests. Each Engram shard contains whole hash
heads and references byte offsets in the original read-only files. It does
not materialize another host copy or put the embedding tables in HBM.

The Q16/S16 layout packs MXFP4 directly for the native decoder, supports
128-element K alignment, and pads the TP-local 1152-element W2 K dimension.
At runtime, each expert tensor is copied directly to its single destination.
When a BF16 linear path is selected, dense block-FP8 matrices are converted
with bounded host chunks to BF16 MME weights without a second persistent FP8
dense device cache. The C1 FP8 sidecars below have their own storage contract.

## Execution

The dedicated entrypoint enables the measured ordinary-C1 fast-path bundle by
default. Prepare the two FP8 sidecars in their standard locations, then launch
without a feature-variable list:

```bash
.venv/bin/python tools/prepare_deepseek_v41_woa_fp8.py PREPARED_DIR
.venv/bin/python tools/prepare_deepseek_v41_dense_fp8.py PREPARED_DIR
.venv/bin/python -m vllm_gaudi.entrypoints.deepseek_v41 PREPARED_DIR \
  --checkpoint-audit CHECKPOINT_AUDIT
```

The standard sidecar directories are `PREPARED_DIR/sidecars/wo_a_fp8` and
`PREPARED_DIR/sidecars/attention_dense_fp8`. The entrypoint discovers both and
fails before model loading if a required artifact is absent. Set
`VLLM_HPU_DSV41_DEFAULT_FASTPATHS=0` to suppress the entrypoint's default
injection; it does not unset existing variables or reconstruct a reference
environment. Individual feature overrides belong in the effective runtime
configuration; see [configuration precedence](../1cat_gaudi_guide.md#runtime-profile).
The generic vLLM entrypoint retains the opt-in defaults from `vllm_gaudi.envs`.

Reserve four free modules using the project's device lease mechanism before
launching. `tools/run_deepseek_v41.py` provides an archived invocation wrapper
that respects existing per-module locks and actual device ownership. The
model implementation resides in the plugin's normal worker, loader and model
registration paths; the wrapper does not inject or replace code.

PP0 owns target layers 0–19, embedding, vision and both host Engram tables.
PP1 owns target layers 20–39, output normalization/head, and the three-layer
DSpark draft. Its draft embedding is part of the prepared PP1 file. Hidden
states and previous mHC mixing coefficients cross the PP boundary together.
Target states from layers 37–39 feed draft context insertion.

Ordinary single-token decode is the dedicated entrypoint's default. It omits
speculative configuration, skips all
`mtp.*` tensors while reading the immutable rank files, and creates no draft
state or target-state auxiliary outputs. The normal runner commits one output
token without proposal or verification. Native warmup captures C1 only;
multi-token prefill retains its compiled compatibility path and tail handling.
Set `VLLM_HPU_DSV41_DSPARK=1` explicitly to select the separate speculative
profile; the ordinary-C1 bundle is not applied in that mode.

### Accelerated C1 candidates

The ordinary decode implementation provides individually overridable expert,
attention and projection optimizations. The dedicated entrypoint enables its
C1 bundle; the generic entrypoint retains opt-in environment defaults. The combined experimental
profile uses N256 FP8 experts with fused activation preparation, channel-scaled
FP8 wo_a, dedicated Router top-6, decoded KV with shared-KV MME attention,
BF16 head operands with FP32 logits, and fused Q/KV input projection.
It retains stage replay, the mHC/TP dependency schedule and native Engram
preparation. See the [environment variable contracts](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/docs/configuration/env_variables.md)
for precision choices and dependencies.

Build these kernels and the host gather extension together with
`tools/build_deepseek_v41.py`. Set `VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR` to that
build's output directory, including its source/binary manifest. A copied
shared library without its matching manifest is insufficient. Rebuild the
native replay adapter against the pinned runtime and use the supplied runtime
source patches where required; do not relax ABI checks to load a different
Bridge, Synapse or HCL build.

N256 expert preparation reuses the original prepared shard files and retains
one resident compressed expert-weight allocation per rank. Original scale encodings support BF16
prefill on that allocation. Prepare wo_a using
`tools/prepare_deepseek_v41_woa_fp8.py PREPARED_DIR`; pass an explicit output
directory only when the sidecar cannot live below the prepared model. FP8 wo_a has no resident BF16
weight duplicate. Precision, layout or weight changes require model reload
and new recipes.

The default C1 bundle does not enable the separate legacy
`VLLM_HPU_DSV41_FP8_DECODE`, coordinate-pipeline, or native PP0 input-capture
experiments. Native PP0 input capture retains its ordinary BF16-only contract.
Neither successful component checks nor relative decode improvements establish
production quality: generated outputs differ from the preceding candidate,
and full quality, prefill numerical consistency and long replay/shutdown
qualification remain open. The dedicated entrypoint is an experimental contract;
use a complete archived reference configuration for compatibility and
production-reference comparisons. The aggregate opt-out only suppresses default
injection and is insufficient to restore that reference on its own.

The scheduler owns one complete request-state block for the bounded context.
Null block 0 and request block 1 each have separate allocations for the actual
SWA, FP4 main/index, candidate, Top-512, compressor and draft arrays. Prefix
caching is disabled. Rebinding state invalidates compiled replay addresses;
errors after state writes abort execution rather than retry another path.

Engram uses `MAP_POPULATE`, `MADV_WILLNEED` and native gather workers. Only its
small DMA staging buffers require HPU-pinned memory. Explicit forced table
locking is available via `engram_force_lock` in the loader configuration.
The default does not assume the full host tables fit the memlock quota.

## Qualification

First check text and image execution, accepted/rejected DSpark verification,
all target/draft layers, state writes, SRAM placement and native replay.
Then capture a complete four-rank trace and absolute request metrics. Report
each kernel's dtype, shape, latency, count and share in milliseconds, with
device-window reconciliation and overlap handled explicitly.

The complete quality gate additionally covers frozen-reference generation,
image spans, accepted-prefix rollback, request changes, state reuse and
shutdown. The dedicated entrypoint enables the experimental ordinary-C1 bundle;
the generic entrypoint retains opt-in defaults, and DSpark remains disabled by
default. Production-quality promotion still requires the complete gate.
