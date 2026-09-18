# DeepSeek V4.1 Flash prepared execution

This experimental profile is under qualification. Enabling a switch is not
evidence of completed model, numerical, memory or performance validation.
The serving contract is Gaudi2, TP2×PP2, one request and greedy sampling.
Paged state supports configuring up to 1,048,576 tokens; prefill uses scheduler
chunks of up to 8192 tokens. This is a capacity contract, not a claim that a
full-window request has passed quality qualification. Unsupported sampling
modifiers fail explicitly.

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

The dedicated entrypoint enables the frozen-reference ordinary-C1 and V2
device-continuation bundle by default. Launch without a feature-variable list:

```bash
.venv/bin/python -m vllm_gaudi.entrypoints.deepseek_v41 PREPARED_DIR \
  --checkpoint-audit CHECKPOINT_AUDIT
```

Set `--no-v2` to retain the synchronous runner, or
`VLLM_HPU_DSV41_DEFAULT_FASTPATHS=0` to suppress the entrypoint's default
injection. The aggregate opt-out does not unset existing variables or
reconstruct a reference environment. Individual feature overrides belong in
the effective runtime configuration; see
[configuration precedence](../1cat_gaudi_guide.md#runtime-profile). The generic
vLLM entrypoint retains the opt-in defaults from `vllm_gaudi.envs`.

Arithmetic-changing FP8, Router, MLA and fused numerical candidates are not in
the default bundle. To test them explicitly, prepare the two FP8 sidecars in
`PREPARED_DIR/sidecars/wo_a_fp8` and
`PREPARED_DIR/sidecars/attention_dense_fp8`, then set
`VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS=1`. The entrypoint validates and
discovers both sidecars before model loading. This aggregate remains an
experimental quality profile, not a production default.

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
multi-token prefill uses bounded expert-grouped BMMs and exact tail handling.
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
`VLLM_HPU_DSV41_FP8_DECODE`, coordinate-pipeline, FP8 projection, N256-FP8
expert, Router-top6, MLA-MME, or fused numerical experiments. Native PP0 input
capture is enabled only as part of the frozen-reference V2 continuation
contract. Successful component checks and relative decode improvements do not
establish production quality; arithmetic-changing candidates stay behind the
experimental numerical aggregate until their full quality, prefill numerical
consistency and long replay/shutdown gates pass.

The scheduler owns the request state. Short contexts use a bounded block;
long contexts use physical page tables for SWA, FP4 main/index, candidates,
Top-512 and compressor state, with null block 0 reserved separately. Prefix
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
shutdown. The structural V2 bundle is the dedicated entrypoint's qualified
ordinary-C1 default; numerical candidates and DSpark remain default-off until
their complete gates pass. The generic entrypoint retains opt-in defaults.

## Default V2 HPU Adapter

The prepared entrypoint selects the HPU adapter for vLLM V2 scheduling and
asynchronous output by default; `--v2` is an explicit equivalent and `--no-v2`
selects the synchronous compatibility runner. This is not the CUDA/Triton GPU runner.
It requires the matching engine platform capability hook, native graph replay,
and direct token IDs. The engine uses its normal async scheduler and output
thread. On PP0, layer-1 Engram preparation continues from the sampled device
token while host C1 prepares only the later layer-14 input.

The incremental hook is provided in
`tools/communication/patches/dsv41-v2-platform.patch`. Apply it to the existing
prepared V4.1 engine, not a stock checkout; the adjacent JSON manifest records
the pinned revision and required parent file hashes. Other platforms still
require Triton, and all remaining V2 feature checks stay enabled.

Only ordinary C1 decode is supported. DSpark, inline completion, resumed
generated-prefix replay, and non-greedy sampling are rejected. Prefill keeps the
synchronous completion path. The full V2 device-continuation bundle passed an
exact three-run end-to-end cohort with a 10.4% median latency improvement; its
trace-localized turnover residual fell by about 60%. Unsupported sampling,
speculative and broader-shape contracts remain rejected explicitly.

The segmented-prefix path retains the complete PP0 compiled
graph and divides only command publication. It registers both late Engram
bindings before capture and stops the prefix before their first recipe read.
Compiler scheduling can place Engram unpacking in an earlier layer's recipe,
so the first Engram exchange is not a sufficient boundary. The versioned native
plan retains cross-boundary TP waits and actual producer completion offsets;
PP1 keeps ordinary replay. The device Engram producer hashes the sampled token,
gathers the production layer-1 rows, and decodes them before the suffix is
published. Request identity, history parity, late-input storage dependencies,
and the real scheduler authorization are checked before continuation. This is
the performance-qualified combination; the constituent switches are not
independent performance claims.

## Prepared runtime weights and long-context serving

Prepare the final N256 runtime layout once from the immutable TP2×PP2 shards:

```bash
.venv/bin/python tools/prepare_deepseek_v41_n256.py PREPARED_DIR RUNTIME_WEIGHT_DIR
.venv/bin/python -m vllm_gaudi.entrypoints.deepseek_v41 PREPARED_DIR \
  --runtime-profile RUNTIME_PROFILE \
  --n256-prepared-dir RUNTIME_WEIGHT_DIR \
  --max-model-len 1048576 --max-num-batched-tokens 8192 --max-num-seqs 1 \
  --block-size 128 --num-gpu-blocks-override 8193
```

The runtime profile supplies the matched native libraries, ABI manifests,
reserved devices, CPU placement and precision configuration. Rebuild both the
V4.1 kernels and the native replay bridge from this revision; previously built
bridges with a fixed PP0 collective count cannot replay long CSA2 buckets.
Do not disable V2 continuation, early device-token commit, device Engram or
segmented input replay in a wrapper around the dedicated entrypoint. Their
existing overrides remain available for diagnostics. Arithmetic-changing
optimizations still require the explicit precision profile.

Runtime preparation writes four rank files and publishes its manifest only
after all files pass exact inverse-layout checks. Layout, quantization, source
manifest and rank identities are validated before loading. Reads are bounded
and copy directly into the single resident compressed expert allocation;
Engram and dense weights retain their existing immutable sources. Partial or
stale prepared files fail explicitly. The cache is optional: omitting it
retains loading-time preparation.

The dedicated entrypoint enables `VLLM_HPU_DSV41_PREFILL_GROUPED`: routing is
bucketed by expert occupancy and each decoded weight is reused across its
selected prompt rows. The implementation preserves clamp, BF16 boundaries,
route order and ordered accumulation, including skewed routes and chunk tails.
It never reduces the scheduler's prompt admission to a single-token loop.
`VLLM_HPU_DSV41_PREFILL_MXFP4` remains a default-off stock-kernel experiment;
it is not the validated grouped-prefill serving path.

Each search bucket owns its native decode plan. Segmented PP0 replay accounts
for CSA2 index exchanges. Ordinary paged native serving captures every reachable
C1 search bucket before API readiness; this adds startup preparation rather than
compiling new decode geometries during a streaming response. Early continuation
requires both a prepared plan and matching current attention bindings. A bucket
transition enters native replay after rebinding; the following tokens resume
early continuation. CPU prefill completion must not bind an old device completion
token into the next C1 input. Packed KV remains canonical; only the active
working set is decoded.

Validation covers a prompt crossing an 8192-token chunk boundary, subsequent
short-request reuse, eight free-generation arithmetic samples, and public
streaming/non-streaming chat. It does not establish full-window quality,
vision qualification or long-duration reliability. Preserve the model and
runtime fingerprints with further qualification results.
