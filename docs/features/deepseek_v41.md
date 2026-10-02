# DeepSeek V4.1 Flash prepared execution

This experimental profile is under qualification. Enabling a switch is not
evidence of completed model, numerical, memory or performance validation.
The qualified ordinary decode contract is Gaudi2, TP4×PP1, one request and greedy sampling.
The shared implementation also accepts TP2×PP2; this change does not qualify its performance again.
Paged state supports configuring up to 1,048,576 tokens; prefill uses scheduler
chunks of up to 8192 tokens. This is a capacity contract, not a claim that a
full-window request has passed quality qualification. Unsupported sampling
modifiers fail explicitly.

With occupancy-based grouped prefill enabled, the BF16 expert path chooses
64/128/192/256-row blocks from the current expert route counts. Larger experts
use multiple bounded slabs. Each slab reuses one decoded weight across its
rows, while the final route reduction preserves token/top-k order. Group width
keeps each route write within the native operator's row limit. These buckets
apply to arbitrary scheduler prompt lengths; they do not select a fixed-length
prompt fast path.

Sparse prefill MLA gathers directly into the KV layout consumed by SDPA.
Invalid indices and the attention sink select a dedicated zero row, appended
once per compiled call and shared by its bounded query tiles. This preserves
NaN isolation and the existing BF16 KV/FP32 mask contract for arbitrary query
lengths, including partial tiles, while avoiding a separate selected-KV
masking and sink-concatenation pass.

## Preparation

Start from a validated V4.1 engine and its matching native replay runtime.
The complete serving engine source delta is pinned in
`tools/communication/patches/dsv41-serving-engine.json`. Apply it to the pinned
clean vLLM checkout with `tools/prepare_deepseek_v41_engine.py` before building
the engine. Preserve the actual engine source and runtime binary fingerprints
with the prepared manifest. See the
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

### Prefix caching

Ordinary paged serving can opt into prefix caching with
`--enable-prefix-caching`, or `"enable_prefix_caching": true` in the installation
settings. Explicit CLI flags take precedence. The matching pinned engine delta
is required; workers reject an older engine before loading model state.
Prompt usage exposes `prompt_tokens_details.cached_tokens` when enabled.

A hit requires retained compressed KV pages and a checkpoint of SWA,
compressor and Engram history acknowledged by every TP/PP rank. Checkpoints
use complete 128-token page boundaries, retain a final uncached prompt page,
and have bounded slot/byte budgets. Request-slot reuse reconstructs decoded
mirrors before native replay consumes them. Equal text alone does not imply a
hit: the token prefix must match. Checkpoints currently capture the prompt
boundary; generated tokens do not automatically extend that auxiliary
checkpoint for the next chat turn. Speculative execution does not support this
checkpoint contract. The option remains off unless requested.

For complete long-prompt tiles, the worker saves the checkpoint's small ring
states directly from the live producers, including a trailing decoder halo.
It preserves the complete prefill invocation and the request's final state;
publication still requires every state, retained page and rank acknowledgment.
Other prompt transactions retain the split-checkpoint path. Full slot-owned
prefill and checkpoint producer coverage warm before API readiness.

Installation settings may set `"max_num_batched_tokens": 16384` to retain the
qualified complete TP4 prompt tile. Explicit CLI flags take precedence. This
changes the scheduler token budget; the context and request-slot limits remain
independent settings.

The dedicated entrypoint enables the frozen-reference ordinary-C1 and V2
device-continuation bundle by default. Launch without a feature-variable list:

```bash
VLLM_ENGINE_READY_TIMEOUT_S=3600 .venv/bin/python -m vllm_gaudi.entrypoints.deepseek_v41 PREPARED_DIR \
  --checkpoint-audit CHECKPOINT_AUDIT
```

Set `--no-v2` to retain the synchronous runner, or
`VLLM_HPU_DSV41_DEFAULT_FASTPATHS=0` to suppress the entrypoint's default
injection. The aggregate opt-out does not unset existing variables or
reconstruct a reference environment. Individual feature overrides belong in
the effective runtime configuration; see
[configuration precedence](../1cat_gaudi_guide.md#runtime-profile). The generic
vLLM entrypoint retains the opt-in defaults from `vllm_gaudi.envs`.

The qualified FP8, Router, MLA and fused numerical profile is part of the
dedicated entrypoint's default bundle. Prepare the two FP8 sidecars in
`PREPARED_DIR/sidecars/wo_a_fp8` and
`PREPARED_DIR/sidecars/attention_dense_fp8`. The entrypoint validates and
discovers both sidecars before model loading and fails before startup when a
required artifact is absent. Set
`VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS=0` to select the structural
compatibility profile for diagnostics.

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
C1 bundle; the generic entrypoint retains opt-in environment defaults. The prepared numerical
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

The default C1 bundle keeps the separate legacy `VLLM_HPU_DSV41_FP8_DECODE`
and coordinate-pipeline experiments disabled. It enables the deployed N256
FP8 expert, Router-top6, MLA-MME and dense FP8 profile through the numerical
aggregate. Native PP0 input capture remains part of the V2 continuation
contract. The decoded SWA mirror uses the same circular row namespace as the
packed cache, and the hot index mirror is populated before its first larger
search bucket; these state repairs are required for long-generation quality.

The scheduler owns the request state. Short contexts use a bounded block;
long contexts use physical page tables for SWA, FP4 main/index, candidates,
Top-512 and compressor state, with null block 0 reserved separately. Prefix
caching is disabled. Rebinding state invalidates compiled replay addresses;
errors after state writes abort execution rather than retry another path.

Engram uses `MAP_POPULATE`, `MADV_WILLNEED` and native gather workers. Only its
small DMA staging buffers require HPU-pinned memory. Explicit forced table
locking is available via `engram_force_lock` in the loader configuration.
The default does not assume the full host tables fit the memlock quota.

### Resident Engram tables

`--engram-residency locked --engram-host-budget-gib 224` makes table residency
a startup contract. Device contexts and communicators are created first; the
table owners then prefault and lock the shared host shards before model loading
continues. An insufficient memory budget or lock quota fails startup rather
than silently accepting a paging-dependent performance profile.

Admission credits checkpoint pages already locked in fully resident, read-only
shared mappings, after checking file identity and byte extents. Overlapping
mappings count once. Reclaimable file cache is already included in the OS memory
estimate and receives no extra credit; device memfd allocations still require
their own budget. Each service keeps independent residency owners for its lifetime.

The CPU gather and device-addressed first Engram layer share one backing region
per TP shard. On drivers that require writable host registration, the backing
is a size-sealed shared memfd populated from the immutable checkpoint, then
mapped read-only for consumers. This avoids private file-mapping COW during
long-term driver pinning. The other Engram layer retains its read-only file
mapping. Neither table is replicated per PP rank or copied to HBM. The original
checkpoint remains unchanged, and its temporary file-cache pages are not locked
alongside the replacement shared backing.

The bridge must advertise `device_engram_shared_mapping_version=1`, and the
normal ABI and binary fingerprint checks still apply. Startup owns the backing
processes until shutdown, records residency, and removes readiness if a table
owner exits. This deployment setting is independent of arithmetic fast paths.

### Bounded Prefill implementation

The Prefill candidates use device route descriptors, compiled BF16 expert
bodies, native recipe replay and ordered route reduction. Descriptors and
workspace addresses are bound explicitly; changing routing values does not
create an occupancy-specific recipe. Full expert weights are not retained as
BF16 tensors between calls. Workspaces are reused only after their consumers
complete, and weight or state rebinding invalidates dependent plans.

The sparse-MLA candidate reuses a bounded decoded index-key prefix across
Reindex query tiles, restores scores to the original candidate-slot order,
and keeps the original top-k merge boundaries. Its Gaudi FlashInfer adapter
uses the installed Habana attention backend; it does not load CUDA kernels.
An explicit zero-valued sink row retains the head-dependent softmax denominator.
Invalid cache rows are zeroed before the PV product, not merely masked.

The dedicated entrypoint enables the qualified C8192 expert transaction,
bounded FlashInfer-Gaudi MLA regions, transaction-local decoded-KV reuse,
two-slot PP wavefront and TP2 index-query partitioning. Query partitioning is
limited to the qualified C1024-C8192 shapes; smaller graphs retain the local
selection path. The complete 32K request and subsequent decode were qualified
together; the generic vLLM entrypoint keeps each feature opt-in. Unqualified
alternatives, including grouped FP8 prompt experts, remain disabled until their
complete serving gates pass.

## Qualification

First check text and image execution, accepted/rejected DSpark verification,
all target/draft layers, state writes, SRAM placement and native replay.
Then capture a complete four-rank trace and absolute request metrics. Report
each kernel's dtype, shape, latency, count and share in milliseconds, with
device-window reconciliation and overlap handled explicitly.

The complete quality gate additionally covers frozen-reference generation,
image spans, accepted-prefix rollback, request changes, state reuse and
shutdown. The structural V2 bundle and prepared numerical profile are the
dedicated entrypoint's qualified ordinary-C1 defaults. DSpark remains
default-off, and the generic entrypoint retains opt-in defaults.

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

Only ordinary C1 decode is supported. DSpark, inline completion and resumed
generated-prefix replay are rejected. Ordinary sampling supports temperature,
top-p, top-k and seed; speculative verification still requires greedy sampling. Prefill keeps the
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
  --max-model-len 1048576 --max-num-batched-tokens 32768 --max-num-seqs 8 \
  --block-size 128 --num-gpu-blocks-override 8193
```

The runtime profile supplies the matched native libraries, ABI manifests,
reserved devices, CPU placement and precision configuration. Rebuild both the
V4.1 kernels and the native replay bridge from this revision; previously built
bridges with a fixed PP0 collective count cannot replay long CSA2 buckets.
Do not disable V2 continuation, early device-token commit, device Engram or
segmented input replay in a wrapper around the dedicated entrypoint. Their
existing overrides remain available for diagnostics. The prepared numerical
profile is selected automatically and remains explicitly disableable before
startup.

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

The example reserves a longer engine initialization timeout for the first full
context bucket preparation. This is an upstream startup timeout, not a request
timeout or a context limit. Monitor bucket progress during initialization.

Validation covers a prompt crossing an 8192-token chunk boundary, subsequent
short-request reuse, eight free-generation arithmetic samples, and public
streaming/non-streaming chat. It does not establish full-window quality,
vision qualification or long-duration reliability. Preserve the model and
runtime fingerprints with further qualification results.

### Prefill decoder column layout

The dedicated entrypoint enables occupancy buckets automatically when N256
expert storage is selected. Every expert uses full 256-row slabs plus a
remainder rounded to 64, 128, 192 or 256 rows. Native replay submits only the
occupied recipe prefix and keeps route workspaces alive through their consumers.
Set `VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS=0` to select fixed-row plans for diagnosis.

The occupancy-bucket path automatically selects a native decoder for
qualified normal-scale BF16 expert weights.
It retains the converter's even/odd column order within each N256 weight tile,
restores W13 columns on the smaller activation tensor, and restores W2 columns
after the original ordered six-route reduction. The prepared compressed weight
layout and allocation are reused. Selection does not depend on prompt length.

Rebuild the native kernel, PyTorch libraries and TP2 Bridge when updating this
code. The Bridge must expose `replay_prepared_groups_prefix`; workers validate
required MoE operators before model allocation.
Component checks cover independent decoder agreement, route bucket boundaries,
changing inputs and the existing BF16 arithmetic boundary. Normal serving
checks cover the frozen prefill request and two other prompt lengths. The
decoder is not selected for optional FP8 grouped-prefill modes or non-normal
scale encodings. Set `VLLM_HPU_DSV41_PREFILL_COLUMN_INTERLEAVE=0` to disable it
for diagnosis.

Grouped FP8 modes remain explicit experiments selected with
`VLLM_HPU_DSV41_PREFILL_GROUPED_FP8`. Dual W13 modes use high and residual
activation terms; `w13_single_prequant` uses one term and has greater
quantization error. `w13_single_bucket` uses the same single-term arithmetic
with occupancy buckets rounded to 32 rows, retaining BF16 W2 and ordered route
reduction. These modes change rounding and require workload quality validation.
The default empty mode keeps BF16 arithmetic.


## Independent C1 installation

The dedicated entrypoint selects the qualified shared C1 stage replay, native
greedy tail, numerical preparation fusions and packed SWA automatically. Only
pipeline-specific operations are excluded for PP1. Router and LM head retain
BF16. The ordinary grouped prompt path retains BF16; grouped FP8 prompt modes
remain explicit diagnostics. This does not enable a DSpark configuration.

The input/shared-expert FP8 sidecar contains every selected projection and a
`precision.json` file. The entrypoint discovers that file alongside the sidecar,
so the accepted dense precision does not require an optimization environment
variable. An explicit precision override remains available for diagnosis.

Build the matching native kernel extension, host gather, peer bridge and pinned
Bridge/Synapse/HCL runtime using the maintained build tools. The concat-axis
cache-key patch or its fingerprinted adapter is required; do not globally disable
shape-agnostic compilation. Every native source in the build manifest must match
this plugin revision. Apply the complete serving engine delta first:

```bash
python tools/prepare_deepseek_v41_engine.py ENGINE_CHECKOUT --check
python tools/prepare_deepseek_v41_engine.py ENGINE_CHECKOUT
```

Materialize the deployment using the matching Python environment. The runtime
profile identifies SDK options and installed artifacts, rather than a list of
performance opt-ins. The installer verifies the engine delta, native source and
input library fingerprints, copies Python dependencies, engine and all plugin
packages, removes editable import hooks, and relocates library/ABI paths.
Model checkpoints and the installed system Gaudi driver/SDK remain prerequisites.

```bash
SERVING_PYTHON tools/install_deepseek_v41_runtime.py \
  --runtime-profile QUALIFIED_ARTIFACT_PROFILE.json \
  --engine-source ENGINE_CHECKOUT --prepared PREPARED_MODEL \
  --machine-settings MACHINE.json --output INSTALLATION
python tools/serve_deepseek_v41.py INSTALLATION --port 18552
```

Use a new installation directory. `--asset-output STABLE_ASSET_DIRECTORY` can
place immutable sidecars on their source filesystem. Independent hard links
avoid duplicating their data and survive removal of the original build directory;
never modify installed asset files in place. Other files are materialized without
workspace symlinks. Retain `installation.json` and the library ABI manifests.

Use `docs/configuration/examples/deepseek_v41_tp4_machine.json` as the machine
settings template; adapt its NUMA allocation to the actual host. Relative lock
paths resolve beside the release directories so successive installations share
their leases. The installer creates a private API key unless a key file is
already configured.

Machine settings contain `cpus`, `device_lock_dir`, `environment` with physical
modules and worker CPU pools, and `cpu_allocation` with `worker_main`,
`worker_helpers`, `engine_main`, `api_main`, and `control_helpers`. They describe
resources, not model optimizations. Include one main and one helper pool per
rank. The supervisor waits for full startup before maintaining per-thread
allocation, including threads created after the first request. Its records are
written atomically. The service keeps normal 1M context capacity, 32 request
slots, 8192-token prompt chunks and all warmup buckets.

Optional `isolate_user_processes` moves same-user background threads off the
reserved physical cores and records their original masks. Processes owned by
other users require machine administration. Qualification uses CPU PSI and
actual process/card ownership; driver D-state threads make load average unsuitable
as a CPU-contention gate. Record competing card and CPU activity during acceptance.
An optional `api_key_file` supplies `VLLM_API_KEY` without exposing the key in
process arguments. Keep that file private; expose a separate authenticated API
port so other services keep their existing routes.

`--settings INSTALLATION/settings.json` on the normal model entrypoint bypasses
the user's historical installation config. Startup, module leases, native ABI
checks, full warmup and normal Chat API execution therefore work without the
checkout's environment variables, temporary graph dumps, manual thread binding,
or an editable source directory. Validate a new installation with one unprofiled
16K-to-natural-EOS request and the fixed semantic cohort before replacing the
running release. Profiled timing and component estimates do not qualify it.


For a public tunnel, run `tools/serve_deepseek_v41_api.py --api-key-file KEY_FILE`
on a separate loopback port and expose that port. This streaming adapter allows
only model queries, Chat/Completion generation and health; it rejects internal
RPC, metrics and profiler routes even with a valid key. The backend remains
loopback-only. Use the published model name and `temperature: 0` for the qualified
ordinary greedy path. Ordinary decode also supports the model-recommended
`temperature: 1.0` with `top_p: 0.95` or `1.0`; probability filtering runs on the
device and request-owned random draws agree across TP ranks. These settings can
change output length and incur sampling overhead, so greedy performance results
do not qualify stochastic sampling. The recommended temperature/top-p pair
passed natural-EOS checks and a fixed semantic cohort separately; stochastic
sampling is not yet qualified against the greedy latency target.
Requests with unsupported penalties,
logprobs or constrained sampling fail explicitly.
