# DeepSeek V4.1 Flash prepared execution

This experimental profile is under qualification. Enabling a switch is not
evidence of completed model, numerical, memory or performance validation.
The initial contract is Gaudi2, TP2×PP2, one request, context up to 512 tokens,
and greedy sampling. Unsupported sampling modifiers fail explicitly.

## Preparation

Use the pinned V4.1 engine revision and the matching native replay runtime.
Preserve their source revisions and binary fingerprints with the prepared
manifest. The original Hugging Face checkpoint is the recovery source;
the four prepared files are immutable derived artifacts.

```bash
.venv/bin/python tools/audit_deepseek_v41_upstream.py --help
.venv/bin/python tools/sync_deepseek_v41_checkpoint.py MODEL_DIR --evidence CHECKPOINT_AUDIT
.venv/bin/python tools/prepare_deepseek_v41_shards.py MODEL_DIR PREPARED_DIR \
  --checkpoint-audit CHECKPOINT_AUDIT --upstream-lock UPSTREAM_LOCK
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
Dense block-FP8 matrices are converted with bounded host chunks to BF16 MME
weights; there is no second persistent FP8 dense device cache.

## Execution

After setting the validated runtime library paths and native bridge variables,
explicitly enable the five profile switches:

```bash
export VLLM_HPU_DSV41_PREPARED_SHARDS=1
export VLLM_HPU_DSV41_ENGRAM_HOST_TABLE=1
export VLLM_HPU_DSV41_GRAPH_REPLAY=1
export VLLM_HPU_DSV41_DSPARK=1
export VLLM_HPU_DSV41_VISION=1
export VLLM_HPU_TP2_STATIC_GROUP_PLAN=1
export VLLM_HPU_TP2_PREPARED_COMM=1
.venv/bin/python -m vllm_gaudi.entrypoints.deepseek_v41 PREPARED_DIR \
  --checkpoint-audit CHECKPOINT_AUDIT
```

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

For ordinary single-token decode, set `VLLM_HPU_DSV41_DSPARK=0` before
starting the same entrypoint. It omits speculative configuration, skips all
`mtp.*` tensors while reading the immutable rank files, and creates no draft
state or target-state auxiliary outputs. The normal runner commits one output
token without proposal or verification. Native warmup captures C1 only;
multi-token prefill retains its compiled compatibility path and tail handling.
This mode is under separate performance and quality qualification.

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
shutdown. Default enablement remains off until these checks pass.
