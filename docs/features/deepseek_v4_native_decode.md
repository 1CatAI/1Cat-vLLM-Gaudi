# Experimental DeepSeek V4 native decoder

The opt-in adapter replays compiled DeepSeek V4 decoder recipes and TP2
communication from the model entry. It remains disabled by default. Full-model
performance, production output equivalence and compatibility qualification are
required before promotion.

The current candidate passes the measured decode performance gate, but it is
not production-qualified. Frozen production token/logprob outputs differ, and
the standalone asynchronous native-chain diagnostic still reports an exit-time
heap failure. Repeated requests pass the tested destination-ordering regression;
that result does not resolve either remaining gate. Native decode, prepared
MXFP4 and Sinkhorn switches therefore remain disabled by default.

## Scope and execution

The adapter supports Gaudi2, TP2/PP1, BF16, one request, single-token decode,
43 layers, top-6 routing and context lengths up to 512. It uses five groups of
eight layers and a final group of three. The decoder retains all 86 layer
reductions and includes the final mHC, HC head and norm. Embedding reduction,
output projection and sampling execute through their normal surrounding paths.
The vocabulary mask and lookup share a compiled embedding program. The exact
mHC replication belongs to the first decoder group. Fixed-buffer input copies
use the native device-copy helper, which retains producer dependencies and
allocation lifetimes before publishing decoder commands.

Attention compilation receives explicit weight, cache and metadata tensors.
SWA and global sparse attention expose BF16 activations as actual graph
outputs. Their existing TPC programs retain the same tensor ordinals and
arithmetic, with the writable output moved from the input list to output zero.
Consumers therefore retain the attention producer and its cache dependencies
through dead-code elimination and partitioning.
Runtime router outputs remain part of the graph, so expert selection changes
with each token. Fixed input buffers receive new activation, token IDs,
positions and packed metadata before replay. Metadata ring slots retain
completion events and generations before reuse.
The physical decoder event is recorded on the execute thread and queried
directly through Synapse. It avoids the Bridge user-event mutex, which can be
held by another stream's result wait. The fixed metadata capture allocation is
separate from transfer-ring storage; retiring a transfer slot does not grant
permission to overwrite a later decoder's fixed destination.

The native C1 decode output copy is queued after the sampler on the existing
lowering and execution queues. The device helper retains producer waits,
substream ordering and the source allocation through DMA completion. A pinned
host buffer becomes readable only after its completion callback. Logical int64
tokens stored as int32 by the Bridge use the actual device wire type before
conversion to Python integers. Prefill keeps the existing output-copy path.
Derived CPU metadata uses integer array operations over the current block table
instead of scalar Tensor indexing loops; the packed transfer format is unchanged.

Preparation snapshots only the state ranges that capture may modify, restores
them after capture, then executes the real token once. Cache allocation,
weight reload, device migration and communicator changes invalidate plans.
Execution errors after state mutation terminate the call without retrying a
different implementation.

The plan explicitly retains KV, compressor and MTP allocations as both input
and output dependencies. Each captured recipe also owns its scratch workspace;
later ordinary launches can resize the Bridge workspace without moving these
captured addresses. Native program allocations cover both physical SCAL blobs
and recipe-declared reserved program ranges. Workspace residency is capped at
2GiB per plan and reported separately as `native_workspace_bytes`.

The prepared MXFP4 layout retains one compressed expert-weight copy. BF16
weight blocks use the native TPC-to-MME recipe. The experimental runtime
predates the stock packed-MXFP4 graph type, so prefill currently invokes the
prepared MoE program for each input row through an explicit compatibility
boundary. Include this path in prefill quality, peak-memory and complete-request
measurements. Its qualification is separate from single-token decode.

## Build and launch

Apply the pinned engine patch series with
`tools/prepare_deepseek_v4_engine.py`, build the DeepSeek native operators with
`tools/build_deepseek_v4.py`, and build the matching private replay runtime and
extension using the
[native runtime instructions](../../tools/communication/patches/native-runtime/README.md).
Keep the ABI sidecar with the extension. Select the matching private
Bridge/Synapse/HCL libraries before starting workers; the runtime loader checks
their fingerprints. Do not replace shared service libraries.

Enable `VLLM_HPU_DSV4_MXFP4_PREPARED_MME=1` and select the built replay extension
using `VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE`. The normal source entrypoint accepts:

```bash
.venv/bin/python -m vllm_gaudi.entrypoints.deepseek_v4 "$MODEL" \
  --native-decode-graph \
  --worker-cpus "$MAIN_CPUS" \
  --worker-helper-cpus "$HELPER_CPUS" \
  --num-gpu-blocks-override "$EXISTING_KV_BLOCKS"
```

The feature flag is `VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH`. It requires the prepared
communication, static group and joint plan flags, and the eight-layer grouping.
The V4 adapter does not require GDN. Keep the previous context, concurrency and
KV block count when comparing performance.

`VLLM_HPU_DSV4_TPC_SINKHORN=1` enables a separate experimental compiler fusion
for the complete39-step FP32 normalization chain in C1 mHC. It preserves the
surrounding projection, gates, softmax, epsilon additions and BF16 boundaries.
The kernel follows the production axis-specific reduction order. Other shapes,
parameters and chains with observed intermediate outputs remain unchanged.
This switch is disabled by default and does not enable the broader
`VLLM_HPU_DSV4_TPC_MHC` transformation.

`MAIN_CPUS` contains one CPU per rank. `HELPER_CPUS` contains one comma/range set
per rank, separated by a semicolon. Choose CPUs on each device's local NUMA node;
helper sets must be disjoint and exclude every main CPU and its SMT siblings.
The process's initial affinity must include all selected CPUs. Worker
initialization applies these bindings before measured requests.

## Execution evidence

`VLLM_HPU_TP2_PLAN_DUMP_DIR` saves prepared graph source and segment order per
rank. `VLLM_TORCH_PROFILER_DIR` also receives native plan statistics at profiler
boundaries and shutdown. Inspect actual compute segments, reductions, compute
pages, queue publications, NIC command storage and memory separately from the
single decoder entry. A single replay call is not a single hardware command.

The profiler HTTP routes require the normal engine profiler configuration;
setting the legacy directory variable alone does not register them. Append
`--profiler-config.profiler torch` and
`--profiler-config.torch_profiler_dir "$TRACE_DIRECTORY"` at startup.
`hcl_shared_stream_snapshots` reports live byte, wrap and publication counters
on the selected HCL streams; differences may include other users of those same
streams. The retained `native_hcl_*` fields describe the first replay snapshot.

Profiler start and stop retire native command copies after device completion,
while keeping compiled recipes. The next decode rebuilds the command copies
using the normal capture snapshot/restore transaction. Validate hardware
attribution independently: cached full-model recipes can still produce anonymous
events after command-only recapture. Counters are identified by
`native_program_generation`; compare deltas within a generation. Exclude that
generation's capture call from steady replay statistics.

For performance qualification, retain complete requests and raw token arrivals,
capture both ranks, and exclude profiler-active requests from latency scoring.
Verify that steady decoder execution contains no group Python calls, graph
compilation or old per-node launches. Inspect compiler SRAM placement and
weight-buffer lifetimes. Validate every attention type and its cache writes:
layer/reduction counts alone cannot establish complete computation. The
`tools/check_deepseek_v4_native_attention.py` device regression requires exact
nonzero attention outputs and updated KV pages across changing positions.
Complete the production token/logprob, state-boundary,
reload and long-replay checks after the performance gate. The separate TPC
decode activity target is not satisfied by reducing host submission overhead.

`VLLM_HPU_DSV4_NATIVE_REPLAY_AUDIT_DIR` enables paired full-decoder diagnosis
on identical inputs and snapshotted write ranges. It records ordinary/native
group outputs and state writes at selected replay steps, restoring state before
the real invocation. These additional executions and CPU copies must be absent
from performance qualification. A paired same-runtime match alone does not
qualify against the frozen production reference.
