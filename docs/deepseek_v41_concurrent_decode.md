# V4.1 ordinary concurrent decode candidates

These experiments preserve the existing N256 FP8 numerical algorithm and run
without DSpark. They do not change the context limit, KV capacity, Engram
history or ordinary chunked-prefill dispatch. They are not enabled by default.

## Batched input preparation

Within request-batch decode, independent one-token Engram hashes are prepared
together while each request retains its own history, generation and pending
transaction. Metadata is filled through the existing pinned allocation, and
the runner reads each committed token by position without copying its prefix.
These preparation changes apply automatically whenever request batching is
selected. Single-request decode retains native replay and V2 completion even
when the service reserves slots for concurrent requests.

The combined serving comparison retained one API process across the concurrency
matrix. Steady per-request latency decreased by about 7% at the larger tested
concurrencies; the largest-concurrency unmatched interval decreased by about
46%. These are measured comparisons with the earlier request-batch path, not
full-request or prefill speed claims. The component consumer outputs and state
contracts matched the saved reference exactly. Full-model outputs at larger
concurrencies still differ from the saved reference; that investigation remains
open, and this result does not establish complete output-quality qualification.

`check_deepseek_v41_batch_gap.py` exercises metadata, request histories, host
gather, pinned DMA and real native TP consumers with changing values and request
orders. Its endpoint is the downstream consumer completion. It does not model
the entire pipeline and must not be reported as end-to-end serving throughput.

## SRAM-bounded expert matrices

`VLLM_HPU_DSV41_CONCURRENT_MOE_ROWS=1` selects bounded direct routing for B64.
Values `4`, `8` and `16` expose grouped alternatives for diagnostics. Zero
retains the existing implementation. B1 through B32 always retain the existing
implementation; shape policies must be supported by a complete-chain result.

The native compound uses at most sixteen expert descriptors per connected
decode/matrix block. It preserves the original activation quantizer, N256
weight decoder, FP32 matrix products, projection rounding, SwiGLU/clamp,
activation rounding, output scaling and ordered top-six reduction. Grouped
variants quantize each original token once, keep route-slot inverses and
restore the original reduction order. No route is dropped under skew.

The grouping planner has fixed capacity independent of occupancy. An inactive
descriptor does **not** imply that MME work was skipped: the current candidate
still describes those matrix blocks. Compiled placement and complete consumer
timing must both be inspected. Group count or a micro GEMM result alone is not
a reuse-performance claim.

The option requires the existing fused N256 FP8 numerical configuration and
the new native operator. Invalid configuration fails explicitly. The execution
choice participates in the model/recipe fingerprint. No additional permanent
expert-weight copy is introduced.

## Reindex compaction and bounded replay capability

The native compactor preserves valid candidate slots in original order and
returns their source slots and device counts. Tail slots are reset on every
invocation. The scheduler-side tile bound is valid only for unique block pools
produced by Full selection; it must not be used for arbitrary duplicate pools.
It consumes already-owned host positions, never device-count readback.

`VLLM_HPU_DSV41_REINDEX_BOUNDED_PLAN=1` isolates the eight pure scoring tiles
from mandatory state writers, mask/Top-512 consumers and TP exchanges. It
requires batch Reindex MME, the mHC dependency partitioner, DSpark disabled,
and the matching versioned native runtime. It is experimental and off by
default. The normal model passes positions already owned by the scheduler;
the execution bound participates in the native replay call, not a tensor
count readback or a history-length compilation key.

The joint plan prepares all nine execution bounds before replay, sharing
captured recipes and device workspace. Each variant remaps mandatory
collective dependencies and completion targets. Only pure scoring segments
can be omitted: producers, consumers, state updates and the terminal segment
remain mandatory. One native entry still publishes multiple command pages.
Inactive scoring outputs may retain earlier values; mandatory downstream
masks exclude them before threshold selection.

The separate diagnostic executor uses independent tile programs, each of
which publishes its own terminal completion. Its timings do not qualify the
production joint-stage executor. Real model/TP integration and service
qualification are separate gates.

Long-context cases must retain the whole-chain tactic when splitting and
compaction cost more than the omitted work. Do not introduce compilation on
history-length changes to select a tactic.

### Building the bounded runtime

Use an isolated source/build tree. Apply the existing locked Synapse patch
bundle, then run `tools/communication/apply_bounded_reindex_runtime.py
--source <synapse-source> --apply`. This checks the exact input file and patch
hashes before changing source and checks the resulting hashes afterward.
Build Synapse and the native bridge from the matching source; do not replace
shared serving libraries. The normal bridge artifact metadata must identify
the actual Bridge, Synapse and HCL binaries. Missing bounded API symbols or
invalid optional-segment dependencies cause an explicit preparation failure.

## Validation tools

The optional two-microbatch runner uses `VLLM_HPU_DSV41_PP_MICROBATCHES=2`.
Both lanes share immutable weights and request-owned KV/history banks, with
separate native scratch, fixed metadata and transport packets. A transfer
stream joins each producer and each receiving consumer. Both forward packets
are submitted before reverse token broadcasts; the scheduler keeps one owned
transaction until all consumers finish. This is not a queue-depth increase.
The default whole-batch runner and small-batch dispatch are preserved.

- `check_deepseek_v41_reindex_compact.py`: device encodings, stable slots,
  request shapes and repeated tail clearing.
- `check_deepseek_v41_reindex_bounded.py`: changing lengths, fixed tile
  preparation, submission counters and the real packed-MLA consumer.
- `check_deepseek_v41_grouped_decode.py`: real expert weights through mHC,
  varied routing, ordinary execution and optional native capture ownership.
  `--capture-only` reuses timing evidence instead of timing the parent again.
- `check_deepseek_v41_batch_block.py --tp2 --compare-concurrent-moe 1`:
  same-precision real layer chains, live state and actual TP communication.
- `check_deepseek_v41_batch_block.py --tp2 --layer-start 24 --layers 4
  --compare-bounded-reindex`: real Reindex, MLA and layer consumers through
  the production partitioner and joined compute/collective executor.
- `check_deepseek_v41_bounded_joint_plan.py`: changing bounds, omitted-output
  poisoning, mandatory TP exchanges, completion targets and clean release.
- `check_deepseek_v41_batch_transport.py`: four-rank transport, pre-mix
  consumption, partial buckets, independent lanes and slot reuse.
- `check_deepseek_v41_pipeline_chain.py`: real four-layer fragments in both
  stages, complete PP transfer and residual sampling. Embedding, Engram and
  the vocabulary head are outside this component's scope. It does not replace
  whole-service qualification.
- `audit_deepseek_v41_sram.py` and `audit_deepseek_v41_reindex_graphs.py`:
  compiled operands and placement. These are not bandwidth counters or device
  event traces.

Use isolated devices/runtime fingerprints and retain unsuccessful runs.
Component equality is relative to the retained algorithm; it does not resolve
pre-existing quality differences in that algorithm. Production promotion still
requires joint-stage integration, combined TP2×PP2 service measurements,
independent quality, lifecycle, memory and concurrency-isolation validation.
