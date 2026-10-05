# TP4 C1 decode roofline — 2026-10-05

Official baseline: **9.564282 ms/token**, T=1, top_p=.95, seed=42,
16384 input → 1858 natural-EOS output, profiler off (`decode-physical-fusion-serving-03/formal/result.json`).
Current trace rank0 compute union **6.981807 ms**, stage noncompute **2.417838 ms**.
The detailed operator accounting below is historical `kernel-fusion-serving-02`, not a fresh attribution of 9.564.
That directory's REPORT still names an earlier 11.021 formal result; its physical timeline is usable independently,
but the folder name alone does not establish which later 9.997 formal request it represents.

All units decimal GB/TB and ms. Bytes/time is useful-byte throughput, not a hardware bandwidth-counter measurement.
A bandwidth lower bound excludes launch, dependency, reduction, address and pipeline costs. Rows overlap and **must not be summed**.

| Class | useful bytes/card/token | measured activity/boundary ms | effective TB/s | optimistic bound ms | gap / next decision |
|---|---:|---:|---:|---:|---|
| Total checkpoint weight demand | ~4.3 GB (user header audit: 2.9 dense + 1.36 routed) | 9.564282 formal | .450 | 2.150 @2.0 TB/s; 1.755 @2.45 | bound only; not all remaining time is removable |
| Peer producer→consumer handoff union | 81×10240 + 8×2048 + 9×256 + 2×3072 B local messages; remote copies separate | **1.577621** | not a bandwidth metric | ≤.600 target for 100×6 µs, not measured minimum | A: about .9 ms hypothesis; excludes double counting simultaneous query/weight points |
| Attention dense GEMV | ~1.54 GB | ~1.32 (user named-kernel aggregation) | 1.167 | .856 @1.8 TB/s; .675 @head's2.28 | B: .464–.645 ideal; qualify shared input and SRAM lifetime first |
| LM head GEMV reference | ~.331 GB | ~.145 (user aggregation) | 2.28 | .135 @2.45 | same-card comparator, larger shape; not guaranteed for small GEMVs |
| mHC FP32 controller, 80 calls | .1572864 GB = 80×24×20480×4 | ~.500 | .315 | .1049 @1.5 TB/s; .0786 @2.0 | C: .395–.421 ideal; official accuracy tolerance replaces legacy bit-exact gate |
| Routed FP4 decode | ~1.36 GB packed reads; expanded writes use SRAM when qualified | **1.779004 TPC** historical | .765 for packed reads only | cannot divide SRAM+HBM traffic by HBM roof | E: inspect load/store versus VPU issue budget; stop unqualified two-op lookup |
| Routed MME | decoded SRAM weights, not new checkpoint HBM reads | **1.099543** | not inferred | shape-dependent | overlaps TPC **.783195**, joint union **2.095352** |
| Attention TPC / CSA2 | tens of MB; exact traffic still unmeasured | ~1.69 / .82 historical user aggregation | unavailable | launch/instruction bound, not HBM roof | D: maintain qualified fusions/vectorization; inspect ISA before timing |
| Short gaps | no bytes | current ~1.30; historical .0–2µs1.628 +2–5µs.471 +5–10µs.423 | n/a | zero is not realistic target | overlaps producer→consumer handoff; never add both full totals |

## Evidence and communication facts

- `20260928_tp4-decode-gap-1p5/kernel-fusion-serving-02/NATIVE_POINT_SKEW.{md,json,csv}`:
  63 tokens, 180 compute calls, 100 peer points, no causality violations;
  producer max−min median3.414µs/P955.777µs, consumer median2.401µs.
  CSV gives producer completion and consumer start, **not NIC send/arrival**; the two missing timestamps
  must remain unknown until an observable device signal exists. No manufactured four-way time split.
- 81 large points are embedding plus Attention/MoE outputs; 8×(2048+256B) are index query/weight,
  2×3072B external Engram, remaining256B tail candidate. These 19 small points are not all removable.
- Current pending qualified micro forecast **.459858570 ms**, formal benefit **unmeasured**.
  Use ~40% realization as planning assumption; next formal trigger remains ≥1ms accumulated forecast.

## Hardware references and limits

- [Gaudi2 characterization](https://arxiv.org/html/2501.00210v1): 24 TPC,48MB shared SRAM,
  2.46TB/s HBM; small/fine-grained and communication behavior differs from bulk transfers.
- [Intel ExaComm](https://nowlab.cse.ohio-state.edu/static/media/workshops/presentations/exacomm23/Habana_ExaComm_2023.pdf):
  HLS2 has21 internal100Gb/s ports per device for7 peers. Standard allreduce uses reduce-scatter/allgather.
  This is **not proof** that our already-captured peer allgather uses that two-phase algorithm.
- [TPC architecture](https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/Processor_Architectural_Overview.html):
  retrieved locally after an initial429; average one256B vector read or write per4cycles, read/write port overlap unspecified.
- [MonoMoE](https://arxiv.org/abs/2609.04244), [Gaudi kernels](https://github.com/tracycam/gaudi-kernels):
  implementation references; no transferred speed claim.
- [Gaudi3 white paper](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf):
  architecture reference only; do not use its SRAM/bandwidth/core-count as Gaudi2 limits.

Priority: **A communication → B dense GEMV → C mHC → D qualified TPC work → E ISA verdict**.
Expected row gaps are hypotheses, outside the gain ledger until measured with production producer/consumer and native replay.

## First decisions from the A→E pass

**A, checked in source and native replay.** `stage_collectives` already uses rank-major
peer AllGather followed by ordered local sum. The captured HCL standalone-AG phase sends
once to the other ranks; it is not the ReduceScatter+AG path. Buffers/commands are already
prepared before replay. The historical trace has no NIC arrival timestamp, so ≤6µs remains
a target, not a demonstrated transport lower bound.
A private HCL candidate pruned otherwise-empty scale-out schedulers for single-box AG.
Production WO/control→peer→FFN/router, five inputs×four ranks exact; three native A/B differences
**−.000093648, −.000118359, −.000099031 ms/point** (negative means slower).
Case `decode-peer-prune-01`: **zero gain credit**, no serving change. Empty scheduler removal
alone is not the cause of the ~16µs producer/consumer handoff.

**B, current feasibility.** `FusedQKVInput.prepare_qkv_input_weight` already joins wq_a/wkv,
including the FP8 channel-scale path. Repeating this fusion adds no saving. Final compiler
symbol graphs in `decode-moe-dual-quant-01` show overlapping SRAM allocations across distinct
recipes (26 tensors, eight concrete collision examples in `decode-roofline-audit-01/SRAM_LIFETIME.json`).
Native replay binds persistent HBM operands; it does not reserve cross-recipe SRAM tensor
lifetimes. Therefore the existing path cannot safely keep an arbitrary prefetched weight
across another recipe. This is a static safety conclusion, not a measurement of a new
persistent-SRAM allocator. A cross-recipe prefetch would require allocation/lifetime and
DMA dependencies shared by all intervening recipes, not just an inserted DMA command.

**C, updated correctness contract.** The old high/low BF16 MME controller passes the official
mHC mathematical equations under DeepGEMM's normalized squared-error limits on five
checkpoint-derived inputs×four ranks. This executes a CPU equation oracle, not CUDA.
The initial complete boundary still loses **.01389 ms**: its post kernel recomputes the
20480-element RMS at each of 40 feature owners. A new producer computes the statistic once
and passes the existing25-value control layout to all consumers; it also passed accuracy but remained slower (below).

**E, stop the two-instruction lookup direction.** The actual unrolled W13 inner loop emits
8192 values with33 vector loads (including scales),32 stores,96 decode vector arithmetic
operations, and109 static VLIW bundles. Even assuming separate read/write throughput of
one vector/4cycles, the memory issue floor is132cycles; a shared port would be260cycles.
The official overview does not resolve those two interpretations, so neither is reported
as measured stall time. Both exceed the decode arithmetic issue count. Reducing96 arithmetic
ops to64 does not reduce either memory floor. Keep the already-qualified pipeline/unrolling,
and stop the speculative two-op LUT. SRAM output traffic is not HBM write traffic.
The architecture page was retrieved successfully to `/opt/ssd960/refs/tpc-architecture.html`;
ISA and calculations: `decode-roofline-audit-01/{expert.s,EXPERT_ISSUE_BUDGET.json}`.


C follow-up: one shared RRMS did **not** recover the MME regression (three rounds
−.013972680/−.013972555/−.013980027ms). Thus redundant RMS scans were real but not
the measured root cause. The compiled candidate places a second GEMM in the WO
producer; the added MME serialization is a structural explanation to investigate, not a separately measured stall attribution. No MME gain credited.
Keeping TPC and separating eight K accumulators passed official equations (maximum
normalized error7.679e-9) and saves .000995863ms per measured boundary; 40-boundary
forecast.039834531ms replaces the old.025901406ms. Total pending.458715133ms.

A follow-up: existing driver DMA-BUF mapping is usable (2/3,16KB exact). The legacy
TPC shared-page probe required a one-input GUID repair and then passed10KB TP2
correctness; ordinary host submissions measured20.86–20.89µs/batch call. This is
**not native TP4 latency** and receives no credit. Native port qualification remains open.

B API evidence: the serving runtime's `synapse/include/synapse_api_types.h` explicitly
marks `MEMORY_ATTRIBUTE_SRAM` as unsupported, while persistent storage is a separate
attribute. Do not model shared SRAM as a persistent cross-recipe cache. An allocator/
ownership extension is needed before cross-recipe prefetch; current recipes overlap.


## Further qualification and hardware incident

- `decode-mhc-split-k-01`: actual eight-way K partition (192 workpoints rather
  than24), combined with partial reduction inside the post consumer. Five
  fixtures×four ranks passed the upstream numerical contract,55 CPU checks passed.
  Three native-chain differences were **−.003891867/−.003892789/−.003883887 ms**.
  No credit; archived patch and restored production sources. K partitioning alone
  duplicates partial-reduction work at the40 post owners and adds scheduling waves.
- `decode-pcie-native-push-14`: two ranks, fixed addresses, double-buffered direct
  push,10KB. Native completion epoch535 verified; completed batch about9.19µs
  (device event includes a final host completion wait). This is a primitive
  qualification, **not** a production TP4 gain or a transport-only lower bound.
- Earlier native push12/13 sub-microsecond timings are invalid: a stream wait
  did not join the raw native replay queue. A recurrence check caught80 actual
  epochs versus535 requested. Those values are excluded from all totals.
- `decode-pcie-native-nrank-15` failed immediately after capture: the probe tried
  to wait on an unset native replay completion (target0). Its error path exited
  without unmapping the three imported DMA-BUFs on each rank. The driver retained
  three context references on each of modules0/1/4/5. No four-rank correctness or
  latency result exists. All owned processes exited; no other tasks were stopped.
- At18:21 targeted resets of these four unowned devices were requested after
  checking all process FDs. Driver reset repeatedly reports `compute_ctx refcount3`
  and cannot finish. Recovery coordination has been requested; **do not launch
  another cyclic DMA-BUF test**. A hard-killed service would expose the same
  lifetime risk, so this mapping design is not production-ready.

Official baseline remains9.564282ms, pending qualified forecast.458715133ms;
no new formal request and no claim of reaching7ms.

D follow-up: vector BF16 quantization passed5×4 complete-chain checks and3 positive
rounds on available modules2/3/6/7. Forecast only.001143438ms/token, not the9.3%
ISA reduction applied to model time. Pending total.459858570ms. User requested
continuing on available cards rather than coordinating a reboot now; affected
0/1/4/5 are not used. Telemetry records N/A instead of crashing or treating it as idle.

Latest native A/B extensions on available2/3/6/7: gain-only index replication
removes8 small collectives in the C1 candidate (forecast.030703250ms);
swizzled FP32 control weights add.005530156ms; direct shared-RRMS post adds only
.000290156ms, too small for a material timing claim. Current literal pending
forecast **0.496382133ms**, formal baseline unchanged.
Same-recipe SRAM weight streaming failed twice (3-slice and fitting1-slice),
so stop that approach. Early gates with the faster controller also regress.
All default-off until the combined formal gate; no bandwidth-counter claim.

Compiler follow-up (static): Gaudi2 TPC null descriptors program the NOP kernel
and completion signal registers (`TpcQueue::createNullDescRegsList`); they cannot
be deleted independently of completion accounting. Formal batch03 logs show
8 extra independent mHC partitions per four-layer group. The existing common
MERGE_LOCAL_SEGMENTS implementation has no completed historical timing (earlier
run interrupted by restart). Qualify it with the faster controller, retaining
native communication dependencies; do not infer savings from null count alone.

Repartitioning with the faster controller (`decode-merged-parallel-control-02`):
73→41 native compute segments with42 unchanged peer points, five states/four
ranks exact. Three real16 rounds slow by0.197/0.175/0.202ms;32 independent
control/communication windows disappear. Fewer recipe boundaries alone do not
win this tradeoff. No gain credit and no additional service trace.

Next B hypothesis: earlier dense-prefetch failures used a TPC weight-copy kernel.
The installed Synapse `MemcpyEngineManager` defaults Gaudi2 contiguous semantic
memcpy to `DmaMemcpy`. A graph-local nonpersistent SRAM section can test DMA
prefetch alongside activation quantization without cross-recipe lifetime
assumptions. Require an actual DRAM→DMA→SRAM→MME edge in the final compiler
graph, then the full WO/control→peer→FFN/router chain; unmeasured.

B DMA follow-up completed: final compiler graph proves10MiB DRAM→DmaMemcpy→SRAM→MME on all four ranks. Five states×four ranks exact, but the complete WO/control→peer→FFN/router chain regresses9.744/9.750/9.746µs. Archived candidate sources and removed unused operator. No gain.

A acyclic star protocol passed612 changing native epochs on2/3 with explicit importer-first cleanup. Moving reduction from the hub to each rank reduces two-rank completed primitive18.1661→16.0945µs, still too slow for6µs target beforeTP4 or real producer/consumer. Stop hardware expansion; no gain credit. This avoids cyclic driver ownership but is not a production-qualified transport.

Producer-side scheduling also regresses: real16 73→41 compute calls,42 peer points; five×four states and feedback exact. Three native pairs lose.142305/.150671/.130405ms. This rules out recipe-count reduction alone as an improvement with the current controller. Both partition variants are archived; further trials require a compiler-dataflow explanation, not another placement guess.

A width diagnosis: failed host protocol was an8-bit counter, not cache-line sharing. GEN_ADDR carries scalar-access type in ADRF; update_addr retains it. UINT32 override plus byte-offset updates repairs256-step wrap. Simulator checks through65536 and native612 epochs pass. Local two-rank completed primitive is12.883us, still above6us before a realTP4 producer/consumer chain. No forecast/serving credit; no hardware expansion.

A host-amortization qualification:100 exchanges in one native compute graph,61200 epochs exact. Two-rank five-owner protocol remains~10.62us/point; per-point host submission was not the missing~4.6us to6us. No production forecast; stop this transport family rather than expanding unsafe imports. Evidence:decode-host-shared-peer-05.
Current qualified pending forecast is0.634172625ms/token; formal baseline remains9.564282ms/token. New publish-mask/native-codec candidates remain outside gain totals until native complete-chain tests finish.
