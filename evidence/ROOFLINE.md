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
- Current pending qualified micro forecast **.444782008 ms**, formal benefit **unmeasured**.
  Use ~40% realization as planning assumption; next formal trigger remains ≥1ms accumulated forecast.

## Hardware references and limits

- [Gaudi2 characterization](https://arxiv.org/html/2501.00210v1): 24 TPC,48MB shared SRAM,
  2.46TB/s HBM; small/fine-grained and communication behavior differs from bulk transfers.
- [Intel ExaComm](https://nowlab.cse.ohio-state.edu/static/media/workshops/presentations/exacomm23/Habana_ExaComm_2023.pdf):
  HLS2 has21 internal100Gb/s ports per device for7 peers. Standard allreduce uses reduce-scatter/allgather.
  This is **not proof** that our already-captured peer allgather uses that two-phase algorithm.
- [TPC architecture](https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/Processor_Architectural_Overview.html):
  initial web fetch returned429; cached SDK/ISA must verify Gaudi2-specific load/store throughput and whether read/write overlap.
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
and passes the existing25-value control layout to all consumers; verification pending.

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
producer; moving TPC work onto the same MME removes engine overlap. No MME gain credited.
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
