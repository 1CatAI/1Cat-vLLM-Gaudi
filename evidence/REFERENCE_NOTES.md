> 2026-10-05 direction update: user accepted batch03 **9.564282 ms/token** as the formal baseline and requested its six active paths as entrypoint defaults. Stop recipe-ID attribution and archive output/post-norm/WOa small fusions. Active work is now **MoE SRAM pipeline first**, then the MLA gather/decode→QK→softmax/sink→PV body. Start a new gain accumulator at0; official acceptance after ≥1ms accumulated expectation.

# TP4 C1 decode: upstream reference notes

Updated 2026-10-05. This is an implementation study and design record, **not a gain ledger**.
The accepted parent formal result is 9.997343 ms/token. Batch03 now measured
9.564282 ms/token with official sampling, natural EOS and 14 semantic facts passed.
Its 0.433061 ms saving is below half the 1.123126 ms forecast; defaults are not
promoted while the companion four-rank trace is analyzed. Earlier interrupted
physical-fusion launches produced no formal result.
Targets of roughly 20 physical nodes/layer and 6 ms/token are unverified targets.
Study is ongoing; this file does not claim that all requested study time or hardware validation is complete.

## Reproducible sources

Local clones are under `/opt/ssd960/refs`; `decode-reference-manifest.json` records revisions.

| Source | Revision / status | Files inspected |
|---|---|---|
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7` | `tests/test_mega_moe.py`, `csrc/apis/{mega_moe,mega_mhc,attention}.hpp`, `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.cuh` and Mega mHC implementation |
| [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend) | `8491bbb4b8c02a094a2318965f50c70438a3e73c` | `deep_gemm/include/deep_gemm/{mega_moe,fp8_gemm}.hpp`, `simd/mega_moe.hpp`, `scheduler/mega_moe.hpp`, `layout/mega_moe.hpp`, `ascend/{sync,copy}.hpp`, `tests/test_mega_moe.py` |
| [TileKernels](https://github.com/deepseek-ai/TileKernels) | `66258df6175d2f630ffecb04c5ab66bff8a2ae6a` | Ascend implementations in `quant/`, `mhc/`, `moe/`, `engram/` |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | `2e5429fc5653bab6e081f09477126f731882a6a9` | `flash_mla/fused_norm_rope_attn_rope_cast.py`, fused-attention and sparse-decode tests, SM100 fused attention implementation |
| [DeepSelect](https://github.com/deepseek-ai/DeepSelect) | `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` | `docs/DeepSelect-deep-dive.zh.md`, `csrc/ascend_kernels/kernel.asc` |
| [SGLang](https://github.com/sgl-project/sglang) | `f385390be593575cda74cda06bc3a82591d52b18` | `python/sglang/srt/layers/attention/dsv4/{metadata,indexer,dsv41_sparse}.py`, `kernels/ops/attention/dsv4/{metadata_kernel,c2_decode_pool}.py`, `kernels/jit/csrc/deepseek_v4/c2.cuh` |
| [SGLang mHC PR 42245](https://github.com/sgl-project/sglang/pull/42245) | **Open, not merged** when checked; head `70830c8ac37f774fd975e4b1e5cf5fc9b6822b2c` | extracted `deepseek_v4_mhc.py` in `/opt/ssd960/refs/sglang-pr42245-mhc.py`; main checkout unchanged |

The requested SGLang filenames `c2.py`, `indexer_postprocess.py`, `q_rope_store.py`,
`wo_a_bf16_small_batch.py` are not present at these pinned revisions. Above are the
actual current equivalents inspected; do not cite nonexistent paths as evidence.

Additional primary sources:
- [V4.1 report §3.2](https://arxiv.org/pdf/2609.19969): reports 11 decode kernels for a CSA2 Reuse layer. This counts its implementation, not Gaudi physical nodes.
- [SGLang optimization article](https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization): plain BS1 approximately 203.3 tokens/s (~4.92 ms); the 873.63 tokens/s result uses DSpark and is not the plain-decode comparison. It describes shared layer metadata and overlapping mHC statistics.
- [vLLM V4 article](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md): combined compressor/cache stores and RoPE/quantization boundaries; its V4 compression ratios cannot be substituted for V4.1's local 1/2 ratios.
- [SGLang day-zero article](https://www.lmsys.org/blog/2026-09-10-deepseek-v41/): architecture/integration context, not a Gaudi performance prediction.

## What transfers to Gaudi

Ascend's Cube/Vector separation is a useful analogy for Gaudi's MME/TPC. It does
not imply identical ISA, matrix scale support, SRAM visibility or synchronization.
Gaudi2 has 24 TPCs, two MMEs and 48 MB shared SRAM; TPC private registers are not
MME operand storage. A CUDA fused kernel can contain matrix instructions. A
normal Gaudi TPC kernel cannot simply invoke MME matrix multiplication.

Our unit of success is the actual TPC/MME/DMA dependency graph, including the
consumer, rather than one Python call or compound-op registration. Preserve
FP32 accumulation order, BF16 rounding boundaries, scales and cache ownership.
Count physical nodes and measure native replay; the historical ~2 microseconds
per removed node is a heuristic, not a guaranteed additive service saving.

## Ascend MegaMoE: concrete pipeline mechanics

`mega_moe.hpp` dispatches a mixed kernel into AIC GEMM work, AIV0 weight/decode
work and AIV1 epilogue/communication work. The important details are:

1. **Bounded buffering with ownership.** The inspected schedule has two L0A/B
   stages, three L1A stages, four routed L1B stages and separate scale/epilogue
   stages. AIV0 uses three packed-FP4 UB buffers and two expanded FP8/NZ buffers.
   Its producer waits for an empty stage, publishes completion after transfer,
   and does not overwrite until the matrix consumer releases that stage.
2. **Prepare the consumer's layout directly.** `vf_fp4_to_fp8_nz` expands nibbles
   with shifts/masks and scatters to padded NZ layout. The exponent compensation
   is carried into scale metadata. Gaudi's current N256/SAT codec and scales
   differ: copying the shifts or +6 compensation without a codec proof is wrong.
3. **One kernel does not mean no global intermediates.** The AIV1 path still
   stores SwiGLU/quantized activation and scales in GM before publishing a ready
   flag. Both traffic and readiness must be checked; source-level fusion alone
   does not prove SRAM residency.
4. **Resumable progress, not host scheduling.** `advance_combine_push` moves a
   bounded number of ready rows and yields to the next pipeline task. Cursor and
   credits persist. `advance_combine_reduce` waits for the actual routed/shared
   producers, then consumes ordered chunks. The coroutine description refers to
   this explicit device state machine; it is not evidence of C++ `co_await`.
5. **Compact routed work.** `ExpertCursor` enumerates nonempty experts and valid
   rows. AIC/AIV replay the same deterministic task sequence. The linear-task
   scheduler allows limited first-linear lookahead for second-linear work.

Gaudi design: keep six selected routes in a compact device plan; decode straight
into MME-consumable tiles; retain explicit full/empty dependencies and fixed
expert reduction order. W13 may cover all six routes; W2 can use two groups of
three only if compiled SRAM lifetimes and physical node counts confirm it.
Do not copy Ascend buffer sizes, NZ descriptors, MX-scale instruction assumptions
or event APIs. The available Gaudi compiler/replay must express the same ownership.

Local failure to avoid: one producer exposing two connecting tensors to the same
MME triggers `producerMultipleConnectingTensorsChainBreaker`; previous batch-GEMM
candidates then split and lose the intended SRAM path. The horizontal W2 implementation computes off-diagonal products; its production
component later passed SRAM, physical-node and paired timing checks (recorded
below). Full serving qualification remains pending.

## NVIDIA DeepGEMM: scheduling and arithmetic boundaries

MegaMoE assigns persistent roles to dispatch, transfer, matrix work and epilogue,
using ready/empty barriers and fixed task buffers. Mega mHC carries previous mix
state, controls residual construction and optionally emits quantized input.
Barrier storage is initialized before capture. These are useful scheduling and
state-lifetime patterns; SM100 thread roles are not Gaudi execution primitives.

`csrc/apis/attention.hpp:174` takes I32 page tables, lengths, optional row indices
and a precomputed schedule. It creates views of packed KV/scales without repacking,
and allocates logits at an aligned physical stride while exposing the logical
visible length. Its block-size and SM-generation restrictions are not model
semantics. Reuse page/schedule metadata across our indexer layers; do not repeat
casts or flatten/repack the cache in each consumer.

## TileKernels: fuse a complete numerical boundary

- `quant/swiglu_forward_asc.py`: clamp, activation, route weighting, amax and FP8
  with consumer-oriented scale layout share a vector pipeline. Its fast division
  has a specific BF16/FP8/power-of-two-scale condition; retain exact fallback and
  our existing rounding, rather than silently adopting approximate arithmetic.
- `quant/norm_forward_asc.py`: norm, optional residual/weight and optional BF16/FP8
  outputs share reduction work. UB budgeting includes staged inputs and scales.
- `mhc/pre_apply_mix_asc.py`: load small mix controls once per tile, preserve FP32
  mixing order, then round output. `mhc/sinkhorn_asc.py` packs small independent
  rows into vector lanes and keeps transpose/reduction work in registers.
- `engram/engram_gate_fwd_asc.py`: combines norm statistics, dot, gate and update;
  masked rows and state ownership remain part of the contract.
- `moe/topk_gate_asc.py`: route scores stay in registers, repeated maxima use
  deterministic minimum-index ties, and invalid values are guarded. This is a
  small expert-router implementation, not a ready-made whole-vocabulary sampler.

Apply the principle “at most one TPC numerical boundary between adjacent MMEs”
where the dependency graph permits it, while retaining every required BF16
rounding point inside that TPC. Separate mandatory branches are not removable
merely because they look like duplicated normalization.

## FlashMLA: fuse addressing and output preparation

The inspected fused API uses I32 positions/page indices and handles invalid rows.
The 288-byte FP4 representation is allowed for the extra/main cache; its primary
SWA representation in this path is 528-byte FP8. Output FP8 uses packed per-32
UE8M0 scales. Projection permutations are prepared outside the hot loop.

Gaudi mapping is a producer/consumer pipeline: gather+decode/address mapping in
the KV producer, QK on MME, softmax/sink/exp/PV preparation in one TPC, PV on MME,
then inverse RoPE and next-consumer quantization in one TPC. Do not replace MME
QK/PV with scalar TPC work merely to report a smaller node count. Our WO scaling
granularity differs; do not change the model's quantization contract by copying
FlashMLA output formats.

## DeepSelect: load once, emit from retained predicates

The inspected design filters using an evolving threshold, retains a bounded
candidate buffer and performs selection on that buffer. Retaining predicate bits
allows emission without repeating comparisons. Ascend's implementation uses
histograms/counters with saturation and invalid-value guards.

Reuse our shared threshold/bitmap path; keep ties, NaNs, invalid rows and top-k
cardinality explicit. A top-k kernel alone does not prove exact nucleus sampling:
coverage certification, identical random variate and complete fallback remain
required. Do not introduce a separate TP4-only selection implementation.

## SGLang: shared metadata and one mHC state contract

`PagedIndexerMetadata` (`metadata.py:176`) computes chunk/schedule decisions once
per forward and shares them with every indexer layer (`rows_per_chunk` at line198).
`indexer.py` caches shape/dtype-specific aranges. Its current generic
`dsv41_sparse.py:73 token_req_indices` still performs conversions; do not infer
that every upstream path has eliminated every cast.

The open mHC state-machine PR explicitly carries residual, previous pre-mix and
optional already-normalized/quantized input. Row selection and residual changes
invalidate/reorder these together. `fork_stats_stream` establishes dependencies;
`mix_stats` records tensor lifetimes on the side stream; the post consumer joins
where statistics are needed. This is safer than scattered independent flags.
Overlap only work whose inputs are already ready. Do not move next-layer
statistics before their residual producer or assume a side stream creates speed.

`c2.cuh` keeps some prefill ring writes separate to avoid read-after-write hazards
when slots alias. That split is deliberate; fusion must respect ring generations.
`c2_decode_pool.py` preserves specific exp/division/product rounding because
index selection is sensitive. Keep local rounding exact inside our fused kernels.

## Local module plan and acceptance

| Order | Module / current issue | Change and invariant | Required evidence |
|---|---|---|---|
| a | `prepare_decode_metadata` runs once per four-layer group; native coordinate producer exists but is not consumed by serving | one device producer per token, fixed storage for positions/page/ring/mask; all groups read it; feedback mutates canonical roots only after consumers | physical producer once, consumers remove actual casts/gathers; five exact fixtures including ring/page crossing and slot reuse; native ABABAB |
| d | paired expert path, SRAM-breaking failed larger compounds | six-route W13, W2 3+3 only if real SRAM placement survives; fused activation and ordered shared reduction | physical nodes and no new DRAM handoff; complete producer→MME→consumer chain |
| c | KV publication/reuse and MLA preparation fragmented | handwritten multi-input/output TPC boundaries around MMEs; retain cache writes and completion dependencies | output plus cache bytes exact; physical count and replay timing |
| b | WO rounding→quantization and other repeated boundaries | preserve rounding inside combined TPC; remove only duplicated materialization | combined-module timing; current WO-only mixed-direction run carries zero gain |
| e | mHC controls, gates, post/norm | MME control projection plus TPC state transition; overlap only independent statistics | numerical order/tolerance contract, lifetime proof and physical count |

The 192-column coordinate prototype cannot be blindly sliced into cross-recipe
inputs: prepared native replay rejects partial-storage aliases in existing paths.
Use full owned output buffers or teach each native consumer to read the shared
buffer directly. A cached host position is invalid under device feedback.
Coordinate refresh must be captured before the first layer, not run on host before
replay. Request-slot generation, EOS/fallback repair and C2–C6 entry must remain
consistent. Original I64 feedback roots and sampling ordinal must not be replaced
by a detached I32 copy.

Each completed module joins already qualified pending candidates for one official
16K→EOS request (temperature1, top_p.95, seed42) plus a companion decode trace.
A profiled request is not the formal TPOT. Pending micro savings remain estimates
until that service test. No DUMP environment variables. Acquire device locks,
recheck device ownership, release only our process groups and locks after use.

### Integration progress and scope, 2026-10-05

The first shared-coordinate stage entry now exists behind default-off
`VLLM_HPU_DSV41_SHARED_COORDINATES`. It carries full owned tensors across native
recipes, keeps sampling's original roots, and leaves ordinary/prefill/speculative
entry signatures unchanged. CPU state/lifecycle checks and the native HPU producer
check cover C1/C2/C6 and five boundary fixtures; they are not performance gains.
The frozen-source real16 attempt stopped before timing; see the failure note below.

The real16 fixture already supplies I32 positions. Therefore its future speed
result must **not** be credited as removal of all historical I64 casts. This first
step primarily shares image masks, ring rows and length metadata across groups.
The packed page/history row still needs actual downstream consumers; computing it
without consumers does not qualify the complete coordinate module. Layer-dependent
selected KV indices cannot be calculated at token entry before their score producer.

For component physical-node evidence, `prepare_physical_audit` uses per-worker SSD
working directories with ordinary `GRAPH_VISUALIZATION`; no DUMP variable is set.
Bridge's default relative symbol directory avoids cross-rank name collisions and
shared post-graph JSON. Logical views and null descriptors remain distinct from
TPC/MME/DMA physical nodes; null descriptors need the module's companion trace.

### Historical count cross-check

The archived `kernel-fusion-serving-02/RANK0_KERNEL_COUNT.json` reports 3064 physical compute nodes/token over 63 cycles. Selected **exact family names** in that artifact:

| Family | Calls/token |
|---|---:|
| `constant_i64` | 85 |
| `constant_i32` | 57 |
| `constant_f32` | 50 |
| `cast_i64_to_i32` | 124 |
| `cast_i32_to_i64` | 11 |
| `clamp_fwd_i32` | 52 |
| `gather_bf16` | 1 |
| `index_copy_fwd_u8` | 48 |
| `or_fwd_i8` | 10 |
| `equal_fwd_i32` | 21 |

The ordinary `gather_bf16` family alone is not the entire KV gather category. Native KV-gather families must remain separate in attribution. Likewise the old 124 cast calls are not all position conversions: the current real16 canonical position tensor is already I32. This is a reason to inspect consumer dependencies, not to reject the coordinate-fusion module.

### Address preparation: preserve the fast consumer

The saved FX join attributes `gather_fwd_bf16` (52 calls) and its clamp predecessors
to `mirror_index_tile` in Reindex scoring, not exclusively to rotary/position work.
The historical custom mirror-gather experiment was exact but regressed
14.894→28.068 microseconds despite 7→5 nodes. Its SRAM consumer cannot be replaced
without checking the resulting compiler schedule.

An I32 tensor-bound clamp screen preserved the stock consumer but showed no gain:
three paired differences were -0.049/-0.065/-0.068 microseconds per chain. It is
not in the gain ledger. The following module candidate instead hand-fuses block
expansion, negative-block masking and safe-address clipping into one TPC producer,
retaining the existing gather/MME. These coordinates are generated **after** the
source layer publishes candidate blocks; they cannot legally be moved before that
producer. Logical rows and safe load addresses are separate owned I32 outputs.

The whole-stage coordinate test did not produce a timing: its old bootstrap first
needed the two-layer Engram contract, then cold initial full-sampler preparation
exhausted host RAM before reaching the coordinate arm. Worker cleanup now tracks
separate torchrun sessions by PID/start time, with a low-headroom stop. Narrowing
the initial sampler's closure to explicit operands is a fixture fix pending
verification, not a demonstrated explanation or performance improvement.

The Reindex geometry must be taken from checkpoint source layers, not from early
ratio-2 Full layers: only 24/28/32/36 reindex the layer-20 candidate pool, with
ratio 1 and a 524288-row reserved mirror. The production-shape coordinate chain
has now passed five exact fixtures and three positive pairs (69→49 nodes).
It eliminates clamp/transpose preparation while retaining the 13 physical gathers
and 15 MMEs in the complete projection/scoring graph. It is a small qualified
piece of module a, not completion of the global 470→60-node target.

### Ascend pipeline mapping: first local physical result

The C1 six-route W13 / two three-route W2 chain now has **24→21** physical
producer nodes, with W2 FP8 weights in SRAM and five exact inputs on four ranks.
W13 remains four TPC/MME slices. W2 uses two ordinary GEMMs (each computes a 3×3
cross product) and reads only their diagonal blocks; local native replay proves
that this specific tradeoff is faster. It does not establish the ≤12-node target.
This is a concrete result of separating producer layout, SRAM lifetime, and
consumer scheduling, rather than counting one compound registration as one node.

The new shared finalizer initially placed `output` before the last `shared`
input in the TPC signature. Gaudi binds inputs before outputs; this mismatch
corrupted the shared row. Correcting the signature restored all five fixtures
bit-for-bit. Three paired four-layer savings were 0.023608125 / 0.023651500 /
0.0236019375 ms. Full-service qualification is pending.

Reference clone paths under `/opt/ssd960/refs` now resolve to dedicated owned
Optane copies after SSD inode exhaustion. Revisions are unchanged. Logs and
physical graphs remain on SSD. Per-rank physical recipe caches avoid another
rank satisfying compilation from cache without generating its local post-graph.

### Output-boundary module under validation

The next candidate composes two genuinely fused TPC boundaries around the same
MMEs: inverse RoPE + WO-a FP8 quantization, then WO-a scale/BF16/group32
roundtrip + WO-b FP8 quantization. It preserves the projection's separate
rounding points in registers. This follows FlashMLA/TileKernels' dataflow, while
leaving QK/PV and WO matrix work on MME. It remains default-off; fewer public
operator calls alone are not performance evidence.

### Gaudi2 ISA check for the qualified expert batch

Disassembled the actual two ELF objects with `tpc-llvm-objdump --triple=tpc
--mcpu=gaudi2`; the default x86 target is invalid even though the container ELF
header says elf32-i386. Artifacts: SSD
`decode-sat-horizontal-w2-fusion-04/isa/`.

The steady 256-value reference codec emits `or.i8 → shuffle.u8 → add.i8 →
sel_eq.u8`; the qualified SAT codec emits `or.i8 → shuffle.u8 → sub.u8 st`.
That is **4→3 vector arithmetic instructions per 256 decoded values (−25%)**,
excluding block-scale preparation, scalar addressing, load/store and loop
control. It is not the earlier proposed −40% reduction and not a measured HBM
bandwidth. Source-level LUT changes are not being added as a separate candidate.
The measured batch also changes route grouping and physical producer count;
its complete-chain saving cannot be attributed solely to those instructions.

### Integration and numerical gates encountered

Q/KV publication's compound initially assigned `syn_out` for three intermediate
results without registering their final output indices in `BuildNode`. Native
serving warmup then failed `OrderOutputTinfos: Unaccounted output`. A two-recipe
component reproduces this with the old binary and passes all four outputs plus
its downstream completion consumer for five fixtures with the corrected indices.
This is output-storage ownership, not a shape-cache toggle. No formal request was
sent by the failed batch02 warmup.

The proposed high/low BF16 MME mHC controller does **not** meet the current exact
acceptance: maximum gate error 7.45e-7 changed BF16 residual values by up to
4.8828125e-4. Router IDs matched, and the hand-fused TPC consumer itself was exact
with the same projection. No timing or gain is credited. The next exact variant
retains the existing FP32 control producer and carries its existing RRMS (25th
value) into the fused gates/post, avoiding duplicated 20480-element statistics.
It is unqualified and remains out of the service batch.

### Ascend scheduling details carried into the next module

Pinned DeepGEMM-Ascend `deep_gemm/include/deep_gemm/scheduler/mega_moe.hpp`
advances Linear2 with one Linear1 wave of lookahead, rather than draining every
Linear1 task first. Shared Linear1/Linear2 may run before routed metadata is
ready. The kernel uses three L1A stages, four routed L1B stages and two L0 stages;
AIV0's three FP4 stages and two FP8-NZ stages hand explicit full/empty credits to
AIC. These are producer/consumer lifetime contracts, not a universal grouping
constant. Our current W13 six-route/W2 three-route result retains SRAM but still
has four W13 physical slices; it is not equivalent to Ascend's persistent launch.

TileKernels `mhc/pre_big_fuse_asc.py` combines reduced controller partials, RMS,
24 mix values, Sinkhorn and pre-apply while preserving the composed Ascend
arithmetic. Its wrapper uses **16-token blocks**, four pipeline stages and a
single internally reduced GEMM partial. Those batch assumptions cannot be
copied to Gaudi C1. The transferable idea is to pass the already-computed RMS
statistic together with control values, and preserve each arithmetic boundary.
The Gaudi exact-controller experiment now accepts a 25-value producer (24
controls plus existing RRMS); all 29 CPU meta contract checks pass across C1,
C2/C6 and TP2/4/8. Hardware exactness and device timing are still pending.

Attention output boundaries plus post/norm are being rechecked as one module.
The first short native intervals had baseline jumps as well as mixed A/B signs;
no gain was credited. The reusable fixture now places 128 complete native
producer/collective/consumer stages inside each device interval, and waits for
other-card weight loading to finish. This changes measurement duration, not
production weights, rounding or the serving implementation.

The default-off `MHC_DEFERRED_GATES` serving prototype now carries the owned
25-value control tensor through the matrix sublayer and defers gate evaluation
until post/collapse. Attention and MoE ready-output dependencies carry that
control tensor instead of already-materialized post/comb views. There is no
new TP4-only branch; native post consumes either a reduced row or the existing
TP-rank peer tensor and retains its BF16 sum boundary. Prefill, draft and C2–C6
keep the existing dispatch. This integration is unqualified and is not present
in the currently frozen batch03 service.

### Fused post/norm instruction-level correction

The actual Gaudi2 disassembly of `peer_post_norm_quant` contains 24 scalar
`ld_g` operations inside its 40-feature-tile loop. Coefficients are immutable
throughout the kernel, but the inlined helper and output stores prevented their
hoisting. The revised kernel explicitly loads one `PostWeights` object before
the tile loop. The new disassembly places the loads before that loop: coefficient
loads per owner fall from 960 to 24 without changing residual/collapse/norm/FP8
arithmetic. ISA before/after is preserved in `decode-attention-output-post-norm-02`
and `decode-post-norm-invariant-01`. Hardware validation is pending; no timing or
ledger credit is inferred from this instruction reduction.

The next mHC micro reference already uses the qualified fused peer/post consumer. Its prospective saving therefore excludes the peer-sum/post saving in batch03; the two ledger entries must not count the same boundary twice. The output/post/norm fixture uses that same reference convention.

The invariant-load change passed five exact fixtures on four ranks. Complete-module A/B is still slower by about 2.6 µs/boundary (three negative pairs), so it remains off and gets no gain credit. Relative to the previous candidate it recovered about 5 µs. The WO scale/roundtrip+quant kernel still recomputes the complete row amax in each of 16 owners; its activation-scale load is already hoisted outside the inner tile loop in emitted ISA. Do not add another scalar-hoist trial without evidence.

## Batch03 raw activity follow-up (2026-10-05)

Same capture, 24 preselected consecutive positions: per-rank periods 9.394–9.400 ms, compute union 6.965–6.982 ms, noncompute 2.418–2.432 ms. Sub-2-us gaps still contribute 1.296–1.320 ms. Null activity is 0.349–0.356 ms and overlaps other columns. Cached recipe-ID collisions prevent trustworthy named node attribution; do not turn maximum-lane descriptor counts into physical-node claims. Evidence: `decode-physical-fusion-serving-03/trace-analysis-window/RAW_ACTIVITY_REPORT.md`.

Next shared-main attention candidate reuses the manual MLA product/RoPE/FP8 consumer on both publish and reuse paths. It retains MME QK/PV/WO and every BF16 rounding boundary, removes the exposed PV BF16 cast and standalone inverse-RoPE/WO quantization boundary, and preserves the main-row/mask public outputs. It deliberately retains the existing distributed group32 WO scale kernel: recomputing the complete global quantization reduction at every tile was measured slower. Default off pending the complete production-shape chain.

### Shared-main complete-chain findings

The sliceable shared-main PV/RoPE/WO producer reduced actual compiler nodes 19→18. Reuse: five inputs × four ranks exact; three native-chain savings 1.369/1.823/1.617 us. Only 27 configured reuse occurrences receive an estimated 0.043661 ms/token. Publish is also exact and 19→18, but paired savings 0.174/0.193/−0.049 us are mixed: zero separate credit, retained for the larger module.

An attempted nonsliceable full-output producer demonstrates the Ascend/Gaudi ownership distinction: it reduced RoPE/quant physical clones 2→1 but introduced two DMA copies and duplicated WO scale, increasing total producer nodes 19→20. Its all-required mapping was reverted. Keep the compiler's SRAM-consumer slicing unless the complete physical graph improves; a single named producer is not by itself an optimization. Evidence: `decode-main-mla-whole-producer-01/DECISION.json`.

### 2026-10-05 MoE pipeline, after accepted batch03

`decode-moe-pipeline3-02`: two independent three-route W13→SiLU→W2 chains
compile to 11 routed physical nodes (17 including shared experts/input prep),
versus 21 in the accepted complete producer. W13 has two 19,660,800-byte SRAM
blocks and W2 two 9,830,400-byte SRAM blocks, each with exactly one MME consumer.
Zero-offset same-size reshape aliases must be followed by the SRAM checker.
Five checkpoint-derived inputs on all four ranks are exact after native peer
exchange and mHC/FFN consumption. Nevertheless ABABAB is slower by
0.052748 / 0.052549 / 0.052575 ms per four-layer group. No gain credited.

The final graph schedules W13-A, W2-A, W13-B, W2-B. A follow-up emits both W13
bundles first and requires that order in the final graph before timing. This
exposes W13-B preparation while W13-A's matrix consumer can execute. A separate
structural issue remains: the shared input norm/quant producer is outside both
W13 bundles when it feeds two MMEs. Compare a single ordinary W13 matrix with
the accepted batch-GEMM form before changing compiler policies.

Explicit Gaudi2 disassembly confirms three hot vector arithmetic instructions
per 256 FP4 values: enable-bit OR, table shuffle, saturated exponent subtraction.
The two-instruction goal remains unmet. The documented shuffle enable bit is
required. Expert source scales vary by output channel; unlike dense FP8 weights,
they cannot be treated as uniform 32×32 blocks to hoist scaling before lookup.
See `decode-moe-pipeline3-01/ISA_REVIEW.json`; no blanket bandwidth claim is made.

### MoE follow-up decisions (same formal baseline)

Two-branch prefetch reduced the regression to3.50us/layer but did not beat the
accepted chain. Single ordinary W13 uses one39.32MB SRAM weight block and gives
15 complete /9 routed nodes, yet is1.45us/layer slower. Both are disabled.
Explicit RMW handoff for W13→SiLU→W2 caused compiler weight spills and was
rejected without timing. The next compiler check retains the original producer
bundle and scopes NON_COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING=2 to the candidate
compile. The local solver starts from4 slices; input/consumer ownership is unchanged.

The four-scale dictionary attempt is numerically invalid. CPU scalar LUT parity
did not model byte SHUFFLE group control or the high half of an unpacked load.
The real HPU chain failed before timing; isolated simulator probes reproduce it.
FORM_FP_NUMBER removes the high-bit contamination but does not repair the
variable group-control behavior. Removed native dispatch and kernel sources;
archived under decode-moe-dictionary-02/unqualified-source and unqualified.diff.
No two-instruction claim or gain credit. Full dictionary metadata also exceeds
production memory headroom; future designs must account for that independently.

### Exact decoder scheduling and the next MoE dependency

`decode-moe-unroll-chain-01` manually expands the three eight-row load/decode
stages. It removes loop-carried vector register copies without changing the
three arithmetic instructions (OR, SHUFFLE, saturated subtract). The simulator
executes 3516→3148 instructions for a six-route K128 block, with zero differing
bytes for normal and invalid route fixtures. Four-rank producer→peer→mHC/FFN
validation is exact on five checkpoint-derived inputs. With two W13 slices,
physical producer nodes are 21→17 and every decoded weight remains in SRAM
with one matrix consumer. Three four-layer savings are
0.0013724375/0.00123634375/0.00124034375 ms: 0.000310086 ms/layer,
0.0124034 ms for 40 occurrences before serving overhead. This is a component
result only, not a new formal baseline.

The two-slice-only control (`decode-moe-two-slice-02`) was slower by
0.00235624 ms/layer. Reducing slice count alone delays first MME consumption.
The current pair-SiLU access pattern processes two routes per index-space tile;
a W13 half contains three routes. `decode-moe-streamed-01` tests whether a
one-route SiLU access pattern can stitch the three-route W13 slice through
the activation boundary. No layout, rounding or scale rule is changed.

`decode-moe-streamed-01` confirms the one-route SiLU is exact and faster in
all three native-chain rounds (0.760 us/layer total versus accepted groups3),
but **does not** stitch activation SRAM: the compiler chose W13 slices
4096/3584, cutting through 1280-column route rows. The next producer maps one
complete route per index-space tile (rather than one N256 block), retaining
K128 partitioning across TPCs. This supplies a genuine 1280-column access
granularity, without adding padding, changing values or imposing fake
whole-tensor access. Expected slice boundaries are 3840/3840; compiler proof
and exact outputs determine acceptance.

The aligned route producer does obtain 3840/3840 slices, but alone is slower
by 0.841 us/layer (`decode-moe-aligned-01`). It does not eliminate activation
DRAM. The compiler's `isLogicalChainBreaker` requires one-to-one matching
slicing dimensions across a reshape, and its generic consumer check rejects
granularity greater than one. A flat-input SiLU removes the reshape boundary
without changing arithmetic; its required compile gate is SRAM W13 output.
If that fails, it is not timed as a claimed SRAM pipeline.

The flat-input SiLU still leaves W13 output in DRAM. The pre-timing gate
rejects `decode-moe-aligned-flat-01`; no timing is taken. Next, the explicit
RMW scratch experiment changes an earlier failed condition: **both** decoded
weights and activations share the same on-chip section. The earlier 16 MiB
experiment reserved only activations, disabling automatic weight bundling.
Source inspection shows 16 MiB is a configurable compiler validation limit
(`SYN_RMW_SECTION_MAX_SIZE_BYTES`), not Gaudi2's full SRAM size. A cold-only
64 MiB limit allows the approximately 59.4 MB complete scratch to be tested.
All actual tensor locations and sole MME consumption remain compile gates;
this is not promoted or assumed fast. It trades automatic slicing for an
explicit one-W13/two-W2 SRAM layout and must be timed before any gain claim.

`decode-moe-full-sram-01` is rejected before timing: its 59.4 MB scratch does
not fit the compiler's actual **47.5 MiB / 49,807,360-byte** available SRAM
pool. Raising the validation maximum does not create physical memory. The
next version reuses the dead W13 weight range for W2 group A and then group B,
with explicit previous-MME-completion→next-decode dependencies. Activations
and both W2 products occupy separate tail offsets, so the section fits a
40 MiB cap. Compilation must prove SRAM storage and no hidden DRAM copy;
five changing inputs must catch reuse hazards before timing.

`decode-moe-full-sram-reuse-02` proves 21→15 total producer nodes and all
routed W13→SiLU→W2 intermediates in SRAM. Five fixtures × four ranks are
exact, including weight-storage reuse. Nevertheless, three four-layer pairs
are slower by 0.05657/0.05680/0.05664 ms. No gain is credited. The physical
schedule puts the entire shared expert before the first routed decoder;
the RMW section pulls the final shared-add consumer into its bundle, so
shared completion becomes an entry dependency. The next graph returns W2
products/scales to the finalizer outside the RMW bundle and uses independent
three-route W13/W2 branches. Only those small terminal operands are DRAM;
all decoded weights and W13→SiLU→W2 activation handoffs must remain SRAM.
This addresses the observed dependency, rather than assuming that more SRAM
residency alone improves latency.

The detailed `aligned-flat-01` compiler log identifies the automatic pipeline
break more precisely: `expert_flat_silu_quant` is rejected with **creates a
circle in BP graph**, after `ffn_norm_quant` has been bundled with W13. Its
second output takes a separate scale broadcast path before re-entering SiLU.
`aligned-scale-01` makes the six routes read the original C1 scale scalar,
already supported by the SiLU kernel, avoiding that external broadcast path.
This is an enabling dependency change for the complete W13→SiLU pipeline;
correctness and actual residency remain mandatory.

## 2026-10-05: router dependency and four-slice policy correction

The fixed-ID MoE fixture allowed FFN norm to join the W13 bundle. Its scale
then became a second direct input to SiLU, which `validateConsumerPaths` rule
4d rejects (an accepted ancestor provides another consumer input). Removing
the sixfold scale broadcast alone did not remove this diamond. With the real
BF16 gate GEMM and top6 router included, the W13 activation in
`decode-moe-router-pipeline-01` is in SRAM; five inputs/four ranks are exact.
That run used two-slice policy and is archived, not credited or promoted.
The next qualification retains policy 4, all 384 checkpoint experts and the
actual router, and reports independent logical stages separately from TPC/MME
pipeline fragments. Compiler fragments are not kernel-fusion savings.

### MoE dual quantizer ISA review

The routed expert uses BF16-rounded epsilon/240 scaling; the shared expert
uses power-of-two scaling and FP8 subnormal flush. Reusing one quantized row
would change arithmetic. The new five-output FFN producer shares normalized
BF16 values and one amax, then emits both formats separately. The Gaudi2 ISA
in `decode-moe-dual-quant-01/kernel.s` has three loops: input load/squares,
weight load/normalization, and one cached VLM read feeding both conversions.
The amax and both scales are outside the output loop; there is no second
tensor read of the normalized row. VLM reads preserve the existing cached-row
implementation, rather than introducing repeated DRAM loads.

Shuffle validation: `check_deepseek_v41_shuffle_lanes.py` matches all 32,768
outputs of the SDK simulator probe. Explicitly masking the raw high nibble
still leaves 10,240/20,480 failures for the flat 64-entry class dictionary.
The first masked counterexample is lane 2, direction 138: group mux returns
42 rather than 10. The two-instruction dictionary is **not qualified**; the
three-instruction exact SAT decoder remains selected. Instruction source:
https://docs.habana.ai/en/latest/TPC/TPC_Intrinsics_Guide/Arithmetic.html#shuffle

## 2026-10-05: retained pipeline, vectorized MLA body

The shared-main gather used 64-value FP32 decoding, while the existing wider
batch path already contained exact 128-value BF16 codecs. Both operands are
BF16-representable before the output boundary; the vector codec preserves
zero/scale rules and writes the identical rounded keys and FP32 PV values.
The helpers now live in `deepseek_v41_selected_kv_codecs.h`, shared with the
existing selected-KV implementation. No matrix, softmax/sink, selection order,
mask, page mapping or exported-row ownership changes. Scale bytes are read
once per row, outside four 128-value chunks; no amax and no loop-local
reload of the full scale row. The source and ISA are archived with each case.

`decode-mla-reuse-vector-01` and `decode-mla-publish-vector-01` each pass five
inputs on four ranks and three positive native A/B rounds. Node counts are
unchanged; classify this as efficiency, not a reduced logical stage count.

### Existing fused quantizer: repeated row work removed

The old disabled `woa_scale_dense_quant` scheduled 16/32 workpoints and each
repeated the full roundtrip/amax before emitting one tile. The corrected
implementation assigns the full row once, hoists scale loads and caches at
most 8 KiB of rounded BF16 values in VLM. The output loop has no tensor
reloads or row amax. Frontend unroll exposes independent tile loads; the
previous backend pragma reported an unsupported vector PHI and did nothing.
`decode-woa-single-amax-02` validates the complete native QKV→MLA→WO→peer/mHC
chain, rather than the isolated quantizer. Its small but consistent gain is
retained under the revised accumulation policy; the default stays disabled.


### 2026-10-06 — parallel PCIe payload protocol: ordering before timing

The archived star protocol02 has one TPC workpoint handling40 vectors. The parallel01 research candidate uses40 workpoints, per-vector ready/reader-ACK generations and separate two-bank payload storage. Every assigned payload range is written and drained before that instance announces readiness; every assigned reduction output is drained before ACK. Epoch comparisons use signed modulo32-bit differences, including wrap. The import graph remains acyclic: hub exports; leaves import only the hub. No reciprocal imports, no production integration or measured gain yet.

Intel documents that vector stores and later loads are not implicitly coherent; ASO commits earlier vector writes before its semaphore update. The candidate waits for that update before publishing scalar readiness, then invalidates scalar cache around incoming ready polls. This is an ordering design, not proof of remote PCIe visibility or native-chain latency. Sources: [TPC coherency](https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/TPC_Coherency.html), [ASO/cache intrinsics](https://docs.habana.ai/en/latest/TPC/TPC_Intrinsics_Guide/Cache.html), [index execution order](https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/TPC_Programming_Model.html). Direct fetches returned429; official indexed text supplied these definitions.

Offline checks: five four-rank randomized/different-partition scheduler fixtures (three include U32 wrap); five checkpoint BF16 rows through production no-bias mode then stale-data diagnostic mode in the Gaudi2 TPC simulator, all exact; complete P2P-output->FFN norm/quant graph compiles with two physical logical compute stages in both arms. Next gate is bounded primitive/cleanup on healthy cards, then the actual WO->TP4 exchange->mHC/FFN consumer native chain. Never map quarantined modules0/1/4/5.
