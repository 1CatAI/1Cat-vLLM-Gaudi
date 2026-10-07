# Archived shared-main output projection candidate

Archived on 2026-10-05 at the user's request to focus on MoE. The switch
`VLLM_HPU_DSV41_MAIN_MLA_PROJECTION` remains disabled by default.

The shared publish/reuse implementation retains QK/PV/WO matrix engines and
performs the two BF16 boundaries, inverse RoPE and WOa input quantization in the
existing hand-written TPC product consumer. Public main rows/mask remain owned
outputs; a changed selection owner invalidates reuse as before. No runner,
communication protocol or non-C1 entry interface changes.

- Reuse: complete QKV→MLA→WO→TP→post/norm producer nodes 19→18, downstream3
  unchanged. Five checkpoint-derived inputs on four ranks are bit-exact.
  Three native A/B savings: 0.001369 / 0.001823 / 0.001617 ms per reuse layer.
- Publish: same exactness and node reduction; savings 0.000174 / 0.000193 /
  −0.000049 ms are inconclusive. No positive credit.
- A nonsliceable producer was rejected and reverted: it introduced two DMA
  copies and a split scale kernel, increasing the producer to20 nodes.
- No formal service measurement includes this candidate. The accepted batch03
  baseline is 9.564282 ms/token and does not include this switch.

Evidence archive cases: `decode-main-mla-projection-01`,
`decode-main-mla-publish-projection-02`, `decode-main-mla-whole-producer-01`.
CPU ownership tests:6 passed. Meta contracts for both TP geometries:7 passed.
This item is removed from the active MoE gain accumulator, not promoted.
