# C6 SiLU/W2 access-map capability (disabled)

The C1 compound shares one TPC invocation between SwiGLU activation quantization and W2 FP4 decoding. The original C6 port lost the six independent W2 producer slices and regressed in the real sixteen-layer chain. A second, private registration narrows activation-scale input5 from all-route dependency to the two route rows actually read. It preserves the common TPC ELF, route tile, weight format, arithmetic, C1 registration, and transport.

Meta checks retain compact TP4 I640 and legacy I1152 checkpoint layouts and reject draft top3. Three actual production cases on four ranks remain byte exact through hidden/pre/SWA. The native replay gate still regresses in all three pairs. Captured recipe debug symbols retain one unsliced compound producer; they do not prove tensor placement or DRAM spill.

`VLLM_HPU_DSV41_DSPARK_SILU_DECODE_AFFINE` defaults to0. This direction is stopped after two measured variants. No standalone or end-to-end saving is credited. Reuse the decision in evidence/20260930_tp4_dspark/SILU_DECODE_AFFINE_NATIVE_REAL16_587_DECISION.json rather than repeating it.
