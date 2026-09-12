# Experimental V4.1 dependency and Engram preparation paths

These independent options are disabled by default and apply to ordinary C1
DeepSeek V4.1 Flash decode with TP2, PP2 and DSpark disabled. They preserve the
existing BF16 computation, routing, cache capacity and normal worker entrypoint.
Neither option enables the FP8 GEMM experiment.

## mHC and TP dependencies

`VLLM_HPU_DSV41_TP_MHC_OVERLAP=1` selects the V4.1-specific compiler transform and
the additive `synNativeComputeGraphPreparePlanV2` runtime API. The transform
extracts control/statistics/mixing work whose storage dependencies are independent
of the current peer result. The native planner records the actual producer,
first consumer and last consumer for each collective. Communication waits for the
producer; the compute fence is placed at the first consumer. Preparation, alias
checks and relocation construction happen before steady replay.

The old adjacent dependency API is preserved. Missing V2 symbols, invalid buffer
dependencies, missing independent work or an incompatible ABI reject preparation.
Input/state mutations and the Engram injection boundary are not moved. Private
reinplaced additions are functionalized only when their allocation and sole-use
contract proves they cannot alias inputs or state. FP32 residual casts are reused
without changing reduction, Sinkhorn or BF16 rounding order.

The semantic seam reference is [vLLM #56513](https://github.com/vllm-project/vllm/pull/56513),
commit `44d680ead86e2f72775851e372a3aa912054825c`, particularly the Engram exclusion
and delayed-pre tests. ROCm kernels and their different accumulation order are
not imported. The implementation uses Gaudi compiler partitions, completion
counters and the existing HCL batch path.

Build the versioned Synapse patch from
`tools/communication/patches/native-runtime/synapse.patch` in an isolated runtime
checkout and build the native communication extension against that library.
The adjacent manifest pins the patch digest. Preserve the locked Bridge/HCL
libraries; the normal runtime profile and extension ABI manifest must identify
the actual candidate binaries. Do not replace shared libraries or disable ABI
checks. The device runner accepts the resulting isolated profile through its
existing `--runtime-profile` argument.

## Native Engram C1 preparation

`VLLM_HPU_DSV41_ENGRAM_NATIVE_C1=1` requires the normally built host extension's
`c1_abi_version()` contract. The C++ entry compresses tokens, computes the existing
wrapped-integer hashes and copies selected mmap rows directly into the registered
final packed staging buffer. The Python history owner remains the sole committed
history; preparation is a transaction identified by request, history generation,
slot and staging generation.

Both layers' row bounds and all destination contracts are checked before writing.
The existing consumer-stream upload and DMA/consumer retirement rules remain.
Image boundaries, short histories, request resets and non-C1 prefill retain their
compatibility paths. No extra Engram table, second history or layer-14 prefetch
worker is introduced. Rebuild through `tools/build_deepseek_v41_host.py` so the
normal package loads the C1 ABI; runtime injection is not supported.

This implementation is retained for experiments. Its measured incremental
end-to-end benefit did not satisfy the selection gate, so the retained overlap
configuration leaves this option disabled.

## Validation and reporting

Focused tests cover planner ranges/aliases and generations, nonadjacent and
shared collective boundaries, changing FX inputs, the real four-layer native
chain, exact KV/CSA2 state, both host shards, integer wraparound, image boundaries,
request resets and repeated staging/DMA reuse. Whole-model short generation
checks compare all emitted token IDs with the respective parent configuration.
These checks do not replace independent model-quality and long-lifecycle gates.

`collect_deepseek_v41_trace.py`, `analyze_deepseek_v41_trace.py`,
`map_deepseek_v41_trace_contracts.py`, `report_deepseek_v41_trace.py` and
`account_deepseek_v41_trace.py` form the offline reporting pipeline. Compiler
program IDs and serialized recipe IDs are different namespaces: use the exact
captured segment order and validate mHC symbols when joining them. The final
ledger unions all four ranks on one trace clock and uses explicit fixed-priority
accounting for overlaps. Unknown gaps and descriptor invocation counts remain
unknown; a single native entry is not a single hardware command.

The options remain experimental pending the complete performance, quality,
compatibility and shutdown/lifecycle qualification. Existing engine shutdown
cleanup warnings must not be treated as resolved merely because a launcher
returns success.
