# Environment Variables

This document lists the supported diagnostic and profiling, as well as performance tuning options.

`VLLM_HPU_NATIVE_DECODE_GRAPH` selects the research native replay path for the
Gaudi2 Qwen3.8 TP2 C1 decoder. Warmup retains eight compiled decoder groups,
fixed tensor bindings, recipe memory, and HCL command templates. Stable decode
then updates the fixed input buffers and publishes the staged compute and
communication queues from one outer `NativeDecodeGraph` replay. The path
requires the version-locked Synapse and HCL builds plus the three structural
options listed below. It also reserves five percent of global HBM from the
Bridge caching allocator for graph-owned recipe programs unless the user has
selected a smaller pool. Missing symbols, a changed address or shape, an invalid
state transition, or incomplete graph coverage raises an error; it never
falls back after a token may have mutated recurrent state. The option remains
disabled and unqualified while the device correctness and performance gates
are in progress.

## Diagnostic and Profiling Parameters

| Parameter name                            | Description                                                                                                                                                                                                                                                                                                                                                                                                             | Default value |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `VLLM_PROFILER_ENABLED`                   | Enables the high-level profiler. You can view resulting JSON traces at [perfetto.habana.ai](https://perfetto.habana.ai/#!/viewer).                                                                                                                                                                                                                                                                                      | `false`       |
| `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION`     | Logs graph compilations for each vLLM engine step, only when a compilation occurs. We recommend using it in conjunction with `PT_HPU_METRICS_GC_DETAILS=1`.                                                                                                                                                                                                                                                             | `false`       |
| `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL` | Logs graph compilations for every vLLM engine step, even if no compilation occurs.                                                                                                                                                                                                                                                                                                                                      | `false`       |
| `VLLM_HPU_LOG_STEP_CPU_FALLBACKS`         | Logs CPU fallbacks for each vLLM engine step, only when a fallback occurs.                                                                                                                                                                                                                                                                                                                                              | `false`       |
| `VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL`     | Logs CPU fallbacks for each vLLM engine step, even if no fallback occurs.                                                                                                                                                                                                                                                                                                                                               | `false`       |
| `VLLM_T_COMPILE_FULLGRAPH`                | Forces the PyTorch compile function to raise an error if any graph breaks happen during compilation. This allows for the easy detection of existing graph breaks, which usually reduce performance.                                                                                                                                                                                                                     | `false`       |
| `VLLM_T_COMPILE_DYNAMIC_SHAPES`           | Forces PyTorch to compile graphs with disabled dynamic options to use dynamic shapes only when needed.                                                                                                                                                                                                                                                                                                                  | `false`       |
| `VLLM_FULL_WARMUP`                        | Forces PyTorch to assume that the warm-up phase fully covers all possible tensor sizes, preventing further compilation. If compilation occurs after warm-up, PyTorch will crash (with this message: `Recompilation triggered with skip_guard_eval_unsafe stance. This usually means that you have not warmed up your model with enough inputs such that you can guarantee no more recompilations.`) and must be disabled. | `false`       |
| `VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY`     | Runs multimodal warmup outside `PT_COMPILE_ONLY_MODE`. Enable this for multimodal paths with data-dependent output shapes that must be materialized during warmup, such as Gemma4. Enabling it increases warmup time because recipes are compiled and executed.                                                                                                                                                | `false`       |

## Performance Tuning Parameters

| Parameter name               | Description                                                   | Default value |
| ---------------------------- | ------------------------------------------------------------- | ------------- |
| `VLLM_GRAPH_RESERVED_MEM`    | Percentage of memory dedicated to HPUGraph capture.           | `0.1`         |
| `VLLM_BUCKETING_STRATEGY`    | Selects the bucketing strategy: `exp`, `lin`, or `pad`.      | `exp`         |
| `VLLM_PROMPT_BUCKETING_STRATEGY` | Overrides the strategy for prompt/prefill buckets: `exp`, `lin`, or `pad`. | `None` (use global strategy) |
| `VLLM_DECODE_BUCKETING_STRATEGY` | Overrides the strategy for decode buckets: `exp`, `lin`, or `pad`. | `None` (use global strategy) |
| `VLLM_EXPONENTIAL_BUCKETING` | Deprecated compatibility flag. If set, it overrides `VLLM_BUCKETING_STRATEGY`: `true` forces `exp`, `false` forces `lin`. It cannot select `pad` and will be removed in a future release. | `None`        |
| `VLLM_BUCKETING_FROM_FILE`   | Enables reading bucket configuration from file.              | `None`        |
| `VLLM_ROW_PARALLEL_CHUNKS`   | Number of chunks to split input into for pipelining matmul with all-reduce in RowParallelLinear layers. Setting to a value greater than 1 enables chunking. See [Row-Parallel Chunking](../features/row_parallel_chunking.md). | `1` (disabled) |
| `VLLM_ROW_PARALLEL_CHUNK_THRESHOLD` | Minimum number of tokens required to activate row-parallel chunking. Inputs below this threshold use the standard non-chunked path. | `8192` |
| `VLLM_HPU_TP2_FUSED_AR_NORM` | Enables experimental deferred all-reduce/RMSNorm boundaries for dense Qwen3.5/Qwen3-Next TP2, including the embedding reduction at the first input norm. Supported RMSNorm decode inputs use the graph-native fused collective. GemmaRMSNorm uses stock HCCL unless the separately gated native experiment below is enabled. Unknown normalization consumers are rejected before reductions are disabled. Requires `PT_HPU_ENABLE_LAZY_COLLECTIVES=1`; prefill, unsupported payloads, and decode batches above 20 tokens also use the standard path. | `false` |
| `VLLM_HPU_TP2_GEMMA_FUSED_AR_NORM` | Experimental Gemma decode fusion; requires the deferred-boundary flag and a matching patched Bridge/extension. Both ranks must pass changing-input compiled startup checks, including actual collective execution and graph-produced `weight + 1`, before any model reduction flags are changed. A failed check aborts initialization; it does not silently fall back. Passing this probe is not model accuracy or performance qualification. See `tools/communication/README.md`. | `false` |
| `VLLM_HPU_TP2_FUSED_AR_NORM_MAX_BYTES` | Largest BF16 activation payload accepted by the experimental TP2 fused path. | `524288` |
| `VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE` | Path to the native bridge built by `tools/communication/build_tp2_fused_ar_norm_bridge.py`; required only when `VLLM_HPU_TP2_FUSED_AR_NORM=true`. | `None` |
| `VLLM_PROMPT_BS_BUCKET_MAX`  | Sets prefill batch size | `1` |
| `VLLM_MULTIMODAL_BUCKETS`    | Overrides the per-model patch-count buckets used to warm up native-resolution vision towers (models where `is_batch_based=False`, e.g. Gemma4, Kimi-K2.5/K2.6, Qwen2.5/3/3.5-VL). Comma-separated list of integers. Set to `None` to disable bucketing for these models. | model-specific |
| `VLLM_MULTIMODAL_RESOLUTIONS` | Pins explicit raw pixel resolutions (comma-separated, e.g. `1024x768,768x1024`) to warm up for native-resolution vision towers. Each entry is `WxH`, `WxHxN` (pin the count-`N` graph), or `WxHxN-M` (warm the item-count range `[N, M]`); `WxH` alone warms one graph at the `--limit-mm-per-prompt` ceiling. See [Warm-up](../features/warmup.md#multimodal-warm-up). | `None` |
| `VLLM_MINIMAX_M3_MOE_TOKEN_TILE` | Maximum number of tokens processed per tile by the MiniMax-M3 dense SwiGLU-OAI expert path. Non-positive values disable tiling. | `512` |
| `VLLM_MINIMAX_M3_MOE_DECODE_GATHER` | Enables the MiniMax-M3 routed-expert gather path for low-token decode. Set to `0` or `false` to use the dense expert path. | `true` |
| `VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS` | Maximum token count for the MiniMax-M3 routed-expert gather path. Larger batches use the dense expert path. | `16` |
| `VLLM_GDN_CHUNK_SIZE` | Overrides the GDN prefill chunk size. Set to a positive multiple of 32; `0` keeps the model-provided value or the HPU default. | `0` |
| `VLLM_GDN_NEUMANN_ITERS` | Sets the iteration budget for the approximate GDN triangular solve. Lower values can improve prefill speed but require model-level quality validation. | `14` |
| `VLLM_GDN_FUSED_STATE_MATMUL` | Fuses the GDN phase-B output and recurrent-state projections into one larger matrix multiplication per chunk. | `false` |
| `VLLM_GDN_DEFERRED_OUTPUT_ADD` | Defers GDN phase-B output accumulation until after the recurrent loop, avoiding in-place writes to chunk views in compiled HPU graphs. | `false` |
| `VLLM_GDN_RECURSIVE_SOLVER_BASE` | Enables recursive block inversion for the GDN triangular solve. Set to `0` to disable it or a power-of-two base size of which the chunk size is a power-of-two multiple. | `0` |
| `VLLM_GDN_COMPACT_REPEATED_KKT` | Computes the GDN KKT product once per unique key head when value heads repeat the same key heads. | `false` |
| `VLLM_GDN_COMPACT_REPEATED_LOCAL_ATTN` | Computes the phase-B local Q/K product once per unique Q/K head, then combines it with each repeated value head's causal decay. | `false` |
| `VLLM_GDN_COMPILED_QK_L2NORM` | Keeps GDN Q/K L2 normalization inside the compiled prefill graph. Leave disabled on HPU compiler versions where this path has not been validated. | `false` |
| `VLLM_GDN_FUSED_RMSNORM_GATED` | Uses Habana FusedRMSNorm for Qwen GDN output normalization before the output gate during prefill. Decode is unchanged. | `false` |
| `VLLM_GDN_FLASHQLA` | Uses the experimental Gaudi-native FlashQLA similarity transforms, factorized local decay, and compact KKT layout for GDN prefill. This does not load FlashQLA's CUDA kernels. Enable `VLLM_GDN_FUSED_STATE_MATMUL` and `VLLM_GDN_DEFERRED_OUTPUT_ADD` for the validated fast path. | `false` |
| `VLLM_GDN_FLASHQLA_FACTORIZE_PHASE_B` | Experimental factorization of FlashQLA's local gate decay in phase B. Disabled because separate positive gate factors can overflow for strongly decaying chunks. | `false` |
| `VLLM_GDN_FLASHQLA_FP32_SCALING` | Computes the gate-free phase-A diagonal scale products in FP32 before returning to BF16 MME inputs and outputs. This reduces the reformulation's numerical drift without moving its BMMs off MME. | `false` |
| `VLLM_GDN_BF16_BMM_F32` | Uses the experimental Synapse BF16-input, FP32-output batch GEMM for the GDN recurrent state projection. | `false` |
| `VLLM_GDN_SOLVE_BF16_BMM_F32` | Uses the experimental BF16-input, FP32-output batch GEMM for recursive GDN KKT block merges while retaining the base 16x16 inverse in FP32. | `false` |
| `VLLM_GDN_BF16_BMM_F32_EXTENSION` | Path to the private-ABI registration extension used by `VLLM_GDN_BF16_BMM_F32`. | empty |
| `VLLM_GDN_QWEN38_NATIVE_QK_PREP` | Uses Gaudi2 TPC kernels to fuse Qwen3.8 prompt Q/K FP32 normalization and final-layout writes. Supports TP1, and TP2 when `VLLM_GDN_QWEN38_COMPACT_QK` is enabled. Expanded native output layouts remain TP1-only. Decode is unchanged. | `false` |
| `VLLM_GDN_QWEN38_BF16_QK` | Makes the native Qwen3.8 Q/K kernel write its expanded 48-head output directly in BF16, matching the optimized GDN input dtype and avoiding a large FP32 intermediate. Requires `VLLM_GDN_QWEN38_NATIVE_QK_PREP`. | `false` |
| `VLLM_GDN_QWEN38_COMPACT_QK` | Keeps native Qwen3.8 Q/K in compact BF16 layout through the GDN core: 16 Q/K heads for TP1 or 8 for TP2, with three grouped value heads per Q/K head. Requires `VLLM_GDN_QWEN38_NATIVE_QK_PREP`; supersedes `VLLM_GDN_QWEN38_BF16_QK`. TP2 requires rebuilding the native extension and TPC library with TP2 support. | `false` |
| `VLLM_GDN_QWEN38_NATIVE_COMPACT_KKT` | Uses the Gaudi2 TPC compact-KKT producer for Qwen3.8 chunk-64 prefill. Requires compact Q/K and the native Q/K extension. | `false` |
| `VLLM_GDN_COMPACT_QK_FACTOR_GATE` | Experimental compact-Q/K phase-A dataflow that applies each value head's gate to the smaller KKT coefficient matrix before the shared physical K-head multiplication. | `false` |
| `VLLM_GDN_NATIVE_RECURRENT_SCAN` | Lowers the fused BF16 phase-B recurrent projection and additions as one mixed MME/TPC Synapse subgraph. Requires `VLLM_GDN_FUSED_STATE_MATMUL`, BF16 state, and `VLLM_GDN_BF16_BMM_F32_EXTENSION`. | `false` |
| `VLLM_GDN_QWEN38_NATIVE_QK_PREP_EXTENSION` | Path to the built PyTorch registration extension used by `VLLM_GDN_QWEN38_NATIVE_QK_PREP`. Its TPC perf library must also be present in `GC_KERNEL_PATH`. | empty |
| `VLLM_HPU_DYNAMIC_QUANT_CGUID` | Uses Habana `calculate_scale_for_cast` for runtime dynamic per-token FP8 activation scales on sufficiently large prefill graphs, while preserving the ordinary path's epsilon for zero and tiny rows. Model-load block-to-channel weight conversion retains the ordinary `abs().amax()` path. | `false` |
| `VLLM_HPU_DYNAMIC_QUANT_CGUID_MIN_TOKENS` | Minimum flattened token count for `VLLM_HPU_DYNAMIC_QUANT_CGUID`. Smaller prompt and decode graphs retain the ordinary reduction path. | `2048` |
| `VLLM_HPU_EXPLICIT_SIGMOID_SILU` | Expresses long-prompt SwiGLU as `gate * sigmoid(gate) * up`, allowing the HPU graph compiler to fuse the activation with its products and following dynamic FP8 quantization. | `false` |
| `VLLM_HPU_EXPLICIT_SIGMOID_SILU_MIN_TOKENS` | Minimum flattened token count for explicit-sigmoid SwiGLU. Smaller prompt and decode graphs retain the ordinary HPU SiLU path. | `2048` |
| `VLLM_GDN_HPU_CAUSAL_CONV1D` | Uses Habana's native causal-conv1d forward op for GDN prompt convolution and SiLU, then explicitly persists its functional cache output. Decode is unchanged. | `false` |
| `VLLM_GDN_TOKEN_MAJOR_CAUSAL_CONV1D` | Keeps the compiled PyTorch GDN prompt convolution in token-major layout, avoiding the packed activation's channel-major round trip. Decode is unchanged. | `false` |
| `VLLM_HPU_FLASHINFER_GDN` | Enables the in-tree FlashInfer-compatible packed GDN decode path for Qwen hybrid models. In `auto`, only the measured contiguous-state fast path is selected; other layouts use the existing vLLM implementation. | `false` |
| `VLLM_HPU_FLASHINFER_DFLASH2` | Enables the experimental HPU V1 DFlash2 proposer and FlashInfer-Gaudi grouped-convolution, candidate-selection, and rollback-safe GDN paths. The initial qualification scope is greedy Qwen3.8/Qwen3.5 hybrid inference, TP1/PP1/DP1, text-only requests, at most 16 sequences, with prefix caching and LoRA disabled. When unset, this follows `VLLM_HPU_FLASHINFER_GDN`. | `false` (`true` with FlashInfer GDN) |
| `VLLM_HPU_DFLASH2_CONV_ROUND_BEFORE_ACTIVATION` | Experimental numerical-contract check: round speculative convolution to the cache dtype before activation, matching ordinary HPU decode. Retains convolution rollback and padding behavior. Off by default pending independent model-quality and acceptance qualification; this is not a kernel promotion. | `false` |
| `VLLM_HPU_DFLASH2_FULL_QUERY_CONV` | Experimental removal of redundant speculative-convolution masks and state reads. Requires a CPU-proven complete DFlash2 verification block, compact request-owned state, and prefix caching disabled. Ragged batches and single-token transitions retain the reference path. Arithmetic and activation rounding are unchanged. | `false` |
| `VLLM_HPU_DFLASH2_DIRECT_CHECKPOINTS` | Experimental internal contiguous destination for the prepared native DFlash2 Target checkpoint writes. Requires full T8 verification, compact state, no prefix caching, and an exact CPU proof of group-major request/checkpoint ownership. Other layouts retain indexed writes. Does not enable or promote the native kernel itself. | `false` |
| `VLLM_HPU_DFLASH2_DEVICE_PREPARE` | Experimental fixed-width DFlash2 target-to-draft bridge. Accepted lengths, context/query positions, KV slots, block lists, and attention bias remain on HPU so draft work can be queued before target samples are copied to the host. Unsupported or ragged decode batches retain the CPU preparation path. | `false` |
| `VLLM_HPU_FLASHINFER_GDN_TP2` | Opt in to the experimental TP2 local-head GDN prefill and fused decode shapes. The parent GDN switch alone keeps the existing geometry. | `false` |
| `VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE` | Enables the qualified Qwen3.8 TP1 fused direct-state decode recipe for batches 1, 2, 4, 8, 16, and 32. Unsupported shapes retain the existing packed GDN path. When unset, this follows `VLLM_HPU_FLASHINFER_GDN`. | `false` (`true` with FlashInfer GDN) |
| `VLLM_HPU_FLASHINFER_GDN_PREFILL` | Enables the offline-promoted FlashQLA graph tactic for Qwen3.8 GDN prefill. Unsupported shapes retain the general HPU path. When unset, this follows `VLLM_HPU_FLASHINFER_GDN`. | `false` (`true` with FlashInfer GDN) |
| `VLLM_HPU_GDN_DIRECT_STATE` | Uses group-major compact recurrent-state spans when a decode bucket is full, prefix caching is disabled, and request slots are contiguous. | `true` |
| `VLLM_HPU_GDN_ACTIVE_STATE_VIEWS` | Experimental: binds only active recurrent-state rows outside compiled Qwen3 decoder groups to reduce state writeback dependencies. Requires FlashInfer fused direct-state ordinary decode; prefill, indexed state, and speculative decode retain their existing cache paths. | `false` |
| `VLLM_HPU_GDN_ASYNC_STATE_DMA` | Experimental: compiled Qwen3 decoder groups return fresh SSM states for native DMA on a separate HPU stream. Disjoint state slices and queued consumer/reset events avoid whole-pool write dependencies. Requires active state views, regional decoder groups, and a TP2 bridge exporting `queue_gdn_state_copies` / `queue_gdn_state_waits`; currently TP2 batch-one ordinary decode with speculation and prefix caching disabled. | `false` |
| `VLLM_HPU_GDN_DIRECT_STATE_UPDATE` | Experimental TP2 C1 state-update epilogue bound to the final FP32 active cache row. Requires active state views, regional compilation, and the matching native bridge. Compiler liveness checks reject unsafe mutation lowering. Overrides queued state DMA; speculation and prefix caching are unsupported. Remains unqualified until exact state and end-to-end checks pass. | `false` |
| `VLLM_HPU_GDN_PRECISE_DMA_EVENTS` | Experimental queued-state DMA tickets use physical producer/copy/consumer events. Ordinary reads wait on compute only; reset and prefill join all generic substreams and retain events through every consumer. Requires queued state DMA and the matching native bridge. Direct state updates remove this DMA path. | `false` |
| `VLLM_HPU_TP2_PREPARED_COMM` | Prepares fixed TP2 exchange buffers and the local reduction/RMSNorm recipe during warmup. Required by the static group plan and native decoder graph. | `false` |
| `VLLM_HPU_TP2_NATIVE_JOINT_PLAN` | Internal Gaudi2 TP2 research path (`1` enables). Instantiates compute command pages and a prepared NIC batch; requires matching native runtime APIs. Unqualified candidates remain disabled. | `0` |
| `VLLM_HPU_TP2_GQA_COMPACT_KV` | Research Gaudi2 TP2 C1 Qwen decode layout: share BF16 K/V between grouped query rows; requires exact MME qualification. | `0` |
| `VLLM_HPU_TP2_COMPILED_CONSUMER_NORM` | Research-only Gaudi2 TP2 C1: keep dedicated exchange as a side-effect boundary and compile local add/residual/RMSNorm into its consumer. Requires the native joint plan and exact qualification. | `0` |
| `VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT` | Research Gaudi2 TP2 C1: one TPC node for the unchanged CGUID BF16 scale/epsilon/reciprocal/FP8 sequence. Requires the separate kernel database, native adapter and complete exact/model qualification. | `0` |
| `VLLM_HPU_TP2_STATIC_GROUP_PLAN` | Retains the eight compiled decoder groups, their fixed bindings, and their compute/collective dependency order. Requires prepared communication and direct GDN state update. | `false` |
| `VLLM_HPU_TP2_PLAN_DUMP_DIR` | Optional directory for prepared compiler graphs, recipe IDs and compute/communication order. Writes only during preparation. | unset |
| `VLLM_HPU_NATIVE_DECODE_GRAPH` | Research-only native replay for the Gaudi2 Qwen3.8 TP2 C1 decoder. Captures the eight compiled groups and 128 layer TP2 reductions into fixed-address Synapse/SCAL/HCL templates; the embedding reduction remains outside the decoder graph. Requires the two TP2 options above, direct GDN state update, matching custom runtime libraries, and `PT_HPU_POOL_MEM_ACQUIRE_PERC<=95` for persistent recipe storage. Explicit requests fail closed when any contract is missing. | `false` |
| `VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH` | Experimental Gaudi2 DeepSeek V4 Flash BF16 TP2 C1 native decoder. Uses six compiled groups for all 43 layers, 86 ordinary AllReduces and the mHC/head/norm tail. Requires the prepared communication and joint-plan runtime, packed q1 metadata and context <=512; does not require GDN. Runtime ABI fingerprints are checked before model loading. | `false` |
| `VLLM_HPU_DSV4_WORKER_HELPER_CPUS` | Semicolon-separated per-rank CPU sets for initialized background worker threads, e.g. `30-34,86-90;38-43,94-99`. Requires one main CPU per rank in `VLLM_HPU_DSV4_WORKER_CPUS`. Helper sets must be disjoint and exclude every main CPU and its SMT sibling. Applied in normal worker warmup. | unset |
| `VLLM_HPU_GDN_PADDED_DIRECT_STATE` | Opts partially filled decode buckets into the direct-state path. Requires FlashInfer GDN and direct state to be enabled, a contiguous active-slot prefix, and every padding slot to be free. Paused requests retain their slots and are never used as padding. Prefix caching and multi-token decode retain the general path. | `false` |
| `VLLM_HPU_CGUID_DYNAMIC_QUANT` | Uses Gaudi's fused scale-calculation CGUID for decode-sized per-token dynamic FP8 quantization. Large prefill matrices retain the existing reduction path to preserve its numerical behavior. When unset, this follows `VLLM_HPU_FLASHINFER_GDN`. | `false` (`true` with FlashInfer GDN) |
| `VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS` | Maximum flattened row count eligible for CGUID dynamic quantization. Keep this below the smallest prefill token bucket; increasing it can change prefill graph fusion. | `32` |
| `VLLM_HPU_FUSED_GREEDY_LOGITS` | Fuses hidden-state selection, the LM-head projection, FP32 conversion, and argmax into one compiled decode region for plain greedy requests. Requests using logprobs, penalties, token masks, logit processors, structured output, or speculative decoding retain the general sampler path. | `false` |
| `FLASHINFER_GAUDI_BACKEND` | Selects `auto`, `native`, `public`, `bridge`, or `pytorch`. Strict policies (`native/public/bridge`) require the entire operation to be native. The SiLU candidate supports `native/public`; GDN strict calls reject the remaining torch prologue/reference graph. Native availability does not imply performance qualification. | `auto` |
| `FLASHINFER_GAUDI_STATE_DTYPE` | Selects the requested recurrent-state precision. `fp32` is the production default; `bf16` remains experimental until model-quality validation succeeds. | `fp32` |
| `FLASHINFER_GAUDI_NATIVE_LIBRARY` | Optional path-list of native host-extension libraries to load. Packaged libraries are discovered automatically. | unset |
| `FLASHINFER_GAUDI_ENABLE_COMPAT_SHIM` | Exposes supported modules under the `flashinfer` namespace when explicitly enabled and the official package is absent. vLLM-Gaudi does not require this shim. | `false` |
| `FLASHINFER_GAUDI_ENABLE_PUBLIC_AUTO` | Deprecated, ignored. Environment flags cannot bypass offline whole-operation correctness and performance qualification. | ignored |
| `FLASHINFER_GAUDI_ENABLE_MTP_AUTO` | Deprecated, ignored. DFlash2 MTP automatic selection is controlled by the offline tactic manifest. | ignored |
| `FLASHINFER_GAUDI_ENABLE_BRIDGE_AUTO` | Deprecated, ignored. The bridge GDN prototype is not qualified for whole-operation native dispatch. | ignored |
| `FLASHINFER_GAUDI_ENABLE_MTP_PREPARED` | Experimental qualification switch for the graph-native Qwen3.8 B1/T8 full-query MTP core. Requires rebuilt native libraries and `auto` or `public`; forced `pytorch`, eager calls, padded/partial queries, and other batches keep their existing routes. It remains off until full-model quality and end-to-end latency gates pass. | `false` |
| `FLASHINFER_GAUDI_ENABLE_DFLASH2_SELECT_AUTO` | Deprecated, ignored. DFlash2 path selection is controlled by the offline tactic manifest. | ignored |
| `FLASHINFER_GAUDI_ENABLE_DFLASH2_TOPK_AUTO` | Deprecated, ignored. DFlash2 TopK selection is controlled by the offline tactic manifest. | ignored |
| `FLASHINFER_GAUDI_ENABLE_DFLASH2_SCORE_SELECT_AUTO` | Deprecated, ignored. DFlash2 score-selection is controlled by the offline tactic manifest. | ignored |
| `VLLM_GAUDI_BUILD_FLASHINFER` | Controls native builds during packaging: `auto` builds when both the TPC compiler and Gaudi PyTorch package exist; `1` requires a successful build; `0` installs reference code only. | `auto` |
| `VLLM_HPU_TRITON_MODE` | Selects Gaudi2-native Triton paths: `off`, performance-safe `hybrid`, or fail-closed `strict`. Hybrid uses only paths that passed the relevant eager/fullgraph gate; strict also exposes ungated kernels for correctness and A/B performance CI. | `off` |
| `VLLM_HPU_TRITON_CACHE_DIR` | Overrides the private, content-addressed TPC ELF cache used by the Triton/Bridge ABI. | `None` |
| `VLLM_HPU_TRITON_BLOCK_SIZE` | Logical Triton block size used by the initial 2048-bit TPC elementwise kernels. | `256` |
| `VLLM_HPU_TRITON_SILU_BLOCK_SIZE` | Power-of-two row chunk used to distribute each fused SiLU-and-mul row across Gaudi2 TPC engines; accepted range is 128–1024. | `128` |
| `VLLM_HPU_TRITON_GDN_VALUE_TILE` | Value rows owned by each experimental Qwen3.5 packed GDN decode program; accepted values are 16, 32, 64, and 128. | `16` |

The Gaudi2 backend routes supported contiguous BF16 residual RMSNorm calls with
hidden sizes up to 8192 to one Triton-generated TPC kernel. It returns both the
normalized activations and rounded residual sum. The production path is a
Bridge custom op inside the current HPU graph and is compatible with
`torch.compile(..., backend="hpu_backend", fullgraph=True)`; the diagnostic
direct-recipe launcher remains outside HPUGraph capture. The eager path has
passed its operator gate, while hybrid mode retains the vendor implementation
inside compiled graphs until the fullgraph gate also passes. vLLM prepares the
fixed-GUID perf library during operator registration, before Synapse graph
compiler initialization.

The default remains `off` until the fast-path operator set passes an end-to-end
model gate. Use `strict` for correctness/performance CI with no fallback, or
`hybrid` when an explicit, counted HPU vendor fallback is acceptable.

Row-wise BF16-to-E4M3 dynamic quantization is available as a strict-mode
candidate for decode shapes with at most 32 rows and hidden sizes up to 16384.
It folds max-abs reduction, F32 scale generation, and FP8 conversion into one
Triton-generated TPC node. Larger prefill shapes and `single_scale=True` retain
the vendor path. Run
`python tools/benchmark_triton_gaudi_dynamic_quant.py` to compare the candidate
against the existing `amax` plus `cast_to_fp8_v2` fullgraph. Hybrid rollout
remains closed until that generated-kernel gate and the model-level output gate
both pass on Gaudi2.

When a strict fullgraph contains an exclusive SiLU-and-mul result immediately
consumed by row-wise dynamic quantization, the HPU compiler pass replaces both
custom nodes and their alias-only view chain with one Triton-generated TPC
node. The fusion is limited to static two-dimensional BF16 shapes with output
widths up to 4096 and fails closed on extra activation consumers or mismatched
specializations. Run
`python tools/benchmark_triton_gaudi_silu_dynamic_quant.py` to exercise the
actual graph rewrite against the vendor fullgraph. Hybrid rollout remains
closed until this combined graph gate passes.

Run `python tools/benchmark_triton_gaudi_rms_norm.py` on Gaudi2 after setting
the matching Triton perf-library and artifact-cache environment variables. The
gate requires at least 1.20x geometric-mean device and wall speedup and rejects
any tested shape below 0.95x. Fullgraph compilation is the default comparison;
use `--eager` only for the diagnostic operator-level comparison.

Packed Qwen3.5 GDN decode remains on the complete vendor graph in `hybrid`.
Both the recurrent-only kernel and the experimental split Q/K-conv plus
value-conv/GDN path are available only in `strict` mode for standalone
diagnostics. The recurrent kernel wins its operator microbenchmark, but its
48-layer full-model graph has not cleared the no-regression gate; the split path
also does not yet preserve convolution state across decode-recipe re-entry.
Weight loading still materializes the TPC-friendly transposed convolution
weights and an FP32 decay-bias view once for strict diagnostics. The Bridge
admits stateful GDN nodes to a native graph only for the validated batch-eight
shape and keeps every other shape on the custom-op path. The strict standalone
diagnostic is `python tools/benchmark_triton_gaudi_gdn_decode.py --include-conv`;
it validates the BF16 output, FP32 recurrent state, and BF16 convolution state
before reporting speedup. Add `--check-recipe-reentry` to validate state after
switching to another batch recipe and back. Because a two-node micrograph can
become host-submit bound, hybrid rollout is decided by the full-model gate
rather than that standalone timing alone. A complete Gaudi Bridge installation
must include the GDN reinplace compiler pass; initialization rejects stale
Python package overlays instead of running the functionalized full-cache-copy
graph. SiLU-and-mul remains a strict-mode candidate.

Triton and FlashInfer-Gaudi coexist in the same checkout. With Triton `off`,
the existing FlashInfer switches select its graph tactics or the general HPU
implementation. Triton `hybrid` also preserves that GDN selection. Triton
`strict` takes precedence for recurrent decode and requires identical load/store
state-index tensors; it cannot silently execute FlashInfer instead. The known
recipe-reentry issue keeps split convolution kernels out of model execution;
they remain available through the standalone diagnostic. TP2 collective/RMSNorm
fusion retains ownership of its communication boundary before local Triton
RMSNorm is considered.

## DeepSeek V4.1

The experimental [prepared TP2×PP2 profile](../features/deepseek_v41.md)
uses explicit switches and rejects unsupported execution contracts.

| Variable | Description | Default |
|---|---|---|
| `VLLM_HPU_DSV41_PREPARED_SHARDS` | Enables the rank-local loader and bounded CSA2 runner on four Gaudi2 devices. Requires the immutable TP2×PP2 manifest. | `false` |
| `VLLM_HPU_DSV41_ENGRAM_HOST_TABLE` | Uses shared read-only host mmap tables, native asynchronous row gather, and generation-owned HPU staging. | `false` |
| `VLLM_HPU_DSV41_GRAPH_REPLAY` | Captures each PP stage with the ABI-locked native compute/communication plan. Requires prepared communication and the static group plan. | `false` |
| `VLLM_HPU_DSV41_DSPARK` | Runs the three-layer draft on PP1 with accepted-prefix context insertion and Engram rollback. Requires `method=dspark`, five speculative tokens. | `false` |
| `VLLM_HPU_DSV41_VISION` | Binds the portable upstream ViT/aligner to prepared PP0 weights. Requires `mm_encoder_tp_mode=data`. | `false` |
| `VLLM_HPU_DSV41_QUANT_ROUNDTRIP` | Fuses BF16 activation group-32 E4M3FN quantization and restoration in TPC. Matrix operands remain BF16. Experimental. | `false` |
| `VLLM_HPU_DSV41_PACKED_ATTENTION` | Uses a C1 native composite to decode selected packed KV rows once for all heads, then consumes them with the existing ordered attention arithmetic. Experimental; other token counts keep their existing path. | `false` |
| `VLLM_HPU_DSV41_BOUNDED_ATTENTION` | Limits C1 packed attention to its runtime visible prefix at context lengths up to 512. Reuses the ordered V4 attention kernel and retains Full/Reindex/Reuse state publication. Requires packed attention; experimental. | `false` |
| `VLLM_HPU_DSV41_FP8_DECODE` | Enables selected routed experts in ordinary C1 native replay to decode Q16/S16 directly into E4M3 SRAM weights for FP8 MME. Requires prepared channel sidecars; prefill keeps the existing precision. Experimental and not quality qualified. | `false` |
| `VLLM_HPU_DSV41_FP8_SIDECAR` | Directory produced by `tools/prepare_deepseek_v41_fp8.py`, with channel scales and a manifest bound to the immutable prepared shards. | empty |
| `VLLM_HPU_DSV41_FP8_CONFIG` | Optional JSON precision configuration with `version: 1` and selected `routed_experts` layer IDs. `attention`, `shared_experts`, and `mhc` must be empty in this candidate. | all routed expert layers |
| `VLLM_HPU_DSV41_FIXED_POSITIONS` | Binds cached views of an immutable device position bank directly to C1 decode native staging. Prefill keeps its disjoint input buffers. Requires the single-in-flight V4.1 runner. | `false` |
| `VLLM_HPU_DSV41_PACKED_PP` | Packs ordinary C1 BF16 hidden and FP32 pre-mix bits into one HCCL message using two generation-owned buffers. DSpark and multi-token transfers retain their existing transport. Experimental. | `false` |
| `VLLM_HPU_DSV41_TPC_MHC` | Uses FP32 TPC GEMV for the C1 mHC control projection. Preserves FP32 operands but changes reduction order from MME; requires separate model quality qualification. Other shapes retain MME. Experimental. | `false` |
| `VLLM_HPU_DSV41_ENGRAM_NATIVE_C1` | Performs C1 compression, integer hash, TP row lookup and final pinned staging copies in the native host extension. Requires C1 ABI version 1; retains consumer-stream DMA, request history transactions and the existing prefill path. Experimental, not performance or quality qualified. | `false` |
| `VLLM_HPU_DSV41_TP_MHC_OVERLAP` | Splits proven independent C1 residual/mHC work between TP production and consumption. Requires the versioned native dependency API, BF16 ordinary decode and joint replay; rejects a capture with no independent segment. Experimental, not performance or quality qualified. | `false` |
| `VLLM_HPU_DSV41_DIRECT_TOKEN_IDS` | Binds the completed int32 device token directly to C1 embedding and native stage inputs, with request/position checks. Prefill retains int64 input storage. Experimental. | `false` |
| `VLLM_HPU_DSV41_DEVICE_COMMIT` | Builds the C1 completion record after the compiled greedy head, broadcasts it through PP, and copies the four integers to the host with a native completion ticket. Requires the matching Bridge API, graph replay, and DSpark disabled. Experimental. | `false` |
| `VLLM_HPU_DSV41_NATIVE_PP_COPY` | Packs contiguous C1 hidden/pre-mix tensors through bounded native DMA with existing storage dependencies. Requires packed PP, graph replay and the matching Bridge API; rejects unsupported physical permutations. Experimental. | `false` |
| `VLLM_HPU_DSV41_PREPARED_OUTPUT` | Prepares the BF16 wo_a K,N layout once after target weights load, replacing its original buffer to avoid repeated weight transposes. Reload invalidates recipes and prepares the new weights again. Experimental. | `false` |
| `VLLM_HPU_DSV41_ISOLATE_CONTROL` | Enables startup CPU isolation for the V4.1 EngineCore and API after worker processes have been spawned. The launcher reserves separate NUMA-local physical cores. Experimental. | `false` |
| `VLLM_HPU_DSV41_ENGINE_CPUS` | EngineCore CPU set, including its I/O threads. Must be allowed by the initial launch affinity and disjoint from worker and API cores and their SMT siblings. | unset |
| `VLLM_HPU_DSV41_API_CPUS` | API CPU set with the same ownership checks. Both control sets are recorded by the launcher. | unset |
| `VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR` | Selects an independently built native kernel directory. Both binaries must match its build manifest; communication runtime ABI checks remain required. | unset |

`VLLM_HPU_DSV4_WORKER_CPUS` and `VLLM_HPU_DSV4_WORKER_HELPER_CPUS`
also apply to this profile, with one entry per rank. Each worker sets
`HLS_MODULE_ID` from its assignment in `HABANA_VISIBLE_MODULES` before allocation.

## DeepSeek V4

The [source-integrated Gaudi2 TP2 profile](../features/deepseek_v4_flash.md)
selects the accepted short-context defaults during normal platform setup.
The table below lists raw environment fallbacks outside that profile.

| Variable | Description | Default |
|---|---|---|
| `VLLM_HPU_DSV4_EARLY_OUTPUT_LOWERING` | Lowers q1 attention output dependencies before HPU graph partitioning, retaining required clones. Enabled by the bounded source profile. | `false` |
| `VLLM_HPU_DSV4_WORKER_CPUS` | Optional comma-separated CPU IDs, one distinct allowed CPU per rank. No affinity change when unset. | unset |
| `VLLM_HPU_MXFP4_DECODE_GATHER` | Gathers only the routed packed experts before single-token MXFP4 MoE decode. Disable to use the full expert TensorList. | `true` |
| `VLLM_HPU_DSV4_TPC_MXFP4_GATHER` | Uses one experimental Gaudi2 TPC launch to gather DeepSeek V4's four packed MXFP4 weight/scale tensors for the six routed experts. | `false` |
| `VLLM_HPU_DSV4_TPC_MXFP4_INDEXED` | Uses the experimental Gaudi2 direct-indexed MXFP4 decode path, reading six routed experts from stacked weights without materializing selected weights. | `false` |
| `VLLM_HPU_DSV4_MXFP4_PREPARED_MME` | Prepares DeepSeek V4 TP2 expert weights in the Gaudi2 Q16/S16 layout at load time and uses the exact BF16 direct-indexed MME path for single-token top-6 decode. Unsupported calls restore the checkpoint layout within a bounded temporary buffer. | `false` |
| `VLLM_HPU_DSV4_BF16_ATTN_WEIGHT_CACHE` | Caches transposed BF16 copies of DeepSeek V4's fused Q/KV and Q-projection weights for single-token decode. Prompt processing stays on the block-FP8 path. | `false` |
| `VLLM_HPU_DSV4_COMPILED_ATTN_FRONTEND` | Runs DeepSeek V4 single-token projection, Q/KV normalization, and Q projection through shared HPU-compiled functions. Requires the BF16 attention weight cache. | `false` |
| `VLLM_HPU_DSV4_INLINE_ATTN_FRONTEND` | Inlines the DeepSeek V4 single-token projection frontend into coarse decoder graphs while keeping cache mutations and MLA behind a custom-op boundary. Requires the BF16 attention weight cache. | `false` |
| `VLLM_HPU_DSV4_NATIVE_FP8_ATTN_FRONTEND` | Uses native FP8 MME for the two DeepSeek V4 attention projections during single-token decode. Keeps the BF16 path available as a quality fallback. | `false` |
| `VLLM_HPU_DSV4_SHORT_INDEXER_SKIP` | Skips DeepSeek V4 indexer scoring when all compressed candidates fit in top-k, while retaining cache writes. | `true` |
| `VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP` | Also skips the unused DeepSeek V4 indexer projection and K-cache write during decode when the configured maximum compressed sequence fits entirely in top-k. Long contexts and mixed prefill/decode batches automatically fall back. | `false` |
| `VLLM_HPU_DSV4_FUSED_SDPA` | Uses Gaudi FusedSDPA plus an explicit attention-sink merge for the gathered DeepSeek V4 sparse-attention core. | `false` |
| `VLLM_HPU_DSV4_TPC_DEQUANT_GATHER` | Uses the experimental Gaudi2 TPC packed-cache dequant/gather kernel for DeepSeek V4 decode. | `false` |
| `VLLM_HPU_DSV4_TPC_SPARSE_ATTN` | Uses the experimental Gaudi2 TPC sparse-attention kernel for DeepSeek V4 decode. | `false` |
| `VLLM_HPU_DSV4_TPC_SPARSE_ATTN_MAX_WIDTH` | Maximum gathered width that uses the experimental DeepSeek V4 sparse-attention TPC kernel; larger widths retain the graph implementation. | `128` |
| `VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN` | Uses the experimental Gaudi2 native-FP8 paged sparse-attention kernel for single-token DeepSeek V4 decode. | `false` |
| `VLLM_HPU_DSV4_TPC_PAIR_HEADS` | Tiles two DeepSeek V4 MQA query heads per Gaudi2 paged-attention program so C4 decode shares each packed KV load and dequantization. | `false` |
| `VLLM_HPU_DSV4_FLASHMLA_SPLIT_KV` | Uses the functional split-KV FlashMLA-style packed-FP8 decode path. | `false` |
| `VLLM_HPU_DSV4_FLASHMLA_TILED` | Selects the experimental four-token tiled-softmax partial kernel for the split-KV FlashMLA path. | `false` |
| `VLLM_HPU_DSV4_FLASHMLA_SPLITS` | Number of split-KV partitions used by the FlashMLA-style decode path. | `2` |
| `VLLM_HPU_DSV4_FLASHMLA_PREFILL` | Uses the experimental length-aware Gaudi2 sparse-prefill kernel, mirroring FlashMLA `flash_mla_sparse_fwd` without scanning aligned index padding. | `false` |
| `VLLM_HPU_DSV4_FLASHMLA_PREFILL_MAX_WIDTH` | Maximum aligned sparse-index width accepted by the experimental length-aware prefill kernel. | `640` |
| `VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN` | Uses TPC packed-cache dual gather with FP32 Gaudi MME QK/PV for single-token DeepSeek V4 decode. | `false` |
| `VLLM_HPU_DSV4_ATTENTION_BACKEND` | Selects the DeepSeek V4 decode backend: `auto`, `flashmla_hpu`, `mme`, or `legacy`. `auto` prefers the direct packed-FP8 FlashMLA-style path. | `auto` |
| `VLLM_HPU_DSV4_DIRECT_DECODE_DISPATCH` | Bypasses generic per-layer backend planning for eligible C4 single-token decode and dispatches directly to the selected packed-FP8 TPC kernel. | `false` |
| `VLLM_HPU_DSV4_TPC_SAVE_COMPRESS_NORM_C4` | Uses the experimental fused C4/C128 state-save, compression, norm, RoPE, and packed-cache producer for single-token DeepSeek V4 decode. | `false` |
| `VLLM_HPU_DSV4_TPC_ORDERED_COMPRESSOR` | Uses the experimental alias-free C4 Compressor write with an explicit device dependency on the following attention operation. | `false` |
| `VLLM_HPU_DSV4_TPC_ORDERED_C128_COMPRESSOR` | Extends the alias-free ordered Compressor path to C128 decode while preserving an explicit dependency on the following sparse attention operation. | `false` |
| `VLLM_HPU_DSV4_FUSED_COMPRESSOR_FLASHMLA` | Places the C4 Compressor cache producer and tiled FlashMLA partial/combine kernels in one ordered Gaudi2 recipe for eligible single-token decode. | `false` |
| `VLLM_HPU_DSV4_FUSED_QNORM_COMPRESSOR` | Co-schedules the independent QNorm/RoPE/SWA-cache and C4 Compressor cache producers in one compiled Gaudi2 recipe while keeping MLA in a downstream recipe. | `false` |
| `VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS` | Experimental precision-changing path that keeps DeepSeek V4 compressor projections and norm inputs in native BF16. | `false` |
| `VLLM_HPU_DSV4_TPC_MIXED_COMPRESS_INPUTS` | Preserves FP32 compressor projections while reading the static norm in BF16 and using the fused vector-RoPE TPC path. | `false` |
| `VLLM_HPU_DSV4_TPC_QNORM_ROPE_KV_PACK` | Uses the experimental per-head fused Q norm/RoPE, KV RoPE/FP8 quantization, and direct paged-cache writer for single-token DeepSeek V4 decode. | `false` |
| `VLLM_HPU_DSV4_TPC_MHC` | Uses the decode-specialized Gaudi2 fused post+pre MHC TPC kernel for the DeepSeek V4 hardware-agnostic model path. | `false` |
| `VLLM_HPU_DSV4_TPC_SINKHORN` | Experimental V4 compiler fusion for the complete39-step FP32 C1 Sinkhorn chain. Preserves the surrounding projection, gates, softmax, norm and BF16 boundaries; requires the matching native kernel library. | `false` |
| `VLLM_HPU_DSV4_PACKED_DECODE_METADATA` | Packs DeepSeek V4 q1 framework attention metadata into one persistent int32 host-to-device transfer per decode step. | `false` |
| `VLLM_HPU_DSV4_DECODE_METADATA_RING_SIZE` | Number of host/device packed metadata buffers rotated by the DeepSeek V4 q1 decode path to avoid overwriting in-flight graph inputs. | `2` |
| `VLLM_HPU_DSV4_Q1_METADATA_FASTPATH` | Enables uniform q1 shortcuts in DeepSeek V4 metadata builders, removing redundant per-step tensor construction and uploads. | `false` |
| `VLLM_HPU_DSV4_TPC_OP_LIBRARY` | Path to the PyTorch registration library for the experimental DeepSeek V4 TPC operators. | unset |

Use `VLLM_BUCKETING_STRATEGY=exp` for the default exponential warm-up, `VLLM_BUCKETING_STRATEGY=lin` for explicitly configured linear ranges, or `VLLM_BUCKETING_STRATEGY=pad` for padding-aware ranges with absolute and relative padding limits.

Set `VLLM_PROMPT_BUCKETING_STRATEGY` or `VLLM_DECODE_BUCKETING_STRATEGY` to override only that phase. For example, `VLLM_BUCKETING_STRATEGY=exp` with `VLLM_DECODE_BUCKETING_STRATEGY=pad` keeps exponential prompt buckets and uses padding-aware decode buckets. An unset phase override falls back to the global strategy. This configuration does not enable padded direct GDN state access.

Leave `VLLM_EXPONENTIAL_BUCKETING` unset when using global or phase-specific strategy settings. The legacy flag still overrides both phases when present. `VLLM_BUCKETING_FROM_FILE` takes precedence over generated buckets for both phases.

## Developer Mode Parameters

To enter developer mode use `VLLM_DEVELOPER_MODE`:

| Parameter name     | Description              | Default value |
| ------------------ | ------------------------ | ------------- |
| `VLLM_SKIP_WARMUP` | Skips the warm-up phase. | `false`       |

## Additional Parameters

| Parameter name                | Description                                                                                                                                                                                   | Default value |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `VLLM_HANDLE_TOPK_DUPLICATES` | Handles duplicates outside top-k.                                                                                                                                                             | `false`       |
| `VLLM_CONFIG_HIDDEN_LAYERS`   | Sets the number of hidden layers to run per HPUGraph for model splitting among hidden layers when TP is 1. It improves throughput by reducing inter-token latency limitations in some models. | `1`           |
| `VLLM_HPU_QWEN3_COMPILE_LAYER_GROUP_SIZE` | Compiles consecutive Qwen3-Next/Qwen3.5 decoder layers as one regional graph during decode and folds the final RMSNorm into the last group. Prefill remains per-layer. Values greater than 1 require non-eager execution and are supported with TP=1, or TP=2 when `VLLM_HPU_TP2_FUSED_AR_NORM=true`. Decode batches above 16 use grouped execution only with the validated direct GDN state layout; other layouts retain per-layer compilation. | `1` |
| `VLLM_WORKER_MULTIPROC_METHOD` | Sets the Python `multiprocessing` start method used by the `mp` distributed executor backend when launching worker processes. The upstream default is `fork`. On HPU, it is automatically overridden to `spawn` with a warning because forked child processes inherit HPU driver state and can hang on exit. The override is applied when `--distributed-executor-backend` is `mp` or `uni`. With `uni`, no subprocess is created, so the value has no practical effect. With `external_launcher` and `ray`, workers are not started through Python `multiprocessing`, so the value is irrelevant. Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` explicitly to suppress the auto-override warning, or set it to `fork` to opt out of the override, which is not recommended. | `spawn` on HPU (auto-overridden from upstream `fork`) |

## Heterogeneous KV Transfer (NIXL)

These variables control the NIXL KV-cache transfer path when an HPU prefill instance serves a GPU decode instance (disaggregated prefill/decode across different accelerators). They apply on the HPU prefill side and require `VLLM_HPU_HETERO_KV_LAYOUT=true` plus `enable_permute_local_kv` in the KV transfer config.

| Parameter name                | Description                                                                                                                                                                                                                                                                                                                                                                     | Default value  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `VLLM_HPU_NIXL_JOINT_KV`      | Enables joint-KV staging so an HPU prefill instance can serve a GPU (FLASH_ATTN, blocks-first) decode instance. The GPU registers one joint `[K\|V]` region per layer and reads V at `block_len // 2`; when enabled, the HPU stages transferred blocks into a matching joint buffer and advertises the same region model. Leave `false` for HPU-to-HPU disaggregation (separate K/V regions). | `false`        |
| `VLLM_HPU_NIXL_STAGING_SLOTS` | Number of joint-KV staging slots (each holds one on-save block for all layers). `0` auto-sizes from `min(max_num_seqs * ceil(max_model_len / block_size_on_save), fraction_of_total_HPU_memory / per_slot_bytes)`. Set a positive value to override; must be identical on prefill scheduler and worker (same process, so a single value applies). | `0` (auto)     |

## Distributed Executor Backend on HPU

vLLM exposes the `--distributed-executor-backend` CLI flag, also available as `distributed_executor_backend` in the Python API. On HPU, the relevant choices are:

- `mp`: Python multiprocessing-based executor. It is used when `world_size > 1` (that is, `TP * PP * DP > 1`), and each worker runs in its own subprocess. This backend honors `VLLM_WORKER_MULTIPROC_METHOD`. On HPU, the start method is forced to `spawn` to avoid teardown hangs caused by forking after HPU driver initialization. `mp` is the recommended backend for single-node, multi-card serving on Gaudi.
- `uni`: In-process (uni-process) executor. It is selected automatically when `world_size == 1` (typically `TP=1`, `PP=1`, `DP=1`), so no subprocess is started and the worker runs inside the engine process. `VLLM_WORKER_MULTIPROC_METHOD` has no effect on `uni` worker creation. However, the HPU platform still sets the environment variable so that engine-adjacent multiprocessing, such as LMCache helpers or plugins, also runs under `spawn`.
- `external_launcher`: vLLM does not start any workers. Instead, the user is expected to launch all processes through an external tool such as torchrun, MPI, or SLURM. This option is available on HPU, but it is not commonly used.
- `ray`: Ray-based executor. Workers run as Ray actors rather than through Python `multiprocessing`. Multi-node serving with Ray on Gaudi has not yet been validated by the Gaudi software product engineering team. Use `mp` for production deployments.

If the flag is not provided, vLLM selects the backend automatically: `uni` when `world_size == 1`, and `mp` otherwise on HPU.

HPU PyTorch bridge environment variables impacting vLLM execution:

| Parameter name                     | Description                                                                                                                                           | Default value                                    |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `PT_HPU_LAZY_MODE`                 | Sets the backend for Gaudi, with `0` for PyTorch Eager and `1` for PyTorch Lazy.                                                                      | `0`                                              |
| `PT_HPU_ENABLE_LAZY_COLLECTIVES`   | Must be set to `true` for tensor parallel inference with HPU Graphs.                                                                                  | `true`                                           |
| `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` | Must be set to `false` for LLaVA, Qwen, and RoBERTa models.                                                                                           | `false`                                          |
| `VLLM_PROMPT_USE_FLEX_ATTENTION`   | Enabled only for the Llama model, allowing usage of `torch.nn.attention.flex_attention` instead of FusedSDPA. Requires `VLLM_PROMPT_USE_FUSEDSDPA=0`. | `false`                                          |
| `RUNTIME_SCALE_PATCHING`           | Enables the runtime scale patching feature, which applies only to FP8 execution and is ignored for BF16.                                              | `true` (Torch Compile mode), `false` (Lazy mode) |
| `ENABLE_EXPERIMENTAL_FLAGS` and `ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES` | Must both be set to `true` for Qwen3.5 (GDN hybrid) models to improve graph compilation performance. | `false`                                          |

## Additional Performance Tuning Parameters for Bucketing Strategies

`VLLM_{phase}_{dim}_BUCKET_{param}` is a collection of environment variables configuring user-defined bucket ranges, where:

- `{phase}` is in `['PROMPT', 'DECODE']`.
- `{dim}` is in `['BS', 'QUERY', 'CTX']` for `PROMPT` phase or in `['BS', 'BLOCK']` for `DECODE` phase.
- `{param}` is in `['MIN', 'STEP', 'MAX']` for the `lin` strategy.
- `{param}` is in `['MIN', 'STEP', 'MAX', 'PAD_MAX', 'PAD_PERCENT']` for the `pad` strategy.

The following table lists the available variables with their default values. `PAD_MAX` and `PAD_PERCENT` are used when the corresponding phase selects `pad`, through either the global strategy or its phase override.

| Phase  | Variable name                                                            | Default value                                                                                                       |
|--------|--------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Prompt | batch size min (`VLLM_PROMPT_BS_BUCKET_MIN`)                             | `1`                                                                                                                 |
| Prompt | batch size step (`VLLM_PROMPT_BS_BUCKET_STEP`)                           | `1`                                                                                                                 |
| Prompt | batch size max (`VLLM_PROMPT_BS_BUCKET_MAX`)                             | `max_num_prefill_seqs`                                                                                              |
| Prompt | batch size max abs padding (`VLLM_PROMPT_BS_BUCKET_PAD_MAX`)             | `ceil(max_num_prefill_seqs / 4)`                                                                                    |
| Prompt | batch size max padding percent (`VLLM_PROMPT_BS_BUCKET_PAD_PERCENT`)     | `25`                                                                                                                |
| Prompt | query length min (`VLLM_PROMPT_QUERY_BUCKET_MIN`)                        | `block_size`                                                                                                        |
| Prompt | query length step (`VLLM_PROMPT_QUERY_BUCKET_STEP`)                      | `block_size`                                                                                                        |
| Prompt | query length max (`VLLM_PROMPT_QUERY_BUCKET_MAX`)                        | `max_num_batched_tokens`                                                                                            |
| Prompt | query length max abs padding (`VLLM_PROMPT_QUERY_BUCKET_PAD_MAX`)        | `ceil(max_num_batched_tokens / 4)`                                                                                  |
| Prompt | query length max padding percent (`VLLM_PROMPT_QUERY_BUCKET_PAD_PERCENT`)| `25`                                                                                                                |
| Prompt | sequence ctx min (`VLLM_PROMPT_CTX_BUCKET_MIN`)                          | `0`                                                                                                                 |
| Prompt | sequence ctx step (`VLLM_PROMPT_CTX_BUCKET_STEP`)                        | `2`                                                                                                                 |
| Prompt | sequence ctx max (`VLLM_PROMPT_CTX_BUCKET_MAX`)                          | `ceil((max_model_len - VLLM_PROMPT_QUERY_BUCKET_MIN) / block_size)`                                                 |
| Prompt | sequence ctx max abs padding (`VLLM_PROMPT_CTX_BUCKET_PAD_MAX`)          | `ceil(max_num_batched_tokens / block_size)`                                                                         |
| Prompt | sequence ctx max padding percent (`VLLM_PROMPT_CTX_BUCKET_PAD_PERCENT`)  | `25`                                                                                                                |
| Decode | batch size min (`VLLM_DECODE_BS_BUCKET_MIN`)                             | `1`                                                                                                                 |
| Decode | batch size step (`VLLM_DECODE_BS_BUCKET_STEP`)                           | `2`                                                                                                                 |
| Decode | batch size max (`VLLM_DECODE_BS_BUCKET_MAX`)                             | `max_num_seqs`                                                                                                      |
| Decode | batch size max abs padding (`VLLM_DECODE_BS_BUCKET_PAD_MAX`)             | `ceil(max_num_seqs / 4)`                                                                                            |
| Decode | batch size max padding percent (`VLLM_DECODE_BS_BUCKET_PAD_PERCENT`)     | `25`                                                                                                                |
| Decode | num blocks min (`VLLM_DECODE_BLOCK_BUCKET_MIN`)                          | `block_size`                                                                                                        |
| Decode | num blocks step (`VLLM_DECODE_BLOCK_BUCKET_STEP`)                        | `block_size`                                                                                                        |
| Decode | num blocks max (`VLLM_DECODE_BLOCK_BUCKET_MAX`)                          | `ceil(max_model_len * max_num_seqs / block_size)` <br>by default or `max_blocks` <br>if `VLLM_CONTIGUOUS_PA = True` |
| Decode | num blocks max abs padding (`VLLM_DECODE_BLOCK_BUCKET_PAD_MAX`)          | `ceil(VLLM_DECODE_BLOCK_BUCKET_MAX / 4)`                                                                            |
| Decode | num blocks max padding percent (`VLLM_DECODE_BLOCK_BUCKET_PAD_PERCENT`)  | `25`                                                                                                                |

`VLLM_PROMPT_BS_BUCKET_MAX` no longer affects only prompt warm-up coverage. It also affects the real prefill batch size used by the Gaudi runner.

The default value of `25` for `VLLM_*_BUCKET_PAD_PERCENT` is a balance of warmup duration and runtime performance. Using smaller value like `10` introduce more buckets and reduces the padding to get better runtime performance. Setting to `0` to fall back to the original linear bucketing with minimum padding. And setting to `50` is close to the exponential bucketing except for the corresponding  `VLLM_*_BUCKET_MIN` is not `0` nor `1`.

Legacy `VLLM_PROMPT_SEQ_BUCKET_*` variables are still accepted as a fallback for prompt query settings when `VLLM_PROMPT_QUERY_BUCKET_*` is not set, but this compatibility path is deprecated and will be removed in a future release.

When a deployed workload does not use the full context a model can handle, we
recommend you to limit the maximum values upfront, based on the expected input
and output token lengths that will be generated after serving the vLLM server.
For example, suppose you want to deploy the text generation model Qwen2.5-1.5B
with `max_position_embeddings` of 131072 (our `max_model_len`) and your workload
pattern will not use the full context length (you expect the maximum input token
size of 1K and predict generating the maximum of 2K tokens as output). In this
case, starting the vLLM server to be ready for the full context length is
unnecessary and you can limit the values upfront. It reduces the startup time
and warm-up. Recommended settings for this case are:

- `--max_model_len`: `3072`, which is the sum of input and output sequences (1+2)*1024.  
- `VLLM_PROMPT_QUERY_BUCKET_MAX`: `1024`, which is the maximum input token size that you expect to handle.

!!! note
    If the model config specifies a high `max_model_len`, set it to the sum of `input_tokens` and `output_tokens`, rounded up to a multiple of `block_size` according to actual requirements.

## Additional Performance Tuning Parameters for the FusedSDPA Kernel with Padding-Aware Bucketing

FusedSDPA can be split into smaller chunks to improve performance while using the padding-aware bucketing strategy which guarantees the max absolute padding in the sequence and context dimensions.

| Parameter name                           | Description                                                                                  | Default value                               |
| ---------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `VLLM_HPU_FSDPA_SLICE_ENABLED`           | Enable the slicing.                                                                          | `True` when using padding-aware bucketing strategy with bucketing enabled, merged prefill disabled, and FusedSDPA kernel available |
| `VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD`      | KV length threshold above which slicing is applied.                                          | `min(max_num_batched_tokens, 8192)`         |
| `VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE`        | Chunk size for `q_len` and `kv_len` in each chunk. Rounded up to the next multiple of 1024.  | `VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD // 2`    |
| `VLLM_HPU_FSDPA_SLICE_WITH_GRAPH_BREAKS` | Places each chunk in a separate graph to reduce compilation time.                            | `true` for lazy mode and `false` otherwise  |

!!! note
    These parameters require generated padding-aware prompt buckets, selected by `VLLM_PROMPT_BUCKETING_STRATEGY="pad"` or inherited from `VLLM_BUCKETING_STRATEGY="pad"`. Decode overrides do not enable prompt slicing. File-based buckets and the deprecated exponential-bucketing flag disable slicing because they do not establish the required padding bounds.

The slicing is only activated if all the following additional conditions are satisfied:
- The batch size should be 1.
- The query length and KV length should be different, i.e. the normal causal prefill will route to the default dispatch for better performance.
- It's a causal attention model.
- The padding side is 'right'.
- No sliding window nor sinks (BF16 only; FP8 does not support sinks).

## Query Tiling for Large Prompt Attention Biases

FusedSDPA addresses the attention bias with a 32-bit signed byte offset, so a dense prompt bias of
2 GiB or more (`batch_size * query_len * (context_blocks * block_size + query_len) * itemsize >=
2**31`) makes the offset wrap and the kernel silently returns `NaN` instead of raising. Long
contexts combined with a wide context bucket can reach this, for example a
`[1, 1, 8192, 131072]` bf16 bias, which is exactly `2**31` bytes.

Query tiling splits the query dimension of prompt attention into the smallest number of tiles that
keeps every per-call bias below the limit. Attention rows are independent, so each tile attends the
full K/V and the results simply concatenate; the output is unchanged apart from the tiling itself.

| Parameter name                    | Description                                                                | Default value |
| --------------------------------- | -------------------------------------------------------------------------- | ------------- |
| `VLLM_HPU_FSDPA_Q_TILE_ENABLE`    | Enable query tiling when the prompt attention bias would reach `2**31` bytes. | `false`     |

!!! note
    This is independent of `VLLM_HPU_FSDPA_SLICE_ENABLED` and works with any bucketing strategy.
    When enabled, shapes whose bias already fits below the limit take the untiled path unchanged.

## DeepSeek V4 indexed MXFP4 MME candidate

`VLLM_HPU_DSV4_MXFP4_INDEXED_MME=1` selects the experimental BF16
indexed MoE implementation for the existing Gaudi2 DeepSeek V4 TP2,
single-token, top-6 shape. It defaults to `0`. Unsupported shapes continue
through the existing path. If the old indexed TPC flag is also set, this
MME candidate takes precedence for matching inputs.

Immutable scales are checked once at weight load. Ordinary E8M0 codes
`2..254` select a shorter exact BF16 decoder; other codes retain the
complete decoder. Both implementations preserve the checkpoint values.

The native compound op addresses packed weights using runtime expert IDs
and emits two logical batched MME operations. It preserves BF16 activation
boundaries and never converts weights through FP8. Decoded weights are
internal to the Synapse recipe; this does **not** guarantee SRAM placement.
Keep this flag disabled for production until compiled placement, correctness
and end-to-end performance have all been qualified. See
[the implementation and qualification contract](../features/deepseek_v4_indexed_mme.md).

### V4.1 fused SWA cache writes

`VLLM_HPU_DSV41_SWA_PACK_WRITE` (default `0`) enables experimental C1 group-32 checkpoint byte encoding and in-place SWA row writes. It requires bounded packed attention, which consumes the explicit write-completion tensor. Other token shapes retain their existing path. The cache encoding, row positions and state ownership are preserved; this does not enable FP8 matrix arithmetic.

`VLLM_HPU_DSV41_FP4_CACHE_WRITE` (default `0`) extends the ordered C1 path to native main-KV and index-K encoding and row writes. It requires `VLLM_HPU_DSV41_SWA_PACK_WRITE=1`. Main rows retain group-16 E4M3FN scales and index rows retain group-32 UE8M0 scales. Incomplete compression groups keep their scratch-row writes. Attention explicitly waits for both cache completions; other token shapes retain the existing path.

- `VLLM_HPU_DSV41_NATIVE_ROPE` (default `0`): C1 BF16 pairwise RoPE through the V4 TPC helper, with prepared FP32 cos/sin tables. Context512, SWA128, RoPE64. Other shapes keep existing execution.
- `VLLM_HPU_DSV41_C1_INDICES` (default `0`): fuse C1 SWA/window, compressed-slot mask/offset and visible-length preparation. Requires bounded packed attention; preserves Full/Reindex/Reuse state publication.

- `VLLM_HPU_DSV41_SELECTED_VALID_ONLY` (default `0`): omit BF16 stores for invalid selected slots inside ordered packed attention. The attention consumer rejects their remapped `-1` IDs before loading; the public selected-KV gather retains zero-filled invalid rows. Requires SWA pack/write.
