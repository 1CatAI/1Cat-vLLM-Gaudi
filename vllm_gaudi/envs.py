# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH: bool = True
    VLLM_HPU_FORCE_CHANNEL_FP8: bool = True
    VLLM_HPU_FP8_JIT_DYNAMIC_QUANT: bool = False
    VLLM_HPU_HETERO_KV_LAYOUT: bool = False
    VLLM_HPU_MULTI_MODEL_CONFIG: Optional[str] = None
    VLLM_NIXL_ABORT_REQUEST_TIMEOUT: float = 300.0
    VLLM_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
    VLLM_NIXL_SIDE_CHANNEL_PORT: int = 5600
    VLLM_HPU_NIXL_JOINT_KV: bool = False
    VLLM_HPU_NIXL_STAGING_SLOTS: int = 0
    VLLM_MINIMAX_M3_MOE_TOKEN_TILE: int = 512
    VLLM_MINIMAX_M3_MOE_DECODE_GATHER: bool = True
    VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS: int = 16
    VLLM_HPU_MXFP4_DECODE_GATHER: bool = True
    VLLM_HPU_DSV4_TPC_MXFP4_GATHER: bool = False
    VLLM_HPU_DSV4_TPC_MXFP4_INDEXED: bool = False
    VLLM_HPU_DSV4_MXFP4_INDEXED_MME: bool = False
    VLLM_HPU_DSV4_MXFP4_PREPARED_MME: bool = False
    VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH: bool = False
    VLLM_HPU_DSV41_PREPARED_SHARDS: bool = False
    VLLM_HPU_DSV41_ENGRAM_HOST_TABLE: bool = False
    VLLM_HPU_DSV41_GRAPH_REPLAY: bool = False
    VLLM_HPU_DSV41_DSPARK: bool = False
    VLLM_HPU_DSV41_VISION: bool = False
    VLLM_HPU_DSV41_QUANT_ROUNDTRIP: bool = False
    VLLM_HPU_DSV41_SWA_PACK_WRITE: bool = False
    VLLM_HPU_DSV41_FP4_CACHE_WRITE: bool = False
    VLLM_HPU_DSV41_NATIVE_ROPE: bool = False
    VLLM_HPU_DSV41_C1_INDICES: bool = False
    VLLM_HPU_DSV41_SELECTED_VALID_ONLY: bool = False
    VLLM_HPU_DSV41_SELECTED_KV_VECTOR: bool = False
    VLLM_HPU_DSV41_PACKED_ATTENTION: bool = False
    VLLM_HPU_DSV41_FIXED_POSITIONS: bool = False
    VLLM_HPU_DSV41_PACKED_PP: bool = False
    VLLM_HPU_DSV41_TPC_MHC: bool = False
    VLLM_HPU_DSV41_DIRECT_TOKEN_IDS: bool = False
    VLLM_HPU_DSV41_DEVICE_COMMIT: bool = False
    VLLM_HPU_DSV41_NATIVE_PP_COPY: bool = False
    VLLM_HPU_DSV41_PREPARED_OUTPUT: bool = False
    VLLM_HPU_DSV41_OUTPUT_GEMM_LAYOUT: bool = False
    VLLM_HPU_DSV41_BOUNDED_ATTENTION: bool = False
    VLLM_HPU_DSV41_ISOLATE_CONTROL: bool = False
    VLLM_HPU_DSV41_ENGINE_CPUS: Optional[str] = None
    VLLM_HPU_DSV41_API_CPUS: Optional[str] = None
    VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR: Optional[str] = None
    VLLM_HPU_DSV4_SHORT_INDEXER_SKIP: bool = True
    VLLM_HPU_DSV4_BF16_SCORE_PROJECTION: bool = False
    VLLM_HPU_DSV4_FUSED_SDPA: bool = False
    VLLM_HPU_DSV4_TPC_DEQUANT_GATHER: bool = False
    VLLM_HPU_DSV4_TPC_SPARSE_ATTN: bool = False
    VLLM_HPU_DSV4_TPC_SPARSE_ATTN_MAX_WIDTH: int = 128
    VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN: bool = False
    VLLM_HPU_DSV4_TPC_PAIR_HEADS: bool = False
    VLLM_HPU_DSV4_FLASHMLA_SPLIT_KV: bool = False
    VLLM_HPU_DSV4_FLASHMLA_TILED: bool = False
    VLLM_HPU_DSV4_FLASHMLA_SPLITS: int = 2
    VLLM_HPU_DSV4_FLASHMLA_PREFILL: bool = False
    VLLM_HPU_DSV4_FLASHMLA_PREFILL_MAX_WIDTH: int = 640
    VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN: bool = False
    VLLM_HPU_DSV4_ATTENTION_BACKEND: str = "auto"
    VLLM_HPU_DSV4_DIRECT_DECODE_DISPATCH: bool = False
    VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP: bool = False
    VLLM_HPU_DSV4_TPC_SAVE_PARTIAL_STATES: bool = False
    VLLM_HPU_DSV4_TPC_SAVE_PARTIAL_STATES_MAX_WIDTH: int = 1024
    VLLM_HPU_DSV4_TPC_SAVE_COMPRESS_NORM_C4: bool = False
    VLLM_HPU_DSV4_TPC_ORDERED_COMPRESSOR: bool = False
    VLLM_HPU_DSV4_TPC_ORDERED_C128_COMPRESSOR: bool = False
    VLLM_HPU_DSV4_FUSED_COMPRESSOR_FLASHMLA: bool = False
    VLLM_HPU_DSV4_FUSED_QNORM_COMPRESSOR: bool = False
    VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS: bool = False
    VLLM_HPU_DSV4_TPC_MIXED_COMPRESS_INPUTS: bool = False
    VLLM_HPU_DSV4_TPC_QNORM_ROPE_KV_PACK: bool = False
    VLLM_HPU_DSV4_INLINE_ATTENTION: bool = False
    VLLM_HPU_DSV4_INLINE_ATTN_FRONTEND: bool = False
    VLLM_HPU_DSV4_COMPILE_CHUNK_SIZE: int = 0
    VLLM_HPU_DSV4_EARLY_OUTPUT_LOWERING: bool = False
    VLLM_HPU_DSV4_BF16_ATTN_WEIGHT_CACHE: bool = False
    VLLM_HPU_DSV4_COMPILED_ATTN_FRONTEND: bool = False
    VLLM_HPU_DSV4_NATIVE_FP8_ATTN_FRONTEND: bool = False
    VLLM_HPU_DSV4_TPC_MHC: bool = False
    VLLM_HPU_DSV4_TPC_SINKHORN: bool = False
    VLLM_HPU_DSV4_PACKED_DECODE_METADATA: bool = False
    VLLM_HPU_DSV4_DECODE_METADATA_RING_SIZE: int = 2
    VLLM_HPU_DSV4_Q1_METADATA_FASTPATH: bool = False
    VLLM_HPU_DSV4_TPC_OP_LIBRARY: Optional[str] = None
    VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY: bool = False
    VLLM_HPU_TRITON_MODE: str = "off"
    VLLM_HPU_TRITON_CACHE_DIR: Optional[str] = None
    VLLM_HPU_TRITON_BLOCK_SIZE: int = 256
    VLLM_HPU_TRITON_SILU_BLOCK_SIZE: int = 128
    VLLM_HPU_TRITON_GDN_VALUE_TILE: int = 16
    VLLM_HPU_NATIVE_DECODE_GRAPH: bool = False

    VLLM_GDN_CHUNK_SIZE: int = 0
    VLLM_GDN_NEUMANN_ITERS: int = 14
    VLLM_GDN_FUSED_STATE_MATMUL: bool = False
    VLLM_GDN_DEFERRED_OUTPUT_ADD: bool = False
    VLLM_GDN_RECURSIVE_SOLVER_BASE: int = 0
    VLLM_GDN_COMPACT_REPEATED_KKT: bool = False
    VLLM_GDN_COMPACT_REPEATED_LOCAL_ATTN: bool = False
    VLLM_GDN_COMPILED_QK_L2NORM: bool = False
    VLLM_GDN_FUSED_RMSNORM_GATED: bool = False
    VLLM_GDN_FLASHQLA: bool = False
    VLLM_GDN_FLASHQLA_FACTORIZE_PHASE_B: bool = False
    VLLM_GDN_FLASHQLA_FP32_SCALING: bool = False
    VLLM_GDN_BF16_BMM_F32: bool = False
    VLLM_GDN_SOLVE_BF16_BMM_F32: bool = False
    VLLM_GDN_BF16_BMM_F32_EXTENSION: str = ""
    VLLM_GDN_QWEN38_NATIVE_QK_PREP: bool = False
    VLLM_GDN_QWEN38_BF16_QK: bool = False
    VLLM_GDN_QWEN38_COMPACT_QK: bool = False
    VLLM_GDN_QWEN38_NATIVE_COMPACT_KKT: bool = False
    VLLM_GDN_COMPACT_QK_FACTOR_GATE: bool = False
    VLLM_GDN_NATIVE_RECURRENT_SCAN: bool = False
    VLLM_GDN_QWEN38_NATIVE_QK_PREP_EXTENSION: str = ""
    VLLM_GDN_HPU_CAUSAL_CONV1D: bool = False
    VLLM_GDN_TOKEN_MAJOR_CAUSAL_CONV1D: bool = False
    VLLM_HPU_DYNAMIC_QUANT_CGUID: bool = False
    VLLM_HPU_DYNAMIC_QUANT_CGUID_MIN_TOKENS: int = 2048
    VLLM_HPU_EXPLICIT_SIGMOID_SILU: bool = False
    VLLM_HPU_EXPLICIT_SIGMOID_SILU_MIN_TOKENS: int = 2048
    VLLM_HPU_FLASHINFER_GDN: bool = False
    VLLM_HPU_FLASHINFER_DFLASH2: bool = False
    VLLM_HPU_DFLASH2_CONV_ROUND_BEFORE_ACTIVATION: bool = False
    VLLM_HPU_DFLASH2_FULL_QUERY_CONV: bool = False
    VLLM_HPU_DFLASH2_DIRECT_CHECKPOINTS: bool = False
    VLLM_HPU_DFLASH2_DEVICE_PREPARE: bool = False
    VLLM_HPU_FLASHINFER_GDN_TP2: bool = False
    VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE: bool = False
    VLLM_HPU_FLASHINFER_GDN_PREFILL: bool = False
    VLLM_HPU_GDN_DIRECT_STATE: bool = True
    VLLM_HPU_GDN_ACTIVE_STATE_VIEWS: bool = False
    VLLM_HPU_GDN_ASYNC_STATE_DMA: bool = False
    VLLM_HPU_GDN_DIRECT_STATE_UPDATE: bool = False
    VLLM_HPU_GDN_PRECISE_DMA_EVENTS: bool = False
    VLLM_HPU_TP2_PREPARED_COMM: bool = False
    VLLM_HPU_TP2_STATIC_GROUP_PLAN: bool = False
    VLLM_HPU_TP2_PLAN_DUMP_DIR: str | None = None
    VLLM_HPU_TP2_NATIVE_JOINT_PLAN: bool = False
    VLLM_HPU_TP2_GQA_COMPACT_KV: bool = False
    VLLM_HPU_TP2_COMPILED_CONSUMER_NORM: bool = False
    VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT: bool = False
    VLLM_HPU_GDN_PADDED_DIRECT_STATE: bool = False
    VLLM_HPU_CGUID_DYNAMIC_QUANT: bool = False
    VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS: int = 32
    VLLM_HPU_FUSED_GREEDY_LOGITS: bool = False

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition
environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_HPU_DSV4_EARLY_OUTPUT_LOWERING":
    lambda: os.environ.get("VLLM_HPU_DSV4_EARLY_OUTPUT_LOWERING", "0").lower() in ("1", "true"),
    "VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT":
    lambda: os.environ.get("VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT", "0") == "1",
    # Require the version-locked native decoder compute/collective replay path.
    "VLLM_HPU_NATIVE_DECODE_GRAPH":
    lambda: os.environ.get("VLLM_HPU_NATIVE_DECODE_GRAPH", "false").strip().lower() in ("1", "true"),

    # Contiguous cache fetching to avoid using costly gather operation on
    # Gaudi3. This is only applicable to HPU contiguous cache. If set to true,
    # contiguous cache fetch will be used.
    "VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH":
    lambda: os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() in ("1", "true"),

    # Convert block fp8 to channel fp8 for HPU
    # If `QUANT_CONFIG` is set, this will be forced to false.
    "VLLM_HPU_FORCE_CHANNEL_FP8":
    lambda: os.environ.get("VLLM_HPU_FORCE_CHANNEL_FP8", "true").lower() in
    ("1", "true") and os.environ.get("QUANT_CONFIG", None) is None,

    # Fuse per-row amax, scale calculation, and FP8 cast on Gaudi2.
    "VLLM_HPU_FP8_JIT_DYNAMIC_QUANT":
    lambda: os.environ.get("VLLM_HPU_FP8_JIT_DYNAMIC_QUANT", "false").lower() in ("1", "true"),

    # Enable prefill side kv_layout and block_size for heterogeneous run.
    "VLLM_HPU_HETERO_KV_LAYOUT":
    lambda: os.environ.get("VLLM_HPU_HETERO_KV_LAYOUT", "false").lower() in ("0", "false"),

    # Path to a YAML config file describing the multi-model setup
    # (model names, weights, tensor-parallel config, etc.).
    # When unset, multi-model mode is disabled.
    "VLLM_HPU_MULTI_MODEL_CONFIG":
    lambda: os.environ.get("VLLM_HPU_MULTI_MODEL_CONFIG", None),

    # Timeout in seconds for NIXL abort request handling
    "VLLM_NIXL_ABORT_REQUEST_TIMEOUT":
    lambda: float(os.environ.get("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", "300")),

    # NIXL side channel host and port for KV transfer
    "VLLM_NIXL_SIDE_CHANNEL_HOST":
    lambda: os.environ.get("VLLM_NIXL_SIDE_CHANNEL_HOST", "localhost"),
    "VLLM_NIXL_SIDE_CHANNEL_PORT":
    lambda: int(os.environ.get("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600")),

    # Enable joint-KV staging so an HPU prefill instance can serve a GPU
    # (FLASH_ATTN, blocks-first) decode instance over NIXL. When the remote
    # decode registers ONE joint [K|V] region per layer (block_len covering
    # both K and V, V at block_len // 2), the HPU must advertise the same
    # region model. Leave false for HPU<->HPU disagg (separate K/V regions).
    "VLLM_HPU_NIXL_JOINT_KV":
    lambda: os.environ.get("VLLM_HPU_NIXL_JOINT_KV", "false").lower() in ("1", "true"),

    # Number of joint-KV staging slots for heterogeneous (HPU prefill ->
    # GPU decode) NIXL transfer. Each slot holds one on-save block for all
    # layers laid out blocks-first [2, n_kv_heads, block_size_on_save,
    # head_size] so the GPU can read K and V from one contiguous region
    # (V at block_len // 2). 0 (default) auto-sizes to the full concurrent
    # workload: max_num_seqs * ceil(max_model_len / block_size_on_save), which
    # guarantees every schedulable request gets a reservation. Registration
    # fails fast if that pool does not fit device memory. Only set a smaller
    # value if you are certain peak concurrent transfer demand stays under it.
    "VLLM_HPU_NIXL_STAGING_SLOTS":
    lambda: int(os.environ.get("VLLM_HPU_NIXL_STAGING_SLOTS", "0")),

    # MiniMax-M3 SwiGLU-OAI expert execution tuning.
    "VLLM_MINIMAX_M3_MOE_TOKEN_TILE":
    lambda: int(os.environ.get("VLLM_MINIMAX_M3_MOE_TOKEN_TILE", "512")),
    "VLLM_MINIMAX_M3_MOE_DECODE_GATHER":
    lambda: os.environ.get("VLLM_MINIMAX_M3_MOE_DECODE_GATHER", "1").lower() in ("1", "true"),
    "VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS":
    lambda: int(os.environ.get("VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS", "16")),

    # Gather only routed packed experts for single-token MXFP4 decode.
    "VLLM_HPU_MXFP4_DECODE_GATHER":
    lambda: os.environ.get("VLLM_HPU_MXFP4_DECODE_GATHER", "1").lower() in ("1", "true"),

    # V4.1 is a separate opt-in contract; none of the V4 defaults enable it.
    "VLLM_HPU_DSV41_PREPARED_SHARDS":
    lambda: os.environ.get("VLLM_HPU_DSV41_PREPARED_SHARDS", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_ENGRAM_HOST_TABLE":
    lambda: os.environ.get("VLLM_HPU_DSV41_ENGRAM_HOST_TABLE", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_GRAPH_REPLAY":
    lambda: os.environ.get("VLLM_HPU_DSV41_GRAPH_REPLAY", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_DSPARK":
    lambda: os.environ.get("VLLM_HPU_DSV41_DSPARK", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_VISION":
    lambda: os.environ.get("VLLM_HPU_DSV41_VISION", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_QUANT_ROUNDTRIP":
    lambda: os.environ.get("VLLM_HPU_DSV41_QUANT_ROUNDTRIP", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_SWA_PACK_WRITE":
    lambda: os.environ.get("VLLM_HPU_DSV41_SWA_PACK_WRITE", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_FP4_CACHE_WRITE":
    lambda: os.environ.get("VLLM_HPU_DSV41_FP4_CACHE_WRITE", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_NATIVE_ROPE":
    lambda: os.environ.get("VLLM_HPU_DSV41_NATIVE_ROPE", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_C1_INDICES":
    lambda: os.environ.get("VLLM_HPU_DSV41_C1_INDICES", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_SELECTED_KV_VECTOR":
    lambda: os.environ.get("VLLM_HPU_DSV41_SELECTED_KV_VECTOR", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_SELECTED_VALID_ONLY":
    lambda: os.environ.get("VLLM_HPU_DSV41_SELECTED_VALID_ONLY", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_PACKED_ATTENTION":
    lambda: os.environ.get("VLLM_HPU_DSV41_PACKED_ATTENTION", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_FIXED_POSITIONS":
    lambda: os.environ.get("VLLM_HPU_DSV41_FIXED_POSITIONS", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_PACKED_PP":
    lambda: os.environ.get("VLLM_HPU_DSV41_PACKED_PP", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_TPC_MHC":
    lambda: os.environ.get("VLLM_HPU_DSV41_TPC_MHC", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_ENGRAM_NATIVE_C1":
    lambda: os.environ.get("VLLM_HPU_DSV41_ENGRAM_NATIVE_C1", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_TP_MHC_OVERLAP":
    lambda: os.environ.get("VLLM_HPU_DSV41_TP_MHC_OVERLAP", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_NATIVE_INPUT_PREFLIGHT":
    lambda: os.environ.get("VLLM_HPU_DSV41_NATIVE_INPUT_PREFLIGHT", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_ATTENTION_PAIRED_EXP":
    lambda: os.environ.get("VLLM_HPU_DSV41_ATTENTION_PAIRED_EXP", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_DIRECT_TOKEN_IDS":
    lambda: os.environ.get("VLLM_HPU_DSV41_DIRECT_TOKEN_IDS", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_DEVICE_COMMIT":
    lambda: os.environ.get("VLLM_HPU_DSV41_DEVICE_COMMIT", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_NATIVE_PP_COPY":
    lambda: os.environ.get("VLLM_HPU_DSV41_NATIVE_PP_COPY", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_PREPARED_OUTPUT":
    lambda: os.environ.get("VLLM_HPU_DSV41_PREPARED_OUTPUT", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_EXPERT_K128": lambda: os.environ.get("VLLM_HPU_DSV41_EXPERT_K128", "0") == "1",
    "VLLM_HPU_DSV41_OUTPUT_GEMM_LAYOUT":
    lambda: os.environ.get("VLLM_HPU_DSV41_OUTPUT_GEMM_LAYOUT", "0") == "1",
    "VLLM_HPU_DSV41_FP8_DECODE": lambda: os.environ.get("VLLM_HPU_DSV41_FP8_DECODE", "0") == "1",
    "VLLM_HPU_DSV41_FP8_SIDECAR": lambda: os.environ.get("VLLM_HPU_DSV41_FP8_SIDECAR", ""),
    "VLLM_HPU_DSV41_FP8_CONFIG": lambda: os.environ.get("VLLM_HPU_DSV41_FP8_CONFIG", ""),
    "VLLM_HPU_DSV41_BOUNDED_ATTENTION":
    lambda: os.environ.get("VLLM_HPU_DSV41_BOUNDED_ATTENTION", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_ISOLATE_CONTROL":
    lambda: os.environ.get("VLLM_HPU_DSV41_ISOLATE_CONTROL", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV41_ENGINE_CPUS":
    lambda: os.environ.get("VLLM_HPU_DSV41_ENGINE_CPUS"),
    "VLLM_HPU_DSV41_API_CPUS":
    lambda: os.environ.get("VLLM_HPU_DSV41_API_CPUS"),
    "VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR":
    lambda: os.environ.get("VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"),

    # Fuse the four selected-expert packed-weight copies into one Gaudi2 TPC
    # launch for DeepSeek V4 single-token decode.
    "VLLM_HPU_DSV4_TPC_MXFP4_GATHER":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_MXFP4_GATHER", "0").lower() in ("1", "true"),

    # Run the DeepSeek V4 single-token MXFP4 experts directly from stacked
    # weights using runtime expert IDs, avoiding selected-weight copies.
    "VLLM_HPU_DSV4_TPC_MXFP4_INDEXED":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_MXFP4_INDEXED", "0").lower() in ("1", "true"),

    # The indexed MME candidate keeps BF16 and uses graph-internal decoded
    # weights. SRAM placement and E2E quality are not yet qualified.
    "VLLM_HPU_DSV4_MXFP4_INDEXED_MME":
    lambda: os.environ.get(
        "VLLM_HPU_DSV4_MXFP4_INDEXED_MME", "0"
    ).lower() in ("1", "true"),

    # Load-time Q16/S16 layout for the exact BF16 Gaudi2 TP2 decoder. This
    # remains opt-in until full-model quality and end-to-end gates pass.
    "VLLM_HPU_DSV4_MXFP4_PREPARED_MME":
    lambda: os.environ.get(
        "VLLM_HPU_DSV4_MXFP4_PREPARED_MME", "0"
    ).lower() in ("1", "true"),

    # Experimental V4 adapter for the version-locked joint compute/NIC plan.
    "VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH":
    lambda: os.environ.get("VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH", "0").lower() in ("1", "true"),

    # Skip DeepSeek V4 indexer scoring when every compressed candidate is
    # guaranteed to fit in top-k.
    "VLLM_HPU_DSV4_SHORT_INDEXER_SKIP":
    lambda: os.environ.get("VLLM_HPU_DSV4_SHORT_INDEXER_SKIP", "1").lower() in ("1", "true"),

    # Keep DeepSeek V4 compressor/indexer score projections on the BF16 MME
    # path and cast only their small outputs to FP32.
    "VLLM_HPU_DSV4_BF16_SCORE_PROJECTION":
    lambda: os.environ.get("VLLM_HPU_DSV4_BF16_SCORE_PROJECTION", "0").lower() in ("1", "true"),

    # Use Gaudi FusedSDPA for the gathered DeepSeek V4 sparse-attention core.
    "VLLM_HPU_DSV4_FUSED_SDPA":
    lambda: os.environ.get("VLLM_HPU_DSV4_FUSED_SDPA", "0").lower() in ("1", "true"),

    # Use the DeepSeek V4 Gaudi2 TPC prototypes for packed-cache dequant
    # gather and sparse attention. The PyTorch registration library is kept
    # external while the kernels are under performance validation.
    "VLLM_HPU_DSV4_TPC_DEQUANT_GATHER":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_DEQUANT_GATHER", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_SPARSE_ATTN":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_SPARSE_ATTN", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_SPARSE_ATTN_MAX_WIDTH":
    lambda: int(os.environ.get("VLLM_HPU_DSV4_TPC_SPARSE_ATTN_MAX_WIDTH", "128")),
    "VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_PAIR_HEADS":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_PAIR_HEADS", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_FLASHMLA_SPLIT_KV":
    lambda: os.environ.get("VLLM_HPU_DSV4_FLASHMLA_SPLIT_KV", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_FLASHMLA_TILED":
    lambda: os.environ.get("VLLM_HPU_DSV4_FLASHMLA_TILED", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_FLASHMLA_SPLITS":
    lambda: int(os.environ.get("VLLM_HPU_DSV4_FLASHMLA_SPLITS", "2")),
    "VLLM_HPU_DSV4_FLASHMLA_PREFILL":
    lambda: os.environ.get("VLLM_HPU_DSV4_FLASHMLA_PREFILL", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_FLASHMLA_PREFILL_MAX_WIDTH":
    lambda: int(os.environ.get("VLLM_HPU_DSV4_FLASHMLA_PREFILL_MAX_WIDTH", "640")),
    "VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN":
    lambda: os.environ.get("VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_ATTENTION_BACKEND":
    lambda: os.environ.get("VLLM_HPU_DSV4_ATTENTION_BACKEND", "auto"),
    "VLLM_HPU_DSV4_DIRECT_DECODE_DISPATCH":
    lambda: os.environ.get("VLLM_HPU_DSV4_DIRECT_DECODE_DISPATCH", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP":
    lambda: os.environ.get("VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_SAVE_PARTIAL_STATES":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_SAVE_PARTIAL_STATES", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_SAVE_PARTIAL_STATES_MAX_WIDTH":
    lambda: int(os.environ.get("VLLM_HPU_DSV4_TPC_SAVE_PARTIAL_STATES_MAX_WIDTH", "1024")),
    "VLLM_HPU_DSV4_TPC_SAVE_COMPRESS_NORM_C4":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_SAVE_COMPRESS_NORM_C4", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_ORDERED_COMPRESSOR":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_ORDERED_COMPRESSOR", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_ORDERED_C128_COMPRESSOR":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_ORDERED_C128_COMPRESSOR", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_FUSED_COMPRESSOR_FLASHMLA":
    lambda: os.environ.get("VLLM_HPU_DSV4_FUSED_COMPRESSOR_FLASHMLA", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_FUSED_QNORM_COMPRESSOR":
    lambda: os.environ.get("VLLM_HPU_DSV4_FUSED_QNORM_COMPRESSOR", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_MIXED_COMPRESS_INPUTS":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_MIXED_COMPRESS_INPUTS", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_QNORM_ROPE_KV_PACK":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_QNORM_ROPE_KV_PACK", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_INLINE_ATTENTION":
    lambda: os.environ.get("VLLM_HPU_DSV4_INLINE_ATTENTION", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_INLINE_ATTN_FRONTEND":
    lambda: os.environ.get("VLLM_HPU_DSV4_INLINE_ATTN_FRONTEND", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_COMPILE_CHUNK_SIZE":
    lambda: int(os.environ.get("VLLM_HPU_DSV4_COMPILE_CHUNK_SIZE", "0")),
    "VLLM_HPU_DSV4_BF16_ATTN_WEIGHT_CACHE":
    lambda: os.environ.get("VLLM_HPU_DSV4_BF16_ATTN_WEIGHT_CACHE", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_COMPILED_ATTN_FRONTEND":
    lambda: os.environ.get("VLLM_HPU_DSV4_COMPILED_ATTN_FRONTEND", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_NATIVE_FP8_ATTN_FRONTEND":
    lambda: os.environ.get("VLLM_HPU_DSV4_NATIVE_FP8_ATTN_FRONTEND", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_MHC":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_MHC", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_SINKHORN":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_SINKHORN", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_PACKED_DECODE_METADATA":
    lambda: os.environ.get("VLLM_HPU_DSV4_PACKED_DECODE_METADATA", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_DECODE_METADATA_RING_SIZE":
    lambda: int(os.environ.get("VLLM_HPU_DSV4_DECODE_METADATA_RING_SIZE", "2")),
    "VLLM_HPU_DSV4_Q1_METADATA_FASTPATH":
    lambda: os.environ.get("VLLM_HPU_DSV4_Q1_METADATA_FASTPATH", "0").lower() in ("1", "true"),
    "VLLM_HPU_DSV4_TPC_OP_LIBRARY":
    lambda: os.environ.get("VLLM_HPU_DSV4_TPC_OP_LIBRARY", None),

    # Run multimodal warmup outside PT_COMPILE_ONLY_MODE for models with
    # data-dependent output shapes that must be materialized during warmup.
    "VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY":
    lambda: os.environ.get("VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY", "false").strip().lower() in ("1", "true"),

    # Gaudi2-native Triton backend policy. `off` preserves the vendor path,
    # `hybrid` permits a recorded vendor fallback, and `strict` fails closed.
    "VLLM_HPU_TRITON_MODE":
    lambda: os.environ.get("VLLM_HPU_TRITON_MODE", "off").strip().lower(),
    "VLLM_HPU_TRITON_CACHE_DIR":
    lambda: os.environ.get("VLLM_HPU_TRITON_CACHE_DIR", None),
    "VLLM_HPU_TRITON_BLOCK_SIZE":
    lambda: int(os.environ.get("VLLM_HPU_TRITON_BLOCK_SIZE", "256")),
    "VLLM_HPU_TRITON_SILU_BLOCK_SIZE":
    lambda: int(os.environ.get("VLLM_HPU_TRITON_SILU_BLOCK_SIZE", "128")),
    "VLLM_HPU_TRITON_GDN_VALUE_TILE":
    lambda: int(os.environ.get("VLLM_HPU_TRITON_GDN_VALUE_TILE", "16")),

    # Enable the in-tree FlashInfer-compatible GDN decode adapter. Auto mode
    # uses only offline-promoted native tactics and otherwise selects the
    # compile-friendly reference implementation.
    "VLLM_HPU_FLASHINFER_GDN":
    lambda: os.environ.get("VLLM_HPU_FLASHINFER_GDN", "false").strip().lower() in ("1", "true"),

    # Enable the HPU DFlash2 V1 proposer and its FlashInfer-Gaudi operator
    # adapters. This follows the parent GDN switch because rollback-safe GDN
    # state handling is part of the DFlash2 correctness contract.
    "VLLM_HPU_FLASHINFER_DFLASH2":
    lambda: os.environ.get(
        "VLLM_HPU_FLASHINFER_DFLASH2",
        os.environ.get("VLLM_HPU_FLASHINFER_GDN", "false"),
    ).strip().lower() in ("1", "true"),

    # Experimental alignment with the ordinary HPU convolution/SiLU dtype
    # boundary. Keep off until independent DFlash2 quality qualification.
    "VLLM_HPU_DFLASH2_CONV_ROUND_BEFORE_ACTIVATION":
    lambda: os.environ.get("VLLM_HPU_DFLASH2_CONV_ROUND_BEFORE_ACTIVATION", "false").strip().lower() in ("1", "true"),

    # Skip speculative-convolution padding work only for CPU-proven full
    # verification blocks with compact, request-owned state rows.
    "VLLM_HPU_DFLASH2_FULL_QUERY_CONV":
    lambda: os.environ.get("VLLM_HPU_DFLASH2_FULL_QUERY_CONV", "false").strip().lower() in ("1", "true"),

    # Specialize native checkpoint writes only after CPU ownership validation.
    "VLLM_HPU_DFLASH2_DIRECT_CHECKPOINTS":
    lambda: os.environ.get("VLLM_HPU_DFLASH2_DIRECT_CHECKPOINTS", "false").strip().lower() in ("1", "true"),

    # Keep accepted lengths and draft input preparation on HPU so the draft
    # can be queued before target samples are materialized on the host.
    "VLLM_HPU_DFLASH2_DEVICE_PREPARE":
    lambda: os.environ.get("VLLM_HPU_DFLASH2_DEVICE_PREPARE", "false").strip().lower() in ("1", "true"),

    # Enable the Qwen3.8 fused direct-state recipe for supported local TP heads. By
    # default this follows the parent FlashInfer-Gaudi GDN switch.
    "VLLM_HPU_FLASHINFER_GDN_TP2":
    lambda: os.environ.get("VLLM_HPU_FLASHINFER_GDN_TP2", "false").strip().lower() in ("1", "true"),
    "VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE":
    lambda: os.environ.get(
        "VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE",
        os.environ.get("VLLM_HPU_FLASHINFER_GDN", "false"),
    ).strip().lower() in ("1", "true"),

    # Enable the shape-gated FlashQLA graph tactic for GDN prefill. By
    # default this follows the parent FlashInfer-Gaudi GDN switch.
    "VLLM_HPU_FLASHINFER_GDN_PREFILL":
    lambda: os.environ.get(
        "VLLM_HPU_FLASHINFER_GDN_PREFILL",
        os.environ.get("VLLM_HPU_FLASHINFER_GDN", "false"),
    ).strip().lower() in ("1", "true"),

    # Use group-major compact GDN state views for full decode buckets.
    "VLLM_HPU_GDN_DIRECT_STATE":
    lambda: os.environ.get("VLLM_HPU_GDN_DIRECT_STATE", "true").strip().lower() in ("1", "true"),

    # Bind active recurrent-state spans before entering compiled decoder groups.
    "VLLM_HPU_GDN_ACTIVE_STATE_VIEWS":
    lambda: os.environ.get("VLLM_HPU_GDN_ACTIVE_STATE_VIEWS", "false").strip().lower() in ("1", "true"),
    "VLLM_HPU_GDN_ASYNC_STATE_DMA":
    lambda: os.environ.get("VLLM_HPU_GDN_ASYNC_STATE_DMA", "false").strip().lower() in ("1", "true"),
    "VLLM_HPU_GDN_DIRECT_STATE_UPDATE":
    lambda: os.environ.get("VLLM_HPU_GDN_DIRECT_STATE_UPDATE", "false").strip().lower() in ("1", "true"),
    "VLLM_HPU_GDN_PRECISE_DMA_EVENTS":
    lambda: os.environ.get("VLLM_HPU_GDN_PRECISE_DMA_EVENTS", "false").strip().lower() in ("1", "true"),
    "VLLM_HPU_TP2_PREPARED_COMM":
    lambda: os.environ.get("VLLM_HPU_TP2_PREPARED_COMM", "false").strip().lower() in ("1", "true"),
    "VLLM_HPU_TP2_STATIC_GROUP_PLAN":
    lambda: os.environ.get("VLLM_HPU_TP2_STATIC_GROUP_PLAN", "false").strip().lower() in ("1", "true"),
    "VLLM_HPU_TP2_PLAN_DUMP_DIR":
    lambda: os.environ.get("VLLM_HPU_TP2_PLAN_DUMP_DIR"),
    "VLLM_HPU_TP2_NATIVE_JOINT_PLAN":
    lambda: os.environ.get("VLLM_HPU_TP2_NATIVE_JOINT_PLAN", "0") == "1",
    "VLLM_HPU_TP2_GQA_COMPACT_KV":
    lambda: os.environ.get("VLLM_HPU_TP2_GQA_COMPACT_KV", "0") == "1",
    "VLLM_HPU_TP2_COMPILED_CONSUMER_NORM":
    lambda: os.environ.get("VLLM_HPU_TP2_COMPILED_CONSUMER_NORM", "0") == "1",

    # Opt in to free-slot padding of the compact direct-state decode path.
    "VLLM_HPU_GDN_PADDED_DIRECT_STATE":
    lambda: os.environ.get("VLLM_HPU_GDN_PADDED_DIRECT_STATE", "false").strip().lower() in ("1", "true"),

    # Use Gaudi's calculate_scale_for_cast CGUID for decode-sized per-token
    # dynamic FP8 scales instead of materializing abs + reduce_max + scale
    # arithmetic. Large prefill matrices retain the existing path because the
    # CGUID changes graph fusion and numerical results there.
    "VLLM_HPU_CGUID_DYNAMIC_QUANT":
    lambda: os.environ.get(
        "VLLM_HPU_CGUID_DYNAMIC_QUANT",
        os.environ.get("VLLM_HPU_FLASHINFER_GDN", "false"),
    ).strip().lower() in ("1", "true"),
    "VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS":
    lambda: int(os.environ.get("VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS", "32")),

    # Fuse hidden-state selection, the LM head, and argmax for plain greedy
    # decode requests. Sampling features that can change logits or require
    # logprobs retain the general sampler path.
    "VLLM_HPU_FUSED_GREEDY_LOGITS":
    lambda: os.environ.get("VLLM_HPU_FUSED_GREEDY_LOGITS", "false").strip().lower() in ("1", "true"),

    # Override the GDN prefill chunk size. Zero preserves the model-provided
    # value, or the HPU default when the model does not specify one.
    "VLLM_GDN_CHUNK_SIZE":
    lambda: int(os.environ.get("VLLM_GDN_CHUNK_SIZE", "0")),

    # Iteration budget for the approximate GDN triangular solve.
    "VLLM_GDN_NEUMANN_ITERS":
    lambda: int(os.environ.get("VLLM_GDN_NEUMANN_ITERS", "14")),

    # Fuse the output and recurrent-state projections in GDN phase B.
    "VLLM_GDN_FUSED_STATE_MATMUL":
    lambda: os.environ.get("VLLM_GDN_FUSED_STATE_MATMUL", "false").lower() in ("1", "true"),

    # Avoid per-chunk in-place writes to views of the phase-B output tensor.
    "VLLM_GDN_DEFERRED_OUTPUT_ADD":
    lambda: os.environ.get("VLLM_GDN_DEFERRED_OUTPUT_ADD", "false").lower() in ("1", "true"),
    # Use recursive block inversion for GDN unit lower-triangular matrices.
    # Zero disables the path; a positive power of two selects the base size.
    "VLLM_GDN_RECURSIVE_SOLVER_BASE":
    lambda: int(os.environ.get("VLLM_GDN_RECURSIVE_SOLVER_BASE", "0")),

    # Compute the GDN K @ K^T product only for unique key heads, then
    # broadcast it across repeated value-head groups.
    "VLLM_GDN_COMPACT_REPEATED_KKT":
    lambda: os.environ.get("VLLM_GDN_COMPACT_REPEATED_KKT", "false").lower() in ("1", "true"),

    # Compute phase-B Q @ K^T only for unique Q/K heads, then combine the
    # compact product with each repeated value head's causal decay.
    "VLLM_GDN_COMPACT_REPEATED_LOCAL_ATTN":
    lambda: os.environ.get("VLLM_GDN_COMPACT_REPEATED_LOCAL_ATTN", "false").lower() in ("1", "true"),

    # Keep GDN Q/K L2 normalization inside the compiled prefill graph.
    "VLLM_GDN_COMPILED_QK_L2NORM":
    lambda: os.environ.get("VLLM_GDN_COMPILED_QK_L2NORM", "false").lower() in ("1", "true"),

    # Use Habana FusedRMSNorm for the Qwen GDN output norm before applying
    # the output gate. This is currently limited to the prefill path.
    "VLLM_GDN_FUSED_RMSNORM_GATED":
    lambda: os.environ.get("VLLM_GDN_FUSED_RMSNORM_GATED", "false").lower() in ("1", "true"),

    # Use the FlashQLA algebraic reformulation for GDN prefill. This is a
    # Gaudi-native implementation and does not load FlashQLA's CUDA kernels.
    "VLLM_GDN_FLASHQLA":
    lambda: os.environ.get("VLLM_GDN_FLASHQLA", "false").lower() in ("1", "true"),

    # Keep the algebraically factorized local-decay matrix in FlashQLA phase
    # B. This research path is disabled because separate positive gate factors
    # can overflow even when their final causal-decay product is finite.
    "VLLM_GDN_FLASHQLA_FACTORIZE_PHASE_B":
    lambda: os.environ.get("VLLM_GDN_FLASHQLA_FACTORIZE_PHASE_B", "false").lower() in ("1", "true"),

    # Evaluate FlashQLA phase-A diagonal scale products in FP32 before
    # returning to the BF16 MME path.
    "VLLM_GDN_FLASHQLA_FP32_SCALING":
    lambda: os.environ.get("VLLM_GDN_FLASHQLA_FP32_SCALING", "false").lower() in ("1", "true"),

    # Use a research Synapse batch_gemm backend with BF16 inputs and an FP32
    # output for the GDN recurrent state projection.
    "VLLM_GDN_BF16_BMM_F32":
    lambda: os.environ.get("VLLM_GDN_BF16_BMM_F32", "false").lower() in ("1", "true"),

    # Use the mixed-output batch_gemm for recursive KKT block merges. The
    # base 16x16 inverse remains FP32 for numerical stability.
    "VLLM_GDN_SOLVE_BF16_BMM_F32":
    lambda: os.environ.get("VLLM_GDN_SOLVE_BF16_BMM_F32", "false").lower() in ("1", "true"),

    # PyTorch registration extension for the mixed-output batch_gemm backend.
    "VLLM_GDN_BF16_BMM_F32_EXTENSION":
    lambda: os.environ.get("VLLM_GDN_BF16_BMM_F32_EXTENSION", ""),

    # Use the fixed-shape Gaudi2 TPC kernel for Qwen3.8 TP1 prompt Q/K
    # normalization and grouped-head expansion.
    "VLLM_GDN_QWEN38_NATIVE_QK_PREP":
    lambda: os.environ.get("VLLM_GDN_QWEN38_NATIVE_QK_PREP", "false").lower() in ("1", "true"),

    # Write the Qwen3.8 native expanded Q/K output directly as BF16, matching
    # the dtype consumed by the optimized GDN graph.
    "VLLM_GDN_QWEN38_BF16_QK":
    lambda: os.environ.get("VLLM_GDN_QWEN38_BF16_QK", "false").lower() in ("1", "true"),

    # Keep Qwen3.8 prompt Q/K at their 16 physical heads. The GDN grouped
    # value-head path maps each of 48 value heads to its compact Q/K head.
    "VLLM_GDN_QWEN38_COMPACT_QK":
    lambda: os.environ.get("VLLM_GDN_QWEN38_COMPACT_QK", "false").lower() in ("1", "true"),

    # Fuse compact Qwen3.8 KKT broadcast, beta scaling, lower masking, and
    # identity insertion in one Gaudi2 TPC kernel.
    "VLLM_GDN_QWEN38_NATIVE_COMPACT_KKT":
    lambda: os.environ.get("VLLM_GDN_QWEN38_NATIVE_COMPACT_KKT", "false").lower() in ("1", "true"),

    # For compact Q/K, apply the per-value-head gate to the smaller KKT
    # coefficient matrix before multiplying by the shared physical K head.
    "VLLM_GDN_COMPACT_QK_FACTOR_GATE":
    lambda: os.environ.get("VLLM_GDN_COMPACT_QK_FACTOR_GATE", "false").lower() in ("1", "true"),

    # Lower the BF16 phase-B recurrent chain as one mixed MME/TPC Synapse
    # subgraph so the compiler can retain state and remove chunk-wise DMA.
    "VLLM_GDN_NATIVE_RECURRENT_SCAN":
    lambda: os.environ.get("VLLM_GDN_NATIVE_RECURRENT_SCAN", "false").lower() in ("1", "true"),

    # PyTorch registration extension for the Qwen3.8 native Q/K TPC kernel.
    "VLLM_GDN_QWEN38_NATIVE_QK_PREP_EXTENSION":
    lambda: os.environ.get("VLLM_GDN_QWEN38_NATIVE_QK_PREP_EXTENSION", ""),

    # Use the native HPU causal-conv1d forward op for GDN prefill. The
    # returned functional cache is persisted by the model wrapper.
    "VLLM_GDN_HPU_CAUSAL_CONV1D":
    lambda: os.environ.get("VLLM_GDN_HPU_CAUSAL_CONV1D", "false").lower() in ("1", "true"),

    # Keep the compiled PyTorch GDN prompt convolution in token-major layout
    # so its large packed activation does not need two DMA transposes.
    "VLLM_GDN_TOKEN_MAJOR_CAUSAL_CONV1D":
    lambda: os.environ.get("VLLM_GDN_TOKEN_MAJOR_CAUSAL_CONV1D", "false").lower() in ("1", "true"),

    # Use Habana's scale-calculation compound GUID for dynamic per-token FP8
    # activation quantization.
    "VLLM_HPU_DYNAMIC_QUANT_CGUID":
    lambda: os.environ.get("VLLM_HPU_DYNAMIC_QUANT_CGUID", "false").lower() in ("1", "true"),

    # Restrict the dynamic-quantization compound GUID to large static prefill
    # graphs. Small prompt and decode graphs use the ordinary reduction path.
    "VLLM_HPU_DYNAMIC_QUANT_CGUID_MIN_TOKENS":
    lambda: int(os.environ.get("VLLM_HPU_DYNAMIC_QUANT_CGUID_MIN_TOKENS", "2048")),

    # Express long-prompt SwiGLU as gate * sigmoid(gate) * up. On Gaudi this
    # lets the graph compiler fuse activation, multiply, and FP8 quantization.
    "VLLM_HPU_EXPLICIT_SIGMOID_SILU":
    lambda: os.environ.get("VLLM_HPU_EXPLICIT_SIGMOID_SILU", "false").lower() in ("1", "true"),

    # Keep the ordinary HPU SiLU kernel for small prompt and decode graphs.
    "VLLM_HPU_EXPLICIT_SIGMOID_SILU_MIN_TOKENS":
    lambda: int(os.environ.get("VLLM_HPU_EXPLICIT_SIGMOID_SILU_MIN_TOKENS", "2048")),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
