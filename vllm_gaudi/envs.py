# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH: bool = True
    VLLM_HPU_FORCE_CHANNEL_FP8: bool = True
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
    VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY: bool = False
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

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition
environment_variables: dict[str, Callable[[], Any]] = {
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

    # Run multimodal warmup outside PT_COMPILE_ONLY_MODE for models with
    # data-dependent output shapes that must be materialized during warmup.
    "VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY":
    lambda: os.environ.get("VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY", "false").strip().lower() in ("1", "true"),

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
