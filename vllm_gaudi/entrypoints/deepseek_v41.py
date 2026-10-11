# SPDX-License-Identifier: Apache-2.0
"""Launch the prepared V4.1 TP4 x PP1 profile through the normal vLLM CLI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

_C1_FASTPATH_DEFAULTS = {
    # This is the quality-qualified numerical profile used by the prepared
    # V4.1 deployment.  Individual switches remain explicit below so a
    # diagnostic override can disable one component before process startup.
    "VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS": "1",
    "VLLM_HPU_DSV41_PREPARED_SHARDS": "1",
    "VLLM_HPU_DSV41_ENGRAM_HOST_TABLE": "1",
    "VLLM_HPU_DSV41_GRAPH_REPLAY": "1",
    "VLLM_HPU_DSV41_VISION": "1",
    "VLLM_HPU_DSV41_QUANT_ROUNDTRIP": "1",
    "VLLM_HPU_DSV41_PACKED_ATTENTION": "1",
    "VLLM_HPU_DSV41_FIXED_POSITIONS": "1",
    "VLLM_HPU_DSV41_PACKED_PP": "1",
    "VLLM_HPU_DSV41_TPC_MHC": "1",
    "VLLM_HPU_DSV41_DIRECT_TOKEN_IDS": "1",
    "VLLM_HPU_DSV41_NATIVE_PP_COPY": "1",
    "VLLM_HPU_DSV41_PREPARED_OUTPUT": "1",
    "VLLM_HPU_DSV41_BOUNDED_ATTENTION": "1",
    "VLLM_HPU_DSV41_SWA_PACK_WRITE": "1",
    "VLLM_HPU_DSV41_FP4_CACHE_WRITE": "1",
    "VLLM_HPU_DSV41_NATIVE_ROPE": "1",
    "VLLM_HPU_DSV41_C1_INDICES": "1",
    "VLLM_HPU_DSV41_SELECTED_VALID_ONLY": "1",
    "VLLM_HPU_DSV41_TP_MHC_OVERLAP": "1",
    "VLLM_HPU_DSV41_ENGRAM_NATIVE_C1": "1",
    "VLLM_HPU_DSV41_ENGRAM_C1_PACKET": "1",
    "VLLM_HPU_DSV41_DECODED_KV_STATE": "1",
    # The long-context model keeps its canonical packed page pool.  Decode
    # selected rows directly from that pool instead of gathering, expanding
    # and materializing a per-layer BF16 cache before MLA.
    "VLLM_HPU_DSV41_PAGED_SELECTED_KV": "1",
    "VLLM_HPU_DSV41_SHARED_PREFIX_KV": "1",
    "VLLM_HPU_DSV41_FUSED_PREFIX_LAYOUT": "1",
    "VLLM_HPU_DSV41_NATIVE_KV_PACK": "1",
    # Keep the scheduler-owned packed pages canonical for the full 1M
    # lifetime, while retaining the active <=512-token prefix in the exact
    # decoded form already produced by the same quantizing writer.
    "VLLM_HPU_DSV41_PAGED_DECODED_KV_STATE": "1",
    # Use one fixed-capacity native CSA2 graph whose device-valued position
    # bounds the real scan.  This keeps long conversations on the same replay
    # path and avoids the generic per-bucket PyTorch score/top-k chain.
    "VLLM_HPU_DSV41_RUNTIME_INDEXER": "1",
    "VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH": "1",
    "VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT": "1",
    "VLLM_HPU_DSV41_V2_EARLY_INPUT_COMMIT": "1",
    "VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX": "1",
    "VLLM_HPU_DSV41_V2_DEVICE_ENGRAM": "1",
    "VLLM_HPU_DSV41_V2": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_HPU_DSV41_EXPERT_K128": "1",
    # Reuse decoded expert weights across each scheduler prompt chunk while
    # C1 decode continues to consume the same resident N256 allocation.  The
    # stock packed-MXFP4 prefill operator leaks its internal packed dtype into
    # later Synapse recipes on this runtime, so it is deliberately excluded.
    "VLLM_HPU_DSV41_PREFILL_GROUPED": "1",
    # Keep route metadata on HPU and make the descriptor shape depend only on
    # the scheduler tile.  Compact route output bounds the shared workspace to
    # real top-6 rows instead of the occupancy-dependent host implementation.
    "VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES": "1",
    "VLLM_HPU_DSV41_PREFILL_ROUTE_OUTPUT": "1",
    "VLLM_HPU_DSV41_PREFILL_EXPERT_ROWS": "128",
    "VLLM_HPU_DSV41_PREFILL_EXPERTS_PER_PLAN": "24",
    # Prefill compute capacity follows the scheduler and topology. Keep the
    # normal transaction intact; an explicit diagnostic cap remains supported.
    # Keep two scheduler-owned PP packets in flight.  The producer waits only
    # when its slot is reused and PP1 waits at the real consumer.  This was
    # qualified with the normal 1M context and max_num_seqs=8 profile; it is
    # not a single-request-only execution path.
    "VLLM_HPU_DSV41_PREFILL_PP_WAVEFRONT": "1",
    "VLLM_HPU_DSV41_PREFILL_COMPACT_CANDIDATES": "1",
    # KV source layers publish immutable compressed rows for several reuse
    # layers in the same prompt transaction. Decode each source once and keep
    # the bounded BF16 workspace alive through its real MLA consumers instead
    # of repeating packed-page gather and FP4 expansion in every layer.
    "VLLM_HPU_DSV41_PREFILL_KV_REUSE": "1",
    # These exact/qualified prompt regions remove generic materialization,
    # repeated packed-KV expansion and Python submissions.  They remain
    # finite-shape recipes and consume runtime request metadata.
    "VLLM_HPU_DSV41_PREFILL_MLA_ROWS": "512",
    "VLLM_HPU_DSV41_PREFILL_REINDEX_REUSE": "1",
    "VLLM_HPU_DSV41_FLASHINFER_PREFILL": "1",
    "VLLM_HPU_DSV41_PREFILL_INDEX_MME": "1",
    "VLLM_HPU_DSV41_PREFILL_INDEX_SHARED": "1",
    "VLLM_HPU_DSV41_PREFILL_INDEX_SRAM": "1",
    "VLLM_HPU_DSV41_PREFILL_REINDEX_SRAM": "1",
    # Query rows are independent after the existing head all-gather. Split
    # C1024-C8192 selection rows across TP2 and gather only final integer
    # choices. Smaller graph shapes retain the established local path.
    "VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP": "1",
    "VLLM_HPU_DSV41_PREFILL_CANDIDATE_GATHER": "1",
    "VLLM_HPU_DSV41_PREFILL_REGIONS": "1",
    "VLLM_HPU_DSV41_PREFILL_MHC_INPUT": "1",
    "VLLM_HPU_DSV41_PREFILL_MHC_POST": "1",
    "VLLM_HPU_DSV41_PREFILL_VECTOR_QUANT": "1",
    "VLLM_HPU_DSV41_PREFILL_ROPE": "1",
    "VLLM_HPU_DSV41_PREFILL_NATIVE_NORM": "1",
    "VLLM_HPU_DSV41_PREFILL_Q_PROJECTION": "1",
    "VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION": "1",
    "VLLM_HPU_DSV41_COMPRESSOR_FUSED_INPUT": "1",
    "VLLM_HPU_TP2_NATIVE_JOINT_PLAN": "1",
    "VLLM_HPU_TP2_PREPARED_COMM": "1",
    "VLLM_HPU_TP2_STATIC_GROUP_PLAN": "1",
}

_PREFILL_MOE_DEFAULTS = {
    "VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN": "1",
    "VLLM_HPU_DSV41_PREFILL_FAST_DEQUANT": "1",
    "VLLM_HPU_DSV41_PREFILL_SKIP_EMPTY": "1",
    "VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS": "1",
}

_NUMERIC_FASTPATH_DEFAULTS = {
    "VLLM_HPU_DSV41_WO_A_FP8": "1",
    # Keep the exact group-32 BF16 roundtrip required by wo_b inside the
    # wo_a scale producer.  This removes one HBM intermediate without
    # changing any model-visible arithmetic boundary.
    "VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP": "1",
    "VLLM_HPU_DSV41_ROUTER_TOP6": "1",
    "VLLM_HPU_DSV41_BF16_LM_HEAD": "1",
    "VLLM_HPU_DSV41_MLA_MME": "1",
    "VLLM_HPU_DSV41_EXPERT_N256_FP8": "1",
    "VLLM_HPU_DSV41_EXPERT_FUSED_QUANT": "1",
    "VLLM_HPU_DSV41_QKV_FUSED_INPUT": "1",
    "VLLM_HPU_DSV41_ATTN_FUSED_NORM": "1",
    "VLLM_HPU_DSV41_SHARED_GATE_UP": "1",
    "VLLM_HPU_DSV41_BF16_ROUTER_GATE": "1",
    "VLLM_HPU_DSV41_MHC_GATES_FUSED": "1",
    "VLLM_HPU_DSV41_MHC_CONTROL_RRMS": "1",
    "VLLM_HPU_DSV41_ATTN_DENSE_FP8": "1",
    "VLLM_HPU_DSV41_ENGRAM_FP8": "1",
    "VLLM_HPU_DSV41_Q_SCALE_ROPE": "1",
    "VLLM_HPU_DSV41_EXPERT_FUSED_REDUCE": "1",
}

_SIDECARS = {
    "wo_a_fp8": ("VLLM_HPU_DSV41_WO_A_FP8", "VLLM_HPU_DSV41_WO_A_FP8_SIDECAR"),
    "attention_dense_fp8": ("VLLM_HPU_DSV41_ATTN_DENSE_FP8", "VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR"),
    "engram_fp8": ("VLLM_HPU_DSV41_ENGRAM_FP8", "VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR"),
}

# Prepared N256 storage retains its qualified prefill and expert defaults.
# Decode computation and replay inherit the shared C1 profile below.
_TP4_FASTPATH_DEFAULTS = {
    "VLLM_HPU_DSV41_PREPARED_SHARDS": "1",
    "VLLM_HPU_DSV41_ENGRAM_HOST_TABLE": "1",
    "VLLM_HPU_DSV41_QUANT_ROUNDTRIP": "1",
    "VLLM_HPU_DSV41_VISION": "1",
    "VLLM_HPU_DSV41_ROUTER_TOP6": "1",
    "VLLM_HPU_DSV41_BF16_LM_HEAD": "1",
    "VLLM_HPU_DSV41_BF16_ROUTER_GATE": "1",
    # The stock fused MoE does not implement V4.1's gate/up clamps or its
    # FP32 routing-before-BF16 boundary. The N256 grouped path preserves both.
    "VLLM_HPU_DSV41_PREFILL_MXFP4": "0",
    "VLLM_HPU_DSV41_PREFILL_REGIONS": "1",
    "VLLM_HPU_DSV41_EXPERT_N256_FP8": "1",
    "VLLM_HPU_DSV41_EXPERT_FUSED_QUANT": "1",
    "VLLM_HPU_DSV41_EXPERT_FUSED_REDUCE": "1",
    "VLLM_HPU_DSV41_PREFILL_GROUPED": "1",
    "VLLM_HPU_DSV41_PREFILL_MHC_POST": "1",
    "VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES": "1",
    "VLLM_HPU_DSV41_PREFILL_ROUTE_OUTPUT": "1",
    "VLLM_HPU_DSV41_PREFILL_FAST_DEQUANT": "1",
    "VLLM_HPU_DSV41_PREFILL_ACTIVE_PLAN": "1",
    "VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN": "1",
    "VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS": "1",
    "VLLM_HPU_DSV41_PREFILL_SKIP_EMPTY": "1",
    "VLLM_HPU_DSV41_PREFILL_EXPERTS_PER_PLAN": "24",
    "VLLM_HPU_DSV41_ENGRAM_NATIVE_C1": "1",
    "VLLM_HPU_DSV41_ENGRAM_C1_PACKET": "1",
    "VLLM_HPU_DSV41_PREFILL_EXPERT_ROWS": "128",
    # Preserve the qualified complete-prefill W13 path. This mode leaves
    # routed W2 and the native C1 decode program unchanged.
    "VLLM_HPU_DSV41_PREFILL_GROUPED_FP8": "w13_single_bucket",
    "VLLM_HPU_DSV41_PREFILL_DECODER_HALO": "1",
    "VLLM_HPU_DSV41_PREFILL_MLA_SEQUENCE": "1",
}

# Shape-aware computation shared by TP2 and TP4. Communication stays HCCL.
_TP4_FASTPATH_DEFAULTS.update(
    {
        "VLLM_HPU_DSV41_V2": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_HPU_DSV41_DIRECT_TOKEN_IDS": "1",
        "VLLM_HPU_DSV41_FIXED_POSITIONS": "1",
        "VLLM_HPU_DSV41_V2_EARLY_INPUT_COMMIT": "1",
        "VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX": "1",
        "VLLM_HPU_DSV41_V2_DEVICE_ENGRAM": "1",
        "VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS": "1",
        "VLLM_HPU_DSV41_ROUTER_TOP6": "1",
        "VLLM_HPU_DSV41_TPC_MHC": "1",
        "VLLM_HPU_DSV41_MHC_GATES_FUSED": "1",
        "VLLM_HPU_DSV41_MHC_CONTROL_RRMS": "1",
        "VLLM_HPU_DSV41_SHARED_GATE_UP": "1",
        "VLLM_HPU_DSV41_WO_A_FP8": "1",
        "VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP": "1",
        "VLLM_HPU_DSV41_ATTN_DENSE_FP8": "1",
        "VLLM_HPU_DSV41_ENGRAM_FP8": "1",
        "VLLM_HPU_DSV41_NATIVE_ROPE": "1",
        "VLLM_HPU_DSV41_Q_SCALE_ROPE": "1",
        "VLLM_HPU_DSV41_QKV_FUSED_INPUT": "1",
        "VLLM_HPU_DSV41_ATTN_FUSED_NORM": "1",
        "VLLM_HPU_DSV41_COMPRESSOR_FUSED_INPUT": "1",
        "VLLM_HPU_DSV41_MLA_MME": "1",
        "VLLM_HPU_DSV41_PAGED_SELECTED_KV": "1",
        "VLLM_HPU_DSV41_SHARED_PREFIX_KV": "1",
        "VLLM_HPU_DSV41_FUSED_PREFIX_LAYOUT": "1",
        "VLLM_HPU_DSV41_NATIVE_KV_PACK": "1",
        "VLLM_HPU_DSV41_PAGED_DECODED_KV_STATE": "1",
        "VLLM_HPU_DSV41_PREFILL_COMPACT_CANDIDATES": "1",
        "VLLM_HPU_DSV41_PREFILL_KV_REUSE": "1",
        "VLLM_HPU_DSV41_PREFILL_REINDEX_REUSE": "1",
        "VLLM_HPU_DSV41_FLASHINFER_PREFILL": "1",
        "VLLM_HPU_DSV41_PREFILL_INDEX_MME": "1",
        "VLLM_HPU_DSV41_PREFILL_INDEX_SHARED": "1",
        "VLLM_HPU_DSV41_PREFILL_INDEX_SRAM": "1",
        "VLLM_HPU_DSV41_PREFILL_REINDEX_SRAM": "1",
        "VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP": "1",
        "VLLM_HPU_DSV41_PREFILL_CANDIDATE_GATHER": "1",
        "VLLM_HPU_DSV41_PREFILL_REGIONS": "1",
        "VLLM_HPU_DSV41_PREFILL_MHC_INPUT": "1",
        "VLLM_HPU_DSV41_PREFILL_VECTOR_QUANT": "1",
        "VLLM_HPU_DSV41_PREFILL_ROPE": "1",
        "VLLM_HPU_DSV41_PREFILL_NATIVE_NORM": "1",
        "VLLM_HPU_DSV41_PREFILL_Q_PROJECTION": "1",
        "VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION": "1",
        "VLLM_HPU_DSV41_PREFILL_MLA_ROWS": "512",
    }
)

# A TP-only stage inherits the qualified C1 compute and replay defaults.
# The three exclusions require a pipeline peer, absent when PP has one rank.
_TP4_FASTPATH_DEFAULTS = {**_C1_FASTPATH_DEFAULTS, **_NUMERIC_FASTPATH_DEFAULTS, **_TP4_FASTPATH_DEFAULTS}
# Qualified together in the ordinary TP-only C1 service. STATIC_COORDINATES
# is already part of the single-stage defaults below. Explicit diagnostics
# can still disable any member; non-C1 execution keeps its existing guards.
_TP4_FASTPATH_DEFAULTS.update({
    "VLLM_HPU_DSV41_COMPRESSOR_FUSED_PUBLISH": "1",
    "VLLM_HPU_DSV41_EXPERT_W2_THREE_ROUTES": "1",
    "VLLM_HPU_DSV41_QKV_FUSED_PROLOGUE": "1",
    "VLLM_HPU_DSV41_CANDIDATE_COORDINATES": "1",
    "VLLM_HPU_DSV41_PEER_POST_COLLAPSE": "1",
})
# Qualified ordinary replay without a pipeline boundary. PP stages retain
# their existing token ownership; diagnostic overrides remain authoritative.
_SINGLE_STAGE_NATIVE_DEFAULTS = {
    "VLLM_HPU_DSV41_DEVICE_SAMPLING": "1",
    "VLLM_HPU_DSV41_DEVICE_NEXT_POSITION": "1",
    "VLLM_HPU_DSV41_DEVICE_INPUT_FEEDBACK": "1",
    "VLLM_HPU_DSV41_DEVICE_CLOSED_LOOP": "1",
    "VLLM_HPU_DSV41_STATIC_COORDINATES": "1",
}

_PIPELINE_ONLY_FASTPATHS = (
    "VLLM_HPU_DSV41_PACKED_PP",
    "VLLM_HPU_DSV41_NATIVE_PP_COPY",
    "VLLM_HPU_DSV41_PREFILL_PP_WAVEFRONT",
)

# DSpark shares the prepared compute and prompt paths. The existing native
# input/prefix, decoded-state and V2 adapters still have C1-only contracts.
_DSPARK_FASTPATH_DEFAULTS = {
    **_C1_FASTPATH_DEFAULTS,
    **_PREFILL_MOE_DEFAULTS,
    "VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS": "0",
    "VLLM_HPU_DSV41_BF16_LM_HEAD": "0",
    "VLLM_HPU_DSV41_BF16_ROUTER_GATE": "0",
    "VLLM_HPU_DSV41_SHARED_GATE_UP": "0",
    # Packed speculative control remains opt-in until final token identity
    # is qualified; hc_pre keeps its prompt arithmetic and decode scope.
    "VLLM_HPU_DSV41_MHC_CONTROL_RRMS": "0",
    # These prompt regions require the dense FP8 sidecars, whose stage
    # contract currently excludes speculative verification. Keep their
    # existing BF16 producers until that shared contract is qualified.
    "VLLM_HPU_DSV41_PREFILL_Q_PROJECTION": "0",
    "VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION": "0",
    "VLLM_HPU_DSV41_V2": "0",
    "VLLM_USE_V2_MODEL_RUNNER": "0",
    "VLLM_HPU_DSV41_RUNTIME_INDEXER": "0",
    "VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH": "0",
    "VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT": "0",
    "VLLM_HPU_DSV41_V2_EARLY_INPUT_COMMIT": "0",
    "VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX": "0",
    "VLLM_HPU_DSV41_V2_DEVICE_ENGRAM": "0",
    "VLLM_HPU_DSV41_DECODED_KV_STATE": "0",
    "VLLM_HPU_DSV41_PAGED_DECODED_KV_STATE": "0",
    "VLLM_HPU_DSV41_ENGRAM_NATIVE_C1": "0",
    "VLLM_HPU_DSV41_ENGRAM_C1_PACKET": "0",
    "VLLM_HPU_DSV41_TP_MHC_OVERLAP": "0",
    "VLLM_HPU_DSV41_DEVICE_VERIFY": "1",
    "VLLM_HPU_DSV41_EXPERT_N256_FP8": "1",
    "VLLM_HPU_DSV41_EXPERT_FUSED_QUANT": "1",
    "VLLM_HPU_DSV41_EXPERT_FUSED_REDUCE": "1",
    "VLLM_HPU_DSV41_MHC_GATES_FUSED": "1",
    "VLLM_HPU_DSV41_MHC_SCHEDULE": "1",
    "VLLM_HPU_DSV41_MLA_MME": "1",
    "VLLM_HPU_DSV41_QKV_FUSED_INPUT": "1",
    # PREPARED_OUTPUT already owns the BF16 wo_a layout. Applying the old
    # loader's four-group transpose first scrambles the target and MTP
    # matrices; TP4 owns two output groups. C1 FP8 sidecars bypass this path.
    "VLLM_HPU_DSV41_PRETRANSPOSE_ATTN": "0",
    # Fused text entry currently reads residual.shape before creating its
    # residual. Use the existing explicit embedding entry until the shared
    # replay boundary receives its row-count fix.
    "VLLM_HPU_DSV41_FUSED_STAGE_IO": "0",
    "VLLM_HPU_DSV41_BATCHED_INPUT_STAGING": "0",
    "VLLM_HPU_TP2_NATIVE_JOINT_PLAN": "1",
    "VLLM_HPU_TP2_PREPARED_COMM": "1",
    "VLLM_HPU_TP2_STATIC_GROUP_PLAN": "1",
}


def _enabled(value):
    return str(value).strip().lower() in ("1", "true")


# The qualified official sampling profile applies to the stage owning both
# input and sampling. Explicit disables retain their priority; PP2 keeps its
# separately qualified defaults.
_DSPARK_SINGLE_STAGE_DEFAULTS = {
    **{key: value for key, value in _TP4_FASTPATH_DEFAULTS.items()
       if (key.startswith("VLLM_HPU_DSV41_PREFILL_") or key == "VLLM_HPU_DSV41_FLASHINFER_PREFILL")
       and key not in _PIPELINE_ONLY_FASTPATHS},
    "VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS": "1",
    "VLLM_HPU_DSV41_BF16_LM_HEAD": "1",
    "VLLM_HPU_DSV41_WO_A_FP8": "1",
    "VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP": "1",
    "VLLM_HPU_DSV41_ATTN_DENSE_FP8": "1",
    "VLLM_HPU_DSV41_MHC_CONTROL_RRMS": "1",
    "VLLM_HPU_DSV41_MHC_CONTROL_MME": "1",
    "VLLM_HPU_DSV41_ROUTER_TOP6": "1",
    "VLLM_HPU_DSV41_FUSED_STAGE_IO": "1",
    "VLLM_HPU_DSV41_BATCHED_INPUT_STAGING": "1",
    "VLLM_HPU_DSV41_DEVICE_ROUNDS": "1",
    "VLLM_HPU_DSV41_DSPARK_NATIVE_SAMPLED_PROTOCOL": "1",
    "VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_MAIN": "1",
    "VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_REPAIR": "1",
    "VLLM_HPU_DSV41_DSPARK_GLOBAL_BOUNDED_SAMPLING": "1",
    "VLLM_HPU_DSV41_DSPARK_LARGE_HCCL": "1",
    "VLLM_HPU_DSV41_DSPARK_FULL_HCCL_MAIN": "1",
    "VLLM_HPU_DSV41_DSPARK_LANE_CANDIDATES": "1",
    "VLLM_HPU_DSV41_DSPARK_FUSED_BOUNDED_NUCLEUS": "1",
    "VLLM_HPU_DSV41_DSPARK_JOURNAL_COPY": "1",
    "VLLM_HPU_DSV41_DSPARK_CONSUMED_TARGET_CERTIFICATE": "1",
    "VLLM_HPU_DSV41_DSPARK_COVERAGE_AUDIT": "1",
    "VLLM_HPU_DSV41_DSPARK_HEAD_INPUT_BOUNDARY": "1",
    "VLLM_HPU_DSV41_DSPARK_WEIGHTED_DRAFT_NUCLEUS": "1",
    "VLLM_HPU_DSV41_DSPARK_LAYER_MAIN_SPLIT": "1",
    "VLLM_HPU_DSV41_DSPARK_NORM_ROUNDTRIP": "1",
    "VLLM_HPU_DSV41_DSPARK_W13_K_PIPELINE": "1",
    "VLLM_HPU_DSV41_DSPARK_W13_K_PIPELINE_STAGES": "2",
    "VLLM_HPU_DSV41_DSPARK_MHC_HIGH_PLANE": "1",
    "VLLM_HPU_DSV41_DSPARK_SHARED_FP8": "1",
    "VLLM_HPU_DSV41_DSPARK_SCALE_CACHE": "1",
    "VLLM_HPU_DSV41_DSPARK_STOCHASTIC_ONLY": "1",
    "VLLM_HPU_DSV41_DSPARK_THRESHOLD_SELECTION": "1",
    "VLLM_HPU_DSV41_DSPARK_BATCH_ENGRAM": "1",
    "VLLM_HPU_DSV41_DSPARK_DRAFT_VOCAB_CDF": "1",
}


def prepare_default_fastpaths(model, sidecars=None, tensor_parallel_size=4, pipeline_parallel_size=1):
    """Select prepared execution defaults for the requested four-card topology."""
    if (tensor_parallel_size, pipeline_parallel_size) not in ((2, 2), (4, 1)):
        raise ValueError("The prepared V4.1 path supports TP4 x PP1 or legacy TP2 x PP2")
    if not _enabled(os.environ.get("VLLM_HPU_DSV41_DEFAULT_FASTPATHS", "1")):
        return
    os.environ.setdefault("VLLM_HPU_DSV41_DSPARK", "0")
    if _enabled(os.environ["VLLM_HPU_DSV41_DSPARK"]):
        defaults = dict(_DSPARK_FASTPATH_DEFAULTS)
        if pipeline_parallel_size == 1:
            defaults.update(_DSPARK_SINGLE_STAGE_DEFAULTS)
        for key, value in defaults.items():
            os.environ.setdefault(key, value)
        if pipeline_parallel_size == 1:
            for key in _PIPELINE_ONLY_FASTPATHS:
                os.environ[key] = "0"
        prepare_precision_sidecars(model, sidecars)
        return
    selected_v2 = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    selected_adapter = os.environ.get("VLLM_HPU_DSV41_V2")
    if selected_v2 is not None and selected_adapter is None:
        os.environ["VLLM_HPU_DSV41_V2"] = "1" if _enabled(selected_v2) else "0"
    elif selected_adapter is not None and selected_v2 is None:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if _enabled(selected_adapter) else "0"
    if (tensor_parallel_size, pipeline_parallel_size) == (4, 1):
        for key, value in _TP4_FASTPATH_DEFAULTS.items():
            os.environ.setdefault(key, value)
    else:
        for key, value in _C1_FASTPATH_DEFAULTS.items():
            os.environ.setdefault(key, value)
        if _enabled(os.environ.get("VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS", "0")):
            for key, value in _NUMERIC_FASTPATH_DEFAULTS.items():
                os.environ.setdefault(key, value)
    if pipeline_parallel_size == 1:
        for key in _PIPELINE_ONLY_FASTPATHS:
            os.environ[key] = "0"
    if pipeline_parallel_size == 1 and all(
        _enabled(os.environ.get(key, "0"))
        for key in ("VLLM_HPU_DSV41_V2", "VLLM_HPU_DSV41_GRAPH_REPLAY", "VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH")
    ):
        for key, value in _SINGLE_STAGE_NATIVE_DEFAULTS.items():
            os.environ.setdefault(key, value)
    # Prepared N256 weights select the qualified BF16 prompt implementation.
    # Profiles without this storage layout retain their existing dispatch.
    if any(
        _enabled(os.environ.get(name, "0")) for name in ("VLLM_HPU_DSV41_EXPERT_N256", "VLLM_HPU_DSV41_EXPERT_N256_FP8")
    ):
        for key, value in _PREFILL_MOE_DEFAULTS.items():
            os.environ.setdefault(key, value)
    prepare_precision_sidecars(model, sidecars)


def prepare_precision_sidecars(model, sidecars=None):
    """Resolve the same prepared precision artifacts for target C1 and C2-C6."""
    model = Path(model).resolve()
    configured = sidecars or {}
    for name, (enabled_key, path_key) in _SIDECARS.items():
        if not _enabled(os.environ.get(enabled_key, "0")) or os.environ.get(path_key):
            continue
        candidates = []
        if configured.get(name):
            candidates.append(Path(configured[name]).expanduser())
        candidates.extend((model / "sidecars" / name, model.parent / f"{model.name}-{name}"))
        directory = next((candidate.resolve() for candidate in candidates if candidate.is_dir()), None)
        if directory is None:
            raise RuntimeError(
                f"Default V4.1 fast paths require the {name} sidecar; prepare {model / 'sidecars' / name} "
                f"or disable {enabled_key}"
            )
        os.environ[path_key] = str(directory)
    dense_dir = os.environ.get("VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR")
    if dense_dir and not os.environ.get("VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG"):
        precision = Path(dense_dir) / "precision.json"
        if precision.is_file():
            os.environ["VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG"] = str(precision)


def prepare_native_libraries():
    configured_library = os.environ.get("VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR")
    library_dir = (
        Path(configured_library).resolve() if configured_library else Path(__file__).resolve().parents[1] / "lib"
    )
    kernel = library_dir / "libdeepseek_v4_gaudi2_kernels.so"
    extensions = list(library_dir.glob("hpu_dsv4_sparse_attn_pt2*.so"))
    if not kernel.is_file() or len(extensions) != 1:
        raise RuntimeError("Build the prepared V4.1 native libraries first")
    if configured_library:
        manifest = json.loads((library_dir / "deepseek_v4_build.json").read_text())
        for library in (kernel, extensions[0]):
            if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["binaries"].get(library.name):
                raise RuntimeError(f"V4.1 native binary differs from its build manifest: {library.name}")
        host_manifest = json.loads((library_dir / "deepseek_v41_build.json").read_text())
        hosts = list(library_dir.glob("dsv41_host_gather*.so"))
        if (
            len(hosts) != 1
            or host_manifest.get("host_gather_abi_version") != 1
            or host_manifest.get("host_c1_abi_version") != 2
            or host_manifest.get("host_gather_packed_output_version") != 1
            or host_manifest.get("host_gather_profiling_version") != 1
            or hashlib.sha256(hosts[0].read_bytes()).hexdigest() != host_manifest["binaries"].get(hosts[0].name)
        ):
            raise RuntimeError("V4.1 host gather differs from its isolated build manifest")
        # The worker imports this exact file through deepseek_v41_host after
        # the HPU environment is initialized and validates the live ABI there.
    configured = os.environ.get("GC_KERNEL_PATH", str(kernel))
    selected_kernel = kernel
    unique_manifest = library_dir / "deepseek_v41_unique_build.json"
    if unique_manifest.is_file():
        addon = json.loads(unique_manifest.read_text())
        selected_kernel = library_dir / "libdeepseek_v41_unique_kernels.so"
        if addon.get("base_kernel_sha256") != hashlib.sha256(kernel.read_bytes()).hexdigest():
            raise RuntimeError("Unique expert database was built for a different base kernel")
        if (hashlib.sha256(selected_kernel.read_bytes()).hexdigest()
                != addon.get("binaries", {}).get(selected_kernel.name)):
            raise RuntimeError("Unique expert database differs from its build manifest")
        configure_probability_repair_parent(addon, selected_kernel)
        pair = addon.get("router_pair")
        if pair:
            parent = Path(pair["parent"]).resolve()
            variable = "VLLM_HPU_DSV41_ROUTER_PAIR_PARENT_KERNEL"
            if (parent == selected_kernel.resolve() or not parent.is_file()
                    or hashlib.sha256(parent.read_bytes()).hexdigest() != pair["parent_sha256"]):
                raise RuntimeError("Router precision database differs from its locked parent")
            if os.environ.get(variable) and Path(os.environ[variable]).resolve() != parent:
                raise RuntimeError("Router precision parent differs from its build manifest")
            os.environ[variable] = str(parent)
        ready = addon.get("router_ready")
        if ready:
            parent = Path(ready["parent"]).resolve()
            variable = "VLLM_HPU_DSV41_ROUTER_READY_PARENT_KERNEL"
            if (parent == selected_kernel.resolve() or not parent.is_file()
                    or hashlib.sha256(parent.read_bytes()).hexdigest() != ready["parent_sha256"]):
                raise RuntimeError("Prepared Router database differs from its locked parent")
            if os.environ.get(variable) and Path(os.environ[variable]).resolve() != parent:
                raise RuntimeError("Prepared Router parent override differs from its build manifest")
            os.environ[variable] = str(parent)
        softmax = addon.get("vocab_softmax")
        if softmax:
            parent = Path(softmax["parent"]).resolve()
            variable = "VLLM_HPU_DSV41_SOFTMAX_PARENT_KERNEL"
            if (parent == selected_kernel.resolve() or not parent.is_file()
                    or hashlib.sha256(parent.read_bytes()).hexdigest() != softmax["parent_sha256"]):
                raise RuntimeError("Vocabulary softmax database differs from its locked parent")
            if os.environ.get(variable) and Path(os.environ[variable]).resolve() != parent:
                raise RuntimeError("Vocabulary softmax parent override differs from its build manifest")
            os.environ[variable] = str(parent)
        factor = addon.get("w2_factor_build")
        if factor:
            parent = Path(factor["parent_gc_path"]).resolve()
            if (parent == selected_kernel.resolve() or not parent.is_file()
                    or hashlib.sha256(parent.read_bytes()).hexdigest() != factor["parent_gc_sha256"]):
                raise RuntimeError("W2 factor database differs from its locked parent")
            variable = "VLLM_HPU_DSV41_W2_FACTOR_PARENT_KERNEL"
            configured_parent = os.environ.get(variable)
            if configured_parent and Path(configured_parent).resolve() != parent:
                raise RuntimeError("W2 factor parent override differs from its build manifest")
            os.environ[variable] = str(parent)
        consumer = addon.get("w2_consumer_build")
        if consumer:
            parent = Path(consumer["parent_gc_path"]).resolve()
            if (parent == selected_kernel.resolve() or not parent.is_file()
                    or hashlib.sha256(parent.read_bytes()).hexdigest() != consumer["parent_gc_sha256"]):
                raise RuntimeError("W2 consumer database differs from its locked parent")
            variable = "VLLM_HPU_DSV41_W2_REDUCE_PARENT_KERNEL"
            configured_parent = os.environ.get(variable)
            if configured_parent and Path(configured_parent).resolve() != parent:
                raise RuntimeError("W2 consumer parent override differs from its build manifest")
            os.environ[variable] = str(parent)
        shared_consumer = addon.get("w2_shared_scale_build")
        if shared_consumer:
            parent = Path(shared_consumer["parent_gc_path"]).resolve()
            variable = "VLLM_HPU_DSV41_W2_SHARED_PARENT_KERNEL"
            if (parent == selected_kernel.resolve() or not parent.is_file()
                    or hashlib.sha256(parent.read_bytes()).hexdigest() != shared_consumer["parent_gc_sha256"]):
                raise RuntimeError("Shared W2 scale database differs from its locked parent")
            if os.environ.get(variable) and Path(os.environ[variable]).resolve() != parent:
                raise RuntimeError("Shared W2 scale parent override differs from its build manifest")
            os.environ[variable] = str(parent)
        channels = addon.get("w2_channels_build")
        if channels:
            parent = Path(channels["parent_gc_path"]).resolve()
            variable = "VLLM_HPU_DSV41_W2_CHANNELS_PARENT_KERNEL"
            if (parent == selected_kernel.resolve() or not parent.is_file()
                    or hashlib.sha256(parent.read_bytes()).hexdigest() != channels["parent_gc_sha256"]):
                raise RuntimeError("Channel W2 database differs from its locked parent")
            if os.environ.get(variable) and Path(os.environ[variable]).resolve() != parent:
                raise RuntimeError("Channel W2 parent differs from its build manifest")
            os.environ[variable] = str(parent)
        interleaved = addon.get("attention_interleave_build")
        if interleaved:
            parent = Path(interleaved["parent_gc_path"]).resolve()
            variable = "VLLM_HPU_DSV41_ATTENTION_INTERLEAVE_PARENT_KERNEL"
            if (parent == selected_kernel.resolve() or not parent.is_file()
                    or hashlib.sha256(parent.read_bytes()).hexdigest() != interleaved["parent_gc_sha256"]):
                raise RuntimeError("Interleaved attention database differs from its locked parent")
            if os.environ.get(variable) and Path(os.environ[variable]).resolve() != parent:
                raise RuntimeError("Interleaved attention parent differs from its build manifest")
            os.environ[variable] = str(parent)
        batched = addon.get("router_batched_build")
        if batched:
            parent = Path(batched["parent_gc_path"]).resolve()
            variable = "VLLM_HPU_DSV41_ROUTER_BATCHED_PARENT_KERNEL"
            if (parent == selected_kernel.resolve() or not parent.is_file()
                    or hashlib.sha256(parent.read_bytes()).hexdigest() != batched["parent_gc_sha256"]):
                raise RuntimeError("Batched Router database differs from its locked parent")
            if os.environ.get(variable) and Path(os.environ[variable]).resolve() != parent:
                raise RuntimeError("Batched Router parent differs from its build manifest")
            os.environ[variable] = str(parent)
        paired = addon.get("dense_bf16_pair_build")
        if paired:
            parent = Path(paired["parent_gc_path"]).resolve()
            variable = "VLLM_HPU_DSV41_DENSE_BF16_PAIR_PARENT_KERNEL"
            if (parent == selected_kernel.resolve() or not parent.is_file()
                    or hashlib.sha256(parent.read_bytes()).hexdigest() != paired["parent_gc_sha256"]):
                raise RuntimeError("Joint BF16 input database differs from its locked parent")
            if os.environ.get(variable) and Path(os.environ[variable]).resolve() != parent:
                raise RuntimeError("Joint BF16 input parent differs from its build manifest")
            os.environ[variable] = str(parent)
        cooperative = addon.get("cooperative_silu_build")
        if cooperative:
            parent = Path(cooperative["parent_gc_path"]).resolve()
            variable = "VLLM_HPU_DSV41_COOPERATIVE_SILU_PARENT_KERNEL"
            if (parent == selected_kernel.resolve() or not parent.is_file()
                    or hashlib.sha256(parent.read_bytes()).hexdigest() != cooperative["parent_gc_sha256"]):
                raise RuntimeError("Cooperative SiLU database differs from its locked parent")
            if os.environ.get(variable) and Path(os.environ[variable]).resolve() != parent:
                raise RuntimeError("Cooperative SiLU parent override differs from its build manifest")
            os.environ[variable] = str(parent)
        os.environ["VLLM_HPU_DSV41_UNIQUE_BASE_KERNEL"] = str(kernel)
    if configured not in (str(kernel), str(selected_kernel), "/usr/lib/habanalabs/libtpc_kernels.so"):
        raise RuntimeError("The V4.1 launch profile requires its combined kernel database")
    os.environ["GC_KERNEL_PATH"] = str(selected_kernel)
    os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"] = str(extensions[0])


def configure_probability_repair_parent(manifest, selected_kernel):
    """Resolve the additive database's locked parent from its build manifest."""
    repair = manifest.get("repair_build")
    if not repair:
        return
    parent = Path(repair["parent_gc_path"]).resolve()
    variable = "VLLM_HPU_DSV41_FULL_CDF_PARENT_KERNEL"
    if (parent == selected_kernel.resolve() or not parent.is_file()
            or hashlib.sha256(parent.read_bytes()).hexdigest() != repair["parent_gc_sha256"]):
        raise RuntimeError("Probability repair database differs from its locked parent")
    configured = os.environ.get(variable)
    if configured and Path(configured).resolve() != parent:
        raise RuntimeError("Probability repair parent override differs from its build manifest")
    os.environ[variable] = str(parent)


def load_native_operators(required=()):
    """Load the fingerprinted V4.1 extension before model/Dynamo imports.

    Spawned workers do not inherit PyTorch's process-local operator registry.
    Loading on demand from the model constructor was also brittle: another
    extension could already have populated part of ``custom_op`` and make a
    single-symbol guard skip the selected library.  The worker therefore
    loads the exact manifest-checked extension once after binding its HPU and
    validates every operator needed by the selected execution profile.
    """
    prepare_native_libraries()
    import torch

    library = os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"]
    torch.ops.load_library(library)
    directory = Path(library).parent
    controls = list(directory.glob("hpu_dsv4_control_mme_pt2*.so"))
    if controls:
        if len(controls) != 1:
            raise RuntimeError("Expected one prepared mHC MME extension")
        manifest = json.loads((directory / "deepseek_v4_build.json").read_text())
        control = controls[0]
        if hashlib.sha256(control.read_bytes()).hexdigest() != manifest["binaries"].get(control.name):
            raise RuntimeError("mHC MME extension differs from its build manifest")
        if not hasattr(torch.ops.custom_op, "custom_deepseek_v41_control_mme_f32_gaudi2"):
            torch.ops.load_library(str(control))
    from vllm_gaudi import envs

    optional = []
    if envs.VLLM_HPU_DSV41_MHC_RRMS_POST:
        optional.append("custom_deepseek_v41_mhc_rrms_post_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_ROUTER_BATCHED_F32:
        optional.append("custom_deepseek_v41_router_batched_f32_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_VOCAB_SOFTMAX:
        optional.append("custom_deepseek_v41_vocab_softmax_f32_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_TENSOR_READY_PEER:
        optional.append("custom_deepseek_v41_peer_ready_identity_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_ROUTER_SHARED_BF16:
        optional.extend(("custom_deepseek_v41_dense_bf16_pair_gaudi2",
                         "custom_deepseek_v41_router_shared_scaled_gaudi2",
                         "custom_deepseek_v41_shared_silu_full_product_gaudi2",
                         "custom_deepseek_v41_bf16_linear_f32_gaudi2"))
    if envs.VLLM_HPU_DSV41_DSPARK_COOPERATIVE_SILU:
        optional.append("custom_deepseek_v41_expert_n256_moe_cooperative_silu_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_W2_CHANNELS:
        optional.append("custom_deepseek_v41_expert_n256_moe_w2_channels_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_W2_REDUCE_N256:
        optional.append("custom_deepseek_v41_expert_n256_moe_w2_reduce_n256_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_W2_READY_SCALE:
        optional.append("custom_deepseek_v41_expert_n256_moe_w2_ready_scale_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_WEIGHTED_STATIC_MASK:
        optional.extend(("custom_deepseek_v41_weighted_finish_mask_gaudi2",
                         "custom_deepseek_v41_weighted_sparse_bins_gaudi2"))
    if envs.VLLM_HPU_DSV41_DSPARK_MAIN_ADJACENT_PV:
        optional.append("custom_deepseek_v41_main_adjacent_reuse_mla_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_SWA_SOURCE_REUSE:
        optional.append("custom_deepseek_v41_swa_source_reuse_mla_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_SILU_FULL_ROWS:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_silu_full_rows_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_COMPRESSOR_SEQUENCE:
        optional.append("custom_deepseek_v41_compressor_sequence_bf16_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_PHYSICAL_ROLE_SILU:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_physical_role_silu_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_JOURNAL_COPY:
        optional.append("custom_deepseek_v41_journal_copy_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_FUSED_BOUNDED_NUCLEUS:
        optional.append("custom_deepseek_v41_bounded_local_nucleus_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_LANE_CANDIDATES:
        optional.append("custom_deepseek_v41_vocab_lane_candidates_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_MHC_CONTROL_TILES:
        optional.append("custom_deepseek_v41_mhc_control_tiles_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_EXP_PV:
        optional.append("custom_deepseek_v41_logical_scale_exp_pv_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_WO_HANDOFF:
        optional.append("custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_N512_DECODE:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_n512_decode_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_CONTROL_FP8:
        optional.append("custom_deepseek_v41_control_fp8_rrms_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_INPUT_QUANT_REMAT:
        optional.append("custom_deepseek_v41_attention_norm_quant_gaudi2_remat")
    if envs.VLLM_HPU_DSV41_DSPARK_MOE_PEER_POST:
        optional.append("custom_deepseek_v41_dspark_moe_peer_post_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_SCALED_W13:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_scaled_w13_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_EXPLICIT_STEPS:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_explicit_steps_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_MHC_DEFERRED:
        optional.append("custom_deepseek_v41_mhc_mme_post_collapse_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_SILU_DECODE:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_silu_decode_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_NORM_ROUNDTRIP:
        optional.append("custom_deepseek_v41_norm_roundtrip_bf16_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_DRAFT_VOCAB_CDF:
        optional.append("custom_deepseek_v41_probability_draw_gaudi2")
        optional.append("custom_deepseek_v41_probability_full_draw_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_MTP_CACHE:
        optional.append("custom_deepseek_v41_mtp_cached_moe_bf16_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_MTP_SAT:
        optional.append("custom_deepseek_v41_mtp_sat_moe_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_MHC_WEIGHT_REUSE:
        optional.append("custom_deepseek_v41_mhc_control_reuse_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_MERGE_CACHE:
        optional.append("custom_deepseek_v41_logical_merge_cache_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_MHC_MME_EPILOGUE:
        optional.append("custom_deepseek_v41_mhc_mme_epilogue_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_UNPAIRED_W13:
        optional.append("custom_deepseek_v41_expert_n256_moe_unpaired_w13_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_CANDIDATE_KEYS:
        optional.append("custom_deepseek_v41_candidate_mirror_keys_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_SILU_AFFINE:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_silu_affine_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_SCALE_CACHE:
        optional.append("custom_deepseek_v41_logical_scale_cache_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_MAIN_MIRROR:
        optional.append("custom_deepseek_v41_logical_main_mirror_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_W13_UNROLL:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_unroll_steps_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_CHANNEL_SILU:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_channel_silu_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_COMPACT_KV_MME:
        optional.append("custom_deepseek_v41_logical_mla_compact_mme_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_GROUP_PIPELINE:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_group_pipe_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_SHARED_KV_MME:
        optional.append("custom_deepseek_v41_logical_mla_shared_mme_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_MLA_COORD_CACHE:
        optional.append("custom_deepseek_v41_logical_mla_coord_cached_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_HASH_MLA:
        optional.append("custom_deepseek_v41_logical_mla_hash_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_NUCLEUS_MASS:
        optional.append("custom_deepseek_v41_nucleus_mass_sample_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_QKV_PUBLISH:
        optional.append("custom_deepseek_v41_dspark_qkv_projection_publish_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_COHESIVE_MLA:
        optional.append("custom_deepseek_v41_logical_mla_cohesive_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_MERGED_MLA:
        optional.append("custom_deepseek_v41_logical_mla_merged_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_PV_ROPE:
        optional.append("custom_deepseek_v41_logical_mla_pv_rope_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_KV_PUBLISH:
        optional.append("custom_deepseek_v41_dspark_kv_norm_publish_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_PHYSICAL_SILU:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_physical_silu_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_SILU_UNROLL:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_silu_unroll_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE_FUSED_QUANT:
        optional.append("custom_deepseek_v41_hw_dense_roundtrip_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE:
        optional.append("custom_deepseek_v41_hw_dense_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_ROUTER_SHARED_FUSED:
        optional.extend(("custom_deepseek_v41_router_shared_scaled_gaudi2",
                         "custom_deepseek_v41_shared_silu_full_product_gaudi2"))
    if envs.VLLM_HPU_DSV41_DSPARK_ROPE_COHERENT:
        optional.extend(("custom_deepseek_v41_rope_coherent_bf16_gaudi2",
                         "custom_deepseek_v41_rope_inverse_coherent_bf16_gaudi2"))
    if (envs.VLLM_HPU_DSV41_DSPARK_PEER_POST_NORM or envs.VLLM_HPU_DSV41_DSPARK_SCHEDULED_PEER):
        optional.append("custom_deepseek_v41_dspark_peer_post_norm_quant_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_W2_THREE_ROUTES:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_three_route_w2_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_FEATURE_SILU:
        optional.append("custom_deepseek_v41_expert_n256_moe_token_wide_feature_silu_fp8_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_PARTITION_RADIX_SAMPLING:
        optional.append("custom_deepseek_v41_vocab_partition_topk_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_REPAIR:
        optional.append("custom_deepseek_v41_sampling_wire_view_gaudi2")
    if envs.VLLM_HPU_DSV41_DSPARK_STREAM_FILTER_SAMPLING:
        optional.append("custom_deepseek_v41_vocab_filter_gaudi2")
    if optional:
        from vllm_gaudi.ops.deepseek_v41_unique_experts import load_unique_expert_operators

        load_unique_expert_operators()
        required = (*required, *optional)
    if os.environ.get("VLLM_HPU_DSV41_NATIVE_MEMORY_READY", "0") == "1":
        directory = Path(library).parent
        manifest = json.loads((directory / "deepseek_v4_build.json").read_text())
        acquiring = directory / "dsv41_memory_ready.so"
        if (not acquiring.is_file() or hashlib.sha256(acquiring.read_bytes()).hexdigest()
                != manifest["binaries"].get(acquiring.name)):
            raise RuntimeError("Acquiring operator differs from its native build manifest")
        torch.ops.load_library(str(acquiring))
        required = (*required, "private_memory_ready_post", "private_memory_flags_zero")
    baseline = (
        "custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2",
        "custom_deepseek_v41_paged_attention_bf16_gaudi2",
    )
    names = tuple(dict.fromkeys((*baseline, *required)))
    missing = [name for name in names if not hasattr(torch.ops.custom_op, name)]
    if missing:
        raise RuntimeError(f"V4.1 native extension {library} is missing required operators: " + ", ".join(missing))
    return library


def prepare_environment(model=None, sidecars=None, tensor_parallel_size=4, pipeline_parallel_size=1):
    if model is not None:
        prepare_default_fastpaths(model, sidecars, tensor_parallel_size, pipeline_parallel_size)
    prepare_native_libraries()
    defaults = {
        "PT_HPU_LAZY_MODE": "0",
        "PT_HPU_ENABLE_LAZY_COLLECTIVES": "0",
        "PT_HPU_EAGER_PIPELINE_ENABLE": "1",
        "PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE": "1",
        "PT_HPU_ENABLE_EAGER_CACHE": "0",
        "PT_HPU_WEIGHT_SHARING": "0",
        "RUNTIME_SCALE_PATCHING": "0",
        "PT_HPU_POOL_MEM_ACQUIRE_PERC": "95",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        # Reserve scratch for the shared native graph path on every TP size.
        "VLLM_GRAPH_RESERVED_MEM": "0.1",
        "VLLM_HPU_FORCE_CHANNEL_FP8": "0",
        "OMP_NUM_THREADS": "1",
        # This is distinct from the API/engine drain timeout. Workers must be
        # allowed to export an active trace and retire their native resources.
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "60",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--settings", type=Path)
    installation, _ = bootstrap.parse_known_args()
    settings_path = installation.settings or Path.home() / ".config/1cat-vllm/deepseek-v41.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    leased_run = bool(os.environ.get("DSV41_RUN_EVIDENCE"))
    runtime_profile = os.environ.get("DSV41_RUNTIME_PROFILE") if leased_run else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, help="Use this installation instead of the user's default settings")
    parser.add_argument("model", nargs="?", default=settings.get("model"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-audit")
    parser.add_argument(
        "--n256-prepared-dir", type=Path, help="Runtime-layout expert cache from prepare_deepseek_v41_n256.py"
    )
    parser.add_argument("--runtime-profile", default=runtime_profile or settings.get("runtime_profile"))
    parser.add_argument("--max-model-len", type=int, default=1048576)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=settings.get("max_num_batched_tokens", 8192))
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(2, 4), default=4)
    parser.add_argument("--pipeline-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--engram-residency",
        choices=("prefetch", "locked"),
        default=settings.get("engram_residency", "prefetch"),
        help="Keep the shared host tables locked for the service lifetime",
    )
    parser.add_argument("--engram-host-budget-gib", type=int, default=224)
    v2 = parser.add_mutually_exclusive_group()
    v2.add_argument("--v2", dest="v2", action="store_true", help="Use the V2 HPU scheduling/completion adapter")
    v2.add_argument("--no-v2", dest="v2", action="store_false", help="Use the synchronous V4.1 model runner")
    parser.set_defaults(v2=None)
    args, extra = parser.parse_known_args()
    if args.v2 is not None:
        selected = "1" if args.v2 else "0"
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = selected
        os.environ["VLLM_HPU_DSV41_V2"] = selected
    if args.model is None:
        parser.error("A prepared model directory is required")
    if args.runtime_profile:
        profile_path = Path(args.runtime_profile).resolve()
        profile = json.loads(profile_path.read_text())
        identity = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        if os.environ.get("DSV41_SERVING_RUNTIME") != identity:
            for item in profile.get("additional_libraries", []) + profile.get("configuration_files", []):
                with Path(item["path"]).open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != item["sha256"]:
                    raise RuntimeError(f"Serving runtime fingerprint changed: {item['path']}")
            environment = dict(os.environ)
            environment.update(profile["environment"])
            # ``tools/run_deepseek_v41.py`` leases a concrete device set and
            # writes per-acquisition paths after loading the profile.  The
            # persistent user config contains defaults for an ordinary
            # install; letting it overwrite those launcher-owned values can
            # make the child try to lock a different (already busy) device
            # set.  Preserve only the values the launcher is responsible for
            # when its evidence marker is present; user settings continue to
            # override the version-locked profile for normal starts.
            launcher_keys = (
                "HABANA_VISIBLE_MODULES",
                "HLS_MODULE_ID",
                "HABANA_LOGS",
                # Candidate evidence runs build their native extension and
                # kernel database in an isolated directory.  Keep those
                # fingerprinted launcher selections across the runtime-profile
                # re-exec instead of silently restoring the profile's older
                # build.  prepare_native_libraries() validates both files
                # against the candidate build manifest before workers start.
                "GC_KERNEL_PATH",
                "VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR",
                "VLLM_HPU_DSV4_TPC_OP_LIBRARY",
                "VLLM_HPU_DSV4_WORKER_CPUS",
                "VLLM_HPU_DSV4_WORKER_HELPER_CPUS",
                "VLLM_HPU_TP2_PLAN_DUMP_DIR",
                "VLLM_TORCH_PROFILER_DIR",
                # Capture settings are selected after the base profile by
                # the launcher and must survive this re-exec as one unit.
                "HABANA_PROFILE",
                "HABANA_PROF_CONFIG",
                "HABANA_PROFILE_WRITE_HLTV",
                "VLLM_HPU_DSV41_RAW_TRACE",
                "VLLM_HPU_DSV41_RAW_SCOPE_ONLY",
                "ENABLE_PROFILER",
                "GRAPH_VISUALIZATION",
                "GRAPH_VISUALIZATION_DIR",
                "DSV41_RUN_EVIDENCE",
                "DSV41_RUNTIME_PROFILE",
                "PT_HPU_RECIPE_CACHE_CONFIG",
            )
            launcher_values = {key: os.environ[key] for key in launcher_keys if key in os.environ} if leased_run else {}
            environment.update(settings.get("environment", {}))
            # A leased evidence run must use the exact profile it was
            # fingerprinted against.  The persistent user runtime file may
            # contain an older direct-exchange ceiling (or bridge path), and
            # letting it overwrite the candidate makes workers fail before
            # model load.  Ordinary starts have no DSV41_RUN_EVIDENCE marker
            # and keep the existing user-settings precedence.
            if leased_run:
                environment.update(profile["environment"])
            environment.update(launcher_values)
            environment["DSV41_RUNTIME_PROFILE"] = str(profile_path)
            environment["DSV41_SERVING_RUNTIME"] = identity
            os.execve(
                sys.executable,
                [sys.executable, "-m", "vllm_gaudi.entrypoints.deepseek_v41", *sys.argv[1:]],
                environment,
            )
    # The evidence launcher already owns the module leases, NUMA affinity and
    # optional recipe cache. Acquiring them again here would either deadlock
    # on its locks or replace them with the ordinary installation defaults.
    if settings and not leased_run:
        from vllm_gaudi.entrypoints.serving_resources import prepare_serving_resources

        prepare_serving_resources(settings, args.model, extra)
    n256_directory = args.n256_prepared_dir or settings.get("n256_prepared_dir")
    if n256_directory:
        directory = Path(n256_directory).resolve()
        if not (directory / "manifest.json").is_file():
            raise ValueError("Runtime N256 preparation has not published a complete manifest")
        os.environ["VLLM_HPU_DSV41_N256_PREPARED_DIR"] = str(directory)
    prepare_environment(args.model, settings.get("sidecars"), args.tensor_parallel_size, args.pipeline_parallel_size)
    loader = {} if args.checkpoint_audit is None else {"checkpoint_audit": args.checkpoint_audit}
    if settings.get("engram_resident_tables"):
        loader["engram_resident_tables"] = settings["engram_resident_tables"]
    trace_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR")
    if trace_dir and not any(value.startswith("--profiler-config") for value in extra):
        # The engine registers /start_profile only from ProfilerConfig. The
        # legacy directory variable alone configures workers, not the API.
        extra += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": trace_dir,
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": True,
                }
            ),
        ]
    if not any(value.startswith("--shutdown-timeout") for value in extra):
        # Native programs and host staging need normal worker teardown. The
        # engine's zero-second default kills its child before cleanup runs.
        extra += ["--shutdown-timeout", "120"]
    if not any(value.startswith("--reasoning-parser") for value in extra):
        extra += ["--reasoning-parser", "deepseek_v41"]
    if not any(value.startswith("--served-model-name") for value in extra):
        extra += ["--served-model-name", "DeepSeek-V4.1-Flash"]
    from vllm_gaudi import envs as gaudi_envs

    speculative = (["--speculative-config", '{"method":"dspark","num_speculative_tokens":5}']
                   if gaudi_envs.VLLM_HPU_DSV41_DSPARK else [])
    scheduling = "--async-scheduling" if gaudi_envs.VLLM_HPU_DSV41_V2 else "--no-async-scheduling"
    prefix_caching = (
        []
        if any(value.split("=")[0] in ("--enable-prefix-caching", "--no-enable-prefix-caching") for value in extra)
        else [
            "--enable-prefix-caching" if settings.get("enable_prefix_caching", False) else "--no-enable-prefix-caching"
        ]
    )
    if "--enable-prefix-caching" in extra + prefix_caching and not any(
            value.split("=")[0] in ("--enable-prompt-tokens-details", "--no-enable-prompt-tokens-details")
            for value in extra):
        extra += ["--enable-prompt-tokens-details"]
    sys.argv = [
        "vllm",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.max_model_len),
        "--generation-config",
        "vllm",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(args.pipeline_parallel_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--load-format",
        "dsv41_prepared",
        "--model-loader-extra-config",
        json.dumps(loader),
        "--mm-encoder-tp-mode",
        "data",
        *prefix_caching,
        scheduling,
        "--block-size",
        str(args.block_size),
        *speculative,
        *extra,
    ]
    residency = None
    try:
        if args.engram_residency == "locked" and not loader.get("engram_resident_tables"):
            from vllm_gaudi.ops.deepseek_v41_residency import (
                EngramDeviceGate, EngramResidency, EngramStartup, table_regions,
            )

            device_layers = (1, ) if gaudi_envs.VLLM_HPU_DSV41_V2_DEVICE_ENGRAM else ()
            tables = EngramResidency(table_regions(args.model),
                                     args.engram_host_budget_gib * 1024**3,
                                     device_layers=device_layers)
            if device_layers:
                bridge = Path(os.environ["VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE"])
                abi = json.loads(bridge.with_suffix(".abi.json").read_text())
                if abi.get("device_engram_shared_mapping_version") != 1:
                    raise RuntimeError("Locked Device Engram requires a bridge with shared-host mapping support")
                report_path = Path(os.environ["DSV41_RUN_EVIDENCE"]) / "engram-residency.json" if leased_run else None
                residency = EngramStartup(tables, os.environ["HABANA_VISIBLE_MODULES"].split(","),
                                          report_path=report_path).start(lambda: os.kill(os.getpid(), signal.SIGTERM))
                loader["engram_startup_directory"] = str(residency.directory)
                sys.argv[sys.argv.index("--model-loader-extra-config") + 1] = json.dumps(loader)
            else:
                def report_ready(report):
                    if leased_run:
                        (Path(os.environ["DSV41_RUN_EVIDENCE"]) / "engram-residency.json").write_text(
                            json.dumps(report, indent=2)
                        )
                    print("Engram residency ready: " + json.dumps(report), flush=True)

                residency = EngramDeviceGate(tables, world_size=args.tensor_parallel_size * args.pipeline_parallel_size,
                                            on_ready=report_ready,
                                            on_failure=lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        from vllm.entrypoints.cli.main import main as serve

        serve()
    finally:
        if residency is not None:
            residency.close()


if __name__ == "__main__":
    main()
