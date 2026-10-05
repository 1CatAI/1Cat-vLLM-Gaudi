# SPDX-License-Identifier: Apache-2.0
"""Paged CSA2 with bounded working state and the checkpoint's two-level indexer.

Selection follows DeepSeek dba1be0 inference/model.py Indexer and
select_candidate_blocks. KV sharing follows vLLM #56214 e47aa780.
The default HPU path gathers compressed rows before decoding attention values.
The opt-in selected-row path decodes only requested rows in TPC.
"""

from functools import lru_cache
from types import FunctionType
import os

import torch
from vllm_gaudi.ops.deepseek_v41_prefill_regions import (
    prefill_main_workspace,
    prefill_q_projection,
    prefill_output_projection,
)
from vllm_gaudi.ops.deepseek_v41_native_trace import prefill_span
from vllm_gaudi.ops.deepseek_v41_prefill_event_trace import span as prefill_event_span
from vllm_gaudi.ops.deepseek_v41_prefill_sequence import can_partition_prefill_mla
import torch.nn.functional as F
from torch import nn

from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.ops.deepseek_v41_indexer import INDEX_MME_HOT_TOKENS
from vllm_gaudi.ops.deepseek_v41_qkv import FusedCompressorInput, FusedQKVInput
from vllm_gaudi.ops.deepseek_v41_trace import prefill_scope
from vllm_gaudi.ops.deepseek_v41_math import (
    _apply_rope_torch,
    fp4_roundtrip,
    pack_fp4,
    pack_swa,
    quantize_activation,
    rms_norm,
    rotary_table,
    unpack_fp4,
    unpack_swa,
)

PAGE_TOKENS = 128
WORK_TOKENS = 8192
PREFILL_ATTN_TOKENS = 128
PREFILL_INDEX_ROWS = 2048
# Decode the reachable packed main cache once while it is still a bounded
# large-M prefill workspace.  Beyond this prefix, retain the selected-row
# tiled path so a 1M request never creates a 1 GiB BF16 layer temporary.
PREFILL_MAIN_CACHE_ROWS = 65536
NATIVE_WORK_TOKENS = 6
SWA_ROWS = 256
HISTORY_ROWS = 8
INDEX_QUERY_TP_MIN_TOKENS = 1024


def can_partition_prefill_index(tokens, source_rows):
    """Admit query tiles independently of the number of source windows."""
    return tokens >= INDEX_QUERY_TP_MIN_TOKENS and source_rows > 512


def candidate_columns(search_length, ratio, capacity):
    """Bound the source layer's non-padding candidate prefix.

    Full selection emits at most ceil(source_rows/8) blocks, then appends -1
    up to capacity. Keep its original ordering, including holes within this
    prefix; only the provably appended padding can be omitted by Reindex.
    """
    return min(capacity, (search_length // ratio + 7) // 8)


def bounded_prefill_mla(query, cache, indices, sink, scale, tile):
    """Submit bounded real query extents to the large-M MME attention op."""
    if tile not in (16, 32, 64):
        raise ValueError("Prefill MLA query tile must be 16, 32 or 64")
    outputs = []
    for start in range(0, query.shape[0], tile):
        count = min(tile, query.shape[0] - start)
        # The native operation accepts a partial final query tile. Padding a
        # one-row slice to a full tile lets Synapse propagate an invalid
        # 16-row bundle slice back into the one-row producer. Preserve the
        # real extent instead; no padded query can write request state.
        q = query[start : start + count].contiguous()
        ids = indices[start : start + count].contiguous()
        lengths = torch.full((count,), indices.shape[-1], dtype=torch.int32, device=query.device)
        output = torch.ops.custom_op.custom_deepseek_v41_prefill_mla_mme_gaudi2(q, cache, ids, sink, scale, lengths)
        outputs.append(output)
    return torch.cat(outputs, 0)


@lru_cache(maxsize=32)
def compiled_prefill_mla(signature):
    entry = FunctionType(
        bounded_prefill_mla.__code__.replace(co_name=f"prefill_mla_{signature}"), bounded_prefill_mla.__globals__
    )
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def flash_prefill_mla(query, cache, indices, sink):
    """Compile bounded KV gathers and BF16 MLA with an FP32 sink together."""
    from vllm_gaudi.ops.deepseek_v41_prefill_mla import sparse_prefill_mla

    lengths = torch.full((query.shape[0],), indices.shape[1], dtype=torch.int32, device=query.device)
    return sparse_prefill_mla(query, cache, indices, sink, lengths, query_tile=1024)


@lru_cache(maxsize=32)
def compiled_flash_prefill_mla(signature):
    entry = FunctionType(
        flash_prefill_mla.__code__.replace(co_name=f"flash_prefill_mla_{signature}"), flash_prefill_mla.__globals__
    )
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def _selected_attention_layout(physical, selected, window, swa_offsets, selected_offsets):
    """Build the direct-TPC row list and local attention indices.

    The row list contains each SWA row once, followed by token-major selected
    main rows. Attention indices point into this list and preserve ``-1`` for
    invalid windows/selections.
    """
    row_ids = (
        torch.cat(
            (
                swa_offsets,
                physical.reshape(-1).clamp_min(0).to(torch.int32) + swa_offsets.numel(),
            ),
            0,
        )
        .reshape(1, -1)
        .contiguous()
    )
    offset = selected_offsets[: selected.numel()].reshape(selected.shape)
    attention_indices = torch.cat(
        (
            window,
            torch.where(selected >= 0, offset + swa_offsets.numel(), -1),
        ),
        -1,
    ).contiguous()
    return row_ids, attention_indices


def _shared_prefix_attention_layout(physical, selected, window, swa_offsets):
    """Reuse decoded rows for the nested prefixes of an ordered C1–C6 call.

    Only the non-ranking bucket may use this layout: valid selected column i
    is logical row i, and consecutive positions make the last prefix contain
    every earlier query's rows. Full/Reindex/Reuse still retain their own
    selection and state dependencies; ranking buckets use the general layout.
    """
    base = swa_offsets.numel()
    main = torch.where(selected[-1] >= 0, physical[-1].to(torch.int32) + base, -1)
    row_ids = torch.cat((swa_offsets, main)).reshape(1, -1).contiguous()
    indices = torch.cat((window, torch.where(selected >= 0, selected + base, -1)), -1).int().contiguous()
    lengths = (window.shape[-1] + (selected >= 0).sum(-1, dtype=torch.int32)).contiguous()
    return row_ids, indices, lengths


class PagedCSA2SharedState(nn.Module):
    def __init__(
        self, config, layer_start, layer_stop, device, max_length, *, prefill_tokens=WORK_TOKENS, tensor_parallel_size=2
    ):
        super().__init__()
        from vllm_gaudi.ops.deepseek_v41_prefill_capacity import MAX_PREFILL_TOKENS

        if not 1 <= prefill_tokens <= MAX_PREFILL_TOKENS:
            raise ValueError("Paged prefill capacity must be between 1 and 16384")
        self.prefill_tokens = prefill_tokens
        self.rotary_on_host = tensor_parallel_size == 4
        self.length, self.layer_start, self.layer_stop = max_length, layer_start, layer_stop
        self.runtime_indexer = gaudi_envs.VLLM_HPU_DSV41_RUNTIME_INDEXER
        if self.runtime_indexer and not hasattr(torch.ops.custom_op, "custom_deepseek_v41_index_scores_gaudi2"):
            raise RuntimeError("Runtime CSA2 indexer requires the native score/select operators")
        # A view of the 1M-row source still exposes its complete backing
        # allocation to Synapse when the native kernel declares all input rows
        # required.  Lazily materialize independent, stage-shared page buckets
        # instead.  Compiled recipes bind these stable tensor addresses, so the
        # cache intentionally keeps strong references for the process lifetime.
        self._rotary_buckets = {}
        self.decoded_kv_state = gaudi_envs.VLLM_HPU_DSV41_PAGED_DECODED_KV_STATE
        self.sources, self.topk = nn.ModuleDict(), nn.ModuleDict()
        self.prefill_kv_generation = 0
        self.prefill_main_workspace = None
        if gaudi_envs.VLLM_HPU_DSV41_PREFILL_KV_REUSE:
            from vllm_gaudi.ops.deepseek_v41_prefill_kv_reuse import PrefillMainWorkspace

            self.prefill_main_workspace = PrefillMainWorkspace()
        for source in config["kv_source_layer_ids"]:
            if layer_start <= source < layer_stop:
                ratio = config["compress_ratios"][source]
                cache = nn.Module()
                cache.ratio = ratio
                cache.register_buffer(
                    "main", torch.zeros(2 * PAGE_TOKENS // ratio, 288, dtype=torch.uint8, device=device), False
                )
                cache.register_buffer(
                    "index", torch.zeros(2 * PAGE_TOKENS // ratio, 68, dtype=torch.uint8, device=device), False
                )
                if self.runtime_indexer and ratio in (1, 2):
                    cache.register_buffer(
                        "decoded_index_hot",
                        torch.zeros(INDEX_MME_HOT_TOKENS // ratio, 128, dtype=torch.bfloat16, device=device),
                        False,
                    )
                if self.decoded_kv_state:
                    # Keep the exact FP4-roundtripped main rows decoded for
                    # the bounded C1 hot bucket.  The packed paged cache
                    # remains authoritative for the complete 1M context;
                    # this small mirror only extends the proven decoded-MLA
                    # path beyond the old 512-row special case.
                    decoded_rows = INDEX_MME_HOT_TOKENS // ratio if self.runtime_indexer else 512
                    cache.register_buffer(
                        "decoded_main", torch.zeros(decoded_rows, 512, dtype=torch.bfloat16, device=device), False
                    )
                self.sources[str(source)] = cache
        shared_selection = None
        for source in config["index_source_layer_ids"]:
            if layer_start <= source < layer_stop:
                selection = nn.Module()
                if shared_selection is None or tensor_parallel_size != 4:
                    shared_selection = torch.full((self.prefill_tokens, 512), -1, dtype=torch.int32, device=device)
                # Each owner publishes before its contiguous consumer layers.
                # The next owner replaces these ephemeral IDs only after those
                # consumers finish on the same model stream. Candidate blocks
                # remain separate because Reindex owners reuse them later.
                selection.register_buffer("indices", shared_selection, False)
                self.topk[str(source)] = selection
        self.register_buffer(
            "candidate_pool",
            # Keep candidate block ids rather than expanding every block to its
            # eight rows.  Reindex layers expand only the row tile they score.
            # This is equivalent to the upstream candidate-slot contract and
            # cuts the persistent C8192 workspace from 512 MiB to 64 MiB.
            torch.full((self.prefill_tokens, config["candidate_topk_blocks"]), -1, dtype=torch.int32, device=device)
            if layer_start <= config["candidate_source_layer_id"] < layer_stop
            else None,
            False,
        )
        table = torch.zeros((max_length + PAGE_TOKENS - 1) // PAGE_TOKENS, dtype=torch.int32, device=device)
        table[0] = 1
        self.register_buffer("block_table", table, False)
        if self.runtime_indexer and self.candidate_pool is None:
            # Full layers do not consume candidate IDs, but the fixed native
            # ABI still requires one persistent correctly shaped input.
            self.register_buffer(
                "index_candidates_unused",
                torch.full((NATIVE_WORK_TOKENS, config["candidate_topk_blocks"]), -1, dtype=torch.int32, device=device),
                False,
            )
        if self.decoded_kv_state:
            self.register_buffer(
                "decoded_swa",
                torch.zeros((layer_stop - layer_start) * 512, 512, dtype=torch.bfloat16, device=device),
                False,
            )
        scaling = config["rope_scaling"]
        for name, compressed in (("swa_rotary", False), ("compressed_rotary", True)):
            table = rotary_table(
                config["qk_rope_head_dim"],
                max_length,
                config["compress_rope_theta"] if compressed else config["rope_theta"],
                scaling["original_max_position_embeddings"] if compressed else 0,
                scaling["factor"],
                scaling["beta_fast"],
                scaling["beta_slow"],
            )
            if self.rotary_on_host:
                # Immutable full-context masters are ordinary CPU attributes,
                # so module migration cannot upload them accidentally. Active
                # buckets retain independent, stable device storage below.
                setattr(self, name, table)
            else:
                self.register_buffer(name, table.to(device), False)
            # Native RoPE needs [cos32, sin32], but a complete 1M-position
            # copy costs another 256 MiB for each table on every rank.  The
            # active search bucket is at most the currently reachable prefix;
            # materialize that correctly laid-out bucket lazily below.

        from vllm_gaudi.ops.deepseek_v41_index_mirror import initialize_index_mirror

        initialize_index_mirror(self, tensor_parallel_size, device)

    def invalidate_index_mirror(self):
        self.index_mirror_valid = False

    def prepare_index_mirror(self, visible_tokens):
        from vllm_gaudi.ops.deepseek_v41_index_mirror import prepare_index_mirror

        prepare_index_mirror(self, visible_tokens)

    def rotary_bucket(self, name, length):
        """Return a stable RoPE buffer whose backing storage is bucket-sized."""
        if length < 1 or length > self.length:
            raise ValueError(f"invalid RoPE bucket length {length} for maximum {self.length}")
        key = (name, int(length))
        bucket = self._rotary_buckets.get(key)
        if bucket is None:
            if name.endswith("_native"):
                # The TPC RoPE kernel consumes [cos0..cos31,sin0..sin31].
                # The checkpoint/reference table is adjacent-pair interleaved
                # [cos0,sin0,...].  Build the native layout once per bucket;
                # passing a reshaped interleaved table silently rotates every
                # pair with the wrong phase in the paged 1M path.
                source = getattr(self, name.removesuffix("_native"))[:length]
                bucket = torch.cat((source[..., 0], source[..., 1]), -1).contiguous()
                if getattr(self, "rotary_on_host", False):
                    bucket = bucket.to(self.block_table.device)
            else:
                # ``clone`` is intentional: contiguous slices still share the full
                # table allocation and reproduce the short-request compile stall.
                source = getattr(self, name)
                if getattr(self, "rotary_on_host", False):
                    bucket = source[:length].to(self.block_table.device, copy=True)
                else:
                    bucket = source if self.runtime_indexer and length == self.length else source[:length].clone()
            self._rotary_buckets[key] = bucket
        return bucket

    def _apply(self, fn, recurse=True):
        if getattr(self, "rotary_on_host", False):
            self._rotary_buckets.clear()
        result = super()._apply(fn, recurse=recurse)
        if getattr(self, "rotary_on_host", False):
            # Module._apply transforms child buffers independently. Restore
            # the shared scratch identity after a real device migration.
            shared = None
            for selection in self.topk.values():
                if shared is None:
                    shared = selection.indices
                else:
                    selection.indices = shared
        return result

    def physical_rows(self, rows, ratio):
        width = PAGE_TOKENS // ratio
        if width & (width - 1):
            raise ValueError("V4.1 paged rows require a power-of-two page width")
        shift = width.bit_length() - 1
        # V4.1 uses ratio 1/2 over a 128-token page.  Division and remainder
        # by these compile-time powers of two were nevertheless lowered to a
        # generic div_mod TPC launch at every CSA2 source layer.  Preserve the
        # exact non-negative row mapping with shifts/masks so the state chain
        # does not pay integer division on each decoded token.
        blocks = self.block_table.index_select(0, torch.bitwise_right_shift(rows.flatten(), shift).long()).reshape(
            rows.shape
        )
        return blocks * width + torch.bitwise_and(rows, width - 1)


class PagedCSA2Attention(FusedCompressorInput, FusedQKVInput, nn.Module):
    def __init__(self, weights, config, layer, shared, linear, reduce, gather, device, tensor_parallel_size=2):
        super().__init__()
        self.weights, self.shared = weights, shared
        if (
            gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_MME or gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_SHARED
        ) and not hasattr(torch.ops.custom_op, "custom_deepseek_v41_index_keys_gaudi2"):
            raise RuntimeError("Prefill index MME requires the native index-key decoder at model initialization")
        self.woa_fp8 = False
        # Bound by PreparedStage after the FP8 sidecars have been loaded.  The
        # paged and bounded attention implementations share the same MLA
        # output contract, so long-context decode must retain the qualified
        # wo_a -> group32 BF16 boundary -> FP8 wo_b chain as well.
        self.woa_output_roundtrip = False
        self.prepared_output = gaudi_envs.VLLM_HPU_DSV41_PREPARED_OUTPUT and layer < config["num_hidden_layers"]
        self.output_gemm_layout = gaudi_envs.VLLM_HPU_DSV41_OUTPUT_GEMM_LAYOUT and layer < config["num_hidden_layers"]
        if self.output_gemm_layout and not self.prepared_output:
            raise ValueError("V4.1 output GEMM layout requires prepared output weights")
        self.qkv_fused_input = gaudi_envs.VLLM_HPU_DSV41_QKV_FUSED_INPUT and layer < config["num_hidden_layers"]
        self._fused_qkv_weight = None
        self._fused_qkv_quantized = False
        self.compressor_fused_input = (
            gaudi_envs.VLLM_HPU_DSV41_COMPRESSOR_FUSED_INPUT and layer < config["num_hidden_layers"]
        )
        self._fused_compressor_weight = None
        self._fused_compressor_kv_width = 0
        self.mla_mme = gaudi_envs.VLLM_HPU_DSV41_MLA_MME and layer < config["num_hidden_layers"]
        self.native_rope = gaudi_envs.VLLM_HPU_DSV41_NATIVE_ROPE
        self.q_scale_rope = gaudi_envs.VLLM_HPU_DSV41_Q_SCALE_ROPE and layer < config["num_hidden_layers"]
        self.batch_c1_numerics = gaudi_envs.VLLM_HPU_DSV41_BATCH_C1_NUMERICS
        if self.q_scale_rope and not (
            gaudi_envs.VLLM_HPU_DSV41_NATIVE_ROPE and gaudi_envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8
        ):
            raise ValueError("Q scale/RoPE fusion requires native RoPE and prepared dense FP8")
        self.fused_norm = gaudi_envs.VLLM_HPU_DSV41_ATTN_FUSED_NORM and layer < config["num_hidden_layers"]
        self.linear, self.reduce, self.gather = linear, reduce, gather
        self.layer, self.ratio, self.length = layer, config["compress_ratios"][layer], shared.length
        self.runtime_indexer = getattr(shared, "runtime_indexer", False)
        self.direct_selected_kv = gaudi_envs.VLLM_HPU_DSV41_PAGED_SELECTED_KV
        self.shared_prefix_kv = gaudi_envs.VLLM_HPU_DSV41_SHARED_PREFIX_KV
        self.packed_attn_exp = gaudi_envs.VLLM_HPU_DSV41_PACKED_ATTN_EXP
        self.vector_kv_scales = gaudi_envs.VLLM_HPU_DSV41_VECTOR_KV_SCALES
        self.head_vector_attn = gaudi_envs.VLLM_HPU_DSV41_HEAD_VECTOR_ATTN
        self.sram_kv = gaudi_envs.VLLM_HPU_DSV41_SRAM_KV
        if (
            tensor_parallel_size < 1
            or config["num_attention_heads"] % tensor_parallel_size
            or config["o_groups"] % tensor_parallel_size
            or config["index_n_heads"] % tensor_parallel_size
        ):
            raise ValueError("V4.1 paged attention head geometry is not divisible by the runtime TP size")
        self.tensor_parallel_size = tensor_parallel_size
        self.shared_decode_metadata = tensor_parallel_size == 4
        self.paged_mla_direct = tensor_parallel_size == 4 and hasattr(
            torch.ops.custom_op, "custom_deepseek_v41_paged_mla_direct_mme_gaudi2"
        )
        self.paged_mla_logical = tensor_parallel_size == 4 and hasattr(
            torch.ops.custom_op, "custom_deepseek_v41_logical_mla_gaudi2"
        )
        self.shared_main_mla = tensor_parallel_size == 4 and all(
            hasattr(torch.ops.custom_op, name)
            for name in ("custom_deepseek_v41_main_publish_mla_gaudi2", "custom_deepseek_v41_main_reuse_mla_gaudi2")
        )
        self.index_mirror_scores = bool(getattr(shared, "index_mirror_tokens", 0))
        self.runtime_index_mme = hasattr(torch.ops.custom_op, "custom_deepseek_v41_prefill_paged_index_scores_gaudi2")
        # Kept unselected until the distributed selection consumer gate passes.
        self.tp4_tile_selection = False
        # Distributed mirror selection regresses the complete decode chain.
        self.tp4_mirror_selection = False
        self.tp4_local_index_queries = False
        self.tp4_packed_index_queries = False
        for shard in range(4):
            self.register_buffer(f"_tp4_index_query_shard_{shard}", None, False)
            self.register_buffer(f"_tp4_index_score_shard_{shard}", None, False)
        self.search_length = 512
        self.decode_visible_rows = None
        self.heads, self.groups = (
            config["num_attention_heads"] // tensor_parallel_size,
            config["o_groups"] // tensor_parallel_size,
        )
        self.index_heads = config["index_n_heads"] // tensor_parallel_size
        self.eps, self.window = config["rms_norm_eps"], config["sliding_window"]
        self.owns_kv = layer in config["kv_source_layer_ids"]
        self.owns_index = layer in config["index_source_layer_ids"]
        self.candidate_source = config["candidate_source_layer_id"]
        self.batch_reindex_group = (
            bool(self.ratio)
            and max(source for source in config["index_source_layer_ids"] if source <= layer) > self.candidate_source
        )
        self.batch_reindex_mme = gaudi_envs.VLLM_HPU_DSV41_BATCH_REINDEX_MME
        self.batch_full_index_mme = gaudi_envs.VLLM_HPU_DSV41_BATCH_FULL_INDEX_MME
        self.batch_packed_mla = gaudi_envs.VLLM_HPU_DSV41_BATCH_PACKED_MLA
        self.batch_packed_mla_sram = gaudi_envs.VLLM_HPU_DSV41_BATCH_PACKED_MLA_SRAM
        self.batch_packed_mla_vector = gaudi_envs.VLLM_HPU_DSV41_BATCH_PACKED_MLA_VECTOR
        if self.batch_packed_mla_vector and not self.batch_packed_mla_sram:
            raise ValueError("Vector packed MLA requires SRAM placement")
        if self.batch_packed_mla_vector and not hasattr(
            torch.ops.custom_op, "custom_deepseek_v41_batch_packed_vector_mla_mme_gaudi2"
        ):
            raise RuntimeError("Vector packed MLA requires the matching native library")
        if self.batch_packed_mla_sram and not self.batch_packed_mla:
            raise ValueError("SRAM packed MLA requires its fused decoder")
        if self.batch_packed_mla_sram and not hasattr(
            torch.ops.custom_op, "custom_deepseek_v41_batch_packed_sram_mla_mme_gaudi2"
        ):
            raise RuntimeError("SRAM packed MLA requires the matching native library")
        if self.batch_packed_mla and not hasattr(
            torch.ops.custom_op, "custom_deepseek_v41_batch_packed_mla_mme_gaudi2"
        ):
            raise RuntimeError("Batch packed MLA requires the fused native decoder library")
        self.batch_index_tiled_keys = (
            gaudi_envs.VLLM_HPU_DSV41_BATCH_INDEX_TILED_KEYS and self.layer > self.candidate_source
        )
        self.batch_compressor_pair = gaudi_envs.VLLM_HPU_DSV41_BATCH_COMPRESSOR_PAIR
        self.batch_compressor_gather = gaudi_envs.VLLM_HPU_DSV41_BATCH_COMPRESSOR_GATHER
        if self.batch_compressor_pair and self.batch_compressor_gather:
            raise ValueError("Select only one request-slot compressor candidate")
        if self.batch_compressor_gather and not hasattr(
            torch.ops.custom_op, "custom_deepseek_v41_compressor_batch_gather_f32_gaudi2"
        ):
            raise RuntimeError("Batch compressor gather requires the request-slot native library")
        if self.batch_compressor_pair and not hasattr(
            torch.ops.custom_op, "custom_deepseek_v41_compressor_batch_bf16_gaudi2"
        ):
            raise RuntimeError("Batch compressor fusion requires the request-slot native pair kernel")
        self.bounded_reindex = gaudi_envs.VLLM_HPU_DSV41_REINDEX_BOUNDED_PLAN
        if self.bounded_reindex and (
            not self.batch_reindex_mme
            or not gaudi_envs.VLLM_HPU_DSV41_TP_MHC_OVERLAP
            or gaudi_envs.VLLM_HPU_DSV41_DSPARK
        ):
            raise ValueError("Bounded Reindex requires ordinary batch MME and explicit native TP dependencies")
        if (
            self.bounded_reindex
            and self.batch_index_tiled_keys
            and not hasattr(torch.ops.custom_op, "custom_deepseek_v41_reindex_tiled_gaudi2")
        ):
            raise RuntimeError("Bounded Reindex SRAM keys require the matching native compound library")
        if self.batch_reindex_mme and not hasattr(torch.ops.custom_op, "custom_deepseek_v41_index_reduce_gaudi2"):
            raise RuntimeError("Batch Reindex MME requires the native ordered index head reducer")
        self.decoded_kv_state = shared.decoded_kv_state
        self.decoded_swa_offset = (layer - shared.layer_start) * 512 if self.decoded_kv_state else 0
        if self.decoded_kv_state and not (self.mla_mme and self.direct_selected_kv and self.shared_prefix_kv):
            raise ValueError("Paged decoded KV requires direct selected rows, shared prefixes and MME MLA")
        self.candidate_source = config["candidate_source_layer_id"]
        # Both selection variants are retained for diagnosis, not enabled:
        # carried-ID sorting regressed; score-only batching has no resolved gain.
        self.decode_batched_selection = False
        self.decode_swa_packed = not self.ratio
        if self.ratio:
            kv_source = max(source for source in config["kv_source_layer_ids"] if source <= layer)
            self.kv_source = kv_source
            index_source = max(source for source in config["index_source_layer_ids"] if source <= layer)
            self.index_source = index_source
            self.index_ratio = config["compress_ratios"][index_source]
            self.cache = shared.sources[str(kv_source)]
            self.selection = shared.topk[str(index_source)]
        self.register_buffer("swa", torch.zeros(SWA_ROWS, 528, dtype=torch.uint8, device=device), False)
        self.register_buffer("_prefill_sequence_sink", None, persistent=False)
        if self.owns_kv and self.ratio == 2:
            self.register_buffer(
                "kv_history", torch.zeros(HISTORY_ROWS, 512, dtype=torch.float32, device=device), False
            )
            self.register_buffer("score_history", torch.zeros_like(self.kv_history), False)
        self._rotary_name = "compressed_rotary" if self.ratio else "swa_rotary"
        self.register_buffer("rotary", shared.rotary_bucket(self._rotary_name, self.search_length), False)
        if self.native_rope or self.q_scale_rope:
            self._rotary_native_name = f"{self._rotary_name}_native"
            self.register_buffer(
                "rotary_native", shared.rotary_bucket(self._rotary_native_name, self.search_length), False
            )
        self.register_buffer("window_offsets", torch.arange(self.window, dtype=torch.int32, device=device), False)
        self.register_buffer("compressed_offsets", torch.arange(512, dtype=torch.int32, device=device), False)
        self.register_buffer("swa_offsets", torch.arange(SWA_ROWS, dtype=torch.int32, device=device), False)
        if not self.ratio:
            # The packed consumer never reads main rows for a SWA-only layer.
            # Retain the ordinary nonempty main operand contract without
            # allocating a compressed cache or building constants per token.
            self.register_buffer("swa_only_main", torch.zeros(1, 288, dtype=torch.uint8, device=device), False)
            self.register_buffer(
                "swa_only_lengths", torch.full((NATIVE_WORK_TOKENS,), self.window,
                                               dtype=torch.int32, device=device), False
            )
        self.register_buffer(
            "selected_offsets", torch.arange(NATIVE_WORK_TOKENS * 512, dtype=torch.int32, device=device), False
        )
        self.register_buffer("scale", torch.tensor([512**-0.5], dtype=torch.float32, device=device), False)

    def prepare_output_weight(self):
        """Bind the same persistent wo_a layouts used by bounded decode.

        Paged CSA2 changes only KV ownership and candidate selection.  The
        attention result and wo_a contract are identical, so keeping the
        checkpoint layout here would otherwise reintroduce a per-layer
        transpose and would make the prepared FP8 sidecar unreachable.
        """
        if self.woa_fp8:
            return
        if self.prepared_output:
            weight = self.weights.wo_a.weight
            if weight.dtype != torch.bfloat16 or weight.numel() != self.heads * 512 * 1024:
                raise ValueError("V4.1 output weight contract changed")
            grouped = weight.reshape(self.groups, 1024, -1)
            self.weights.wo_a.weight = grouped if self.output_gemm_layout else grouped.transpose(1, 2).contiguous()

    def prepare_tp4_index_query_weights(self):
        """Keep small immutable query shards local across normal TP4 decode."""
        if self.tensor_parallel_size == 4 and self.owns_index:
            from vllm_gaudi.ops.deepseek_v41_tp4_selection import prepare_local_index_queries

            prepare_local_index_queries(self, release_local=True)

    def invalidate_tp4_index_query_weights(self):
        from vllm_gaudi.ops.deepseek_v41_tp4_selection import invalidate_local_index_queries

        invalidate_local_index_queries(self)

    def _rotary_table(self):
        return self.rotary

    def _rotary_native_table(self):
        return self.rotary_native

    def set_search_length(self, length):
        """Bind this layer to the shared independent-storage RoPE bucket."""
        self.search_length = int(length)
        # Runtime-position RoPE reads a sparse row of one persistent table;
        # prompt/decode transitions do not replace captured table addresses.
        rotary_length = self.length if getattr(self, "runtime_indexer", False) else self.search_length
        self.rotary = self.shared.rotary_bucket(self._rotary_name, rotary_length)
        if self.native_rope or self.q_scale_rope:
            self.rotary_native = self.shared.rotary_bucket(self._rotary_native_name, rotary_length)

    def set_decode_visible_tokens(self, token_end):
        """Prune future source tiles while retaining candidate-block merge order."""
        from vllm_gaudi.ops.deepseek_v41_config import decode_source_prefix_bound

        self.decode_visible_rows = None
        if token_end is not None and self.ratio and self.layer <= self.candidate_source:
            bound = decode_source_prefix_bound(
                token_end, self.search_length, self.tensor_parallel_size, runtime_indexer=self.runtime_indexer
            )
            if bound is not None:
                self.decode_visible_rows = bound // self.ratio

    def _rope(self, value, positions, inverse=False, *, request_batch=False):
        # Ordinary concurrent decode must keep the C1 FP32 multiply/FMA
        # sequence before BF16 rounding, including inverse RoPE before WO.
        maximum_rows = 64 if request_batch else NATIVE_WORK_TOKENS
        if (
            not request_batch
            and self.native_rope
            and gaudi_envs.VLLM_HPU_DSV41_PREFILL_ROPE
            and value.dtype == torch.bfloat16
            and value.ndim in (2, 3)
            and (value.ndim == 2 or 1 <= value.shape[1] <= 128)
            and NATIVE_WORK_TOKENS < value.shape[0] <= 16384
            and 128 <= value.shape[-1] <= 512
            and value.shape[-1] % 128 == 0
        ):
            # Large-query eager RoPE rounds separate FP32 products. The small
            # compiled path has a different FMA boundary and keeps its GUID.
            op = (
                torch.ops.custom_op.custom_deepseek_v41_prefill_rope_inverse_bf16_gaudi2
                if inverse
                else torch.ops.custom_op.custom_deepseek_v41_prefill_rope_bf16_gaudi2
            )
            shaped = value.reshape(value.shape[0], -1, value.shape[-1]).contiguous()
            return op(shaped, positions.to(torch.int32).contiguous(), self._rotary_native_table()).reshape(value.shape)
        if (
            self.native_rope
            and value.dtype == torch.bfloat16
            and value.ndim in (2, 3)
            and (value.ndim == 2 or 1 <= value.shape[1] <= 128)
            and 1 <= value.shape[0] <= maximum_rows
            and 128 <= value.shape[-1] <= 512
            and value.shape[-1] % 128 == 0
        ):
            op = (
                torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2
                if inverse
                else torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2
            )
            shaped = value.reshape(value.shape[0], -1, value.shape[-1]).contiguous()
            return op(shaped, positions.to(torch.int32).contiguous(), self._rotary_native_table()).reshape(value.shape)
        return _apply_rope_torch(value, positions, self._rotary_table(), inverse)

    def project_output(self, value):
        """Project MLA output without restoring a bounded-context weight path."""
        if self.woa_fp8:
            operation = (
                torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2
                if self.woa_output_roundtrip
                else torch.ops.custom_op.custom_deepseek_v41_woa_fp8_gaudi2
            )
            if self.woa_output_roundtrip and gaudi_envs.VLLM_HPU_DSV41_PREFILL_VECTOR_QUANT and value.shape[0] >= 16:
                operation = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_wide_gaudi2
            return operation(value.contiguous(), self.weights.wo_a.weight, self.weights.wo_a.channel_scale)
        if self.output_gemm_layout:
            weight = self.weights.wo_a.weight
            if value.shape[0] == 1:
                return torch.cat(
                    tuple(F.linear(value[:, group], weight[group]) for group in range(self.groups)), dim=-1
                )
            return torch.einsum("tgd,grd->tgr", value, weight).flatten(1)
        if self.prepared_output:
            return torch.einsum("tgd,gdr->tgr", value, self.weights.wo_a.weight).flatten(1)
        weight = self.weights.wo_a.weight.reshape(self.groups, 1024, -1)
        return torch.einsum("tgd,grd->tgr", value, weight).flatten(1)

    def project_output_consumer(self, value):
        """Consume the qualified wo_a roundtrip without a BF16 HBM detour."""
        if self.woa_output_roundtrip:
            weight = self.weights.wo_b
            return torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(
                value.contiguous(), weight.weight, weight.channel_scale
            )
        return self.linear(value, self.weights.wo_b)

    def project_query(self, value, positions, *, decode=False, request_batch=False):
        weight = self.weights.wq_b
        if not decode and gaudi_envs.VLLM_HPU_DSV41_PREFILL_Q_PROJECTION and 6 < value.shape[0] <= 16384:
            if not getattr(weight, "dense_fp8", False) or self.heads not in (16, 32):
                raise ValueError("Prefill Q projection requires prepared FP8 weights and 16/32 local heads")
            with prefill_event_span("attention_query_quantization", self.layer, value.shape[0]):
                value = quantize_activation(value) if hasattr(weight, "scale") else value
            with prefill_event_span("attention_query_projection_rope", self.layer, value.shape[0]):
                return prefill_q_projection(
                    value.contiguous(),
                    weight.weight,
                    weight.channel_scale,
                    positions.to(torch.int32).contiguous(),
                    self._rotary_native_table(),
                )
        if self.q_scale_rope and value.shape[0] == 1 and getattr(weight, "dense_fp8", False):
            value = quantize_activation(value) if hasattr(weight, "scale") else value
            return torch.ops.custom_op.custom_deepseek_v41_q_projection_rope_gaudi2(
                value, weight.weight, weight.channel_scale, positions, self._rotary_native_table()
            ).reshape(-1, self.heads, 512)
        return self._rope(
            self.linear(value, weight).reshape(-1, self.heads, 512), positions, request_batch=request_batch
        )

    def project_query_input(self, value, positions):
        """Apply Q norm directly in the FP8 projection producer.

        The normalized query row is otherwise written and reread solely by
        the dynamic quantizer. Reindex owner layers still materialize it
        because their index-query projection is a second real consumer.
        Request batches can share this row-independent producer; each row
        carries its own scale and absolute rotary position.
        """
        weight = self.weights.wq_b
        return torch.ops.custom_op.custom_deepseek_v41_q_norm_projection_rope_gaudi2(
            value.contiguous(),
            self.weights.q_norm.weight,
            weight.weight,
            weight.channel_scale,
            positions.to(torch.int32).contiguous(),
            self._rotary_native_table(),
            self.eps,
        ).reshape(-1, self.heads, 512)

    def project_kv(self, value, positions, *, decode=False):
        """Normalize and rotate KV without materializing the C1 norm output."""
        weight = self.weights.kv_norm.weight
        if decode and self.fused_norm and self.native_rope and value.shape[0] == 1:
            # KV has no consumer between its BF16 RMSNorm boundary and RoPE.
            # The compound TPC kernel preserves that boundary bit-for-bit and
            # feeds the existing cache writer directly.  Wider batches retain
            # the batch-generic implementation used by prefill and B2+.
            return torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_bf16_gaudi2(
                value.contiguous(),
                weight,
                positions.to(torch.int32).contiguous(),
                self._rotary_native_table(),
                self.eps,
            )
        norm = (
            torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2
            if self.fused_norm
            and (
                value.shape[0] == 1
                or (
                    not decode
                    and gaudi_envs.VLLM_HPU_DSV41_PREFILL_NATIVE_NORM
                    and NATIVE_WORK_TOKENS < value.shape[0] <= 16384
                )
            )
            else rms_norm
        )
        return self._rope(norm(value.contiguous(), weight, self.eps), positions)

    def _uses_fused_compressor_publish(self, value):
        return (
            gaudi_envs.VLLM_HPU_DSV41_COMPRESSOR_FUSED_PUBLISH
            and value.device.type == "hpu" and value.shape[0] == 1
            and self.ratio in (1, 2) and self.native_rope
            and hasattr(self.cache, "decoded_main")
            and hasattr(self.cache, "decoded_index_hot")
            and hasattr(torch.ops.custom_op, "custom_deepseek_v41_fp4_norm_rope_publish_gaudi2")
        )

    def _compress(self, value, positions, decoded=False):
        compressor = self.weights.compressor
        if self.ratio == 2:
            if (
                self.tp4_local_index_queries
                and value.shape[0] == 1
                and self.search_length // self.ratio > 512
                and self.search_length <= 32768
            ):
                # Removing the query collectives joins this FP32 projection
                # to its BF16 producer. Synapse otherwise cancels the adjacent
                # FP32 -> BF16 -> FP32 casts and changes compressor history.
                # Keep the original BF16 input boundary inside one opaque TPC.
                value = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(value.contiguous())
            kv, score = self._project_compressor_input(value)
            # Positions and all circular capacities are non-negative powers
            # of two.  The stock `%` expressions emitted a div_mod kernel for
            # every occurrence, including several copies in each ratio-2
            # source layer.  Masks are mathematically identical on the full
            # 1M position domain and keep this preparation inexpensive for
            # both decode and prefill.
            first = torch.bitwise_and(positions, -2)
            if positions.numel() == 1:
                # The archived decode trace exposes a repeated seven-launch
                # history/gather/softmax/reduction chain at each ratio-2
                # source layer.  Keep the same FP32 state and BF16 boundary,
                # but consume the two rows in registers so the temporary
                # gathers and probability tensor never reach HBM.  Wider
                # batches retain the request-slot-safe generic path below.
                latent = torch.ops.custom_op.custom_deepseek_v41_compressor_pair_bf16_gaudi2(
                    self.kv_history,
                    self.score_history,
                    kv.contiguous(),
                    score.contiguous(),
                    positions.to(torch.int32).contiguous(),
                )
            elif positions.numel() <= 6:
                # Preserve the qualified C1/C6 graph and its exact state
                # mutation order.
                ring = torch.bitwise_and(positions, HISTORY_ROWS - 1).long()
                self.kv_history.index_copy_(0, ring, kv)
                self.score_history.index_copy_(0, ring, score)
                a = torch.bitwise_and(first, HISTORY_ROWS - 1).long()
                b = torch.bitwise_and(first + 1, HISTORY_ROWS - 1).long()
                gates = torch.stack((self.score_history[a], self.score_history[b]), 1).softmax(1)
                latent = (self.kv_history[a] * gates[:, 0] + self.kv_history[b] * gates[:, 1]).to(value.dtype)
            else:
                # A C128 block is wider than the eight-row history ring. Read
                # pairs from the current block whenever available and from the
                # prior ring only at the leading boundary; update the ring
                # after all reads so rows cannot alias within this transaction.
                capture = getattr(self.shared, "inline_prefix_capture", None)
                if capture is not None:
                    capture.record(self.layer, "kv_history", kv)
                    capture.record(self.layer, "score_history", score)
                base, tokens = positions[0], positions.numel()
                local_a, local_b = first - base, first + 1 - base
                valid_a = (local_a >= 0) & (local_a < tokens)
                valid_b = (local_b >= 0) & (local_b < tokens)
                current_a = kv.index_select(0, local_a.clamp(0, tokens - 1).long())
                current_b = kv.index_select(0, local_b.clamp(0, tokens - 1).long())
                score_a = score.index_select(0, local_a.clamp(0, tokens - 1).long())
                score_b = score.index_select(0, local_b.clamp(0, tokens - 1).long())
                history_a = self.kv_history[torch.bitwise_and(first, HISTORY_ROWS - 1).long()]
                history_b = self.kv_history[torch.bitwise_and(first + 1, HISTORY_ROWS - 1).long()]
                history_score_a = self.score_history[torch.bitwise_and(first, HISTORY_ROWS - 1).long()]
                history_score_b = self.score_history[torch.bitwise_and(first + 1, HISTORY_ROWS - 1).long()]
                pair_a = torch.where(valid_a.unsqueeze(-1), current_a, history_a)
                pair_b = torch.where(valid_b.unsqueeze(-1), current_b, history_b)
                pair_score_a = torch.where(valid_a.unsqueeze(-1), score_a, history_score_a)
                pair_score_b = torch.where(valid_b.unsqueeze(-1), score_b, history_score_b)
                gates = torch.stack((pair_score_a, pair_score_b), 1).softmax(1)
                latent = (pair_a * gates[:, 0] + pair_b * gates[:, 1]).to(value.dtype)
                tail = min(tokens, HISTORY_ROWS)
                tail_positions = torch.bitwise_and(positions[-tail:], HISTORY_ROWS - 1).long()
                self.kv_history.index_copy_(0, tail_positions, kv[-tail:])
                self.score_history.index_copy_(0, tail_positions, score[-tail:])
        else:
            first, latent = positions, self.linear(value, compressor.wkv)
        latent = rms_norm(latent, compressor.norm.weight, self.eps)
        if self._uses_fused_compressor_publish(value):
            indexer = self.weights.indexer
            raw_index = self.linear(latent, indexer.wk)
            mirror_enabled = bool(
                getattr(self.shared, "index_mirror_valid", False)
                and self.search_length <= self.shared.index_mirror_tokens
            )
            hot_enabled = self.runtime_indexer and self.search_length <= INDEX_MME_HOT_TOKENS
            mirror = self.cache.index_mirror if mirror_enabled else self.cache.decoded_index_hot
            return torch.ops.custom_op.custom_deepseek_v41_fp4_norm_rope_publish_gaudi2(
                latent.contiguous(), raw_index.contiguous(), indexer.k_norm.weight,
                positions.to(torch.int32).contiguous(), self._rotary_native_table(),
                self.shared.block_table, self.cache.main, self.cache.index,
                self.cache.decoded_main, self.cache.decoded_index_hot, mirror,
                self.eps, self.ratio, bool(decoded), bool(hot_enabled), mirror_enabled
            )
        if self.ratio & (self.ratio - 1):
            raise ValueError("V4.1 compressor ratio must be a power of two")
        ratio_shift = self.ratio.bit_length() - 1
        rows = torch.bitwise_right_shift(positions, ratio_shift)
        visible = torch.bitwise_and(positions, self.ratio - 1) == self.ratio - 1
        # Incomplete groups use unique rows in the reserved null page.
        slots = torch.where(
            visible, self.shared.physical_rows(rows, self.ratio), torch.bitwise_and(rows, PAGE_TOKENS // self.ratio - 1)
        )
        indexer = self.weights.indexer
        index = rms_norm(self.linear(latent, indexer.wk), indexer.k_norm.weight, self.eps)
        index, latent = self._rope(index, first), self._rope(latent, first)
        if (
            getattr(self.shared, "index_mirror_valid", False)
            and positions.numel() <= NATIVE_WORK_TOKENS
            and self.search_length <= self.shared.index_mirror_tokens
        ):
            # Maintain the exact packed-key value, including its BF16 FP4
            # roundtrip. The derived buffer is never request-snapshotted.
            self.cache.index_mirror.index_copy_(0, rows.long(), fp4_roundtrip(index, 32))
        if (
            self.runtime_indexer
            and self.ratio in (1, 2)
            # The <=512 static bucket is the producer for the later hot
            # MME bucket.  Populate its prefix before the search geometry
            # switches at token 513; otherwise rows 0..511 remain zero.
            and self.search_length <= INDEX_MME_HOT_TOKENS
            and hasattr(self.cache, "decoded_index_hot")
        ):
            # The mirror must contain the same values as the packed index
            # cache.  Keeping the pre-quantization key would silently change
            # candidate selection when this MME path replaces packed scoring.
            self.cache.decoded_index_hot.index_copy_(0, rows.long(), fp4_roundtrip(index, 32))
        if decoded and value.shape[0] == 1:
            return torch.ops.custom_op.custom_deepseek_v41_fp4_paged_decoded_write_bf16_gaudi2(
                self.cache.main,
                self.cache.index,
                latent.contiguous(),
                index.contiguous(),
                slots.to(torch.int32).contiguous(),
                rows.to(torch.int32).contiguous(),
                self.cache.decoded_main,
            )
        packed_index, packed_main = pack_fp4(index, 32), pack_fp4(latent, 16)
        self.cache.index.index_copy_(0, slots.long(), packed_index)
        self.cache.main.index_copy_(0, slots.long(), packed_main)
        if decoded:
            self.cache.decoded_main.index_copy_(0, rows.long(), unpack_fp4(packed_main))
        return None

    def _prepare_index_queries(self, value, qr, positions, *, prefill=False, request_batch=False):
        indexer = self.weights.indexer
        local_queries = (
            not prefill and getattr(self, "tp4_local_index_queries", False)
            and value.shape[0] == 1 and self.search_length <= 32768
        )
        if local_queries:
            from vllm_gaudi.ops.deepseek_v41_tp4_selection import local_index_query_projections

            q, weights = local_index_query_projections(self, value, qr)
        else:
            q = self.linear(qr, indexer.wq_b).reshape(-1, self.index_heads, 128)
        q = self._rope(q, positions, request_batch=request_batch)
        # Quantize and restore in one TPC pass.  The codec keeps packed FP4
        # codes and UE8M0 scales in registers, so prefill does not materialize
        # a packed query in HBM or compile a generic bit-unpack graph.  Tile by
        # the native flattened-row contract; for TP2's 16 heads this is C512.
        codec_tokens = max(1, 8192 // q.shape[1])
        pieces = [fp4_roundtrip(q[start : start + codec_tokens], 32) for start in range(0, q.shape[0], codec_tokens)]
        q = pieces[0] if len(pieces) == 1 else torch.cat(pieces, 0)
        global_heads = self.index_heads * self.tensor_parallel_size
        if not local_queries:
            weights = self.linear(value, indexer.weights_proj)
        weights = weights * (128**-0.5 * global_heads**-0.5)
        if (
            prefill
            and self.tensor_parallel_size == 4
            and gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP
            and can_partition_prefill_index(positions.numel(), self.search_length // self.ratio)
        ):
            from vllm.distributed import get_tp_group
            from vllm_gaudi.ops.deepseek_v41_prefill_index_exchange import exchange_prefill_index_queries

            group = get_tp_group()
            rank = getattr(self, "prefill_tp_rank", None)
            if rank != group.rank_in_group:
                raise RuntimeError("Prefill index query exchange has no matching TP owner")
            return exchange_prefill_index_queries(q, weights, rank, group=group.device_group)
        # Exchange only small query/head tensors, not one score per cached token.
        if not local_queries:
            if getattr(self, "tp4_packed_index_queries", False):
                from vllm_gaudi.ops.deepseek_v41_index_packet import gather_index_query_packet

                q, weights = gather_index_query_packet(q, weights, self.gather, self.tensor_parallel_size)
            else:
                q, weights = self.gather(q, 1), self.gather(weights, 1)
        return q, weights

    def _uses_index_mirror(self, q):
        return (
            getattr(self, "index_mirror_scores", False)
            and q.shape[0] == 1
            and self.shared.index_mirror_valid
            and self.search_length <= self.shared.index_mirror_tokens
        )

    def _scores(self, positions, logical_rows, q, weights):
        if self.tensor_parallel_size == 4 and q.shape[0] <= NATIVE_WORK_TOKENS:
            from vllm_gaudi.ops.deepseek_v41_decode_index import native_index_tile, supports_native_index_tile

            if supports_native_index_tile(q, logical_rows):
                if self._uses_index_mirror(q):
                    from vllm_gaudi.ops.deepseek_v41_index_mirror import mirror_index_tile

                    return mirror_index_tile(
                        q, weights, self.cache.index_mirror, positions, logical_rows, self.ratio, self.index_heads
                    )
                return native_index_tile(
                    q,
                    weights,
                    self.cache.index,
                    self.shared.block_table,
                    positions,
                    logical_rows,
                    self.ratio,
                    self.index_heads,
                )
        physical = self.shared.physical_rows(logical_rows.clamp_min(0), self.ratio)
        packed = self.cache.index.index_select(0, physical.flatten().long()).reshape(*physical.shape, 68)
        keys = unpack_fp4(packed, 128, 32)
        # Use explicit contiguous GEMMs.  The generic einsum accepted strided
        # query/key views and could require a new, much larger recipe beside
        # the 1M-context resident set.  These forms produce bitwise-identical
        # BF16 scores while giving Bridge the final MME geometry directly.
        scores = (
            torch.matmul(q.contiguous(), keys.transpose(0, 1).contiguous())
            if keys.ndim == 2
            else torch.bmm(q.contiguous(), keys.transpose(1, 2).contiguous())
        )
        scores = scores.relu() * weights.unsqueeze(-1)
        # Preserve the local-head BF16 sums before reducing across all TP
        # ranks. The gathered head axis includes four shards for TP4.
        scores = scores.reshape(q.shape[0], self.tensor_parallel_size, self.index_heads, -1).sum(2).sum(1)
        count = ((positions + 1) // self.ratio).unsqueeze(-1)
        valid = (logical_rows >= 0) & (logical_rows < count)
        return scores.float().masked_fill(~valid, -torch.inf)

    def _prefill_scores(self, positions, rows, q, weights):
        if gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_SRAM and rows.ndim == 1:
            from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import full_prefill_sram_scores

            return full_prefill_sram_scores(
                q, weights, self.cache.index, self.shared.block_table, positions, rows, self.ratio, self.index_heads
            )
        if rows.ndim != 2 or q.shape[0] > PREFILL_ATTN_TOKENS:
            return self._scores(positions, rows, q, weights)
        # The compiled MME helper flattens every BF16 rounding boundary into
        # one identity-TPC row.  A normal 128-query x 2048-key tile therefore
        # exceeds that kernel's compilable geometry, and its formal 1M-cache
        # input also consumes compile workspace beside ~80 GiB resident state.
        # The existing eager scorer materializes BF16 after each operation, so
        # it preserves the same rounding points without the identity kernel.
        # It was qualified at 77.39 GB resident allocation with the full tile.
        return self._scores(positions, rows, q, weights)

    @staticmethod
    def _merge_topk(best_scores, best_rows, scores, rows, width):
        take = min(width, scores.shape[-1])
        values, offsets = scores.topk(take, dim=-1, sorted=False)
        selected = rows.expand_as(scores).gather(1, offsets) if rows.ndim == 1 else rows.gather(1, offsets)
        if best_scores is not None:
            values = torch.cat((best_scores, values), -1)
            selected = torch.cat((best_rows, selected), -1)
            take = min(width, values.shape[-1])
            values, offsets = values.topk(take, dim=-1, sorted=False)
            selected = selected.gather(1, offsets)
        return values, selected

    def _stream_topk(
        self,
        positions,
        rows,
        q,
        weights,
        width=512,
        collect_blocks=False,
        native_scores=False,
        visible_rows=None,
        partition_tiles=False,
        partition_mirror=False,
    ):
        """Exact bounded-memory index selection for a large prompt block."""
        columns = rows.shape[-1]
        if visible_rows is not None and (rows.ndim != 1 or not 0 <= visible_rows <= columns):
            raise ValueError("Visible source prefix requires contiguous logical source rows")
        active_columns = columns if visible_rows is None else visible_rows
        if partition_mirror and not native_scores:
            from vllm_gaudi.ops.deepseek_v41_tp4_selection import supports_mirror_partition, tp4_mirror_stream_topk

            if supports_mirror_partition(self, rows, q, active_columns, width):
                return tp4_mirror_stream_topk(
                    self,
                    positions,
                    rows,
                    q,
                    weights,
                    width=width,
                    collect_blocks=collect_blocks,
                    visible_rows=visible_rows,
                )
        if (
            partition_tiles
            and self.tensor_parallel_size == 4
            and q.shape[0] == 1
            and 4 * PREFILL_INDEX_ROWS <= active_columns <= columns <= 32768
            and columns % PREFILL_INDEX_ROWS == 0
        ):
            from vllm_gaudi.ops.deepseek_v41_tp4_selection import tp4_stream_topk

            return tp4_stream_topk(
                self, positions, rows, q, weights, width=width, collect_blocks=collect_blocks, visible_rows=visible_rows
            )
        best_scores = best_rows = None
        block_scores = block_ids = None
        invalid_scores = None
        prefix_scores = None
        if (
            not native_scores
            and rows.ndim == 1
            and getattr(self, "index_mirror_scores", False)
            and self._uses_index_mirror(q)
            and 128 <= active_columns <= self.cache.index_mirror.shape[0]
            and active_columns % 128 == 0
        ):
            from vllm_gaudi.ops.deepseek_v41_index_mirror import mirror_source_scores

            prefix_scores = mirror_source_scores(q, weights, self.cache.index_mirror, positions,
                                                 rows[:active_columns], self.ratio, self.index_heads)
        from vllm_gaudi.ops.deepseek_v41_decode_selection import can_batch_full_r1

        batched_selection = can_batch_full_r1(self, q, rows, prefix_scores, native_scores, active_columns)
        selection_scores = []
        for start in range(0, columns, PREFILL_INDEX_ROWS):
            current_rows = (
                rows[start : start + PREFILL_INDEX_ROWS]
                if rows.ndim == 1
                else rows[:, start : start + PREFILL_INDEX_ROWS]
            )
            scorer = self._prefill_scores if native_scores else self._scores
            invisible_tile = visible_rows is not None and start >= visible_rows
            if invisible_tile:
                # The scheduler supplies this conservative prefix bound; no
                # device position or route is read back to the host. Candidate
                # block merges remain below because their unsorted order is a
                # published dependency of later Reindex layers.
                shape = (q.shape[0], current_rows.shape[-1])
                if invalid_scores is None or invalid_scores.shape != shape:
                    invalid_scores = q.new_full(shape, -torch.inf, dtype=torch.float32)
                scores = invalid_scores
            else:
                with prefill_event_span("index_scores", getattr(self, "layer", None), positions.numel()):
                    scores = (
                        scorer(positions, current_rows, q, weights)
                        if prefix_scores is None
                        else prefix_scores[:, start : start + current_rows.shape[-1]]
                    )
            if batched_selection:
                selection_scores.append(scores)
                continue
            # An all-invalid tile cannot change valid Top512 rows. Omitting the
            # large merge is exact after invalid rows are normalized to -1,
            # while retaining candidate-block merges preserves their tie order.
            if not invisible_tile:
                with prefill_event_span("index_topk", getattr(self, "layer", None), positions.numel()):
                    best_scores, best_rows = self._merge_topk(best_scores, best_rows, scores, current_rows, width)
            if collect_blocks:
                if scores.shape[-1] % 8:
                    pad = 8 - scores.shape[-1] % 8
                    scores = F.pad(scores, (0, pad), value=-torch.inf)
                grouped = scores.reshape(scores.shape[0], -1, 8).amax(-1)
                ids = torch.arange(
                    start // 8, start // 8 + grouped.shape[-1], device=positions.device, dtype=torch.int32
                )
                count = ((positions + 1) // self.ratio).unsqueeze(-1)
                # Match the reference: keep the block containing the newest
                # visible token so its causal prefix cannot be dropped.
                grouped = grouped.masked_fill(ids == ((count - 1) // 8), torch.inf)
                with prefill_event_span("index_candidates", getattr(self, "layer", None), positions.numel()):
                    block_scores, block_ids = self._merge_topk(block_scores, block_ids, grouped, ids, 2048)

        if batched_selection:
            from vllm_gaudi.ops.deepseek_v41_decode_selection import batched_decode_selection

            return batched_decode_selection(
                selection_scores,
                rows,
                positions,
                self.ratio,
                width=width,
                collect_blocks=collect_blocks,
                selected_tiles=active_columns // PREFILL_INDEX_ROWS,
            )
        return best_rows, block_scores, block_ids

    def _select(self, value, qr, positions, *, buffer_start=0, prepared=None, prefill=False):
        count, tokens = (positions + 1) // self.ratio, positions.numel()
        target = slice(buffer_start, buffer_start + tokens)
        if self.owns_index:
            if (
                prefill
                and gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_SHARED
                and self.layer > self.candidate_source
                and 512 < self.search_length // self.ratio <= 16384
            ):
                from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import compiled_shared_reindex

                q, weights = self._prepare_index_queries(value, qr, positions) if prepared is None else prepared
                pool = self.shared.candidate_pool[target]
                columns = candidate_columns(self.search_length, self.ratio, pool.shape[-1])
                pool = pool[:, :columns].contiguous()
                source_rows = self.search_length // self.ratio
                signature = (
                    q.shape,
                    pool.shape,
                    self.cache.index.shape,
                    self.shared.block_table.shape,
                    self.ratio,
                    self.index_heads,
                    source_rows,
                )
                indices = compiled_shared_reindex(signature)(
                    q.clone(),
                    weights.clone(),
                    self.cache.index,
                    self.shared.block_table,
                    positions.clone(),
                    pool,
                    self.ratio,
                    self.index_heads,
                    source_rows,
                )
                self.selection.indices[target].copy_(indices)
                return self.selection.indices[target]
            # The dynamic indexer is required once the visible prefix exceeds
            # the fixed 512-row candidate window.  For a short C1 prefix the
            # ordered compressed offsets are already the exact candidate set;
            # dispatching score/threshold/emit here adds four device launches
            # per owning layer and regresses the historical C1 replay.  Keep
            # this fast path valid for the long-context runtime-indexer build;
            # only the >512-row case needs dynamic selection.
            if self.runtime_indexer and tokens == 1 and not prefill and self.search_length // self.ratio > 512:
                from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select

                q, weights = self._prepare_index_queries(value, qr, positions) if prepared is None else prepared
                pool = (
                    self.shared.index_candidates_unused[:tokens]
                    if self.shared.candidate_pool is None
                    else self.shared.candidate_pool[target].contiguous()
                )
                decoded_hot = (
                    self.cache.decoded_index_hot
                    if (
                        self.ratio in (1, 2)
                        and self.search_length == INDEX_MME_HOT_TOKENS
                        and hasattr(self.cache, "decoded_index_hot")
                    )
                    else None
                )
                indices, blocks = runtime_index_select(
                    q.contiguous(),
                    weights.contiguous(),
                    self.cache.index,
                    self.shared.block_table,
                    positions.to(torch.int32).contiguous(),
                    pool,
                    ratio=self.ratio,
                    # Packed scoring beyond the decoded mirror must scan the
                    # same bounded geometry as this stage, not every reserved
                    # context row. Device positions still mask future rows.
                    capacity=(self.search_length if self.runtime_index_mme else self.length) // self.ratio,
                    reindex=self.layer > self.candidate_source,
                    publish_candidates=self.layer == self.candidate_source,
                    decoded_hot=decoded_hot,
                    ordered_candidates=True,
                    local_heads=self.index_heads,
                    # The paged-key/MME tile consumer already supports long
                    # source ranges. Crossing the mirror capacity changes the
                    # key producer, not the score arithmetic or selection.
                    search_rows=(self.decode_visible_rows or self.search_length // self.ratio)
                    if decoded_hot is None and self.runtime_index_mme else None,
                    decoded_keys=(self.cache.index_mirror if self._uses_index_mirror(q) else None),
                )
                if blocks is not None:
                    self.shared.candidate_pool[target].copy_(blocks)
                self.selection.indices[target].copy_(indices)
                return self.selection.indices[target]
            if self.search_length // self.ratio <= 512:
                indices = self.compressed_offsets.unsqueeze(0).expand(tokens, -1)
                indices = torch.where(indices < count.unsqueeze(-1), indices, -1)
                if self.layer == self.candidate_source:
                    blocks = torch.arange(2048, device=positions.device, dtype=torch.int32).expand(tokens, -1)
                    self.shared.candidate_pool[target].copy_(torch.where(blocks * 8 < count.unsqueeze(-1), blocks, -1))
            else:
                if self.layer > self.candidate_source:
                    blocks = self.shared.candidate_pool[target]
                    if gaudi_envs.VLLM_HPU_DSV41_PREFILL_COMPACT_CANDIDATES and prefill:
                        blocks = blocks[:, : candidate_columns(self.search_length, self.ratio, blocks.shape[-1])]
                    rows = blocks.unsqueeze(-1) * 8 + torch.arange(8, device=positions.device, dtype=torch.int32)
                    rows = torch.where(blocks.unsqueeze(-1) >= 0, rows, -1).flatten(1)
                else:
                    rows = torch.arange(self.search_length // self.ratio, device=positions.device, dtype=torch.int32)
                q, weights = self._prepare_index_queries(value, qr, positions) if prepared is None else prepared
                indices, candidate_scores, candidate_blocks = self._stream_topk(
                    positions,
                    rows,
                    q,
                    weights,
                    collect_blocks=self.layer == self.candidate_source,
                    native_scores=prefill and gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_MME,
                    visible_rows=None if prefill else self.decode_visible_rows,
                    partition_tiles=not prefill and tokens == 1 and getattr(self, "tp4_tile_selection", False),
                    partition_mirror=getattr(self, "tp4_mirror_selection", False) and not prefill and tokens == 1,
                )
                if self.layer == self.candidate_source:
                    valid = candidate_scores > -torch.inf
                    candidate_blocks = torch.where(valid, candidate_blocks, -1).to(torch.int32)
                    candidate_blocks = F.pad(
                        candidate_blocks,
                        (0, self.shared.candidate_pool.shape[-1] - candidate_blocks.shape[-1]),
                        value=-1,
                    )
                    self.shared.candidate_pool[target].copy_(candidate_blocks)
                # Sort logical positions, with invalid picks after reachable ones.
                indices = torch.where((indices >= 0) & (indices < count.unsqueeze(-1)), indices, self.length)
                indices = indices.sort(-1).values
                indices = torch.where(indices < self.length, indices, -1).int()
            self.selection.indices[target].copy_(indices)
        return self.selection.indices[target]

    def _reduce_prefill_output(self, partial, ready_outputs, prefill_sequence):
        if prefill_sequence:
            if self.tensor_parallel_size != 4 or ready_outputs:
                raise ValueError("Token-owned attention output requires ordinary TP4 prefill")
            from vllm_gaudi.ops.deepseek_v41_prefill_sequence_state import reduce_owned_tokens

            return reduce_owned_tokens(partial, self.reduce)
        return self.reduce(partial, ready_outputs=ready_outputs) if ready_outputs else self.reduce(partial)

    def _finish_output(
        self, output, positions, ready_outputs=(), *, prefill=False, prefill_sequence=False, request_batch=False,
        decode=False
    ):
        """Apply the shared output projection after either attention path."""
        if prefill:
            with prefill_event_span("attention_output_inverse_rope", self.layer, output.shape[0]):
                output = self._rope(output, positions, inverse=True, request_batch=request_batch)
        else:
            output = self._rope(output, positions, inverse=True, request_batch=request_batch)
        output = output.reshape(-1, self.groups, self.heads // self.groups * 512)
        if prefill and gaudi_envs.VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION and 6 < output.shape[0] <= 16384:
            wa, wb = self.weights.wo_a, self.weights.wo_b
            if not (self.woa_fp8 and self.woa_output_roundtrip and getattr(wb, "dense_fp8", False)):
                raise ValueError("Prefill output projection requires prepared FP8 weights and the group codec")
            with prefill_event_span("attention_output_projection", self.layer, output.shape[0]):
                partial = prefill_output_projection(
                    output.contiguous(), wa.weight, wa.channel_scale, wb.weight, wb.channel_scale
                )
            with prefill_event_span("attention_output_reduce", self.layer, output.shape[0]):
                return self._reduce_prefill_output(partial, ready_outputs, prefill_sequence)
        if prefill:
            with prefill_event_span("attention_output_projection", self.layer, output.shape[0]):
                output = self.project_output(output)
                partial = self.project_output_consumer(output)
            with prefill_event_span("attention_output_reduce", self.layer, output.shape[0]):
                return self._reduce_prefill_output(partial, ready_outputs, prefill_sequence)
        output = self.project_output(output)
        return self._finish_projected_output(output, ready_outputs, decode=decode)

    def _finish_projected_output(self, output, ready_outputs=(), *, decode=False):
        """Consume an already inverse-rotated and wo_a-projected row."""
        partial = self.project_output_consumer(output)
        return self._reduce_decode_output(partial, ready_outputs, decode=decode)

    def _reduce_decode_output(self, partial, ready_outputs=(), *, decode=False):
        if decode and getattr(self, "peer_post_collapse", False) and partial.shape[0] <= 2:
            return self.reduce(partial, ready_outputs=ready_outputs, defer=True)
        return self.reduce(partial, ready_outputs=ready_outputs) if ready_outputs else self.reduce(partial)

    def _can_fuse_mla_woa(self, positions):
        """Return whether the exact MLA-product/wo_a producer is available."""
        return (
            positions.numel() == 1 and self.mla_mme and self.woa_fp8 and self.woa_output_roundtrip and self.native_rope
        )

    def _paged_mla_operation(self, tokens):
        if self.paged_mla_direct and tokens == 1:
            return torch.ops.custom_op.custom_deepseek_v41_paged_mla_direct_mme_gaudi2
        return torch.ops.custom_op.custom_deepseek_v41_paged_mla_mme_gaudi2

    def _output(self, query, cache, indices, positions, ready_outputs=(), *, decode=False):
        if decode and self.mla_mme and 1 <= query.shape[0] <= NATIVE_WORK_TOKENS:
            lengths = torch.full((query.shape[0],), indices.shape[1], dtype=torch.int32, device=query.device)
            output = torch.ops.custom_op.custom_deepseek_v41_selected_mla_mme_gaudi2(
                query.contiguous(),
                cache.contiguous(),
                indices.contiguous(),
                self.weights.attn_sink,
                self.scale,
                lengths,
            )
            return self._finish_output(output, positions, ready_outputs, decode=decode)
        output, _, _ = torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(
            query.contiguous(), cache.contiguous(), indices.contiguous(), self.weights.attn_sink, self.scale
        )
        return self._finish_output(output, positions, ready_outputs, decode=decode)

    @prefill_scope("indexer")
    def _prefill_selections(self, value, qr, positions, prepared):
        """Publish one transaction's selection state with bounded Reindex memory."""
        if not self.ratio:
            return None
        source_rows = self.search_length // self.ratio
        # Partitioning removes duplicated large-M index work, but the Gaudi2
        # graph compiler rejects the small two-tile Reindex concat produced
        # after splitting C512 into C256 per rank.  Keep the established local
        # path for C512 and smaller buckets; production C1024-C8192 buckets
        # still split cleanly and carry the useful work reduction.
        if (
            self.owns_index
            and gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP
            and can_partition_prefill_index(positions.numel(), source_rows)
        ):
            from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import (
                tp_full_prefill_index_selection,
                tp_prefill_reindex_selection,
            )

            rank = getattr(self, "prefill_tp_rank", None)
            if rank is None or not 0 <= rank < self.tensor_parallel_size:
                raise RuntimeError("Prefill index query partition has no prepared TP rank")
            from vllm_gaudi.ops.deepseek_v41_prefill_index_exchange import PrefillIndexQueryPartition

            prepared = self._prepare_index_queries(value, qr, positions, prefill=True) if prepared is None else prepared
            partitioned = isinstance(prepared, PrefillIndexQueryPartition)
            if partitioned:
                prepared.validate(positions.numel(), rank)
                q, weights = prepared.query, prepared.weights
            else:
                q, weights = prepared
            if self.layer <= self.candidate_source:
                visible = None
                if gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_VISIBLE:
                    token_end = getattr(self, "prefill_token_end", None)
                    if token_end is None or not positions.numel() <= token_end <= self.search_length:
                        raise RuntimeError("Prefill source visibility has no valid scheduler interval")
                    visible = token_end // self.ratio
                selected, blocks = tp_full_prefill_index_selection(
                    q,
                    weights,
                    self.cache.index,
                    self.shared.block_table,
                    positions.to(torch.int32),
                    self.ratio,
                    source_rows,
                    self.layer == self.candidate_source,
                    rank,
                    self.gather,
                    visible,
                    self.tensor_parallel_size,
                    query_partitioned=partitioned,
                )
                if blocks is not None:
                    self.shared.candidate_pool[: positions.numel()].copy_(blocks)
            else:
                pool = self.shared.candidate_pool[: positions.numel()]
                if gaudi_envs.VLLM_HPU_DSV41_PREFILL_COMPACT_CANDIDATES:
                    pool = pool[:, : candidate_columns(self.search_length, self.ratio, pool.shape[-1])]
                selected = tp_prefill_reindex_selection(
                    q,
                    weights,
                    self.cache.index,
                    self.shared.block_table,
                    positions.to(torch.int32),
                    pool,
                    self.ratio,
                    source_rows,
                    rank,
                    self.gather,
                    tensor_parallel_size=self.tensor_parallel_size,
                    query_partitioned=partitioned,
                )
            self.selection.indices[: positions.numel()].copy_(selected)
            return self.selection.indices[: positions.numel()]
        # Full layers score the same one-dimensional source row range for all
        # query rows.  Streaming source rows while keeping all 8192 queries
        # removes 64 repeated Python submissions. The reduced score tile is
        # 64 MiB, but its 32-head BF16 producer reaches 1 GiB at 8192 queries.
        # Reindex rows are token-dependent (T x 16384), so keep only
        # that selection calculation on the smaller query tile.
        if not self.owns_index or self.layer <= self.candidate_source:
            return self._select(value, qr, positions, prepared=prepared, prefill=True)
        if gaudi_envs.VLLM_HPU_DSV41_PREFILL_REINDEX_REUSE:
            from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import (
                prefill_reindex_selection,
            )

            source_rows = self.search_length // self.ratio
            if source_rows > 512:
                q, weights = self._prepare_index_queries(value, qr, positions) if prepared is None else prepared
                pool = self.shared.candidate_pool[:positions.numel()]
                if gaudi_envs.VLLM_HPU_DSV41_PREFILL_COMPACT_CANDIDATES:
                    pool = pool[:, :candidate_columns(self.search_length, self.ratio, pool.shape[-1])]
                selected = prefill_reindex_selection(
                    q, weights, self.cache.index, self.shared.block_table, positions, pool,
                    self.ratio, source_rows, gaudi_envs.VLLM_HPU_DSV41_PREFILL_CANDIDATE_GATHER,
                    gaudi_envs.VLLM_HPU_DSV41_PREFILL_REINDEX_SRAM, self.tensor_parallel_size,
                )
                self.selection.indices[:positions.numel()].copy_(selected)
                return self.selection.indices[: positions.numel()]
        for start in range(0, positions.numel(), PREFILL_ATTN_TOKENS):
            stop = min(start + PREFILL_ATTN_TOKENS, positions.numel())
            current_prepared = (prepared[0][start:stop], prepared[1][start:stop]) if prepared is not None else None
            self._select(
                value[start:stop],
                qr[start:stop],
                positions[start:stop],
                buffer_start=start,
                prepared=current_prepared,
                prefill=True,
            )
        return self.selection.indices[: positions.numel()]

    def _prefill_swa_workspace(self, kv, positions, *, decoded=False):
        """Bind request state explicitly to the reusable SWA tensor region."""
        from vllm_gaudi.ops.deepseek_v41_prefill_regions import prefill_swa_workspace

        return prefill_swa_workspace(
            kv,
            positions,
            self.swa,
            self.window_offsets,
            self.shared.decoded_swa if decoded else None,
            self.decoded_swa_offset if decoded else 0,
        )

    def _prefill_attention_tiled(self, value, query, kv, positions, selected, ready_outputs=(), prefill_sequence=False):
        """Bound BF16 selected-row storage for prefixes above the dense-cache cap."""
        outputs = []
        for start in range(0, positions.numel(), PREFILL_ATTN_TOKENS):
            stop = min(start + PREFILL_ATTN_TOKENS, positions.numel())
            pos = positions[start:stop]
            packed_swa = pack_swa(kv[start:stop])
            self.swa.index_copy_(0, pos.remainder(SWA_ROWS).long(), packed_swa)
            window = pos.unsqueeze(-1) - self.window + 1 + self.window_offsets.unsqueeze(0)
            indices = torch.where(window >= 0, window.remainder(SWA_ROWS), -1).int()
            cache = None
            if self.ratio:
                current_selected = selected[start:stop]
                physical = self.shared.physical_rows(current_selected.clamp_min(0), self.ratio)
                packed = self.cache.main.index_select(0, physical.flatten().long())
                cache = torch.cat((unpack_swa(self.swa), unpack_fp4(packed)), 0)
                offset = torch.arange(current_selected.numel(), device=value.device, dtype=torch.int32).reshape(
                    current_selected.shape
                )
                indices = torch.cat((indices, torch.where(current_selected >= 0, offset + SWA_ROWS, -1)), -1)
            if cache is None:
                cache = unpack_swa(self.swa)
            output = self._prefill_sparse(query[start:stop], cache, indices)
            outputs.append(output)
        return self._finish_output(
            torch.cat(outputs, 0), positions, ready_outputs, prefill=True, prefill_sequence=prefill_sequence
        )

    @prefill_span("attention_mla")
    def _prefill_sparse(self, query, cache, indices, *, indices_partitioned=False):
        if gaudi_envs.VLLM_HPU_DSV41_FLASHINFER_PREFILL:
            if gaudi_envs.VLLM_HPU_DSV41_PREFILL_MLA_SEQUENCE and can_partition_prefill_mla(
                query.shape, indices.shape[1], self.tensor_parallel_size, self.search_length
            ):
                from vllm.distributed import get_tp_group
                from vllm_gaudi.ops.deepseek_v41_prefill_sequence import sequence_prefill_mla

                group = get_tp_group()
                if self._prefill_sequence_sink is None:
                    self._prefill_sequence_sink = self.gather(self.weights.attn_sink, 0).contiguous()
                return sequence_prefill_mla(
                    query,
                    cache,
                    indices,
                    self._prefill_sequence_sink,
                    group.rank_in_group,
                    group=group.device_group,
                    retire_chunks=False,
                    indices_partitioned=indices_partitioned,
                )
            if indices_partitioned:
                raise ValueError("Query-owned MLA indices cannot fall back to a replicated consumer")
            outputs = []
            # Keep the gathered working set bounded inside a larger compiled
            # recipe, while the caller retains the complete query sequence.
            outer_rows = 2048
            with prefill_event_span("mla_local_tiles", self.layer, query.shape[0]):
                for start in range(0, query.shape[0], outer_rows):
                    stop = min(start + outer_rows, query.shape[0])
                    signature = (stop - start, tuple(query.shape[1:]), tuple(cache.shape), indices.shape[1])
                    outputs.append(
                        compiled_flash_prefill_mla(signature)(
                            query[start:stop].contiguous(),
                            cache,
                            indices[start:stop].contiguous(),
                            self.weights.attn_sink,
                        )
                    )
            return torch.cat(outputs, 0)
        if indices_partitioned:
            raise ValueError("Query-owned MLA indices require the sequence prefill path")
        tile = gaudi_envs.VLLM_HPU_DSV41_PREFILL_MLA_ROWS
        if tile:
            # The long-context bridge used by the qualified decode path loses
            # the dtype of an internal reinterpret_cast when this helper is
            # lowered as a nested torch.compile recipe.  The identical native
            # MME op is valid (including C2052 and its four-row tail), so submit
            # the bounded C64 tiles directly.  This affects prefill only; C1
            # decode still uses the captured native joint-replay program below.
            return bounded_prefill_mla(query, cache, indices, self.weights.attn_sink, self.scale, tile)
        return torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(
            query.contiguous(), cache.contiguous(), indices.contiguous(), self.weights.attn_sink, self.scale
        )[0]

    def _prefill_main_workspace(self, cache, indices, selected, logical):
        workspace = self.shared.prefill_main_workspace
        if workspace is None:
            return prefill_main_workspace(
                self.cache.main, self.shared.block_table, cache, indices, selected, logical, self.ratio
            )
        main = workspace.get(
            self.kv_source, self.cache.main, self.shared.block_table, logical, self.ratio, producer=self.owns_kv
        )
        return workspace.assemble(main, cache, indices, selected)

    def _prefill_attention(self, value, qr, query, kv, positions, ready_outputs=(), prefill_sequence=False):
        """Run one configured prefill transaction with large-M operators.

        Each query carries causal logical KV indices. Reuse a flat decoded
        source while it fits the temporary workspace; otherwise load selected
        paged rows inside bounded tiles of the same query-owner MLA chain.
        """
        if positions.numel() > self.shared.prefill_tokens:
            raise ValueError(f"V4.1 prefill transaction exceeds C{self.shared.prefill_tokens}")
        diagnostic = (
            os.getenv("VLLM_HPU_DSV41_PREFILL_DIAG_BOUNDARIES", "0") == "1"
            and positions.numel() == 256
            and self.search_length == 32768
        )

        def boundary(name):
            if diagnostic:
                print(f"PREFILL_DIAG layer={self.layer} boundary={name} submitted", flush=True)
                torch.hpu.synchronize()
                print(f"PREFILL_DIAG layer={self.layer} boundary={name} complete", flush=True)

        boundary("input")
        decoded = (
            self.decoded_kv_state
            and self.ratio in (1, 2)
            and self.search_length // self.ratio <= self.cache.decoded_main.shape[0]
        )
        if self.ratio and self.owns_kv:
            with prefill_event_span("attention_compress_kv", self.layer, value.shape[0]):
                self._compress(value, positions, decoded=decoded)
        boundary("compress")
        with prefill_event_span("attention_index_query", self.layer, value.shape[0]):
            prepared = (
                self._prepare_index_queries(value, qr, positions, prefill=True)
                if self.ratio and self.owns_index
                else None
            )
        boundary("index_query")
        # Selection is shared by the attention workspace tiles. Full layers
        # can stream each index-K source tile once for all query rows; Reindex
        # layers retain their bounded query tiling in _prefill_selections.
        selected = self._prefill_selections(value, qr, positions, prepared)
        selected_dump = os.getenv("VLLM_HPU_DSV41_DUMP_SELECTED")
        if selected_dump and self.layer == 2 and selected is not None and getattr(self, "prefill_tp_rank", 0) == 0:
            # Diagnostic only: a real selected-ID distribution is needed to
            # decide whether nearby queries can share one compact KV union.
            # The host synchronization here must never enter a timed run.
            from pathlib import Path

            destination = Path(selected_dump)
            destination.mkdir(parents=True, exist_ok=True)
            first_position = int(positions[0].item())
            torch.save(
                {"positions": positions.detach().cpu(), "selected": selected.detach().cpu()},
                destination / f"layer2-start{first_position}-pid{os.getpid()}.pt",
            )
        boundary("selection")
        # Both the flat workspace and the bounded selected-paged consumer use
        # the same normalized chronological KV producer. Save its boundary
        # rows before either path commits the final sliding-window tail.
        capture = getattr(getattr(self, "shared", None), "inline_prefix_capture", None)
        if capture is not None:
            capture.record(self.layer, "swa", kv, pack=pack_swa)
        main_rows = self.search_length // self.ratio if self.ratio else 0
        if main_rows > PREFILL_MAIN_CACHE_ROWS:
            if (gaudi_envs.VLLM_HPU_DSV41_FLASHINFER_PREFILL
                    and gaudi_envs.VLLM_HPU_DSV41_PREFILL_MLA_SEQUENCE
                    and can_partition_prefill_mla(query.shape, 640, self.tensor_parallel_size, self.search_length)):
                from vllm.distributed import get_tp_group
                from vllm_gaudi.ops.deepseek_v41_prefill_sequence import sequence_prefill_mla

                group = get_tp_group()
                if self._prefill_sequence_sink is None:
                    self._prefill_sequence_sink = self.gather(self.weights.attn_sink, 0).contiguous()
                cache, indices = self._prefill_swa_workspace(kv, positions, decoded=decoded)
                # The same query-owner exchange consumes logical selected IDs.
                # Only its bounded local MLA recipe decodes the selected rows;
                # no full-context BF16 main cache is allocated or exchanged.
                indices = torch.cat((indices, selected), -1).contiguous()
                output = sequence_prefill_mla(
                    query, cache, indices, self._prefill_sequence_sink, group.rank_in_group,
                    group=group.device_group, retire_chunks=False,
                    paged_main=(self.cache.main, self.shared.block_table, self.ratio),
                )
                return self._finish_output(output, positions, ready_outputs,
                                           prefill=True, prefill_sequence=prefill_sequence)
            return self._prefill_attention_tiled(value, query, kv, positions, selected, ready_outputs, prefill_sequence)

        with prefill_event_span("attention_swa_workspace", self.layer, value.shape[0]):
            cache, indices = self._prefill_swa_workspace(kv, positions, decoded=decoded)
        boundary("swa")
        columns = indices.shape[1] + (selected.shape[1] if self.ratio else 0)
        indices_partitioned = (
            gaudi_envs.VLLM_HPU_DSV41_FLASHINFER_PREFILL
            and gaudi_envs.VLLM_HPU_DSV41_PREFILL_MLA_SEQUENCE
            and can_partition_prefill_mla(query.shape, columns, self.tensor_parallel_size, self.search_length)
        )
        if indices_partitioned:
            from vllm.distributed import get_tp_group

            group = get_tp_group()
            rank = getattr(self, "prefill_tp_rank", None)
            if rank != group.rank_in_group:
                raise RuntimeError("Prefill MLA indices have no matching TP owner")
            rows = query.shape[0] // 4
            owner = slice(rank * rows, (rank + 1) * rows)
            # Persistent selection/candidate state stays replicated for later
            # layers and decode. Only this invocation's assembled IDs shrink.
            indices = indices[owner]
            selected = selected[owner]
        if self.ratio:
            logical = torch.arange(main_rows, device=value.device, dtype=torch.int32)
            with prefill_event_span("attention_main_workspace", self.layer, value.shape[0]):
                cache, indices = self._prefill_main_workspace(cache, indices, selected, logical)
        boundary("main")
        output = self._prefill_sparse(query, cache, indices, indices_partitioned=indices_partitioned)
        boundary("mla")
        return self._finish_output(output, positions, ready_outputs, prefill=True, prefill_sequence=prefill_sequence)

    def _uses_logical_mla(self, value, decoded):
        return (
            self.paged_mla_logical
            and self.ratio in (1, 2)
            and not decoded
            and self.mla_mme
            and self.direct_selected_kv
            and self.window == 128
            and value.device.type == "hpu"
            and 1 <= value.shape[0] <= NATIVE_WORK_TOKENS
            and not (self.shared_prefix_kv and self.search_length // self.index_ratio <= 512)
        )

    @prefill_span("attention")
    @prefill_scope("attention")
    def forward(
        self, value, positions, ready_outputs=(), *, decode=False, prefill_sequence=False,
        prefill_qkv_sequence=False, selected_main=None, decode_metadata=None, input_prequant=None
    ):
        metadata = decode_metadata if decode and self.shared_decode_metadata else None
        prefill = not decode and value.shape[0] > NATIVE_WORK_TOKENS
        if prefill_sequence and not prefill:
            raise ValueError("Token ownership is confined to the prompt path")
        token_group = None
        if input_prequant is not None and (not decode or prefill_sequence or prefill_qkv_sequence):
            raise ValueError("Attention input prequantization is confined to ordinary decode")
        projection_value = value
        if prefill_qkv_sequence:
            if (
                not prefill_sequence
                or self.tensor_parallel_size != 4
                or value.shape[0] not in (positions.numel(), positions.numel() // 4)
            ):
                raise ValueError("Owned QKV requires matching full or token-owned TP4 prompt input")
            if (self.owns_kv or self.owns_index) and value.shape[0] != positions.numel():
                raise ValueError("KV/index producers require full normalized hidden rows")
            from vllm.distributed import get_tp_group
            from vllm_gaudi.ops.deepseek_v41_prefill_sequence_state import token_owner

            group = get_tp_group()
            token_group = group.device_group
            if value.shape[0] == positions.numel():
                projection_value = value[token_owner(positions.numel(), group.rank_in_group)]
        if prefill:
            with prefill_event_span("attention_input_projection", self.layer, projection_value.shape[0]):
                query_input, kv_input = self._project_qkv_input(projection_value, token_group=token_group)
        else:
            query_input, kv_input = (self._project_qkv_input(value) if input_prequant is None
                                     else self._project_qkv_input(value, prequant=input_prequant))
        norm = (
            torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2
            if self.fused_norm
            and (
                value.shape[0] == 1
                or (
                    not decode
                    and gaudi_envs.VLLM_HPU_DSV41_PREFILL_NATIVE_NORM
                    and NATIVE_WORK_TOKENS < value.shape[0] <= 16384
                )
            )
            else rms_norm
        )
        # The Q/KV split returns offset views. Bridge lowers large-prefill
        # float casts from those views to reinterpret_cast nodes whose dtype
        # metadata is invalid under the resident 1M graph. Materialize the
        # two bounded inputs before normalization; the qualified C1 path
        # already receives contiguous buffers and is unchanged.
        # A dynamic Reindex owner consumes the normalized Q row twice: once
        # for attention and once for candidate scoring. Other C1 layers can
        # keep this exact BF16 boundary inside the projection compound op.
        needs_index_query = self.ratio and self.owns_index and self.search_length // self.ratio > 512
        fused_query_norm = (
            decode
            and value.shape[0] == 1
            and self.fused_norm
            and self.q_scale_rope
            and self.native_rope
            and getattr(self.weights.wq_b, "dense_fp8", False)
            and not needs_index_query
        )
        if fused_query_norm:
            qr = None
            query = self.project_query_input(query_input, positions)
        else:
            if prefill:
                with prefill_event_span("attention_query_norm_projection", self.layer, value.shape[0]):
                    with prefill_event_span("attention_query_normalize", self.layer, value.shape[0]):
                        qr = norm(query_input.contiguous(), self.weights.q_norm.weight, self.eps)
                    query = self.project_query(qr, positions, decode=decode)
            else:
                qr = norm(query_input.contiguous(), self.weights.q_norm.weight, self.eps)
                query = self.project_query(qr, positions, decode=decode)
        decoded = (
            self.decoded_kv_state
            and self.ratio in (1, 2)
            and self.search_length // self.ratio <= self.cache.decoded_main.shape[0]
        )
        shared_key = ((self.kv_source, self.index_source, self.ratio)
                      if gaudi_envs.VLLM_HPU_DSV41_KV_REUSE_FUSION and self.ratio else None)
        fused_reuse = (
            decode and value.shape[0] == 1 and self.fused_norm and self.native_rope
            and not decoded and self._uses_logical_mla(value, decoded)
            and self.shared_main_mla and selected_main is not None and shared_key in selected_main
            and gaudi_envs.VLLM_HPU_DSV41_KV_REUSE_FUSION
            and hasattr(torch.ops.custom_op, "custom_deepseek_v41_kv_norm_reuse_mla_gaudi2")
        )
        fused_publish = (
            not fused_reuse and decode and value.shape[0] == 1 and self.fused_norm and self.native_rope
            and self.decoded_kv_state and self.shared.decoded_swa is not None
            and self.ratio in (1, 2) and (decoded or self._uses_logical_mla(value, decoded))
            and gaudi_envs.VLLM_HPU_DSV41_ATTN_FUSED_PROLOGUE
        )
        completion = None
        if fused_reuse:
            kv = None  # The current row is normalized/encoded in its consuming gather.
        elif fused_publish:
            kv, completion = torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_publish_gaudi2(
                kv_input.contiguous(), self.weights.kv_norm.weight,
                positions.to(torch.int32).contiguous(), self._rotary_native_table(),
                self.swa, self.shared.decoded_swa, self.eps,
                self.decoded_swa_offset if decoded else -1)
            # The emitted logical position is also the cache-consumer edge.
            # Alias-only writes do not order a packed gather in Synapse.
            positions = completion[:1]
        elif prefill:
            with prefill_event_span("attention_kv_norm_rope", self.layer, value.shape[0]):
                kv = self.project_kv(kv_input, positions, decode=decode)
        else:
            kv = self.project_kv(kv_input, positions, decode=decode)
        if not decode and value.shape[0] > NATIVE_WORK_TOKENS:
            del query_input, kv_input
            return self._prefill_attention(value, qr, query, kv, positions, ready_outputs, prefill_sequence)
        if fused_publish or fused_reuse:
            pass  # The fused producer owns the canonical SWA ring write.
        elif decoded and value.shape[0] == 1:
            logical_position = positions.to(torch.int32).contiguous()
            completion = torch.ops.custom_op.custom_deepseek_v41_swa_paged_decoded_write_bf16_gaudi2(
                self.swa, kv.contiguous(), logical_position, logical_position,
                self.shared.decoded_swa, self.decoded_swa_offset)
        else:
            packed_swa = pack_swa(kv)
            ring_rows = positions.remainder(SWA_ROWS).long() if metadata is None else metadata[0]
            self.swa.index_copy_(0, ring_rows, packed_swa)
            if decoded:
                self.shared.decoded_swa.index_copy_(
                    0, positions.remainder(SWA_ROWS).long() + self.decoded_swa_offset, unpack_swa(packed_swa)
                )
        native_prefix = (
            decode
            and gaudi_envs.VLLM_HPU_DSV41_FUSED_PREFIX_LAYOUT
            and self.ratio in (1, 2)
            and self.direct_selected_kv
            and self.shared_prefix_kv
            and self.window == 128
            and value.device.type == "hpu"
            and 1 <= value.shape[0] <= NATIVE_WORK_TOKENS
            and self.search_length // self.index_ratio <= 512
        )
        native_logical = decode and self._uses_logical_mla(value, decoded)
        indices = None
        if not native_prefix and not native_logical:
            window = positions.unsqueeze(-1) - self.window + 1 + self.window_offsets.unsqueeze(0)
            indices = torch.where(window >= 0, window.remainder(SWA_ROWS), -1).int()
        cache = None
        compressed_completion = None
        if self.ratio:
            if self.owns_kv:
                compressed_completion = self._compress(value, positions, decoded)
                if self._uses_fused_compressor_publish(value):
                    # The writer publishes the full logical token position;
                    # dependent scoring/gather cannot overtake its cache stores.
                    positions = compressed_completion[:1]
            selected = self._select(value, qr, positions)
            if decoded and value.shape[0] == 1:
                main_done = compressed_completion if compressed_completion is not None else completion
                if self._can_fuse_mla_woa(positions):
                    # The decoded mirror is already addressed by logical
                    # compressed row.  Let its first real consumer derive the
                    # fixed 128-row SWA prefix and selected-row mapping from
                    # position/selection directly.  This removes one TPC
                    # launch plus the [640] index and length intermediates in
                    # every decoded Attention layer without changing the
                    # B2+/prefill layout contract.
                    partial = torch.ops.custom_op.custom_deepseek_v41_mla_selected_woa_wob_fp8_roundtrip_gaudi2(
                        query.contiguous(),
                        self.shared.decoded_swa,
                        self.cache.decoded_main,
                        selected.contiguous(),
                        self.weights.attn_sink,
                        self.scale,
                        completion,
                        main_done,
                        self.decoded_swa_offset,
                        self.cache.decoded_main.shape[0],
                        self.weights.wo_a.weight,
                        self.weights.wo_a.channel_scale,
                        positions.to(torch.int32).contiguous(),
                        self._rotary_native_table(),
                        self.weights.wo_b.weight,
                        self.weights.wo_b.channel_scale,
                    )
                    return self._reduce_decode_output(partial, ready_outputs, decode=decode)
                # The paged state keeps SWA as a 256-row ring and main KV as
                # logical compressed rows.  Reuse the fixed prefix layout so
                # the decoded mirror sees [SWA ring, logical main] rather than
                # the legacy non-paged [512 absolute SWA, main] namespace.
                layout = (
                    torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r1_i32_gaudi2
                    if self.ratio == 1
                    else torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r2_i32_gaudi2
                )
                _, indices, lengths = layout(
                    selected.contiguous(), positions.to(torch.int32).contiguous(), self.shared.block_table
                )
                output = torch.ops.custom_op.custom_deepseek_v41_mla_mme_gaudi2(
                    query.contiguous(),
                    self.shared.decoded_swa,
                    self.cache.decoded_main,
                    indices.contiguous(),
                    self.weights.attn_sink,
                    self.scale,
                    lengths.contiguous(),
                    completion,
                    main_done,
                    self.decoded_swa_offset,
                    self.cache.decoded_main.shape[0],
                    SWA_ROWS,
                )
                return self._finish_output(output, positions, ready_outputs, decode=decode)
            if native_logical:
                # Keep page lookup and the sliding-window mapping inside the
                # first packed-KV consumer, alongside its exact BF16 codec.
                lengths = (
                    torch.full((value.shape[0],), 640, dtype=torch.int32, device=value.device)
                    if metadata is None
                    else metadata[1]
                )
                if self.shared_main_mla and value.shape[0] == 1 and selected_main is not None:
                    from vllm_gaudi.ops.deepseek_v41_shared_main import shared_main_attention

                    if fused_reuse:
                        main_rows, main_mask = selected_main[shared_key]
                        output = torch.ops.custom_op.custom_deepseek_v41_kv_norm_reuse_mla_gaudi2(
                            query.contiguous(), kv_input.contiguous(), self.weights.kv_norm.weight,
                            self.swa, main_rows, main_mask, positions.to(torch.int32).contiguous(),
                            self._rotary_native_table(), self.weights.attn_sink, self.scale, lengths, self.eps
                        )
                    else:
                        projection = (gaudi_envs.VLLM_HPU_DSV41_MAIN_MLA_PROJECTION
                                      and self._can_fuse_mla_woa(positions)
                                      and getattr(self.weights.wo_b, "dense_fp8", False))
                        output = shared_main_attention(self, query, positions, selected, lengths, selected_main,
                                                       projection=projection)
                        if projection:
                            return self._reduce_decode_output(output, ready_outputs, decode=decode)
                    return self._finish_output(output, positions, ready_outputs, decode=decode)
                output = torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(
                    query.contiguous(),
                    self.swa,
                    self.cache.main,
                    selected.contiguous(),
                    positions.to(torch.int32).contiguous(),
                    self.shared.block_table,
                    self.weights.attn_sink,
                    self.scale,
                    lengths,
                    self.ratio,
                )
                return self._finish_output(output, positions, ready_outputs, decode=decode)
            physical = None if native_prefix else self.shared.physical_rows(selected.clamp_min(0), self.ratio)
            if (
                decode
                and self.direct_selected_kv
                and value.device.type == "hpu"
                and value.shape[0] <= NATIVE_WORK_TOKENS
                and selected.numel() <= self.selected_offsets.numel()
            ):
                # Decode the fixed SWA prefix and all selected paged rows in
                # one graph entry, then consume its internal BF16 value directly
                # in sparse attention. This removes unpack_swa, cat, and the
                # exposed selected-row temporary from the Python graph.
                if self.shared_prefix_kv and self.search_length // self.index_ratio <= 512:
                    if native_prefix:
                        layout = (
                            torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r1_i32_gaudi2
                            if self.ratio == 1
                            else torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r2_i32_gaudi2
                        )
                        row_ids, attention_indices, lengths = layout(
                            selected.contiguous(), positions.to(torch.int32).contiguous(), self.shared.block_table
                        )
                    else:
                        row_ids, attention_indices, lengths = _shared_prefix_attention_layout(
                            physical, selected, indices, self.swa_offsets
                        )
                    attention_op = (
                        self._paged_mla_operation(value.shape[0])
                        if self.mla_mme
                        else torch.ops.custom_op.custom_deepseek_v41_paged_attention_head_vector_bf16_gaudi2
                        if self.head_vector_attn
                        else torch.ops.custom_op.custom_deepseek_v41_paged_attention_sram_bf16_gaudi2
                        if self.sram_kv
                        else torch.ops.custom_op.custom_deepseek_v41_paged_attention_vector_scales_bf16_gaudi2
                        if self.vector_kv_scales
                        else torch.ops.custom_op.custom_deepseek_v41_paged_attention_packed_exp_bf16_gaudi2
                        if self.packed_attn_exp
                        else torch.ops.custom_op.custom_deepseek_v41_paged_attention_lengths_bf16_gaudi2
                    )
                    output = attention_op(
                        query.contiguous(),
                        self.swa.contiguous(),
                        self.cache.main.contiguous(),
                        row_ids,
                        attention_indices,
                        self.weights.attn_sink,
                        self.scale,
                        lengths,
                    )
                    return self._finish_output(output, positions, ready_outputs, decode=decode)
                row_ids, attention_indices = _selected_attention_layout(
                    physical, selected, indices, self.swa_offsets, self.selected_offsets
                )
                if self.mla_mme:
                    lengths = torch.full(
                        (value.shape[0],), attention_indices.shape[1], dtype=torch.int32, device=value.device
                    )
                    output = self._paged_mla_operation(value.shape[0])(
                        query.contiguous(),
                        self.swa.contiguous(),
                        self.cache.main.contiguous(),
                        row_ids,
                        attention_indices,
                        self.weights.attn_sink,
                        self.scale,
                        lengths,
                    )
                    return self._finish_output(output, positions, ready_outputs, decode=decode)
                output = torch.ops.custom_op.custom_deepseek_v41_paged_attention_bf16_gaudi2(
                    query.contiguous(),
                    self.swa.contiguous(),
                    self.cache.main.contiguous(),
                    row_ids,
                    attention_indices,
                    self.weights.attn_sink,
                    self.scale,
                )
                return self._finish_output(output, positions, ready_outputs, decode=decode)
            else:
                packed = self.cache.main.index_select(0, physical.flatten().long())
                cache = torch.cat((unpack_swa(self.swa), unpack_fp4(packed)), 0)
            offset = torch.arange(selected.numel(), device=value.device, dtype=torch.int32).reshape(selected.shape)
            indices = torch.cat((indices, torch.where(selected >= 0, offset + SWA_ROWS, -1)), -1)
        if cache is None:
            if (
                decode and self.decode_swa_packed and not self.ratio and self.mla_mme
                and self.direct_selected_kv and 1 <= value.shape[0] <= NATIVE_WORK_TOKENS
            ):
                output = self._paged_mla_operation(value.shape[0])(
                    query.contiguous(), self.swa, self.swa_only_main,
                    self.swa_offsets.reshape(1, -1), indices.contiguous(),
                    self.weights.attn_sink, self.scale, self.swa_only_lengths[:value.shape[0]],
                )
                return self._finish_output(output, positions, ready_outputs, decode=decode)
            cache = unpack_swa(self.swa)
        return self._output(query, cache, indices, positions, ready_outputs, decode=decode)

    def _compress_batch(self, value, positions, slots, pages):
        from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_physical_rows, read_state_rows, write_state_rows

        state, compressor = self.batch_state, self.weights.compressor
        active = (slots >= 0) & (positions >= 0)
        first = positions - positions.remainder(self.ratio)
        if self.ratio == 2:
            kv, score = self._project_compressor_input(value)
            if self.batch_compressor_pair:
                latent = torch.ops.custom_op.custom_deepseek_v41_compressor_batch_bf16_gaudi2(
                    state.kv_history,
                    state.score_history,
                    kv.contiguous(),
                    score.contiguous(),
                    positions.contiguous(),
                    slots.contiguous(),
                )
            elif self.batch_compressor_gather:
                pair = torch.ops.custom_op.custom_deepseek_v41_compressor_batch_gather_f32_gaudi2(
                    state.kv_history,
                    state.score_history,
                    kv.contiguous(),
                    score.contiguous(),
                    positions.contiguous(),
                    slots.contiguous(),
                )
                gates = torch.stack((pair[:, 2], pair[:, 3]), 1).softmax(1)
                latent = (pair[:, 0] * gates[:, 0] + pair[:, 1] * gates[:, 1]).to(value.dtype)
            else:
                rows = torch.where(active, slots * HISTORY_ROWS + positions.remainder(HISTORY_ROWS), -1).int()
                kv_done = write_state_rows(state.kv_history, kv, rows)
                score_done = write_state_rows(state.score_history, score, rows)
                a = torch.where(active, slots * HISTORY_ROWS + first.remainder(HISTORY_ROWS), -1).int()
                b = torch.where(active, slots * HISTORY_ROWS + (first + 1).remainder(HISTORY_ROWS), -1).int()
                ka = read_state_rows(state.kv_history, a, kv_done)
                kb = read_state_rows(state.kv_history, b, kv_done)
                sa = read_state_rows(state.score_history, a, score_done)
                sb = read_state_rows(state.score_history, b, score_done)
                gates = torch.stack((sa, sb), 1).softmax(1)
                latent = (ka * gates[:, 0] + kb * gates[:, 1]).to(value.dtype)
        else:
            latent = self.linear(value, compressor.wkv)
        latent = rms_norm(latent, compressor.norm.weight, self.eps)
        indexer = self.weights.indexer
        index = rms_norm(self.linear(latent, indexer.wk), indexer.k_norm.weight, self.eps)
        index = self._rope(index, first.clamp_min(0), request_batch=True)
        latent = self._rope(latent, first.clamp_min(0), request_batch=True)
        visible = active & ((positions + 1).remainder(self.ratio) == 0)
        physical = batch_physical_rows((positions // self.ratio).reshape(-1, 1), pages, self.ratio).flatten()
        rows = torch.where(visible, physical, -1).int()
        main_done = write_state_rows(self.cache.main, pack_fp4(latent, 16), rows)
        index_done = write_state_rows(self.cache.index, pack_fp4(index, 32), rows)
        return main_done, index_done

    def forward_batch(
        self, value, positions, slots, pages, selected, candidates, main_done, index_done, ready_outputs=()
    ):
        """Ordinary C1 per request; query rows are not speculative positions."""
        from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_packed_mla, read_state_rows, write_state_rows
        from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select

        safe_positions = positions.clamp_min(0)
        query_input, kv_input = self._project_qkv_input(value)
        norm = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2 if self.fused_norm else rms_norm
        needs_index_query = self.ratio and self.owns_index
        if (
            self.batch_c1_numerics
            and self.fused_norm
            and self.q_scale_rope
            and self.native_rope
            and getattr(self.weights.wq_b, "dense_fp8", False)
            and not needs_index_query
        ):
            qr = None
            query = self.project_query_input(query_input, safe_positions)
        else:
            qr = norm(query_input.contiguous(), self.weights.q_norm.weight, self.eps)
            query = self.project_query(qr, safe_positions, request_batch=True)
        kv = self._rope(
            norm(kv_input.contiguous(), self.weights.kv_norm.weight, self.eps), safe_positions, request_batch=True
        )
        active = (slots >= 0) & (positions >= 0)
        rows = torch.where(active, slots * SWA_ROWS + positions.remainder(SWA_ROWS), -1).int()
        swa_done = write_state_rows(self.batch_state.swa, pack_swa(kv), rows)
        blocks = candidates
        if self.ratio:
            if self.owns_kv:
                main_done, index_done = self._compress_batch(value, positions, slots, pages)
            if self.owns_index:
                q, weights = self._prepare_index_queries(value, qr, safe_positions, request_batch=True)
                # Tie the index-cache mutation to its scoring consumer. An
                # incomplete pair is a completed no-write; prior KV stays
                # visible and the current query must still be scored.
                q = read_state_rows(
                    q.reshape(q.shape[0], -1).contiguous(),
                    torch.arange(q.shape[0], device=q.device, dtype=torch.int32),
                    index_done.clamp_min(0),
                ).reshape(q.shape)
                if self.batch_full_index_mme and q.shape[0] > 1 and self.layer <= self.candidate_source:
                    from vllm_gaudi.ops.deepseek_v41_full_index_mme import full_index_mme_select

                    selected, updated = full_index_mme_select(
                        q.contiguous(),
                        weights.contiguous(),
                        self.cache.index,
                        pages,
                        positions.contiguous(),
                        candidates,
                        ratio=self.ratio,
                        capacity=self.length // self.ratio,
                        publish_candidates=self.layer == self.candidate_source,
                        key_ready=index_done,
                        tiled_keys=self.batch_index_tiled_keys and q.shape[0] in (8, 32),
                    )
                elif self.batch_reindex_mme and q.shape[0] > 1 and self.layer > self.candidate_source:
                    from vllm_gaudi.ops.deepseek_v41_reindex_mme import (
                        bounded_reindex_mme_select,
                        reindex_mme_select,
                    )
                    from vllm_gaudi.ops.deepseek_v41_reindex_compact import bounded_reindex_bucket

                    # Only B8 has qualified a complete-chain benefit from the
                    # optional tile plan. Larger buckets keep the retained
                    # scorer; the same graph retains long-context support.
                    select = (
                        bounded_reindex_mme_select
                        if self.bounded_reindex and bounded_reindex_bucket(q.shape[0])
                        else reindex_mme_select
                    )
                    options = {"tiled_keys": self.batch_index_tiled_keys and q.shape[0] in (8, 32)}
                    selected = select(
                        q.contiguous(),
                        weights.contiguous(),
                        self.cache.index,
                        pages,
                        positions.contiguous(),
                        candidates,
                        ratio=self.ratio,
                        **options,
                    )
                    updated = None
                else:
                    selected, updated = runtime_index_select(
                        q.contiguous(),
                        weights.contiguous(),
                        self.cache.index,
                        pages,
                        positions.contiguous(),
                        candidates,
                        ratio=self.ratio,
                        capacity=self.length // self.ratio,
                        reindex=self.layer > self.candidate_source,
                        publish_candidates=self.layer == self.candidate_source,
                    )
                if updated is not None:
                    blocks = updated
            # Reuse may consume an index source with another compression
            # ratio. Match the existing selection contract after source IDs.
            selected = torch.where(selected < ((positions + 1) // self.ratio)[:, None], selected, -1).int()
            main_cache = self.cache.main
        else:
            main_cache = self.batch_main_unused
        # Keep only complete-chain qualified combinations. Reindex B8 and
        # both B16 chains did not improve; Full/Reuse B8 and both B32 did.
        packed_gather = (
            self.batch_packed_mla
            and bool(self.ratio)
            and query.shape[0] in (8, 32)
            and (query.shape[0] == 32 or not self.batch_reindex_group)
        )
        output = batch_packed_mla(
            query.contiguous(),
            self.batch_state.swa,
            main_cache,
            selected,
            pages,
            positions,
            slots,
            self.weights.attn_sink,
            self.scale,
            swa_done,
            main_done,
            ratio=self.ratio,
            window=self.window,
            fused_gather=packed_gather,
            sram_gather=packed_gather and self.batch_packed_mla_sram,
            vector_gather=packed_gather and self.batch_packed_mla_vector,
        )
        result = self._finish_output(output, safe_positions, ready_outputs, request_batch=True)
        result = torch.where(active[:, None], result, 0)
        return result, selected, blocks, main_done, index_done

    def insert_context(self, value, positions, valid_count=None):
        kv = rms_norm(self.linear(value, self.weights.wkv), self.weights.kv_norm.weight, self.eps)
        indices = positions.remainder(SWA_ROWS).long()
        packed = pack_swa(self._rope(kv, positions))
        if valid_count is not None:
            old = self.swa.index_select(0, indices)
            mask = torch.arange(indices.numel(), device=indices.device) < valid_count.reshape(1)
            packed = torch.where(mask.reshape(-1, 1), packed, old)
        self.swa.index_copy_(0, indices, packed)
        return self.swa

    def draft(self, value, positions):
        query = rms_norm(self.linear(value, self.weights.wq_a), self.weights.q_norm.weight, self.eps)
        query = self._rope(self.linear(query, self.weights.wq_b).reshape(-1, self.heads, 512), positions)
        kv = rms_norm(self.linear(value, self.weights.wkv), self.weights.kv_norm.weight, self.eps)
        kv = unpack_swa(pack_swa(self._rope(kv, positions)))
        cache = torch.cat((unpack_swa(self.swa), kv), 0)
        window = positions[:1] - self.window + self.window_offsets
        window = torch.where(window >= 0, window.remainder(SWA_ROWS), -1)
        draft_slots = SWA_ROWS + positions - positions[:1]
        indices = torch.cat((window, draft_slots), 0).unsqueeze(0).expand(5, -1).int()
        return self._output(query, cache, indices, positions)
