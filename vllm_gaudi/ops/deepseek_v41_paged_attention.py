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
from vllm_gaudi.ops.deepseek_v41_prefill_regions import (prefill_main_workspace, prefill_q_projection,
                                                         prefill_output_projection)
from vllm_gaudi.ops.deepseek_v41_native_trace import prefill_span
from vllm_gaudi.ops.deepseek_v41_prefill_event_trace import span as prefill_event_span
import torch.nn.functional as F
from torch import nn

from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.ops.deepseek_v41_indexer import INDEX_MME_HOT_TOKENS
from vllm_gaudi.ops.deepseek_v41_qkv import FusedCompressorInput, FusedQKVInput
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
    """Use TP query partition only for graph shapes qualified on Gaudi2."""
    return tokens >= INDEX_QUERY_TP_MIN_TOKENS and 512 < source_rows <= 32768


def candidate_columns(search_length, ratio, capacity):
    """Drop only the appended padding after the reachable candidate prefix."""
    return min(capacity, (search_length // ratio + 7) // 8)


def bounded_prefill_mla(query, cache, indices, sink, scale, tile):
    """Submit bounded real query extents to the large-M MME attention op."""
    if tile not in (16, 32, 64):
        raise ValueError("Prefill MLA query tile must be 16, 32 or 64")
    outputs = []
    for start in range(0, query.shape[0], tile):
        count = min(tile, query.shape[0] - start)
        q = query[start:start + count].contiguous()
        ids = indices[start:start + count].contiguous()
        lengths = torch.full((count, ), indices.shape[-1], dtype=torch.int32, device=query.device)
        outputs.append(
            torch.ops.custom_op.custom_deepseek_v41_prefill_mla_mme_gaudi2(q, cache, ids, sink, scale, lengths))
    return torch.cat(outputs, 0)


@lru_cache(maxsize=32)
def compiled_prefill_mla(signature):
    entry = FunctionType(bounded_prefill_mla.__code__.replace(co_name=f"prefill_mla_{signature}"),
                         bounded_prefill_mla.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def flash_prefill_mla(query, cache, indices, sink):
    """Compile shared-KV gather, sink mask and Gaudi Flash Attention together."""
    from flashinfer_gaudi.mla import sparse_mla_prefill
    lengths = torch.full((query.shape[0], ), indices.shape[1], dtype=torch.int32, device=query.device)
    return sparse_mla_prefill(query, cache, indices, sink, lengths, query_tile=512)


@lru_cache(maxsize=32)
def compiled_flash_prefill_mla(signature):
    entry = FunctionType(flash_prefill_mla.__code__.replace(co_name=f"flash_prefill_mla_{signature}"),
                         flash_prefill_mla.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def _selected_attention_layout(physical, selected, window, swa_offsets, selected_offsets):
    """Build the direct-TPC row list and local attention indices.

    The row list contains each SWA row once, followed by token-major selected
    main rows. Attention indices point into this list and preserve ``-1`` for
    invalid windows/selections.
    """
    row_ids = torch.cat((
        swa_offsets,
        physical.reshape(-1).clamp_min(0).to(torch.int32) + swa_offsets.numel(),
    ), 0).reshape(1, -1).contiguous()
    offset = selected_offsets[:selected.numel()].reshape(selected.shape)
    attention_indices = torch.cat((
        window,
        torch.where(selected >= 0, offset + swa_offsets.numel(), -1),
    ), -1).contiguous()
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

    def __init__(self, config, layer_start, layer_stop, device, max_length):
        super().__init__()
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
                cache.register_buffer("main",
                                      torch.zeros(2 * PAGE_TOKENS // ratio, 288, dtype=torch.uint8, device=device),
                                      False)
                cache.register_buffer("index",
                                      torch.zeros(2 * PAGE_TOKENS // ratio, 68, dtype=torch.uint8, device=device),
                                      False)
                if self.runtime_indexer and ratio in (1, 2):
                    cache.register_buffer(
                        "decoded_index_hot",
                        torch.zeros(INDEX_MME_HOT_TOKENS // ratio, 128, dtype=torch.bfloat16, device=device), False)
                if self.decoded_kv_state:
                    # Keep the exact FP4-roundtripped main rows decoded for
                    # the bounded C1 hot bucket.  The packed paged cache
                    # remains authoritative for the complete 1M context;
                    # this small mirror only extends the proven decoded-MLA
                    # path beyond the old 512-row special case.
                    decoded_rows = (INDEX_MME_HOT_TOKENS // ratio if self.runtime_indexer else 512)
                    cache.register_buffer("decoded_main",
                                          torch.zeros(decoded_rows, 512, dtype=torch.bfloat16, device=device), False)
                self.sources[str(source)] = cache
        for source in config["index_source_layer_ids"]:
            if layer_start <= source < layer_stop:
                selection = nn.Module()
                selection.register_buffer("indices", torch.full((WORK_TOKENS, 512),
                                                                -1,
                                                                dtype=torch.int32,
                                                                device=device), False)
                self.topk[str(source)] = selection
        self.register_buffer(
            "candidate_pool",
            # Keep candidate block ids rather than expanding every block to its
            # eight rows.  Reindex layers expand only the row tile they score.
            # This is equivalent to the upstream candidate-slot contract and
            # cuts the persistent C8192 workspace from 512 MiB to 64 MiB.
            torch.full((WORK_TOKENS, config["candidate_topk_blocks"]), -1, dtype=torch.int32, device=device)
            if layer_start <= config["candidate_source_layer_id"] < layer_stop else None,
            False)
        table = torch.zeros((max_length + PAGE_TOKENS - 1) // PAGE_TOKENS, dtype=torch.int32, device=device)
        table[0] = 1
        self.register_buffer("block_table", table, False)
        if self.runtime_indexer and self.candidate_pool is None:
            # Full layers do not consume candidate IDs, but the fixed native
            # ABI still requires one persistent correctly shaped input.
            self.register_buffer("index_candidates_unused",
                                 torch.full((1, config["candidate_topk_blocks"]), -1, dtype=torch.int32, device=device),
                                 False)
        if self.decoded_kv_state:
            self.register_buffer(
                "decoded_swa", torch.zeros((layer_stop - layer_start) * 512, 512, dtype=torch.bfloat16, device=device),
                False)
        scaling = config["rope_scaling"]
        for name, compressed in (("swa_rotary", False), ("compressed_rotary", True)):
            table = rotary_table(config["qk_rope_head_dim"], max_length,
                                 config["compress_rope_theta"] if compressed else config["rope_theta"],
                                 scaling["original_max_position_embeddings"] if compressed else 0, scaling["factor"],
                                 scaling["beta_fast"], scaling["beta_slow"])
            self.register_buffer(name, table.to(device), False)
            # Native RoPE needs [cos32, sin32], but a complete 1M-position
            # copy costs another 256 MiB for each table on every rank.  The
            # active search bucket is at most the currently reachable prefix;
            # materialize that correctly laid-out bucket lazily below.

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
            else:
                # ``clone`` is intentional: contiguous slices still share the full
                # table allocation and reproduce the short-request compile stall.
                source = getattr(self, name)
                bucket = (source if self.runtime_indexer and length == self.length else source[:length].clone())
            self._rotary_buckets[key] = bucket
        return bucket

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
        blocks = self.block_table.index_select(0,
                                               torch.bitwise_right_shift(rows.flatten(),
                                                                         shift).long()).reshape(rows.shape)
        return blocks * width + torch.bitwise_and(rows, width - 1)


class PagedCSA2Attention(FusedCompressorInput, FusedQKVInput, nn.Module):

    def __init__(self, weights, config, layer, shared, linear, reduce, gather, device):
        super().__init__()
        self.weights, self.shared = weights, shared
        self.woa_fp8 = False
        # Bound by PreparedStage after the FP8 sidecars have been loaded.  The
        # paged and bounded attention implementations share the same MLA
        # output contract, so long-context decode must retain the qualified
        # wo_a -> group32 BF16 boundary -> FP8 wo_b chain as well.
        self.woa_output_roundtrip = False
        self.prepared_output = (gaudi_envs.VLLM_HPU_DSV41_PREPARED_OUTPUT and layer < config["num_hidden_layers"])
        self.output_gemm_layout = (gaudi_envs.VLLM_HPU_DSV41_OUTPUT_GEMM_LAYOUT and layer < config["num_hidden_layers"])
        if self.output_gemm_layout and not self.prepared_output:
            raise ValueError("V4.1 output GEMM layout requires prepared output weights")
        self.qkv_fused_input = gaudi_envs.VLLM_HPU_DSV41_QKV_FUSED_INPUT and layer < config["num_hidden_layers"]
        self._fused_qkv_weight = None
        self._fused_qkv_quantized = False
        self.compressor_fused_input = (gaudi_envs.VLLM_HPU_DSV41_COMPRESSOR_FUSED_INPUT
                                       and layer < config["num_hidden_layers"])
        self._fused_compressor_weight = None
        self._fused_compressor_kv_width = 0
        self.mla_mme = gaudi_envs.VLLM_HPU_DSV41_MLA_MME and layer < config["num_hidden_layers"]
        self.native_rope = gaudi_envs.VLLM_HPU_DSV41_NATIVE_ROPE
        self.q_scale_rope = (gaudi_envs.VLLM_HPU_DSV41_Q_SCALE_ROPE and layer < config["num_hidden_layers"])
        if self.q_scale_rope and not (gaudi_envs.VLLM_HPU_DSV41_NATIVE_ROPE
                                      and gaudi_envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8):
            raise ValueError("Q scale/RoPE fusion requires native RoPE and prepared dense FP8")
        self.fused_norm = (gaudi_envs.VLLM_HPU_DSV41_ATTN_FUSED_NORM and layer < config["num_hidden_layers"])
        self.linear, self.reduce, self.gather = linear, reduce, gather
        self.layer, self.ratio, self.length = layer, config["compress_ratios"][layer], shared.length
        self.runtime_indexer = shared.runtime_indexer
        self.direct_selected_kv = gaudi_envs.VLLM_HPU_DSV41_PAGED_SELECTED_KV
        self.shared_prefix_kv = gaudi_envs.VLLM_HPU_DSV41_SHARED_PREFIX_KV
        self.packed_attn_exp = gaudi_envs.VLLM_HPU_DSV41_PACKED_ATTN_EXP
        self.vector_kv_scales = gaudi_envs.VLLM_HPU_DSV41_VECTOR_KV_SCALES
        self.head_vector_attn = gaudi_envs.VLLM_HPU_DSV41_HEAD_VECTOR_ATTN
        self.sram_kv = gaudi_envs.VLLM_HPU_DSV41_SRAM_KV
        self.search_length = 512
        self.heads, self.groups = config["num_attention_heads"] // 2, config["o_groups"] // 2
        self.index_heads = config["index_n_heads"] // 2
        self.eps, self.window = config["rms_norm_eps"], config["sliding_window"]
        self.owns_kv = layer in config["kv_source_layer_ids"]
        self.owns_index = layer in config["index_source_layer_ids"]
        self.decoded_kv_state = shared.decoded_kv_state
        self.decoded_swa_offset = (layer - shared.layer_start) * 512 if self.decoded_kv_state else 0
        if self.decoded_kv_state and not (self.mla_mme and self.direct_selected_kv and self.shared_prefix_kv):
            raise ValueError("Paged decoded KV requires direct selected rows, shared prefixes and MME MLA")
        self.candidate_source = config["candidate_source_layer_id"]
        if self.ratio:
            kv_source = max(source for source in config["kv_source_layer_ids"] if source <= layer)
            self.kv_source = kv_source
            index_source = max(source for source in config["index_source_layer_ids"] if source <= layer)
            self.index_ratio = config["compress_ratios"][index_source]
            self.cache = shared.sources[str(kv_source)]
            self.selection = shared.topk[str(index_source)]
        self.register_buffer("swa", torch.zeros(SWA_ROWS, 528, dtype=torch.uint8, device=device), False)
        if self.owns_kv and self.ratio == 2:
            self.register_buffer("kv_history", torch.zeros(HISTORY_ROWS, 512, dtype=torch.float32, device=device),
                                 False)
            self.register_buffer("score_history", torch.zeros_like(self.kv_history), False)
        self._rotary_name = "compressed_rotary" if self.ratio else "swa_rotary"
        self.register_buffer("rotary", shared.rotary_bucket(self._rotary_name, self.search_length), False)
        if self.native_rope or self.q_scale_rope:
            self._rotary_native_name = f"{self._rotary_name}_native"
            self.register_buffer("rotary_native", shared.rotary_bucket(self._rotary_native_name, self.search_length),
                                 False)
        self.register_buffer("window_offsets", torch.arange(self.window, dtype=torch.int32, device=device), False)
        self.register_buffer("compressed_offsets", torch.arange(512, dtype=torch.int32, device=device), False)
        self.register_buffer("swa_offsets", torch.arange(SWA_ROWS, dtype=torch.int32, device=device), False)
        self.register_buffer("selected_offsets", torch.arange(WORK_TOKENS * 512, dtype=torch.int32, device=device),
                             False)
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
            self.weights.wo_a.weight = (grouped if self.output_gemm_layout else grouped.transpose(1, 2).contiguous())

    def _rotary_table(self):
        return self.rotary

    def _rotary_native_table(self):
        return self.rotary_native

    def set_search_length(self, length):
        """Bind this layer to the shared independent-storage RoPE bucket."""
        self.search_length = int(length)
        rotary_length = self.length if self.runtime_indexer else self.search_length
        self.rotary = self.shared.rotary_bucket(self._rotary_name, rotary_length)
        if self.native_rope or self.q_scale_rope:
            self.rotary_native = self.shared.rotary_bucket(self._rotary_native_name, rotary_length)

    def _rope(self, value, positions, inverse=False):
        if (self.native_rope and gaudi_envs.VLLM_HPU_DSV41_PREFILL_ROPE and value.dtype == torch.bfloat16
                and value.ndim in (2, 3) and (value.ndim == 2 or 1 <= value.shape[1] <= 128)
                and NATIVE_WORK_TOKENS < value.shape[0] <= 8192 and 128 <= value.shape[-1] <= 512
                and value.shape[-1] % 128 == 0):
            # Large-query eager RoPE rounds separate FP32 products. The small
            # compiled path has a different FMA boundary and keeps its GUID.
            op = (torch.ops.custom_op.custom_deepseek_v41_prefill_rope_inverse_bf16_gaudi2
                  if inverse else torch.ops.custom_op.custom_deepseek_v41_prefill_rope_bf16_gaudi2)
            shaped = value.reshape(value.shape[0], -1, value.shape[-1]).contiguous()
            return op(shaped, positions.to(torch.int32).contiguous(), self._rotary_native_table()).reshape(value.shape)
        if (self.native_rope and value.dtype == torch.bfloat16 and value.ndim in (2, 3)
                and (value.ndim == 2 or 1 <= value.shape[1] <= 128) and 1 <= value.shape[0] <= NATIVE_WORK_TOKENS
                and 128 <= value.shape[-1] <= 512 and value.shape[-1] % 128 == 0):
            op = (torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2
                  if inverse else torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2)
            shaped = value.reshape(value.shape[0], -1, value.shape[-1]).contiguous()
            return op(shaped, positions.to(torch.int32).contiguous(), self._rotary_native_table()).reshape(value.shape)
        return _apply_rope_torch(value, positions, self._rotary_table(), inverse)

    def project_output(self, value):
        """Project MLA output without restoring a bounded-context weight path."""
        if self.woa_fp8:
            operation = (torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2
                         if self.woa_output_roundtrip else torch.ops.custom_op.custom_deepseek_v41_woa_fp8_gaudi2)
            if (self.woa_output_roundtrip and gaudi_envs.VLLM_HPU_DSV41_PREFILL_VECTOR_QUANT and value.shape[0] > 6):
                operation = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_wide_gaudi2
            return operation(value.contiguous(), self.weights.wo_a.weight, self.weights.wo_a.channel_scale)
        if self.output_gemm_layout:
            weight = self.weights.wo_a.weight
            if value.shape[0] == 1:
                return torch.cat(tuple(F.linear(value[:, group], weight[group]) for group in range(self.groups)),
                                 dim=-1)
            return torch.einsum("tgd,grd->tgr", value, weight).flatten(1)
        if self.prepared_output:
            return torch.einsum("tgd,gdr->tgr", value, self.weights.wo_a.weight).flatten(1)
        weight = self.weights.wo_a.weight.reshape(self.groups, 1024, -1)
        return torch.einsum("tgd,grd->tgr", value, weight).flatten(1)

    def project_output_consumer(self, value):
        """Consume the qualified wo_a roundtrip without a BF16 HBM detour."""
        if self.woa_output_roundtrip:
            weight = self.weights.wo_b
            return torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(value.contiguous(), weight.weight,
                                                                            weight.channel_scale)
        return self.linear(value, self.weights.wo_b)

    def project_query(self, value, positions, *, decode=False):
        weight = self.weights.wq_b
        if not decode and gaudi_envs.VLLM_HPU_DSV41_PREFILL_Q_PROJECTION and 6 < value.shape[0] <= 8192:
            if not getattr(weight, "dense_fp8", False) or self.heads != 32:
                raise ValueError("Prefill Q projection requires prepared FP8 weights and 32 local heads")
            value = quantize_activation(value) if hasattr(weight, "scale") else value
            return prefill_q_projection(value.contiguous(), weight.weight, weight.channel_scale,
                                        positions.to(torch.int32).contiguous(), self._rotary_native_table())
        if self.q_scale_rope and value.shape[0] == 1 and getattr(weight, "dense_fp8", False):
            value = quantize_activation(value) if hasattr(weight, "scale") else value
            return torch.ops.custom_op.custom_deepseek_v41_q_projection_rope_gaudi2(
                value, weight.weight, weight.channel_scale, positions,
                self._rotary_native_table()).reshape(-1, self.heads, 512)
        return self._rope(self.linear(value, weight).reshape(-1, self.heads, 512), positions)

    def project_query_input(self, value, positions):
        """Apply the exact C1 Q norm directly in the FP8 projection producer.

        The normalized query row is otherwise written and reread solely by
        the dynamic quantizer. Reindex owner layers still materialize it
        because their index-query projection is a second real consumer.
        Wider decode batches and prefill retain the batch-generic path.
        """
        weight = self.weights.wq_b
        return torch.ops.custom_op.custom_deepseek_v41_q_norm_projection_rope_gaudi2(
            value.contiguous(), self.weights.q_norm.weight, weight.weight, weight.channel_scale,
            positions.to(torch.int32).contiguous(), self._rotary_native_table(), self.eps).reshape(-1, self.heads, 512)

    def project_kv(self, value, positions, *, decode=False):
        """Normalize and rotate KV without materializing the C1 norm output."""
        weight = self.weights.kv_norm.weight
        if decode and self.fused_norm and self.native_rope and value.shape[0] == 1:
            # KV has no consumer between its BF16 RMSNorm boundary and RoPE.
            # The compound TPC kernel preserves that boundary bit-for-bit and
            # feeds the existing cache writer directly.  Wider batches retain
            # the batch-generic implementation used by prefill and B2+.
            return torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_bf16_gaudi2(
                value.contiguous(), weight,
                positions.to(torch.int32).contiguous(), self._rotary_native_table(), self.eps)
        norm = (torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2 if self.fused_norm and
                (value.shape[0] == 1 or (not decode and gaudi_envs.VLLM_HPU_DSV41_PREFILL_NATIVE_NORM
                                         and NATIVE_WORK_TOKENS < value.shape[0] <= 8192)) else rms_norm)
        return self._rope(norm(value.contiguous(), weight, self.eps), positions)

    def _compress(self, value, positions, decoded=False):
        compressor = self.weights.compressor
        if self.ratio == 2:
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
                    self.kv_history, self.score_history, kv.contiguous(), score.contiguous(),
                    positions.to(torch.int32).contiguous())
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
        if self.ratio & (self.ratio - 1):
            raise ValueError("V4.1 compressor ratio must be a power of two")
        ratio_shift = self.ratio.bit_length() - 1
        rows = torch.bitwise_right_shift(positions, ratio_shift)
        visible = (torch.bitwise_and(positions, self.ratio - 1) == self.ratio - 1)
        # Incomplete groups use unique rows in the reserved null page.
        slots = torch.where(visible, self.shared.physical_rows(rows, self.ratio),
                            torch.bitwise_and(rows, PAGE_TOKENS // self.ratio - 1))
        indexer = self.weights.indexer
        index = rms_norm(self.linear(latent, indexer.wk), indexer.k_norm.weight, self.eps)
        index, latent = self._rope(index, first), self._rope(latent, first)
        if (self.runtime_indexer and self.ratio in (1, 2)
                # The <=512 static bucket is the producer for the later hot
                # MME bucket.  Populate its prefix before the search geometry
                # switches at token 513; otherwise rows 0..511 remain zero.
                and self.search_length <= INDEX_MME_HOT_TOKENS and hasattr(self.cache, "decoded_index_hot")):
            # The mirror must contain the same values as the packed index
            # cache.  Keeping the pre-quantization key would silently change
            # candidate selection when this MME path replaces packed scoring.
            self.cache.decoded_index_hot.index_copy_(0, rows.long(), fp4_roundtrip(index, 32))
        if decoded and value.shape[0] == 1:
            return torch.ops.custom_op.custom_deepseek_v41_fp4_paged_decoded_write_bf16_gaudi2(
                self.cache.main, self.cache.index, latent.contiguous(), index.contiguous(),
                slots.to(torch.int32).contiguous(),
                rows.to(torch.int32).contiguous(), self.cache.decoded_main)
        packed_index, packed_main = pack_fp4(index, 32), pack_fp4(latent, 16)
        self.cache.index.index_copy_(0, slots.long(), packed_index)
        self.cache.main.index_copy_(0, slots.long(), packed_main)
        if decoded:
            self.cache.decoded_main.index_copy_(0, rows.long(), unpack_fp4(packed_main))
        return None

    def _prepare_index_queries(self, value, qr, positions):
        indexer = self.weights.indexer
        q = self.linear(qr, indexer.wq_b).reshape(-1, self.index_heads, 128)
        q = self._rope(q, positions)
        # Quantize and restore in one TPC pass.  The codec keeps packed FP4
        # codes and UE8M0 scales in registers, so prefill does not materialize
        # a packed query in HBM or compile a generic bit-unpack graph.  Tile by
        # the native flattened-row contract; for TP2's 16 heads this is C512.
        codec_tokens = max(1, 8192 // self.index_heads)
        pieces = [fp4_roundtrip(q[start:start + codec_tokens], 32) for start in range(0, q.shape[0], codec_tokens)]
        q = pieces[0] if len(pieces) == 1 else torch.cat(pieces, 0)
        weights = self.linear(value, indexer.weights_proj) * (128**-0.5 * (self.index_heads * 2)**-0.5)
        # Exchange only small query/head tensors, not one score per cached token.
        q, weights = self.gather(q, 1), self.gather(weights, 1)
        return q, weights

    def _scores(self, positions, logical_rows, q, weights):
        physical = self.shared.physical_rows(logical_rows.clamp_min(0), self.ratio)
        packed = self.cache.index.index_select(0, physical.flatten().long()).reshape(*physical.shape, 68)
        keys = unpack_fp4(packed, 128, 32)
        # Use explicit contiguous GEMMs.  The generic einsum accepted strided
        # query/key views and could require a new, much larger recipe beside
        # the 1M-context resident set.  These forms produce bitwise-identical
        # BF16 scores while giving Bridge the final MME geometry directly.
        scores = (torch.matmul(q.contiguous(),
                               keys.transpose(0, 1).contiguous()) if keys.ndim == 2 else torch.bmm(
                                   q.contiguous(),
                                   keys.transpose(1, 2).contiguous()))
        scores = scores.relu() * weights.unsqueeze(-1)
        # Retain the two TP partial-sum BF16 boundaries of the checkpoint reference.
        scores = scores.reshape(q.shape[0], 2, self.index_heads, -1).sum(2).sum(1)
        count = ((positions + 1) // self.ratio).unsqueeze(-1)
        valid = (logical_rows >= 0) & (logical_rows < count)
        return scores.float().masked_fill(~valid, -torch.inf)

    def _prefill_scores(self, positions, rows, q, weights):
        if gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_SRAM and rows.ndim == 1:
            from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import full_prefill_sram_scores
            return full_prefill_sram_scores(q, weights, self.cache.index, self.shared.block_table, positions, rows,
                                            self.ratio)
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

    def _stream_topk(self, positions, rows, q, weights, width=512, collect_blocks=False, native_scores=False,
                     visible_rows=None):
        """Exact bounded-memory index selection for a large prompt block."""
        columns = rows.shape[-1]
        if visible_rows is not None and (rows.ndim != 1 or not 0 <= visible_rows <= columns):
            raise ValueError("Visible source prefix requires contiguous logical source rows")
        best_scores = best_rows = None
        block_scores = block_ids = None
        invalid_scores = None
        for start in range(0, columns, PREFILL_INDEX_ROWS):
            current_rows = (rows[start:start + PREFILL_INDEX_ROWS] if rows.ndim == 1 else rows[:, start:start +
                                                                                               PREFILL_INDEX_ROWS])
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
                scores = scorer(positions, current_rows, q, weights)
            # An all-invalid tile cannot change valid Top512 rows. Omitting the
            # large merge is exact after invalid rows are normalized to -1,
            # while retaining candidate-block merges preserves their tie order.
            if not invisible_tile:
                best_scores, best_rows = self._merge_topk(best_scores, best_rows, scores, current_rows, width)
            if collect_blocks:
                if scores.shape[-1] % 8:
                    pad = 8 - scores.shape[-1] % 8
                    scores = F.pad(scores, (0, pad), value=-torch.inf)
                grouped = scores.reshape(scores.shape[0], -1, 8).amax(-1)
                ids = torch.arange(start // 8,
                                   start // 8 + grouped.shape[-1],
                                   device=positions.device,
                                   dtype=torch.int32)
                count = ((positions + 1) // self.ratio).unsqueeze(-1)
                # Match the reference: keep the block containing the newest
                # visible token so its causal prefix cannot be dropped.
                grouped = grouped.masked_fill(ids == ((count - 1) // 8), torch.inf)
                block_scores, block_ids = self._merge_topk(block_scores, block_ids, grouped, ids, 2048)
        return best_rows, block_scores, block_ids

    def _select(self, value, qr, positions, *, buffer_start=0, prepared=None, prefill=False):
        count, tokens = (positions + 1) // self.ratio, positions.numel()
        target = slice(buffer_start, buffer_start + tokens)
        if self.owns_index:
            # The dynamic indexer is required once the visible prefix exceeds
            # the fixed 512-row candidate window.  For a short C1 prefix the
            # ordered compressed offsets are already the exact candidate set;
            # dispatching score/threshold/emit here adds four device launches
            # per owning layer and regresses the historical C1 replay.  Keep
            # this fast path valid for the long-context runtime-indexer build;
            # only the >512-row case needs dynamic selection.
            if (self.runtime_indexer and tokens == 1 and not prefill and self.search_length // self.ratio > 512):
                from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select
                q, weights = self._prepare_index_queries(value, qr, positions) if prepared is None else prepared
                pool = (self.shared.index_candidates_unused
                        if self.shared.candidate_pool is None else self.shared.candidate_pool[target].contiguous())
                decoded_hot = (self.cache.decoded_index_hot if
                               (self.ratio in (1, 2) and self.search_length == INDEX_MME_HOT_TOKENS
                                and hasattr(self.cache, "decoded_index_hot")) else None)
                indices, blocks = runtime_index_select(q.contiguous(),
                                                       weights.contiguous(),
                                                       self.cache.index,
                                                       self.shared.block_table,
                                                       positions.to(torch.int32).contiguous(),
                                                       pool,
                                                       ratio=self.ratio,
                                                       capacity=self.length // self.ratio,
                                                       reindex=self.layer > self.candidate_source,
                                                       publish_candidates=self.layer == self.candidate_source,
                                                       decoded_hot=decoded_hot)
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
                        blocks = blocks[:, :candidate_columns(self.search_length, self.ratio, blocks.shape[-1])]
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
                    native_scores=prefill and gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_MME)
                if self.layer == self.candidate_source:
                    valid = candidate_scores > -torch.inf
                    candidate_blocks = torch.where(valid, candidate_blocks, -1).to(torch.int32)
                    candidate_blocks = F.pad(candidate_blocks,
                                             (0, self.shared.candidate_pool.shape[-1] - candidate_blocks.shape[-1]),
                                             value=-1)
                    self.shared.candidate_pool[target].copy_(candidate_blocks)
                # Sort logical positions, with invalid picks after reachable ones.
                indices = torch.where((indices >= 0) & (indices < count.unsqueeze(-1)), indices, self.length)
                indices = indices.sort(-1).values
                indices = torch.where(indices < self.length, indices, -1).int()
            self.selection.indices[target].copy_(indices)
        return self.selection.indices[target]

    def _finish_output(self, output, positions, ready_outputs=(), *, prefill=False):
        """Apply the shared output projection after either attention path."""
        if prefill:
            with prefill_event_span("attention_output_inverse_rope", self.layer, output.shape[0]):
                output = self._rope(output, positions, inverse=True)
        else:
            output = self._rope(output, positions, inverse=True)
        output = output.reshape(-1, self.groups, self.heads // self.groups * 512)
        if prefill and gaudi_envs.VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION and 6 < output.shape[0] <= 8192:
            wa, wb = self.weights.wo_a, self.weights.wo_b
            if not (self.woa_fp8 and self.woa_output_roundtrip and getattr(wb, "dense_fp8", False)):
                raise ValueError("Prefill output projection requires prepared FP8 weights and the group codec")
            with prefill_event_span("attention_output_projection", self.layer, output.shape[0]):
                partial = prefill_output_projection(output.contiguous(), wa.weight, wa.channel_scale, wb.weight,
                                                    wb.channel_scale)
            with prefill_event_span("attention_output_reduce", self.layer, output.shape[0]):
                return self.reduce(partial, ready_outputs=ready_outputs) if ready_outputs else self.reduce(partial)
        if prefill:
            with prefill_event_span("attention_output_projection", self.layer, output.shape[0]):
                output = self.project_output(output)
                partial = self.project_output_consumer(output)
            with prefill_event_span("attention_output_reduce", self.layer, output.shape[0]):
                return self.reduce(partial, ready_outputs=ready_outputs) if ready_outputs else self.reduce(partial)
        output = self.project_output(output)
        return self._finish_projected_output(output, ready_outputs)

    def _finish_projected_output(self, output, ready_outputs=()):
        """Consume an already inverse-rotated and wo_a-projected row."""
        partial = self.project_output_consumer(output)
        return self.reduce(partial, ready_outputs=ready_outputs) if ready_outputs else self.reduce(partial)

    def _can_fuse_mla_woa(self, positions):
        """Return whether the exact MLA-product/wo_a producer is available."""
        return (positions.numel() == 1 and self.mla_mme and self.woa_fp8 and self.woa_output_roundtrip
                and self.native_rope)

    def _output(self, query, cache, indices, positions, ready_outputs=(), *, decode=False):
        if decode and self.mla_mme and 1 <= query.shape[0] <= NATIVE_WORK_TOKENS:
            lengths = torch.full((query.shape[0], ), indices.shape[1], dtype=torch.int32, device=query.device)
            output = torch.ops.custom_op.custom_deepseek_v41_selected_mla_mme_gaudi2(
                query.contiguous(), cache.contiguous(), indices.contiguous(), self.weights.attn_sink, self.scale,
                lengths)
            return self._finish_output(output, positions, ready_outputs)
        output, _, _ = torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(
            query.contiguous(), cache.contiguous(), indices.contiguous(), self.weights.attn_sink, self.scale)
        return self._finish_output(output, positions, ready_outputs)

    @prefill_span("index_selection")
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
        if (self.owns_index and gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP
                and can_partition_prefill_index(positions.numel(), source_rows)):
            from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import (
                tp_full_prefill_index_selection, tp_prefill_reindex_selection)
            rank = getattr(self, "prefill_tp_rank", None)
            if rank not in (0, 1):
                raise RuntimeError("Prefill index query partition has no prepared TP rank")
            q, weights = self._prepare_index_queries(value, qr, positions) if prepared is None else prepared
            if self.layer <= self.candidate_source:
                visible = None
                if gaudi_envs.VLLM_HPU_DSV41_PREFILL_INDEX_VISIBLE:
                    token_end = getattr(self, "prefill_token_end", None)
                    if token_end is None or not positions.numel() <= token_end <= self.search_length:
                        raise RuntimeError("Prefill source visibility has no valid scheduler interval")
                    visible = token_end // self.ratio
                selected, blocks = tp_full_prefill_index_selection(
                    q, weights, self.cache.index, self.shared.block_table, positions.to(torch.int32), self.ratio,
                    source_rows, self.layer == self.candidate_source, rank, self.gather, visible)
                if blocks is not None:
                    self.shared.candidate_pool[:positions.numel()].copy_(blocks)
            else:
                pool = self.shared.candidate_pool[:positions.numel()]
                if gaudi_envs.VLLM_HPU_DSV41_PREFILL_COMPACT_CANDIDATES:
                    pool = pool[:, :candidate_columns(self.search_length, self.ratio, pool.shape[-1])]
                selected = tp_prefill_reindex_selection(
                    q, weights, self.cache.index, self.shared.block_table, positions.to(torch.int32), pool,
                    self.ratio, source_rows, rank, self.gather)
            self.selection.indices[:positions.numel()].copy_(selected)
            return self.selection.indices[:positions.numel()]
        # Full layers score the same one-dimensional source row range for all
        # query rows.  Streaming source rows while keeping all 8192 queries
        # removes repeated per-query-tile Python submissions. The C8192 x
        # 32-head x 2048-key BF16 score tile occupies 1 GiB before reduction.
        # Reindex rows are token-dependent (T x 16384), so keep only
        # that selection calculation on the smaller query tile.
        if not self.owns_index or self.layer <= self.candidate_source:
            return self._select(value, qr, positions, prepared=prepared, prefill=True)
        if gaudi_envs.VLLM_HPU_DSV41_PREFILL_REINDEX_REUSE:
            from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import (SHARED_INDEX_MAX_ROWS,
                                                                          compiled_decode_shared_index_keys,
                                                                          compiled_decoded_reindex)
            source_rows = self.search_length // self.ratio
            if 512 < source_rows <= SHARED_INDEX_MAX_ROWS:
                q, weights = self._prepare_index_queries(value, qr, positions) if prepared is None else prepared
                signature = (tuple(self.cache.index.shape), tuple(self.shared.block_table.shape), self.ratio,
                             source_rows)
                keys = compiled_decode_shared_index_keys(signature)(self.cache.index, self.shared.block_table,
                                                                    self.ratio, source_rows)
                for start in range(0, positions.numel(), PREFILL_ATTN_TOKENS):
                    stop = min(start + PREFILL_ATTN_TOKENS, positions.numel())
                    pool = self.shared.candidate_pool[start:stop]
                    if gaudi_envs.VLLM_HPU_DSV41_PREFILL_COMPACT_CANDIDATES:
                        pool = pool[:, :candidate_columns(self.search_length, self.ratio, pool.shape[-1])]
                    native_gather = gaudi_envs.VLLM_HPU_DSV41_PREFILL_CANDIDATE_GATHER
                    native_scores = gaudi_envs.VLLM_HPU_DSV41_PREFILL_REINDEX_SRAM
                    signature = (stop - start, tuple(pool.shape), source_rows, self.ratio, self.index_heads,
                                 native_gather, native_scores)
                    selected = compiled_decoded_reindex(signature)(q[start:stop].clone(), weights[start:stop].clone(),
                                                                   keys, positions[start:stop].clone(), pool.clone(),
                                                                   self.ratio, self.index_heads, native_gather,
                                                                   native_scores)
                    self.selection.indices[start:stop].copy_(selected)
                # Each tile already publishes to the transaction's persistent
                # selection buffer. Consume that buffer directly instead of
                # concatenating outputs after mutating their published views.
                return self.selection.indices[:positions.numel()]
        for start in range(0, positions.numel(), PREFILL_ATTN_TOKENS):
            stop = min(start + PREFILL_ATTN_TOKENS, positions.numel())
            current_prepared = ((prepared[0][start:stop], prepared[1][start:stop]) if prepared is not None else None)
            self._select(value[start:stop],
                         qr[start:stop],
                         positions[start:stop],
                         buffer_start=start,
                         prepared=current_prepared,
                         prefill=True)
        return self.selection.indices[:positions.numel()]

    def _prefill_swa_workspace(self, kv, positions, *, decoded=False):
        """Bind request state explicitly to the reusable SWA tensor region."""
        from vllm_gaudi.ops.deepseek_v41_prefill_regions import prefill_swa_workspace
        return prefill_swa_workspace(kv, positions, self.swa, self.window_offsets,
                                     self.shared.decoded_swa if decoded else None,
                                     self.decoded_swa_offset if decoded else 0)

    def _prefill_attention_tiled(self, value, query, kv, positions, selected, ready_outputs=()):
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
                offset = torch.arange(current_selected.numel(), device=value.device,
                                      dtype=torch.int32).reshape(current_selected.shape)
                indices = torch.cat((indices, torch.where(current_selected >= 0, offset + SWA_ROWS, -1)), -1)
            if cache is None:
                cache = unpack_swa(self.swa)
            output = self._prefill_sparse(query[start:stop], cache, indices)
            outputs.append(output)
        return self._finish_output(torch.cat(outputs, 0), positions, ready_outputs, prefill=True)

    @prefill_span("attention_mla")
    def _prefill_sparse(self, query, cache, indices):
        if gaudi_envs.VLLM_HPU_DSV41_FLASHINFER_PREFILL:
            outputs = []
            # Retain C512 HPU FusedSDPA tiles inside one larger compiled
            # recipe. A complete C8192 prefill then publishes four recipes
            # instead of sixteen without changing attention arithmetic.
            outer_rows = 2048
            for start in range(0, query.shape[0], outer_rows):
                stop = min(start + outer_rows, query.shape[0])
                signature = (stop - start, tuple(query.shape[1:]), tuple(cache.shape), indices.shape[1])
                outputs.append(
                    compiled_flash_prefill_mla(signature)(query[start:stop].clone(), cache, indices[start:stop].clone(),
                                                          self.weights.attn_sink))
            return torch.cat(outputs, 0)
        tile = gaudi_envs.VLLM_HPU_DSV41_PREFILL_MLA_ROWS
        if tile:
            # The long-context bridge used by the qualified decode path loses
            # the dtype of an internal reinterpret_cast when this helper is
            # lowered as a nested torch.compile recipe.  The identical native
            # MME op is valid (including C2052 and its four-row tail), so submit
            # the bounded C64 tiles directly.  This affects prefill only; C1
            # decode still uses the captured native joint-replay program below.
            return bounded_prefill_mla(query, cache, indices, self.weights.attn_sink, self.scale, tile)
        return torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(query.contiguous(), cache.contiguous(),
                                                                              indices.contiguous(),
                                                                              self.weights.attn_sink, self.scale)[0]

    def _prefill_main_workspace(self, cache, indices, selected, logical):
        workspace = self.shared.prefill_main_workspace
        if workspace is None:
            return prefill_main_workspace(self.cache.main, self.shared.block_table, cache, indices, selected, logical,
                                          self.ratio)
        main = workspace.get(self.kv_source,
                             self.cache.main,
                             self.shared.block_table,
                             logical,
                             self.ratio,
                             producer=self.owns_kv)
        return workspace.assemble(main, cache, indices, selected)

    def _prefill_attention(self, value, qr, query, kv, positions, ready_outputs=()):
        """Run one normal C8192 prefill transaction with large-M operators.

        As in upstream V4.1 sparse prefill, prompt KV lives in a flat workspace
        and each query carries causal indices into it. For the common <=64K
        prefix, decode the small compressed main cache once and submit one MLA
        operation per layer. A bounded selected-row implementation remains for
        the rest of the 1M contract.
        """
        if positions.numel() > WORK_TOKENS:
            raise ValueError(f"V4.1 prefill transaction exceeds C{WORK_TOKENS}")
        diagnostic = (os.getenv("VLLM_HPU_DSV41_PREFILL_DIAG_BOUNDARIES", "0") == "1" and positions.numel() == 256
                      and self.search_length == 32768)

        def boundary(name):
            if diagnostic:
                print(f"PREFILL_DIAG layer={self.layer} boundary={name} submitted", flush=True)
                torch.hpu.synchronize()
                print(f"PREFILL_DIAG layer={self.layer} boundary={name} complete", flush=True)

        boundary("input")
        decoded = (self.decoded_kv_state and self.ratio in (1, 2)
                   and self.search_length // self.ratio <= self.cache.decoded_main.shape[0])
        if self.ratio and self.owns_kv:
            with prefill_event_span("attention_compress_kv", self.layer, value.shape[0]):
                self._compress(value, positions, decoded=decoded)
        boundary("compress")
        with prefill_event_span("attention_index_query", self.layer, value.shape[0]):
            prepared = self._prepare_index_queries(value, qr, positions) if self.ratio and self.owns_index else None
        boundary("index_query")
        # Selection is shared by the attention workspace tiles. Full layers
        # can stream each index-K source tile once for all query rows; Reindex
        # layers retain their bounded query tiling in _prefill_selections.
        selected = self._prefill_selections(value, qr, positions, prepared)
        selected_dump = os.getenv("VLLM_HPU_DSV41_DUMP_SELECTED")
        if (selected_dump and self.layer == 2 and selected is not None
                and getattr(self, "prefill_tp_rank", 0) == 0):
            # Diagnostic only: a real selected-ID distribution is needed to
            # decide whether nearby queries can share one compact KV union.
            # The host synchronization here must never enter a timed run.
            from pathlib import Path
            destination = Path(selected_dump)
            destination.mkdir(parents=True, exist_ok=True)
            first_position = int(positions[0].item())
            torch.save({"positions": positions.detach().cpu(), "selected": selected.detach().cpu()},
                       destination / f"layer2-start{first_position}-pid{os.getpid()}.pt")
        boundary("selection")
        main_rows = self.search_length // self.ratio if self.ratio else 0
        if main_rows > PREFILL_MAIN_CACHE_ROWS:
            return self._prefill_attention_tiled(value, query, kv, positions, selected, ready_outputs)

        with prefill_event_span("attention_swa_workspace", self.layer, value.shape[0]):
            cache, indices = self._prefill_swa_workspace(kv, positions, decoded=decoded)
        boundary("swa")
        if self.ratio:
            logical = torch.arange(main_rows, device=value.device, dtype=torch.int32)
            with prefill_event_span("attention_main_workspace", self.layer, value.shape[0]):
                cache, indices = self._prefill_main_workspace(cache, indices, selected, logical)
        boundary("main")
        output = self._prefill_sparse(query, cache, indices)
        boundary("mla")
        return self._finish_output(output, positions, ready_outputs, prefill=True)

    @prefill_span("attention")
    def forward(self, value, positions, ready_outputs=(), *, decode=False):
        prefill = not decode and value.shape[0] > NATIVE_WORK_TOKENS
        if prefill:
            with prefill_event_span("attention_input_projection", self.layer, value.shape[0]):
                query_input, kv_input = self._project_qkv_input(value)
        else:
            query_input, kv_input = self._project_qkv_input(value)
        norm = (torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2 if self.fused_norm and
                (value.shape[0] == 1 or (not decode and gaudi_envs.VLLM_HPU_DSV41_PREFILL_NATIVE_NORM
                                         and NATIVE_WORK_TOKENS < value.shape[0] <= 8192)) else rms_norm)
        # The Q/KV split returns offset views. Bridge lowers large-prefill
        # float casts from those views to reinterpret_cast nodes whose dtype
        # metadata is invalid under the resident 1M graph. Materialize the
        # two bounded inputs before normalization; the qualified C1 path
        # already receives contiguous buffers and is unchanged.
        # A dynamic Reindex owner consumes the normalized Q row twice: once
        # for attention and once for candidate scoring. Other C1 layers can
        # keep this exact BF16 boundary inside the projection compound op.
        needs_index_query = (self.ratio and self.owns_index and self.search_length // self.ratio > 512)
        fused_query_norm = (decode and value.shape[0] == 1 and self.fused_norm and self.q_scale_rope
                            and self.native_rope and getattr(self.weights.wq_b, "dense_fp8", False)
                            and not needs_index_query)
        if fused_query_norm:
            qr = None
            query = self.project_query_input(query_input, positions)
        else:
            if prefill:
                with prefill_event_span("attention_query_norm_projection", self.layer, value.shape[0]):
                    qr = norm(query_input.contiguous(), self.weights.q_norm.weight, self.eps)
                    query = self.project_query(qr, positions, decode=decode)
            else:
                qr = norm(query_input.contiguous(), self.weights.q_norm.weight, self.eps)
                query = self.project_query(qr, positions, decode=decode)
        if prefill:
            with prefill_event_span("attention_kv_norm_rope", self.layer, value.shape[0]):
                kv = self.project_kv(kv_input, positions, decode=decode)
        else:
            kv = self.project_kv(kv_input, positions, decode=decode)
        if not decode and value.shape[0] > NATIVE_WORK_TOKENS:
            return self._prefill_attention(value, qr, query, kv, positions, ready_outputs)
        decoded = (self.decoded_kv_state and self.ratio in (1, 2)
                   and self.search_length // self.ratio <= self.cache.decoded_main.shape[0])
        completion = None
        if decoded and value.shape[0] == 1:
            # The native writer maps this logical position into the shared
            # 256-row circular namespace for both packed and decoded state.
            # Keeping the modulo inside that existing TPC pass avoids one
            # generic div/mod launch in every decoded Attention layer.
            logical_position = positions.to(torch.int32).contiguous()
            completion = torch.ops.custom_op.custom_deepseek_v41_swa_paged_decoded_write_bf16_gaudi2(
                self.swa, kv.contiguous(), logical_position, logical_position, self.shared.decoded_swa,
                self.decoded_swa_offset)
        else:
            packed_swa = pack_swa(kv)
            self.swa.index_copy_(0, positions.remainder(SWA_ROWS).long(), packed_swa)
            if decoded:
                self.shared.decoded_swa.index_copy_(0,
                                                    positions.remainder(SWA_ROWS).long() + self.decoded_swa_offset,
                                                    unpack_swa(packed_swa))
        native_prefix = (decode and gaudi_envs.VLLM_HPU_DSV41_FUSED_PREFIX_LAYOUT and self.ratio in (1, 2)
                         and self.direct_selected_kv and self.shared_prefix_kv and self.window == 128
                         and value.device.type == "hpu" and 1 <= value.shape[0] <= NATIVE_WORK_TOKENS
                         and self.search_length // self.index_ratio <= 512)
        indices = None
        if not native_prefix:
            window = positions.unsqueeze(-1) - self.window + 1 + self.window_offsets.unsqueeze(0)
            indices = torch.where(window >= 0, window.remainder(SWA_ROWS), -1).int()
        cache = None
        compressed_completion = None
        if self.ratio:
            if self.owns_kv:
                compressed_completion = self._compress(value, positions, decoded)
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
                        query.contiguous(), self.shared.decoded_swa, self.cache.decoded_main, selected.contiguous(),
                        self.weights.attn_sink, self.scale, completion, main_done, self.decoded_swa_offset,
                        self.cache.decoded_main.shape[0], self.weights.wo_a.weight, self.weights.wo_a.channel_scale,
                        positions.to(torch.int32).contiguous(), self._rotary_native_table(), self.weights.wo_b.weight,
                        self.weights.wo_b.channel_scale)
                    return (self.reduce(partial, ready_outputs=ready_outputs)
                            if ready_outputs else self.reduce(partial))
                # The paged state keeps SWA as a 256-row ring and main KV as
                # logical compressed rows.  Reuse the fixed prefix layout so
                # the decoded mirror sees [SWA ring, logical main] rather than
                # the legacy non-paged [512 absolute SWA, main] namespace.
                layout = (torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r1_i32_gaudi2
                          if self.ratio == 1 else torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r2_i32_gaudi2)
                _, indices, lengths = layout(selected.contiguous(),
                                             positions.to(torch.int32).contiguous(), self.shared.block_table)
                output = torch.ops.custom_op.custom_deepseek_v41_mla_mme_gaudi2(
                    query.contiguous(), self.shared.decoded_swa,
                    self.cache.decoded_main, indices.contiguous(), self.weights.attn_sink, self.scale,
                    lengths.contiguous(), completion, main_done, self.decoded_swa_offset,
                    self.cache.decoded_main.shape[0], SWA_ROWS)
                return self._finish_output(output, positions, ready_outputs)
            physical = None if native_prefix else self.shared.physical_rows(selected.clamp_min(0), self.ratio)
            if (decode and self.direct_selected_kv and value.device.type == "hpu"
                    and value.shape[0] <= NATIVE_WORK_TOKENS and selected.numel() <= self.selected_offsets.numel()):
                # Decode the fixed SWA prefix and all selected paged rows in
                # one graph entry, then consume its internal BF16 value directly
                # in sparse attention. This removes unpack_swa, cat, and the
                # exposed selected-row temporary from the Python graph.
                if self.shared_prefix_kv and self.search_length // self.index_ratio <= 512:
                    if native_prefix:
                        layout = (torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r1_i32_gaudi2 if self.ratio == 1
                                  else torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r2_i32_gaudi2)
                        row_ids, attention_indices, lengths = layout(selected.contiguous(),
                                                                     positions.to(torch.int32).contiguous(),
                                                                     self.shared.block_table)
                    else:
                        row_ids, attention_indices, lengths = _shared_prefix_attention_layout(
                            physical, selected, indices, self.swa_offsets)
                    attention_op = (
                        torch.ops.custom_op.custom_deepseek_v41_paged_mla_mme_gaudi2 if self.mla_mme else
                        torch.ops.custom_op.custom_deepseek_v41_paged_attention_head_vector_bf16_gaudi2
                        if self.head_vector_attn else
                        torch.ops.custom_op.custom_deepseek_v41_paged_attention_sram_bf16_gaudi2 if self.sram_kv else
                        torch.ops.custom_op.custom_deepseek_v41_paged_attention_vector_scales_bf16_gaudi2 if self.
                        vector_kv_scales else torch.ops.custom_op.
                        custom_deepseek_v41_paged_attention_packed_exp_bf16_gaudi2 if self.packed_attn_exp else torch.
                        ops.custom_op.custom_deepseek_v41_paged_attention_lengths_bf16_gaudi2)
                    output = attention_op(query.contiguous(), self.swa.contiguous(), self.cache.main.contiguous(),
                                          row_ids, attention_indices, self.weights.attn_sink, self.scale, lengths)
                    return self._finish_output(output, positions, ready_outputs)
                row_ids, attention_indices = _selected_attention_layout(physical, selected, indices, self.swa_offsets,
                                                                        self.selected_offsets)
                if self.mla_mme:
                    lengths = torch.full((value.shape[0], ),
                                         attention_indices.shape[1],
                                         dtype=torch.int32,
                                         device=value.device)
                    output = torch.ops.custom_op.custom_deepseek_v41_paged_mla_mme_gaudi2(
                        query.contiguous(), self.swa.contiguous(), self.cache.main.contiguous(), row_ids,
                        attention_indices, self.weights.attn_sink, self.scale, lengths)
                    return self._finish_output(output, positions, ready_outputs)
                output = torch.ops.custom_op.custom_deepseek_v41_paged_attention_bf16_gaudi2(
                    query.contiguous(), self.swa.contiguous(), self.cache.main.contiguous(), row_ids, attention_indices,
                    self.weights.attn_sink, self.scale)
                return self._finish_output(output, positions, ready_outputs)
            else:
                packed = self.cache.main.index_select(0, physical.flatten().long())
                cache = torch.cat((unpack_swa(self.swa), unpack_fp4(packed)), 0)
            offset = torch.arange(selected.numel(), device=value.device, dtype=torch.int32).reshape(selected.shape)
            indices = torch.cat((indices, torch.where(selected >= 0, offset + SWA_ROWS, -1)), -1)
        if cache is None:
            cache = unpack_swa(self.swa)
        return self._output(query, cache, indices, positions, ready_outputs, decode=decode)

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
