# SPDX-License-Identifier: Apache-2.0
"""Paged CSA2 with bounded working state and the checkpoint's two-level indexer.

Selection follows DeepSeek dba1be0 inference/model.py Indexer and
select_candidate_blocks. KV sharing follows vLLM #56214 e47aa780.
The default HPU path gathers compressed rows before decoding attention values.
The opt-in selected-row path decodes only requested rows in TPC.
"""

import torch
import torch.nn.functional as F
from torch import nn

from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.ops.deepseek_v41_qkv import FusedCompressorInput, FusedQKVInput
from vllm_gaudi.ops.deepseek_v41_math import (
    apply_rope,
    pack_fp4,
    pack_swa,
    rms_norm,
    rotary_table,
    unpack_fp4,
    unpack_swa,
)

PAGE_TOKENS = 128
WORK_TOKENS = 6
SWA_ROWS = 256
HISTORY_ROWS = 8


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
        self.length = max_length
        self.sources, self.topk = nn.ModuleDict(), nn.ModuleDict()
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
            torch.full((WORK_TOKENS, config["candidate_topk_blocks"], config["candidate_block_size"]),
                       -1,
                       dtype=torch.int32,
                       device=device) if layer_start <= config["candidate_source_layer_id"] < layer_stop else None,
            False)
        table = torch.zeros((max_length + PAGE_TOKENS - 1) // PAGE_TOKENS, dtype=torch.int32, device=device)
        table[0] = 1
        self.register_buffer("block_table", table, False)
        scaling = config["rope_scaling"]
        for name, compressed in (("swa_rotary", False), ("compressed_rotary", True)):
            table = rotary_table(config["qk_rope_head_dim"], max_length,
                                 config["compress_rope_theta"] if compressed else config["rope_theta"],
                                 scaling["original_max_position_embeddings"] if compressed else 0, scaling["factor"],
                                 scaling["beta_fast"], scaling["beta_slow"])
            self.register_buffer(name, table.to(device), False)

    def physical_rows(self, rows, ratio):
        width = PAGE_TOKENS // ratio
        blocks = self.block_table.index_select(0, (rows.flatten() // width).long()).reshape(rows.shape)
        return blocks * width + rows.remainder(width)


class PagedCSA2Attention(FusedCompressorInput, FusedQKVInput, nn.Module):

    def __init__(self, weights, config, layer, shared, linear, reduce, gather, device):
        super().__init__()
        self.weights, self.shared = weights, shared
        self.qkv_fused_input = gaudi_envs.VLLM_HPU_DSV41_QKV_FUSED_INPUT and layer < config["num_hidden_layers"]
        self._fused_qkv_weight = None
        self._fused_qkv_quantized = False
        self.compressor_fused_input = (gaudi_envs.VLLM_HPU_DSV41_COMPRESSOR_FUSED_INPUT
                                       and layer < config["num_hidden_layers"])
        self._fused_compressor_weight = None
        self._fused_compressor_kv_width = 0
        self.mla_mme = gaudi_envs.VLLM_HPU_DSV41_MLA_MME and layer < config["num_hidden_layers"]
        self.linear, self.reduce, self.gather = linear, reduce, gather
        self.layer, self.ratio, self.length = layer, config["compress_ratios"][layer], shared.length
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
        self.candidate_source = config["candidate_source_layer_id"]
        if self.ratio:
            kv_source = max(source for source in config["kv_source_layer_ids"] if source <= layer)
            index_source = max(source for source in config["index_source_layer_ids"] if source <= layer)
            self.index_ratio = config["compress_ratios"][index_source]
            self.cache = shared.sources[str(kv_source)]
            self.selection = shared.topk[str(index_source)]
        self.register_buffer("swa", torch.zeros(SWA_ROWS, 528, dtype=torch.uint8, device=device), False)
        if self.owns_kv and self.ratio == 2:
            self.register_buffer("kv_history", torch.zeros(HISTORY_ROWS, 512, dtype=torch.float32, device=device),
                                 False)
            self.register_buffer("score_history", torch.zeros_like(self.kv_history), False)
        self.register_buffer("rotary", shared.compressed_rotary if self.ratio else shared.swa_rotary, False)
        self.register_buffer("window_offsets", torch.arange(self.window, dtype=torch.int32, device=device), False)
        self.register_buffer("compressed_offsets", torch.arange(512, dtype=torch.int32, device=device), False)
        self.register_buffer("swa_offsets", torch.arange(SWA_ROWS, dtype=torch.int32, device=device), False)
        self.register_buffer("selected_offsets", torch.arange(WORK_TOKENS * 512, dtype=torch.int32, device=device),
                             False)
        self.register_buffer("scale", torch.tensor([512**-0.5], dtype=torch.float32, device=device), False)

    def _compress(self, value, positions):
        compressor = self.weights.compressor
        if self.ratio == 2:
            kv, score = self._project_compressor_input(value)
            self.kv_history.index_copy_(0, positions.remainder(HISTORY_ROWS).long(), kv)
            self.score_history.index_copy_(0, positions.remainder(HISTORY_ROWS).long(), score)
            first = positions - positions.remainder(2)
            a, b = first.remainder(HISTORY_ROWS).long(), (first + 1).remainder(HISTORY_ROWS).long()
            gates = torch.stack((self.score_history[a], self.score_history[b]), 1).softmax(1)
            latent = (self.kv_history[a] * gates[:, 0] + self.kv_history[b] * gates[:, 1]).to(value.dtype)
        else:
            first, latent = positions, self.linear(value, compressor.wkv)
        latent = rms_norm(latent, compressor.norm.weight, self.eps)
        rows = positions // self.ratio
        visible = (positions + 1).remainder(self.ratio) == 0
        # Incomplete groups use unique rows in the reserved null page.
        slots = torch.where(visible, self.shared.physical_rows(rows, self.ratio),
                            rows.remainder(PAGE_TOKENS // self.ratio))
        indexer = self.weights.indexer
        index = rms_norm(self.linear(latent, indexer.wk), indexer.k_norm.weight, self.eps)
        self.cache.index.index_copy_(0, slots.long(), pack_fp4(apply_rope(index, first, self.rotary), 32))
        self.cache.main.index_copy_(0, slots.long(), pack_fp4(apply_rope(latent, first, self.rotary), 16))

    def _scores(self, value, qr, positions, logical_rows):
        indexer = self.weights.indexer
        q = self.linear(qr, indexer.wq_b).reshape(-1, self.index_heads, 128)
        q = unpack_fp4(pack_fp4(apply_rope(q, positions, self.rotary), 32), 128, 32)
        weights = self.linear(value, indexer.weights_proj) * (128**-0.5 * (self.index_heads * 2)**-0.5)
        # Exchange only small query/head tensors, not one score per cached token.
        q, weights = self.gather(q, 1), self.gather(weights, 1)
        physical = self.shared.physical_rows(logical_rows.clamp_min(0), self.ratio)
        packed = self.cache.index.index_select(0, physical.flatten().long()).reshape(*physical.shape, 68)
        keys = unpack_fp4(packed, 128, 32)
        scores = (torch.einsum("thd,nd->thn", q, keys) if keys.ndim == 2 else torch.einsum("thd,tnd->thn", q, keys))
        scores = scores.relu() * weights.unsqueeze(-1)
        # Retain the two TP partial-sum BF16 boundaries of the checkpoint reference.
        scores = scores.reshape(value.shape[0], 2, self.index_heads, -1).sum(2).sum(1)
        count = ((positions + 1) // self.ratio).unsqueeze(-1)
        valid = (logical_rows >= 0) & (logical_rows < count)
        return scores.float().masked_fill(~valid, -torch.inf)

    def _select(self, value, qr, positions):
        count, tokens = (positions + 1) // self.ratio, positions.numel()
        if self.owns_index:
            if self.search_length // self.ratio <= 512:
                indices = self.compressed_offsets.unsqueeze(0).expand(tokens, -1)
                indices = torch.where(indices < count.unsqueeze(-1), indices, -1)
                if self.layer == self.candidate_source:
                    rows = torch.arange(16384, device=positions.device, dtype=torch.int32).expand(tokens, -1)
                    self.shared.candidate_pool[:tokens].copy_(
                        torch.where(rows < count.unsqueeze(-1), rows, -1).reshape(tokens, 2048, 8))
            else:
                if self.layer > self.candidate_source:
                    rows = self.shared.candidate_pool[:tokens].flatten(1)
                else:
                    rows = torch.arange(self.search_length // self.ratio, device=positions.device, dtype=torch.int32)
                scores = self._scores(value, qr, positions, rows)
                if self.layer == self.candidate_source:
                    blocks = scores.reshape(tokens, -1, 8).amax(-1)
                    block_ids = torch.arange(blocks.shape[-1], device=positions.device)
                    blocks = blocks.masked_fill(block_ids == ((count - 1) // 8).unsqueeze(-1), torch.inf)
                    values, selected = blocks.topk(min(2048, blocks.shape[-1]), dim=-1, sorted=False)
                    candidate = selected.unsqueeze(-1) * 8 + torch.arange(8, device=positions.device)
                    candidate = torch.where((values > -torch.inf).unsqueeze(-1)
                                            & (candidate < count[:, None, None]), candidate, -1).flatten(1)
                    candidate = F.pad(candidate, (0, 16384 - candidate.shape[-1]), value=-1)
                    self.shared.candidate_pool[:tokens].copy_(candidate.reshape(tokens, 2048, 8).int())
                selected = scores.topk(512, dim=-1, sorted=False).indices
                indices = rows.expand(tokens, -1).gather(1, selected)
                # Sort logical positions, with invalid picks after reachable ones.
                indices = torch.where((indices >= 0) & (indices < count.unsqueeze(-1)), indices, self.length)
                indices = indices.sort(-1).values
                indices = torch.where(indices < self.length, indices, -1).int()
            self.selection.indices[:tokens].copy_(indices)
        return self.selection.indices[:tokens]

    def _finish_output(self, output, positions, ready_outputs=()):
        """Apply the shared output projection after either attention path."""
        output = apply_rope(output, positions, self.rotary, inverse=True)
        output = output.reshape(-1, self.groups, self.heads // self.groups * 512)
        weight = self.weights.wo_a.weight
        if gaudi_envs.VLLM_HPU_DSV41_PRETRANSPOSE_ATTN and weight.ndim == 3:
            # Load-time layout is [groups, D, R]; this form has no transpose
            # view for Synapse to materialize in every decode graph.
            output = torch.einsum("tgd,gdr->tgr", output, weight).flatten(1)
        else:
            weight = weight.reshape(self.groups, 1024, -1)
            output = torch.einsum("tgd,grd->tgr", output, weight).flatten(1)
        partial = self.linear(output, self.weights.wo_b)
        return self.reduce(partial, ready_outputs=ready_outputs) if ready_outputs else self.reduce(partial)

    def _output(self, query, cache, indices, positions, ready_outputs=()):
        if self.mla_mme and 1 <= query.shape[0] <= WORK_TOKENS:
            lengths = torch.full((query.shape[0], ), indices.shape[1], dtype=torch.int32, device=query.device)
            output = torch.ops.custom_op.custom_deepseek_v41_selected_mla_mme_gaudi2(
                query.contiguous(), cache.contiguous(), indices.contiguous(), self.weights.attn_sink, self.scale,
                lengths)
            return self._finish_output(output, positions, ready_outputs)
        output, _, _ = torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(
            query.contiguous(), cache.contiguous(), indices.contiguous(), self.weights.attn_sink, self.scale)
        return self._finish_output(output, positions, ready_outputs)

    def forward(self, value, positions, ready_outputs=()):
        query_input, kv_input = self._project_qkv_input(value)
        qr = rms_norm(query_input, self.weights.q_norm.weight, self.eps)
        query = apply_rope(self.linear(qr, self.weights.wq_b).reshape(-1, self.heads, 512), positions, self.rotary)
        kv = rms_norm(kv_input, self.weights.kv_norm.weight, self.eps)
        self.swa.index_copy_(0, positions.remainder(SWA_ROWS).long(), pack_swa(apply_rope(kv, positions, self.rotary)))
        native_prefix = (gaudi_envs.VLLM_HPU_DSV41_FUSED_PREFIX_LAYOUT and self.ratio in (1, 2)
                         and self.direct_selected_kv and self.shared_prefix_kv and self.window == 128
                         and value.device.type == "hpu" and 1 <= value.shape[0] <= WORK_TOKENS
                         and self.search_length // self.index_ratio <= 512)
        indices = None
        if not native_prefix:
            window = positions.unsqueeze(-1) - self.window + 1 + self.window_offsets.unsqueeze(0)
            indices = torch.where(window >= 0, window.remainder(SWA_ROWS), -1).int()
        cache = None
        if self.ratio:
            if self.owns_kv:
                self._compress(value, positions)
            selected = self._select(value, qr, positions)
            physical = None if native_prefix else self.shared.physical_rows(selected.clamp_min(0), self.ratio)
            if (self.direct_selected_kv and value.device.type == "hpu"
                    and selected.numel() <= self.selected_offsets.numel()):
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
        return self._output(query, cache, indices, positions, ready_outputs)

    def insert_context(self, value, positions, valid_count=None):
        kv = rms_norm(self.linear(value, self.weights.wkv), self.weights.kv_norm.weight, self.eps)
        indices = positions.remainder(SWA_ROWS).long()
        packed = pack_swa(apply_rope(kv, positions, self.rotary))
        if valid_count is not None:
            old = self.swa.index_select(0, indices)
            mask = torch.arange(indices.numel(), device=indices.device) < valid_count.reshape(1)
            packed = torch.where(mask.reshape(-1, 1), packed, old)
        self.swa.index_copy_(0, indices, packed)
        return self.swa

    def draft(self, value, positions):
        query = rms_norm(self.linear(value, self.weights.wq_a), self.weights.q_norm.weight, self.eps)
        query = apply_rope(self.linear(query, self.weights.wq_b).reshape(-1, self.heads, 512), positions, self.rotary)
        kv = rms_norm(self.linear(value, self.weights.wkv), self.weights.kv_norm.weight, self.eps)
        kv = unpack_swa(pack_swa(apply_rope(kv, positions, self.rotary)))
        cache = torch.cat((unpack_swa(self.swa), kv), 0)
        window = positions[:1] - self.window + self.window_offsets
        window = torch.where(window >= 0, window.remainder(SWA_ROWS), -1)
        draft_slots = SWA_ROWS + positions - positions[:1]
        indices = torch.cat((window, draft_slots), 0).unsqueeze(0).expand(5, -1).int()
        return self._output(query, cache, indices, positions)
