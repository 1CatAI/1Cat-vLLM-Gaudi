# SPDX-License-Identifier: Apache-2.0
"""Bounded CSA2 tensor program, with explicit cache ownership and state writes.

Layer sharing and short-context selection follow vLLM #56214 e47aa780,
deepseek_v4_1/attention.py. Compression and packed formats follow the frozen
DeepSeek dba1be0 reference. No CUDA/Triton implementation is imported.
"""

import torch
from torch import nn
from vllm_gaudi.ops.deepseek_v41_qkv import FusedQKVInput
from vllm_gaudi import envs as gaudi_envs

from vllm_gaudi.ops.deepseek_v41_math import (
    apply_rope, pack_fp4, pack_swa, rms_norm, rotary_table, unpack_fp4, unpack_swa,
)


class CSA2SharedState(nn.Module):
    def __init__(self, config, layer_start, layer_stop, device, max_length=512):
        super().__init__()
        if not 1 <= max_length <= config["index_topk"] or max_length > 512:
            raise ValueError("This CSA2 plan requires context <= 512 and exact short-context selection")
        self.length = max_length
        self.sources = nn.ModuleDict()
        self.topk = nn.ModuleDict()
        for source in config["kv_source_layer_ids"]:
            if layer_start <= source < layer_stop:
                ratio = config["compress_ratios"][source]
                cache = nn.Module()
                # Incomplete groups write unique scratch rows; there are no
                # duplicate scatter destinations inside a multi-token verify.
                rows = max_length // ratio + max_length
                cache.register_buffer("main", torch.zeros(rows, 288, dtype=torch.uint8, device=device), False)
                cache.register_buffer("index", torch.zeros(rows, 68, dtype=torch.uint8, device=device), False)
                self.sources[str(source)] = cache
        for source in config["index_source_layer_ids"]:
            if layer_start <= source < layer_stop:
                cache = nn.Module()
                cache.register_buffer("indices", torch.full((max_length, 512), -1, dtype=torch.int32, device=device), False)
                self.topk[str(source)] = cache
        candidate_source = config["candidate_source_layer_id"]
        if layer_start <= candidate_source < layer_stop:
            self.register_buffer("candidate_pool", torch.full(
                (max_length, config["candidate_topk_blocks"], config["candidate_block_size"]),
                -1, dtype=torch.int32, device=device), False)
        else:
            self.register_buffer("candidate_pool", None, False)


class CSA2Attention(FusedQKVInput, nn.Module):
    def __init__(self, weights, config, layer, shared, linear, reduce, device):
        super().__init__()
        self.weights = weights
        self.qkv_fused_input = gaudi_envs.VLLM_HPU_DSV41_QKV_FUSED_INPUT and layer < config["num_hidden_layers"]
        self._fused_qkv_weight = None
        self._fused_qkv_quantized = False
        self.linear, self.reduce = linear, reduce
        self.layer, self.ratio, self.length = layer, config["compress_ratios"][layer], shared.length
        self.heads, self.groups = config["num_attention_heads"] // 2, config["o_groups"] // 2
        self.eps, self.window = config["rms_norm_eps"], config["sliding_window"]
        self.owns_kv = layer in config["kv_source_layer_ids"]
        self.owns_index = layer in config["index_source_layer_ids"]
        self.candidate_source = config["candidate_source_layer_id"]
        if self.ratio:
            kv_source = max(source for source in config["kv_source_layer_ids"] if source <= layer)
            index_source = max(source for source in config["index_source_layer_ids"] if source <= layer)
            if str(kv_source) not in shared.sources or str(index_source) not in shared.topk:
                raise ValueError("CSA2 PP boundary crosses a shared cache owner")
            self.cache = shared.sources[str(kv_source)]
            self.selection = shared.topk[str(index_source)]
            self.shared = shared
        self.register_buffer("swa", torch.zeros(self.length, 528, dtype=torch.uint8, device=device), False)
        if self.owns_kv and self.ratio == 2:
            self.register_buffer("kv_history", torch.zeros(self.length, 512, dtype=torch.float32, device=device), False)
            self.register_buffer("score_history", torch.zeros_like(self.kv_history), False)
        scaling = config["rope_scaling"]
        table = rotary_table(config["qk_rope_head_dim"], self.length,
                             config["compress_rope_theta"] if self.ratio else config["rope_theta"],
                             scaling["original_max_position_embeddings"] if self.ratio else 0,
                             scaling["factor"], scaling["beta_fast"], scaling["beta_slow"])
        self.register_buffer("rotary", table.to(device), False)
        self.register_buffer("window_offsets", torch.arange(self.window, device=device, dtype=torch.int32), False)
        self.register_buffer("compressed_offsets", torch.arange(512, device=device, dtype=torch.int32), False)
        self.register_buffer("scale", torch.tensor([512**-0.5], device=device, dtype=torch.float32), False)
        if layer == self.candidate_source:
            self.register_buffer("candidate_offsets", torch.arange(
                config["candidate_topk_blocks"] * config["candidate_block_size"],
                device=device, dtype=torch.int32), False)

    def _compress(self, value, positions):
        compressor = self.weights.compressor
        if self.ratio == 2:
            kv = self.linear(value.float(), compressor.wkv)
            score = self.linear(value.float(), compressor.wgate)
            self.kv_history.index_copy_(0, positions.long(), kv.float())
            self.score_history.index_copy_(0, positions.long(), score.float())
            first = positions - positions.remainder(2)
            kv0, kv1 = self.kv_history[first.long()], self.kv_history[(first + 1).long()]
            s0, s1 = self.score_history[first.long()], self.score_history[(first + 1).long()]
            gates = torch.stack((s0, s1), 1).softmax(1)
            latent = (kv0 * gates[:, 0] + kv1 * gates[:, 1]).to(value.dtype)
        else:
            first = positions
            latent = self.linear(value, compressor.wkv)
        latent = rms_norm(latent, compressor.norm.weight, self.eps)
        visible = (positions + 1).remainder(self.ratio) == 0
        slots = torch.where(visible, positions // self.ratio, self.length // self.ratio + positions)
        indexer = self.weights.indexer
        index = rms_norm(self.linear(latent, indexer.wk), indexer.k_norm.weight, self.eps)
        index = apply_rope(index, first, self.rotary)
        self.cache.index.index_copy_(0, slots.long(), pack_fp4(index, 32))
        rotated = apply_rope(latent, first, self.rotary)
        self.cache.main.index_copy_(0, slots.long(), pack_fp4(rotated, 16))

    def _select(self, positions):
        count = (positions + 1) // self.ratio
        if self.layer == self.candidate_source:
            candidate = self.candidate_offsets.unsqueeze(0).expand(positions.numel(), -1)
            candidate = torch.where(candidate < count.unsqueeze(-1), candidate, -1)
            pool = candidate.reshape(positions.numel(), *self.shared.candidate_pool.shape[1:])
            self.shared.candidate_pool.index_copy_(0, positions.long(), pool)
        if self.owns_index:
            if self.layer >= self.candidate_source:
                # In the bounded profile every visible candidate survives;
                # Reindex still consumes the candidate source's published slots.
                indices = self.shared.candidate_pool.index_select(0, positions.long()).flatten(1)[:, :512]
            else:
                indices = self.compressed_offsets.unsqueeze(0).expand(positions.numel(), -1)
                indices = torch.where(indices < count.unsqueeze(-1), indices, -1)
            self.selection.indices.index_copy_(0, positions.long(), indices)
        return self.selection.indices.index_select(0, positions.long())

    def forward(self, value, positions, ready_outputs=()):
        q_input, kv_input = self._project_qkv_input(value)
        query = rms_norm(q_input, self.weights.q_norm.weight, self.eps)
        query = self.linear(query, self.weights.wq_b).reshape(-1, self.heads, 512)
        query = apply_rope(query, positions, self.rotary)
        kv = rms_norm(kv_input, self.weights.kv_norm.weight, self.eps)
        kv = apply_rope(kv, positions, self.rotary)
        self.swa.index_copy_(0, positions.long(), pack_swa(kv))
        cache = unpack_swa(self.swa)
        window = positions.unsqueeze(-1) - self.window + 1 + self.window_offsets.unsqueeze(0)
        indices = torch.where(window >= 0, window, -1).to(torch.int32)
        if self.ratio:
            if self.owns_kv:
                self._compress(value, positions)
            compressed_indices = self._select(positions)
            compressed = unpack_fp4(self.cache.main[:self.length // self.ratio])
            cache = torch.cat((cache, compressed), 0)
            compressed_indices = torch.where(compressed_indices >= 0, compressed_indices + self.length, -1)
            indices = torch.cat((indices, compressed_indices), -1)
        output, _, _ = torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(
            query.contiguous(), cache.contiguous(), indices.contiguous(), self.weights.attn_sink, self.scale)
        output = apply_rope(output, positions, self.rotary, inverse=True)
        output = output.reshape(-1, self.groups, self.heads // self.groups * 512)
        weight = self.weights.wo_a.weight.reshape(self.groups, 1024, -1)
        output = torch.einsum("tgd,grd->tgr", output, weight).flatten(1)
        partial = self.linear(output, self.weights.wo_b)
        return self.reduce(partial, ready_outputs=ready_outputs) if ready_outputs else self.reduce(partial)

    def insert_context(self, main_value, positions, valid_count=None):
        """DSpark context-only insert, reusing vLLM #55654's projection boundary."""
        if self.ratio:
            raise RuntimeError("DSpark context insert requires an uncompressed draft SWA cache")
        kv = rms_norm(self.linear(main_value, self.weights.wkv), self.weights.kv_norm.weight, self.eps)
        kv = apply_rope(kv, positions, self.rotary)
        packed = pack_swa(kv)
        if valid_count is not None:
            # The fixed C6 graph computes all six rows.  Preserve the old
            # state in padded lanes so C1--C5 do not insert speculative rows.
            indices = positions.long()
            old = self.swa.index_select(0, indices)
            mask = torch.arange(indices.numel(), device=indices.device) < valid_count.reshape(1)
            packed = torch.where(mask.reshape(-1, 1), packed, old)
        self.swa.index_copy_(0, positions.long(), packed)
        return self.swa

    def draft(self, value, positions):
        """Bidirectional draft block; draft KV never mutates committed context."""
        if self.ratio or value.shape[0] != 5:
            raise RuntimeError("The prepared DSpark program requires a five-token draft block")
        query = rms_norm(self.linear(value, self.weights.wq_a), self.weights.q_norm.weight, self.eps)
        query = self.linear(query, self.weights.wq_b).reshape(-1, self.heads, 512)
        query = apply_rope(query, positions, self.rotary)
        kv = rms_norm(self.linear(value, self.weights.wkv), self.weights.kv_norm.weight, self.eps)
        kv = unpack_swa(pack_swa(apply_rope(kv, positions, self.rotary)))
        cache = torch.cat((unpack_swa(self.swa), kv), 0)
        window = positions[:1] - self.window + self.window_offsets
        window = torch.where(window >= 0, window, -1)
        draft_slots = self.length + positions - positions[:1]
        indices = torch.cat((window, draft_slots), 0).unsqueeze(0).expand(5, -1).contiguous()
        output, _, _ = torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(
            query.contiguous(), cache.contiguous(), indices, self.weights.attn_sink, self.scale)
        output = apply_rope(output, positions, self.rotary, inverse=True)
        output = output.reshape(-1, self.groups, self.heads // self.groups * 512)
        weight = self.weights.wo_a.weight.reshape(self.groups, 1024, -1)
        output = torch.einsum("tgd,grd->tgr", output, weight).flatten(1)
        return self.reduce(self.linear(output, self.weights.wo_b))
