# SPDX-License-Identifier: Apache-2.0
"""Bounded CSA2 tensor program, with explicit cache ownership and state writes.

Layer sharing and short-context selection follow vLLM #56214 e47aa780,
deepseek_v4_1/attention.py. Compression and packed formats follow the frozen
DeepSeek dba1be0 reference. No CUDA/Triton implementation is imported.
"""

import torch
from torch import nn
import torch.nn.functional as F

from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.ops.deepseek_v41_math import (
    apply_rope,
    pack_fp4,
    pack_swa,
    quantize_activation,
    rms_norm,
    rotary_table,
    unpack_fp4,
    unpack_swa,
)


class CSA2SharedState(nn.Module):

    def __init__(self, config, layer_start, layer_stop, device, max_length=512):
        super().__init__()
        if not 1 <= max_length <= config["index_topk"] or max_length > 512:
            raise ValueError("This CSA2 plan requires context <= 512 and exact short-context selection")
        self.length = max_length
        self.layer_start = layer_start
        self.decoded_kv_state = gaudi_envs.VLLM_HPU_DSV41_DECODED_KV_STATE
        if self.decoded_kv_state:
            if max_length != 512 or gaudi_envs.VLLM_HPU_DSV41_DSPARK:
                raise ValueError("Decoded KV state requires context512 and DSpark disabled")
            # One stage-owned allocation survives request block rebinding.
            self.register_buffer(
                "decoded_swa",
                torch.zeros((layer_stop - layer_start) * max_length, 512, dtype=torch.bfloat16, device=device), False)
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
                if self.decoded_kv_state:
                    cache.register_buffer("decoded_main", torch.zeros(rows, 512, dtype=torch.bfloat16, device=device),
                                          False)
                self.sources[str(source)] = cache
        for source in config["index_source_layer_ids"]:
            if layer_start <= source < layer_stop:
                cache = nn.Module()
                cache.register_buffer("indices", torch.full((max_length, 512), -1, dtype=torch.int32, device=device),
                                      False)
                self.topk[str(source)] = cache
        candidate_source = config["candidate_source_layer_id"]
        if layer_start <= candidate_source < layer_stop:
            self.register_buffer(
                "candidate_pool",
                torch.full((max_length, config["candidate_topk_blocks"], config["candidate_block_size"]),
                           -1,
                           dtype=torch.int32,
                           device=device), False)
        else:
            self.register_buffer("candidate_pool", None, False)


class CSA2Attention(nn.Module):

    def __init__(self, weights, config, layer, shared, linear, reduce, device):
        super().__init__()
        self.weights = weights
        self.woa_fp8 = False
        self.packed_decode = gaudi_envs.VLLM_HPU_DSV41_PACKED_ATTENTION
        self.bounded_decode = gaudi_envs.VLLM_HPU_DSV41_BOUNDED_ATTENTION
        self.swa_pack_write = gaudi_envs.VLLM_HPU_DSV41_SWA_PACK_WRITE
        self.fp4_cache_write = gaudi_envs.VLLM_HPU_DSV41_FP4_CACHE_WRITE
        self.native_rope = gaudi_envs.VLLM_HPU_DSV41_NATIVE_ROPE
        self.qkv_fused_input = gaudi_envs.VLLM_HPU_DSV41_QKV_FUSED_INPUT
        self._fused_qkv_weight = None
        self._fused_qkv_quantized = False
        self.c1_indices = gaudi_envs.VLLM_HPU_DSV41_C1_INDICES
        self.selected_valid_only = gaudi_envs.VLLM_HPU_DSV41_SELECTED_VALID_ONLY
        self.selected_kv_vector = gaudi_envs.VLLM_HPU_DSV41_SELECTED_KV_VECTOR
        self.paired_exp = gaudi_envs.VLLM_HPU_DSV41_ATTENTION_PAIRED_EXP
        self.head_pair = gaudi_envs.VLLM_HPU_DSV41_ATTENTION_HEAD_PAIR
        self.mla_mme = gaudi_envs.VLLM_HPU_DSV41_MLA_MME
        self.block_exp = gaudi_envs.VLLM_HPU_DSV41_ATTENTION_BLOCK_EXP
        self.prepared_output = (gaudi_envs.VLLM_HPU_DSV41_PREPARED_OUTPUT and layer < config["num_hidden_layers"])
        self.output_gemm_layout = (gaudi_envs.VLLM_HPU_DSV41_OUTPUT_GEMM_LAYOUT and layer < config["num_hidden_layers"])
        if self.output_gemm_layout and not self.prepared_output:
            raise ValueError("V4.1 output GEMM layout requires prepared output weights")
        self.decoded_kv_state = shared.decoded_kv_state
        if self.mla_mme and (not self.decoded_kv_state or self.block_exp):
            raise ValueError("MME MLA requires decoded KV and excludes the blocked-exponential candidate")
        if self.block_exp and not self.decoded_kv_state:
            raise ValueError("Blocked exponentials require the exact decoded-KV C1 state")
        if self.decoded_kv_state:
            if not (self.c1_indices and self.selected_valid_only and self.fp4_cache_write):
                raise ValueError("Decoded KV requires ordered C1 indices and fused packed cache writes")
            if self.head_pair or self.paired_exp or self.selected_kv_vector:
                raise ValueError("Decoded KV state is an independent attention candidate")
            self.shared = shared
            self.decoded_swa_offset = (layer - shared.layer_start) * shared.length
        if self.head_pair and self.paired_exp:
            raise ValueError("Choose one V4.1 attention variant")
        if (self.paired_exp or self.head_pair) and not (self.selected_valid_only and self.bounded_decode):
            raise ValueError("V4.1 paired exponentials require the ordered bounded attention path")
        if self.selected_kv_vector and not self.selected_valid_only:
            raise ValueError("V4.1 vector selected KV requires SELECTED_VALID_ONLY")
        if self.selected_valid_only and not self.swa_pack_write:
            raise ValueError("Valid-only selected KV requires the ordered SWA/attention path")
        if self.bounded_decode and not self.packed_decode:
            raise ValueError("Bounded V4.1 attention requires packed KV consumption")
        if self.swa_pack_write and not (self.packed_decode and self.bounded_decode):
            raise ValueError("Fused V4.1 SWA writes require bounded packed attention and its completion dependency")
        if self.fp4_cache_write and not self.swa_pack_write:
            raise ValueError("Fused V4.1 FP4 cache writes require ordered SWA and bounded attention")
        if self.c1_indices and not self.bounded_decode:
            raise ValueError("Native C1 index preparation requires bounded packed attention")
        self.linear, self.reduce = linear, reduce
        self.layer, self.ratio, self.length = layer, config["compress_ratios"][layer], shared.length
        self.heads, self.groups = config["num_attention_heads"] // 2, config["o_groups"] // 2
        self.eps, self.window = config["rms_norm_eps"], config["sliding_window"]
        if (self.c1_indices or self.native_rope) and (self.length != 512 or self.window != 128
                                                      or config["qk_rope_head_dim"] != 64):
            raise ValueError("Native V4.1 C1 preparation requires context512, SWA128 and RoPE64")
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
                             scaling["original_max_position_embeddings"] if self.ratio else 0, scaling["factor"],
                             scaling["beta_fast"], scaling["beta_slow"])
        self.register_buffer("rotary", table.to(device), False)
        # V4's verified TPC helper consumes [cos32, sin32]. Prepare both signs
        # once; preserve the original interleaved table for non-C1 execution.
        if self.native_rope:
            native = torch.cat((table[..., 0], table[..., 1]), -1).contiguous()
            inverse = torch.cat((table[..., 0], -table[..., 1]), -1).contiguous()
            self.register_buffer("rotary_native", native.to(device), False)
            self.register_buffer("rotary_native_inverse", inverse.to(device), False)
        self.register_buffer("window_offsets", torch.arange(self.window, device=device, dtype=torch.int32), False)
        self.register_buffer("compressed_offsets", torch.arange(512, device=device, dtype=torch.int32), False)
        self.register_buffer("inactive_compressed", self.compressed_offsets.view(1, 512), False)
        self.register_buffer("scale", torch.tensor([512**-0.5], device=device, dtype=torch.float32), False)
        if layer == self.candidate_source:
            self.register_buffer(
                "candidate_offsets",
                torch.arange(config["candidate_topk_blocks"] * config["candidate_block_size"],
                             device=device,
                             dtype=torch.int32), False)

    def prepare_output_weight(self):
        if self.woa_fp8:
            return
        if self.prepared_output:
            weight = self.weights.wo_a.weight
            if weight.dtype != torch.bfloat16 or weight.numel() != self.heads * 512 * 1024:
                raise ValueError("V4.1 output weight contract changed")
            grouped = weight.reshape(self.groups, 1024, -1)
            self.weights.wo_a.weight = (grouped if self.output_gemm_layout else grouped.transpose(1, 2).contiguous())

    def prepare_qkv_input_weight(self):
        """Bind one persistent Q/KV input matrix for the C1 attention path.

        ``wq_a`` and ``wkv`` consume the same activation and currently each
        invokes the block quantization helper before its own GEMM.  The fused
        matrix keeps the original output ordering (Q first, KV second), so
        splitting the result is algebraically identical while allowing one
        quantization and one MME launch.  The two legacy weight attributes are
        rebound as views to avoid retaining a second device allocation.
        """
        if not self.qkv_fused_input or self._fused_qkv_weight is not None:
            return
        q_module, kv_module = self.weights.wq_a, self.weights.wkv
        q_weight, kv_weight = q_module.weight, kv_module.weight
        if q_weight.ndim != 2 or kv_weight.ndim != 2 or q_weight.shape[1] != kv_weight.shape[1]:
            raise ValueError("V4.1 QKV fusion requires matching input K dimensions")
        if q_weight.dtype != torch.bfloat16 or kv_weight.dtype != torch.bfloat16:
            raise ValueError("V4.1 QKV fusion requires prepared BF16 input weights")
        q_quantized = hasattr(q_module, "scale")
        kv_quantized = hasattr(kv_module, "scale")
        if q_quantized != kv_quantized:
            raise ValueError("V4.1 QKV fusion cannot combine mismatched activation contracts")
        fused = torch.cat((q_weight, kv_weight), dim=0).contiguous()
        self.register_buffer("fused_wqa_wkv", fused, False)
        # Keep compatibility with code which introspects the individual
        # matrices, while making both views point at the single allocation.
        q_module.weight = self.fused_wqa_wkv[:q_weight.shape[0]]
        kv_module.weight = self.fused_wqa_wkv[q_weight.shape[0]:]
        self._fused_qkv_weight = self.fused_wqa_wkv
        self._fused_qkv_quantized = q_quantized

    def invalidate_qkv_input_weight(self):
        """Drop a fused view before a prepared weight reload or migration."""
        if "fused_wqa_wkv" in self._buffers:
            self._buffers.pop("fused_wqa_wkv")
        self._fused_qkv_weight = None
        self._fused_qkv_quantized = False

    def _project_qkv_input(self, value):
        if self._fused_qkv_weight is None:
            query = self.linear(value, self.weights.wq_a)
            kv = self.linear(value, self.weights.wkv)
            return query, kv
        fused_value = quantize_activation(value) if self._fused_qkv_quantized else value
        qkv = F.linear(fused_value, self._fused_qkv_weight)
        q_width = self.weights.wq_a.weight.shape[0]
        return qkv[..., :q_width], qkv[..., q_width:]

    def project_output(self, value):
        if self.woa_fp8:
            return torch.ops.custom_op.custom_deepseek_v41_woa_fp8_gaudi2(value.contiguous(), self.weights.wo_a.weight,
                                                                          self.weights.wo_a.channel_scale)
        if self.output_gemm_layout:
            weight = self.weights.wo_a.weight
            if value.shape[0] == 1:
                # Statically unrolled inside the stage graph. Each projection
                # retains complete K and the input's existing BF16 boundary.
                return torch.cat(tuple(F.linear(value[:, group], weight[group]) for group in range(self.groups)),
                                 dim=-1)
            return torch.einsum("tgd,grd->tgr", value, weight).flatten(1)
        if self.prepared_output:
            return torch.einsum("tgd,gdr->tgr", value, self.weights.wo_a.weight).flatten(1)
        weight = self.weights.wo_a.weight.reshape(self.groups, 1024, -1)
        return torch.einsum("tgd,grd->tgr", value, weight).flatten(1)

    def _rope(self, value, positions, inverse=False):
        if self.native_rope and value.shape[0] == 1:
            shape = value.shape
            table = self.rotary_native_inverse if inverse else self.rotary_native
            return torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(
                value.reshape(1, -1, shape[-1]).contiguous(), positions, table).reshape(shape)
        return apply_rope(value, positions, self.rotary, inverse=inverse)

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
        index = self._rope(index, first)
        rotated = self._rope(latent, first)
        if self.decoded_kv_state and value.shape[0] == 1:
            return torch.ops.custom_op.custom_deepseek_v41_fp4_decoded_write_bf16_gaudi2(
                self.cache.main, self.cache.index, rotated.contiguous(), index.contiguous(), slots.contiguous(),
                self.cache.decoded_main)
        if self.fp4_cache_write and value.shape[0] == 1:
            return torch.ops.custom_op.custom_deepseek_v41_fp4_cache_write_bf16_gaudi2(
                self.cache.main, self.cache.index, rotated.contiguous(), index.contiguous(), slots.contiguous())
        self.cache.index.index_copy_(0, slots.long(), pack_fp4(index, 32))
        packed_main = pack_fp4(rotated, 16)
        self.cache.main.index_copy_(0, slots.long(), packed_main)
        if self.decoded_kv_state:
            self.cache.decoded_main.index_copy_(0, slots.long(), unpack_fp4(packed_main))
        return None

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
        query_input, kv_input = self._project_qkv_input(value)
        query = rms_norm(query_input, self.weights.q_norm.weight, self.eps)
        query = self.linear(query, self.weights.wq_b).reshape(-1, self.heads, 512)
        query = self._rope(query, positions)
        kv = rms_norm(kv_input, self.weights.kv_norm.weight, self.eps)
        kv = self._rope(kv, positions)
        completion = None
        if self.decoded_kv_state and value.shape[0] == 1:
            completion = torch.ops.custom_op.custom_deepseek_v41_swa_decoded_write_bf16_gaudi2(
                self.swa, kv.contiguous(), positions, self.shared.decoded_swa, self.decoded_swa_offset)
        elif self.swa_pack_write and value.shape[0] == 1:
            completion = torch.ops.custom_op.custom_deepseek_v41_swa_pack_write_bf16_gaudi2(
                self.swa, kv.contiguous(), positions)
        else:
            packed_swa = pack_swa(kv)
            self.swa.index_copy_(0, positions.long(), packed_swa)
            if self.decoded_kv_state:
                self.shared.decoded_swa.index_copy_(0,
                                                    positions.long() + self.decoded_swa_offset, unpack_swa(packed_swa))
        use_c1_indices = self.c1_indices and value.shape[0] == 1
        if not use_c1_indices:
            window = positions.unsqueeze(-1) - self.window + 1 + self.window_offsets.unsqueeze(0)
            indices = torch.where(window >= 0, window, -1).to(torch.int32)
        compressed_completion = None
        compressed_indices = self.inactive_compressed
        if self.ratio:
            if self.owns_kv:
                compressed_completion = self._compress(value, positions)
            compressed_indices = self._select(positions)
            if not use_c1_indices:
                compressed_indices = torch.where(compressed_indices >= 0, compressed_indices + self.length, -1)
                indices = torch.cat((indices, compressed_indices), -1)
        if use_c1_indices:
            indices, lengths = torch.ops.custom_op.custom_deepseek_v41_c1_indices_i32_gaudi2(
                positions, compressed_indices.contiguous(), self.ratio)
        if self.decoded_kv_state and value.shape[0] == 1:
            main = self.cache.decoded_main if self.ratio else self.shared.decoded_swa
            # Reuse layers are downstream of their source layer's completed
            # attention and hidden state. Source layers additionally consume
            # their own FP4 writer completion in this recipe.
            main_done = compressed_completion if compressed_completion is not None else completion
            if self.mla_mme:
                output = torch.ops.custom_op.custom_deepseek_v41_mla_mme_gaudi2(
                    query.contiguous(), self.shared.decoded_swa, main, indices.contiguous(), self.weights.attn_sink,
                    self.scale, lengths.contiguous(), completion, main_done, self.decoded_swa_offset,
                    self.length // self.ratio if self.ratio else 0)
            else:
                attention = (torch.ops.custom_op.custom_deepseek_v41_decoded_attn_block_bf16_gaudi2
                             if self.block_exp else torch.ops.custom_op.custom_deepseek_v41_decoded_attn_bf16_gaudi2)
                output, _, _ = attention(query.contiguous(), self.shared.decoded_swa,
                                         main, indices.contiguous(), self.weights.attn_sink, self.scale,
                                         lengths.contiguous(), completion, main_done, self.decoded_swa_offset,
                                         self.length // self.ratio if self.ratio else 0)
        elif self.packed_decode and value.shape[0] == 1:
            # A SWA-width second input denotes an inactive main cache. It
            # aliases existing storage and cannot admit a compressed row.
            main = self.cache.main[:self.length // self.ratio] if self.ratio else self.swa
            if self.bounded_decode:
                vector_options = (True, ) if self.selected_kv_vector else ()
                if self.paired_exp or self.head_pair:
                    if completion is None:
                        raise RuntimeError("V4.1 paired attention requires the SWA write completion")
                    vector_options = ((self.selected_kv_vector, False, True) if self.head_pair else
                                      (self.selected_kv_vector, True))
                # At context <=512, Full/Reindex/Reuse retain every visible
                # compressed row followed by invalid padding. Keep its order.
                if not use_c1_indices:
                    lengths = (self.window +
                               (positions + 1) // self.ratio if self.ratio else torch.full_like(positions, self.window))
                if compressed_completion is not None:
                    output = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_ordered_cache_lengths_gaudi2(
                        query.contiguous(), self.swa, main, indices.contiguous(), self.weights.attn_sink, self.scale,
                        lengths.contiguous(), completion, compressed_completion, self.selected_valid_only,
                        *vector_options)
                elif completion is not None:
                    output = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_ordered_lengths_gaudi2(
                        query.contiguous(), self.swa, main, indices.contiguous(), self.weights.attn_sink, self.scale,
                        lengths.contiguous(), completion, self.selected_valid_only, *vector_options)
                else:
                    output = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_lengths_gaudi2(
                        query.contiguous(), self.swa, main, indices.contiguous(), self.weights.attn_sink, self.scale,
                        lengths.contiguous())
            else:
                output = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_gaudi2(
                    query.contiguous(), self.swa, main, indices.contiguous(), self.weights.attn_sink, self.scale)
        else:
            cache = unpack_swa(self.swa)
            if self.ratio:
                cache = torch.cat((cache, unpack_fp4(self.cache.main[:self.length // self.ratio])), 0)
            output, _, _ = torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(
                query.contiguous(), cache.contiguous(), indices.contiguous(), self.weights.attn_sink, self.scale)
        output = self._rope(output, positions, inverse=True)
        output = output.reshape(-1, self.groups, self.heads // self.groups * 512)
        output = self.project_output(output)
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
            # A C6 verification graph always materializes six rows. Preserve
            # committed cache contents for padded lanes in C1-C5 buckets.
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
        output = self.project_output(output)
        return self.reduce(self.linear(output, self.weights.wo_b))
