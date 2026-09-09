# SPDX-License-Identifier: Apache-2.0
"""Bounded DeepSeek V4 adapter for compiled native decoder replay.

Metadata lookup and allocation discovery happen before compilation. The tensor
program retains the existing BF16 projections, cache kernels and MLA kernels.
"""
import os
import weakref

import torch

from vllm_gaudi.ops.tp2_model_adapter import DEEPSEEK_V4


def validate_config(config):
    model = config.model_config
    hf = model.hf_config
    if (hf.model_type != "deepseek_v4" or hf.num_hidden_layers != 43
            or hf.hidden_size != 4096 or hf.num_experts_per_tok != 6
            or config.parallel_config.tensor_parallel_size != 2
            or config.parallel_config.pipeline_parallel_size != 1
            or config.scheduler_config.max_num_seqs != 1 or model.max_model_len > 512
            or model.dtype != torch.bfloat16 or model.enforce_eager
            or config.speculative_config is not None):
        raise RuntimeError("V4 native decode requires BF16, 43 layers, TP2/PP1, C1 and context <=512")
    for name in ("VLLM_HPU_TP2_STATIC_GROUP_PLAN", "VLLM_HPU_TP2_PREPARED_COMM", "VLLM_HPU_TP2_NATIVE_JOINT_PLAN"):
        os.environ.setdefault(name, "1")
        if os.environ[name].lower() not in ("1", "true"):
            raise RuntimeError(f"V4 native decode requires {name}")
    if os.environ.get("VLLM_HPU_DSV4_COMPILE_CHUNK_SIZE", "8") != "8":
        raise RuntimeError("V4 native decode requires the qualified 8+8+8+8+8+3 grouping")
    os.environ.setdefault("PT_HPU_POOL_MEM_ACQUIRE_PERC", "95")
    if not 1 <= int(os.environ["PT_HPU_POOL_MEM_ACQUIRE_PERC"]) <= 95:
        raise RuntimeError("Native recipes require PT_HPU_POOL_MEM_ACQUIRE_PERC in [1, 95]")


def _tensor(module, name, value):
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"V4 native attention has no prepared {name}")
    module.register_buffer(name, value, persistent=False)


def canonical_storage(value, bindings):
    """One TensorImpl for every shared byte pool, retaining the same storage.

    Framework cache views can have different autograd bases despite sharing
    one allocation. AOT cannot merge those mutated aliases. The native tensor
    program instead binds the allocation once and keeps all per-layer geometry.
    """
    if (value.dtype != torch.uint8 or value.ndim != 1 or not value.is_contiguous()
            or value.storage_offset() or value.numel() != value.untyped_storage().nbytes()):
        raise RuntimeError("Native cache binding requires a complete contiguous byte storage")
    key = value.untyped_storage()._cdata
    if key not in bindings:
        bindings[key] = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
            value.untyped_storage(), 0, (value.numel(),), (1,))
    return bindings[key]


def embedding_partial(input_ids, weight, org_start, org_end, org_padding, added_start, added_end):
    """Retain the hardware-agnostic vocabulary mask and gather arithmetic."""
    original = (input_ids >= org_start) & (input_ids < org_end)
    added = (input_ids >= added_start) & (input_ids < added_end)
    added_offset = added_start - (org_end - org_start) - org_padding
    offset = org_start * original + added_offset * added
    valid = original | added
    local_ids = valid * (input_ids - offset)
    output = torch.nn.functional.embedding(local_ids.long(), weight)
    return output.masked_fill((~valid).unsqueeze(-1), 0)


class NativeAttention(torch.nn.Module):
    """Only explicit tensor inputs enter this compiled attention body."""

    def __init__(self, attention, frontend_no_scores, frontend_compressor, storage_bindings):
        super().__init__()
        self.compress_ratio = attention.compress_ratio
        self.heads = attention.n_local_heads
        self.output_heads = attention.padded_heads
        if self.heads != 32 or self.output_heads != 64 or self.compress_ratio not in (1, 4, 128):
            raise RuntimeError(f"Unsupported V4 native attention geometry: heads={self.heads}, "
                               f"output_heads={self.output_heads}, ratio={self.compress_ratio}")
        self.frontend = frontend_no_scores if attention.compressor is None else frontend_compressor
        self.swa_prefix = attention.swa_cache_layer.prefix
        self.mla_prefix = attention.mla_attn.prefix
        for name, value in (
                ("qkv_weight", getattr(attention.fused_wqa_wkv, "_hpu_block_fp8_weight_dequant_transposed", None)),
                ("q_weight", getattr(attention.wq_b, "_hpu_block_fp8_weight_dequant_transposed", None)),
                ("q_norm", attention.q_norm.weight), ("kv_norm", attention.kv_norm.weight),
                ("swa_storage", attention.swa_cache_layer.kv_cache_storage),
                ("swa_geometry", attention.swa_cache_layer.kv_cache_geometry),
                ("rotary_cache", attention.rotary_emb.cos_sin_cache.float().contiguous()),
                ("sink", attention.mla_attn.attn_sink.float().contiguous())):
            _tensor(self, name, canonical_storage(value, storage_bindings) if name.endswith("_storage") else value)
        self.state_prefix = None
        self.split_count = int(os.environ.get("VLLM_HPU_DSV4_FLASHMLA_SPLITS", "4"))
        if attention.compressor is not None:
            compressor = attention.compressor
            self.state_prefix = compressor.state_cache.prefix
            self.compressor_width = compressor.coff * compressor.head_dim
            for name, value in (
                    ("compressor_weight", compressor.fused_wkv_wgate.weight),
                    ("state_storage", compressor.state_cache.kv_cache_storage),
                    ("state_geometry", compressor.state_cache.kv_cache_geometry),
                    ("compressed_storage", attention.mla_attn.kv_cache_storage),
                    ("compressed_geometry", attention.mla_attn.kv_cache_geometry),
                    ("ape", compressor.ape), ("compressor_norm", compressor.norm.weight.float().contiguous()),
                    ("compressor_eps", compressor._rms_norm_eps_tensor)):
                _tensor(self, name, canonical_storage(value, storage_bindings) if name.endswith("_storage") else value)
            if self.compress_ratio == 4:
                if attention.indexer is None or attention.indexer.max_model_len > attention.indexer.topk_tokens:
                    raise RuntimeError("Native C4 requires the bounded sequential top-k contract")
                _tensor(self, "topk_shape", attention.mla_attn.topk_indices_buffer[:1])
                _tensor(self, "split_shape", self.topk_shape[0, :self.split_count])

    def bind_metadata(self, metadata):
        swa = metadata[self.swa_prefix]
        if swa.num_decodes != 1 or swa.num_prefills or swa.num_decode_tokens != 1:
            raise RuntimeError("V4 native attention requires a single real decode token")
        result = [swa.slot_mapping[:1], swa.decode_swa_indices.reshape(1, -1), swa.decode_swa_lens[:1]]
        if self.state_prefix is not None:
            state, mla = metadata[self.state_prefix], metadata[self.mla_prefix]
            result.extend((state.slot_mapping[:1], state.token_to_req_indices[:1], state.block_table,
                           mla.slot_mapping[:1]))
            if self.compress_ratio == 4:
                result.extend((swa.token_to_req_indices[:1], mla.block_table[:1],
                               swa.is_valid_token[:1], swa.seq_lens))
            else:
                result.extend((mla.c128a_global_decode_topk_indices.reshape(1, -1), mla.c128a_decode_topk_lens[:1]))
        if any(value.dtype != torch.int32 or not value.is_contiguous() for value in result):
            raise RuntimeError("Native attention metadata must be persistent contiguous int32 tensors")
        return tuple(result)

    def forward(self, hidden_states, positions, metadata):
        if self.compress_ratio == 1:
            q, _, kv = self.frontend(hidden_states, self.qkv_weight, self.q_norm, self.kv_norm, self.q_weight)
        else:
            q, _, kv, kv_score = self.frontend(hidden_states, self.qkv_weight, self.compressor_weight,
                                              self.q_norm, self.kv_norm, self.q_weight)
        q = q.reshape(1, self.heads, 512).contiguous()
        slots, swa_indices, swa_lens = metadata[:3]
        ack = torch.ops.custom_op.custom_deepseek_v4_native_qnorm_rope_kv_pack_bf16_gaudi2(
            q, kv, self.swa_storage, self.swa_geometry, slots, positions.to(torch.int32), self.rotary_cache)
        # Q and the paged KV write are side effects of the same native kernel.
        # Keep a real device edge to the attention consumer across partitioning.
        q = torch.where(ack.reshape(1, 1, 1) <= 1, q, torch.zeros_like(q))
        if self.compress_ratio == 1:
            out, _ = torch.ops.custom_op.custom_deepseek_v4_native_paged_swa_attn_fp8_gaudi2(
                q, self.swa_storage, self.swa_geometry, swa_indices, swa_lens, swa_indices, swa_lens, swa_lens,
                self.swa_storage, self.swa_geometry, swa_indices, swa_lens, self.sink)
            return out
        state_slots, state_token, state_blocks, compressed_slots = metadata[3:7]
        compressor_kv, compressor_score = kv_score.split(self.compressor_width, dim=-1)
        completion = torch.ops.custom_op.custom_deepseek_v4_save_compress_norm_c4_f32_noclone_gaudi2(
            self.state_storage, self.state_geometry, self.compressed_geometry, compressor_kv, compressor_score,
            self.ape, positions.to(torch.int32), state_slots, state_token, state_blocks, self.compressor_norm,
            self.compressor_eps, self.rotary_cache, compressed_slots)
        if self.compress_ratio == 4:
            token, blocks, valid, lengths = metadata[7:]
            valid = torch.where(completion <= 1, valid, torch.zeros_like(valid))
            out = torch.empty((1, self.output_heads, 512), dtype=q.dtype, device=q.device)
            result, _, _, _ = torch.ops.custom_op.custom_deepseek_v4_flashmla_splitkv_tiled_fp8_gaudi2(
                q, self.compressed_storage, self.compressed_geometry, self.topk_shape, self.split_shape,
                token, blocks, valid, lengths, self.swa_storage, self.swa_geometry, swa_indices, swa_lens,
                self.sink, out)
            return result
        topk, lengths = metadata[7:]
        lengths = torch.where(completion <= 1, lengths, torch.zeros_like(lengths))
        out, _ = torch.ops.custom_op.custom_deepseek_v4_native_paged_sparse_attn_fp8_gaudi2(
            q, self.compressed_storage, self.compressed_geometry, topk, lengths, self.swa_storage,
            self.swa_geometry, swa_indices, swa_lens, self.sink)
        return out


class NativeDecoderTail(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.post = model.mhc_post_op
        self.head = model.hc_head_op
        self.norm = model.norm
        self.fn, self.scale, self.base = model.hc_head_fn, model.hc_head_scale, model.hc_head_base
        self.rms_eps, self.hc_eps = model.rms_norm_eps, model.hc_eps
        _tensor(self, "mtp_buffer", model._mtp_hidden_buffer[:1])

    def forward(self, hidden, residual, post_mix, res_mix):
        hidden = self.post(hidden, residual, post_mix, res_mix)
        self.mtp_buffer[:1].copy_(hidden.flatten(1))
        return self.norm(self.head(hidden, self.fn, self.scale, self.base, self.rms_eps, self.hc_eps))


class NativeDecodeMetadata(dict):
    """Per-ring metadata identity and its consumer completion generation."""
    def __init__(self, metadata, entry, slot):
        super().__init__(metadata)
        self.entry, self.slot = entry, slot
        self.native_key = id(entry), slot
        self.native_completion = None

    def consumed(self):
        if self.native_completion is None:
            raise RuntimeError("Native metadata has no new decoder completion generation")
        self.entry["completion"][self.slot] = self.native_completion
        self.entry["generations"][self.slot] += 1
        self.native_completion = None


class NativeStateSnapshot:
    """Save only pages that the capture token can modify, never model weights."""
    def __init__(self, owner, metadata):
        pages = {}
        for layer in owner.layers:
            attention = layer.attn.mla_attn
            caches = [attention.swa_cache_layer]
            if attention.compressor is not None:
                caches.extend((attention.compressor.state_cache, attention.mla_attn))
            for cache in caches:
                info = metadata[cache.prefix]
                slot = int(info.slot_mapping[:1].cpu()[0])
                if slot < 0:
                    continue
                tensor = cache.kv_cache
                block = slot // tensor.shape[1]
                # Cache backing storage is uint8; framework views can have
                # holes between pages and between layers in the shared pool.
                start = (tensor.storage_offset() + block * tensor.stride(0)) * tensor.element_size()
                width = (1 + sum((size - 1) * stride for size, stride in
                                 zip(tensor.shape[1:], tensor.stride()[1:], strict=True))) * tensor.element_size()
                storage = cache.kv_cache_storage
                if start + width > storage.numel():
                    raise RuntimeError("Capture snapshot page exceeds its KV allocation")
                pages[(storage.untyped_storage()._cdata, start, width)] = storage[start:start + width]
        pages[(id(owner), 0, 0)] = owner._mtp_hidden_buffer[:1]
        self.saved = [(page, page.clone()) for page in pages.values()]
        self.bytes = sum(value.numel() * value.element_size() for _, value in self.saved)

    def restore(self):
        for destination, saved in self.saved:
            destination.copy_(saved)


class NativeDecoder:
    def __init__(self, owner, chunks):
        self.owner = weakref.ref(owner)
        self.chunks = chunks
        self.generation = 0
        self.ready = False
        self.metadata = {}
        self.capture_bytes = 0
        self.storage_bindings = {}
        self.state_tensors = ()
        self.embedding_program = None
        self.fixed_metadata_pack = None
        self.attention_audit_dir = os.environ.get("VLLM_HPU_DSV4_NATIVE_ATTENTION_AUDIT_DIR")
        self.attention_audit_done = False
        self.replay_audit_dir = os.environ.get("VLLM_HPU_DSV4_NATIVE_REPLAY_AUDIT_DIR")
        self.replay_audit_positions = set()

    def embed(self, input_ids):
        from vllm.distributed import tensor_model_parallel_all_reduce
        layer = self.owner().embed_tokens
        if (layer.tp_size != 2 or layer.weight.dtype != torch.bfloat16
                or type(layer.quant_method).__name__ != "UnquantizedEmbeddingMethod"):
            raise RuntimeError("Native V4 embedding requires the BF16 TP2 vocabulary contract")
        if self.embedding_program is None:
            self.embedding_program = torch.compile(embedding_partial, backend="hpu_backend",
                                                   fullgraph=True, dynamic=False)
        shard = layer.shard_indices
        partial = self.embedding_program(input_ids, layer.weight, shard.org_vocab_start_index,
                                         shard.org_vocab_end_index, shard.num_org_vocab_padding,
                                         shard.added_vocab_start_index, shard.added_vocab_end_index)
        return tensor_model_parallel_all_reduce(partial)

    @staticmethod
    def is_decode():
        from vllm.forward_context import get_forward_context
        return isinstance(get_forward_context().attn_metadata, NativeDecodeMetadata)

    def invalidate(self):
        from vllm_gaudi.ops.tp2_prepared_plan import invalidate_prepared_group_plans
        invalidate_prepared_group_plans()
        self.generation += 1
        self.ready = False
        self.metadata.clear()
        self.storage_bindings.clear()
        self.state_tensors = ()
        self.embedding_program = None
        self.fixed_metadata_pack = None
        owner = self.owner()
        for layer in owner.layers:
            attention = layer.attn.mla_attn
            if hasattr(attention, "_hpu_native_decode"):
                del attention._hpu_native_decode
        self.chunks[-1].native_tail = None

    def _prepare(self, metadata):
        owner = self.owner()
        source_pack = metadata.entry["buffers"][metadata.slot][1]
        if self.fixed_metadata_pack is None:
            # A capture destination must never also be an H2D ring slot:
            # the next token can still consume that destination after the
            # original ring slot's earlier consumer has completed.
            self.fixed_metadata_pack = torch.empty_like(source_pack)
        if (source_pack.shape != self.fixed_metadata_pack.shape
                or source_pack.dtype != self.fixed_metadata_pack.dtype
                or source_pack.device != self.fixed_metadata_pack.device):
            raise RuntimeError("Native metadata pack layout changed before state mutation")
        if not self.ready:
            for layer in owner.layers:
                attention = layer.attn.mla_attn
                attention._hpu_native_decode = attention.make_hpu_native_decode(self.storage_bindings)
            self.chunks[-1].native_tail = NativeDecoderTail(owner)
            self.state_tensors = (*self.storage_bindings.values(), owner._mtp_hidden_buffer)
            self.ready = True
        key = metadata.native_key
        if key not in self.metadata:
            bindings = tuple(layer.attn.mla_attn._hpu_native_decode.bind_metadata(metadata) for layer in owner.layers)
            source_storage = source_pack.untyped_storage()._cdata

            def fixed_view(value):
                if value.untyped_storage()._cdata != source_storage:
                    raise RuntimeError("Native attention metadata escaped the prepared pack")
                offset = value.storage_offset() - source_pack.storage_offset()
                return self.fixed_metadata_pack.as_strided(value.shape, value.stride(), offset)

            bindings = tuple(tuple(fixed_view(value) for value in layer) for layer in bindings)
            groups, start = [], 0
            for count in DEEPSEEK_V4.group_layers:
                groups.append(bindings[start:start + count])
                start += count
            self.metadata[key] = tuple(groups), tuple(value for layer in bindings for value in layer)
        return self.metadata[key]

    def snapshot(self, metadata):
        saved = NativeStateSnapshot(self.owner(), metadata)
        self.capture_bytes = saved.bytes
        return saved

    def __call__(self, hidden_states, positions, input_ids):
        from vllm.forward_context import get_forward_context
        from vllm_gaudi.ops.tp2_prepared_plan import (
            collect_prepared_group_replays, record_native_decoder_outputs, replay_native_decoder)
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, NativeDecodeMetadata):
            # Prefill/profile calls use their normal model interface.
            return None
        owner = self.owner()
        groups, attention_inputs = self._prepare(metadata)
        if self.attention_audit_dir and not self.attention_audit_done and int(positions[:1].cpu()[0]) == 1:
            from vllm_gaudi.ops.deepseek_v4_native_audit import audit_attention
            self.attention_audit_done = True
            audit_attention(self, hidden_states, positions, input_ids, metadata, self.attention_audit_dir)
        roots = dict(hidden_states=hidden_states, positions=positions, input_ids=input_ids, residual=None,
                     metadata=metadata, attention_inputs=attention_inputs, state_generation=self.generation,
                     state_tensors=self.state_tensors,
                     metadata_destination=self.fixed_metadata_pack,
                     metadata_pack=metadata.entry["buffers"][metadata.slot][1])
        if self.replay_audit_dir:
            from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
            if owner in _native_entries:
                position = int(positions[:1].cpu()[0])
                if position in (2, 3, 4, 15, 16, 127, 128) and position not in self.replay_audit_positions:
                    from vllm_gaudi.ops.deepseek_v4_native_audit import audit_decoder_replay
                    audit_decoder_replay(self, roots, groups, self.replay_audit_dir, position)
                    self.replay_audit_positions.add(position)
        diagnostic_host = os.environ.get("VLLM_HPU_DSV4_NATIVE_DIAGNOSTIC_HOST_REPLAY") == "1"
        outputs = None if diagnostic_host else replay_native_decoder(owner, **roots)
        if outputs is None:
            # Cold discovery/recapture must consume the current token's pack
            # before any chunk runs or its modified state is snapshotted.
            self.fixed_metadata_pack.copy_(roots["metadata_pack"])
            residual, post_mix, res_mix = None, None, None
            with collect_prepared_group_replays(owner=owner, adapter=DEEPSEEK_V4,
                                               diagnostic_host_replay=diagnostic_host,
                                               snapshot=lambda: self.snapshot(metadata), **roots) as context:
                for index, (chunk, inputs) in enumerate(zip(owner._hpu_compiled_layer_chunks, groups, strict=True)):
                    context["group_index"] = index
                    hidden_states, residual, post_mix, res_mix = chunk(
                        hidden_states, positions, input_ids, post_mix, res_mix, residual, inputs)
                record_native_decoder_outputs(hidden_states)
            outputs = hidden_states, None
        metadata.consumed()
        return outputs[0]


def invalidate_model(model):
    for module in model.modules():
        decoder = getattr(module, "_hpu_native_decoder", None)
        if decoder is not None:
            decoder.invalidate()
