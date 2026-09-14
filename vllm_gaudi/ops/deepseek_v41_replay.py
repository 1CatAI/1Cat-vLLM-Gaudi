# SPDX-License-Identifier: Apache-2.0
"""V4.1 stage binding for the maintained native compute/HCL replay plan."""

from dataclasses import dataclass, replace
import weakref

import torch

from vllm_gaudi.ops.tp2_model_adapter import DEEPSEEK_V41_PP0, DEEPSEEK_V41_PP1


def stage_collectives(tp_rank, native):
    from vllm.distributed import tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce

    def reduce(value, *, ready_outputs=()):
        if native and value.dtype == torch.bfloat16 and value.numel() <= 32768:
            flat = value.reshape(1, -1).contiguous()
            peer = (torch.ops.vllm_gaudi.tp2_exchange_peer_scheduled(flat, list(ready_outputs))
                    if ready_outputs else torch.ops.vllm_gaudi.tp2_exchange_peer(flat))
            return (flat + peer).reshape(value.shape)
        return tensor_model_parallel_all_reduce(value)

    def gather(value, dim):
        if native and dim == 1 and value.dtype == torch.bfloat16 and value.numel() <= 32768:
            flat = value.reshape(1, -1).contiguous()
            # The native command path transfers 128 BF16 elements per unit.
            # Indexer head weights can contain only 16..96 elements.
            if value.numel() % 128:
                flat = torch.nn.functional.pad(flat, (0, -value.numel() % 128))
            peer = torch.ops.vllm_gaudi.tp2_exchange_peer(flat)[:, :value.numel()].reshape(value.shape)
            first, second = (value, peer) if tp_rank == 0 else (peer, value)
            return torch.cat((first, second), dim=dim)
        return tensor_model_parallel_all_gather(value, dim=dim)
    return reduce, gather


@dataclass
class _Metadata:
    native_completion: object = None


class _Snapshot:
    def __init__(self, tensors):
        self.tensors = tensors
        self.saved = tuple(value.clone() for value in tensors)
        self.bytes = sum(value.numel() * value.element_size() for value in self.saved)

    def restore(self):
        for destination, source in zip(self.tensors, self.saved, strict=True):
            destination.copy_(source)


def stage_state_tensors(program):
    mutable = {"swa", "main", "index", "indices", "candidate_pool", "kv_history", "score_history", "block_table"}
    return tuple(value for name, value in program.named_buffers()
                 if not name.startswith("draft.") and name.rsplit(".", 1)[-1] in mutable)


class StageVariant(torch.nn.Module):
    def __init__(self, program, hidden, pre_mix, positions, input_ids, engram, pp_wire=None, fused_text_io=False):
        super().__init__()
        self.program = program
        self.adapter = DEEPSEEK_V41_PP0 if program.pp_rank == 0 else DEEPSEEK_V41_PP1
        if program.length > 512:
            extra = sum(2 for layer in program.layers if layer.attention.owns_index
                        and layer.attention.search_length // layer.attention.ratio > 512)
            self.adapter = replace(self.adapter, extra_collectives=self.adapter.extra_collectives + extra)
        from vllm_gaudi.models.deepseek_v41_program import CompiledStage
        self.wire_input = pp_wire is not None
        self.fused_text_io = fused_text_io
        if fused_text_io:
            self.adapter = replace(self.adapter, extra_collectives=self.adapter.extra_collectives + 1)
        self.compiled = CompiledStage(program, native=True, pp_wire_input=self.wire_input,
                                      fused_text_io=fused_text_io)
        self.fixed = tuple(value.clone() if value is not None else None
                           for value in (hidden, pre_mix, positions, input_ids))
        from vllm_gaudi import envs
        # The PP receive tensor has a persistent allocation. Native input
        # dependencies wait for its producer and register its last consumer.
        self.pp_wire = (pp_wire if envs.VLLM_HPU_DSV41_DIRECT_PP_WIRE else pp_wire.clone()) if self.wire_input else None
        self.engram = tuple(value.clone() for value in engram)
        self.states = stage_state_tensors(program)
        self.metadata = _Metadata()
        self.capture_bytes = 0
        self.warm_calls = 0

    def snapshot(self):
        if self.program.length > 512:
            snapshot = _PagedSnapshot(self.program, self.fixed[2], self.states)
        else:
            snapshot = _Snapshot(self.states)
        self.capture_bytes = snapshot.bytes
        return snapshot

    def forward(self, hidden, pre_mix, positions, input_ids, engram, pp_wire=None, fused_text_io=False):
        from vllm_gaudi.ops.tp2_prepared_plan import (
            collect_prepared_group_replays, record_native_decoder_outputs, replay_native_decoder,
        )
        if (pp_wire is not None) != self.wire_input or fused_text_io != self.fused_text_io:
            raise RuntimeError("V4.1 stage input contract changed without preparing its variant")
        roots = dict(hidden_states=hidden, pre_mix=pre_mix, positions=positions, input_ids=input_ids, pp_wire=pp_wire,
                     attention_inputs=engram, metadata=self.metadata, state_generation=self.program.generation,
                     state_tensors=self.states)
        outputs = replay_native_decoder(self, **roots)
        if outputs is not None:
            return outputs
        for destination, source in zip(self.fixed, (hidden, pre_mix, positions, input_ids), strict=True):
            if destination is not None:
                destination.copy_(source)
        if self.wire_input and self.pp_wire is not pp_wire:
            self.pp_wire.copy_(pp_wire)
        for destination, source in zip(self.engram, engram, strict=True):
            destination.copy_(source)
        fixed_hidden, fixed_pre, fixed_positions, fixed_ids = self.fixed
        fixed_roots = dict(roots, hidden_states=fixed_hidden, pre_mix=fixed_pre, positions=fixed_positions,
                           input_ids=fixed_ids, attention_inputs=self.engram, pp_wire=self.pp_wire)
        if self.wire_input:
            fixed_hidden = self.pp_wire
        with collect_prepared_group_replays(owner=self, adapter=self.adapter, snapshot=self.snapshot,
                                           **fixed_roots) as context:
            for index, chunk in enumerate(self.compiled.chunks):
                context["group_index"] = index
                fixed_hidden, fixed_pre, aux = chunk(fixed_hidden, fixed_pre, fixed_positions, fixed_ids, self.engram)
            outputs = fixed_hidden, fixed_pre, aux
            record_native_decoder_outputs(*outputs)
        self.warm_calls += 1
        return outputs


class StageReplay:
    def __init__(self, program):
        self.program = weakref.ref(program)
        self.variants = {}

    def __call__(self, hidden, pre_mix, positions, input_ids, engram, pp_wire=None, fused_text_io=False):
        tokens = input_ids.numel()
        if not 1 <= tokens <= 6:
            raise ValueError("Native V4.1 verification requires between one and six real tokens")
        program = self.program()
        search = getattr(program, "search_length", 512)
        key = (tokens, search, fused_text_io)
        if key not in self.variants:
            self.variants[key] = StageVariant(program, hidden, pre_mix, positions, input_ids, engram, pp_wire, fused_text_io)
        return self.variants[key](hidden, pre_mix, positions, input_ids, engram, pp_wire, fused_text_io)

    def require_ready(self, tokens, search=512):
        from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
        from vllm_gaudi import envs
        fused = self.program().pp_rank == 0 and envs.VLLM_HPU_DSV41_FUSED_STAGE_IO
        variant = self.variants.get((tokens, search, fused))
        if variant is None or variant not in _native_entries:
            raise RuntimeError("V4.1 warmup did not capture its complete native stage; serving cannot start")

    def close(self):
        from vllm_gaudi.ops.tp2_prepared_plan import invalidate_prepared_group_plans
        for variant in self.variants.values():
            invalidate_prepared_group_plans(owner=variant, reason="stage_close")
        self.variants.clear()


class _PagedSnapshot:
    """Capture saves only the rows a target invocation can overwrite."""
    def __init__(self, program, positions, states):
        from vllm_gaudi.ops.deepseek_v41_paged_attention import PAGE_TOKENS
        pooled = {id(getattr(cache, name)) for cache in program.shared.sources.values() for name in ("main", "index")}
        self.small = _Snapshot(tuple(value for value in states if id(value) not in pooled))
        self.rows = []
        for cache in program.shared.sources.values():
            logical = positions // cache.ratio
            physical = program.shared.physical_rows(logical, cache.ratio)
            # Deduplicate on the host only during capture; steady replay stays native.
            indices = torch.cat((physical, logical.remainder(PAGE_TOKENS // cache.ratio))).cpu().unique()
            indices = indices.to(device=positions.device, dtype=torch.int64)
            for name in ("main", "index"):
                value = getattr(cache, name)
                self.rows.append((value, indices, value.index_select(0, indices).clone()))
        self.bytes = self.small.bytes + sum(value.numel() * value.element_size() for _, _, value in self.rows)

    def restore(self):
        self.small.restore()
        for destination, indices, saved in self.rows:
            destination.index_copy_(0, indices, saved)
