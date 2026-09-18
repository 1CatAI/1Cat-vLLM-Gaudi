# SPDX-License-Identifier: Apache-2.0
"""V4.1 stage binding for the maintained native compute/HCL replay plan."""

from dataclasses import dataclass, replace
import weakref

import torch

from vllm_gaudi.ops.deepseek_v41_diagnostics import trace_phase
from vllm_gaudi.ops.tp2_model_adapter import (
    DEEPSEEK_V41_PP0,
    DEEPSEEK_V41_PP0_INPUT,
    DEEPSEEK_V41_PP1,
)


def _native_input_precision_compatible(program):
    return not program.dspark and not (program.fp8_decode and not getattr(program, "expert_n256", False))


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
    mutable = {
        "swa", "main", "decoded_swa", "decoded_main", "index", "indices", "candidate_pool", "kv_history",
        "score_history", "block_table"
    }
    if getattr(program, "length", 512) > 512 and getattr(program, "search_length", 512) > 512:
        # The decoded working set belongs only to the <=512 prefix variant.
        # Longer variants read/write the canonical packed pages. Binding an
        # inactive decoded allocation fails native ownership validation; it
        # is neither a producer nor a consumer of that captured program.
        mutable.difference_update(("decoded_swa", "decoded_main"))
    return tuple(value for name, value in program.named_buffers()
                 if not name.startswith("draft.") and name.rsplit(".", 1)[-1] in mutable)


def capture_engram_inputs(engram, *, direct=False, device_layer1=False):
    if not direct:
        return tuple(value.clone() for value in engram)
    if len(engram) != 2:
        raise ValueError("Direct Engram capture requires both layer inputs")
    if device_layer1:
        first, late = engram
        if (first.dtype != torch.bfloat16 or tuple(first.shape) != (1, 12, 256) or not first.is_contiguous()
                or late.dtype != torch.uint8 or tuple(late.shape) != (1, 12, 264) or not late.is_contiguous()
                or first.device != late.device):
            raise ValueError("Device Engram capture requires fixed BF16 layer-1 and packed U8 layer-14 inputs")
        return tuple(engram)
    first = engram[0]
    offset = 0
    for value in engram:
        if (value.dtype != torch.uint8 or value.ndim != 3 or value.shape[0] != 1 or not value.is_contiguous()
                or value.device != first.device or value.shape[-1] != first.shape[-1]
                or value.untyped_storage()._cdata != first.untyped_storage()._cdata or value.storage_offset() != offset
                or value.numel() == 0):
            raise ValueError("Direct Engram inputs must be ordered views of one complete C1 packet")
        offset += value.numel()
    if first.untyped_storage().nbytes() != offset:
        raise ValueError("Direct Engram capture requires the complete packet allocation")
    return tuple(engram)


class StageVariant(torch.nn.Module):

    def __init__(self,
                 program,
                 hidden,
                 pre_mix,
                 positions,
                 input_ids,
                 engram,
                 pp_wire=None,
                 fused_text_io=False,
                 *,
                 native_input=False):
        super().__init__()
        self.program = program
        self.native_input = native_input
        self.adapter = (DEEPSEEK_V41_PP0_INPUT
                        if native_input else DEEPSEEK_V41_PP0 if program.pp_rank == 0 else DEEPSEEK_V41_PP1)
        if program.length > 512:
            extra = sum(2 for layer in program.layers
                        if layer.attention.owns_index and layer.attention.search_length // layer.attention.ratio > 512)
            self.adapter = replace(self.adapter, extra_collectives=self.adapter.extra_collectives + extra)
        from vllm_gaudi.models.deepseek_v41_program import CompiledStage
        self.wire_input = pp_wire is not None
        self.fused_text_io = fused_text_io
        if fused_text_io:
            self.adapter = replace(self.adapter, extra_collectives=self.adapter.extra_collectives + 1)
        self.compiled = CompiledStage(program,
                                      native=True,
                                      pp_wire_input=self.wire_input,
                                      fused_text_io=fused_text_io,
                                      native_input=native_input)
        self.fixed = tuple(value.clone() if value is not None else None
                           for value in (hidden, pre_mix, positions, input_ids))
        from vllm_gaudi import envs
        # The PP receive tensor has a persistent allocation. Native input
        # dependencies wait for its producer and register its last consumer.
        self.pp_wire = (pp_wire if envs.VLLM_HPU_DSV41_DIRECT_PP_WIRE else pp_wire.clone()) if self.wire_input else None
        self.direct_engram = bool(engram) and native_input and envs.VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT
        self.device_engram = bool(engram) and native_input and envs.VLLM_HPU_DSV41_V2_DEVICE_ENGRAM
        self.engram = capture_engram_inputs(engram, direct=self.direct_engram, device_layer1=self.device_engram)
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

    @trace_phase
    def forward(self,
                hidden,
                pre_mix,
                positions,
                input_ids,
                engram,
                pp_wire=None,
                fused_text_io=False,
                native_input=False):
        from vllm_gaudi.ops.tp2_prepared_plan import (
            collect_prepared_group_replays,
            record_native_decoder_outputs,
            replay_native_decoder,
        )
        if ((pp_wire is not None) != self.wire_input or fused_text_io != self.fused_text_io
                or native_input != self.native_input):
            raise RuntimeError("V4.1 stage input contract changed without preparing its variant")
        roots = dict(hidden_states=hidden,
                     pre_mix=pre_mix,
                     positions=positions,
                     input_ids=input_ids,
                     pp_wire=pp_wire,
                     attention_inputs=engram,
                     metadata=self.metadata,
                     state_generation=(self.program.generation, self.program.precision_fingerprint),
                     state_tensors=self.states)
        outputs = replay_native_decoder(self, **roots)
        if outputs is not None:
            return outputs
        for destination, source in zip(self.fixed, (hidden, pre_mix, positions, input_ids), strict=True):
            if destination is not None:
                destination.copy_(source)
        if self.wire_input and self.pp_wire is not pp_wire:
            self.pp_wire.copy_(pp_wire)
        if not self.direct_engram:
            for destination, source in zip(self.engram, engram, strict=True):
                destination.copy_(source)
        fixed_hidden, fixed_pre, fixed_positions, fixed_ids = self.fixed
        fixed_roots = dict(roots,
                           hidden_states=fixed_hidden,
                           pre_mix=fixed_pre,
                           positions=fixed_positions,
                           input_ids=fixed_ids,
                           attention_inputs=self.engram,
                           pp_wire=self.pp_wire)
        if self.wire_input:
            fixed_hidden = self.pp_wire
        # A normal vLLM scheduler transaction may contain thousands of
        # prompt tokens.  Do not capture all of its bounded N256 tiles into a
        # single native recipe: that retains every tile workspace until the
        # four-layer group closes and can exhaust HPU memory.  The compiled
        # stage has an eager large-M path that keeps the same layer order,
        # state updates and 128-token resource tiles, while C1/C6 continues
        # through the replay capture below.
        from vllm_gaudi.models.deepseek_v41_program import PreparedMoE
        if fixed_hidden is not None and fixed_hidden.shape[0] > PreparedMoE.N256_PREFILL_TILE:
            return self.compiled(fixed_hidden, fixed_pre, fixed_positions, fixed_ids, self.engram)
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
        from vllm_gaudi import envs
        self.program = weakref.ref(program)
        self.variants = {}
        self.native_input_enabled = (envs.VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH and program.pp_rank == 0)
        self.segmented_prefix_enabled = envs.VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX and program.pp_rank == 0
        if self.segmented_prefix_enabled and not self.native_input_enabled:
            raise ValueError("V2 segmented prefix requires native PP0 input replay")
        if (envs.VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT
                and (not _native_input_precision_compatible(program) or not envs.VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH)):
            raise ValueError("Direct Engram capture requires ordinary BF16 native input replay")
        self.input_seed = None

    def _input_seed(self, input_ids):
        if self.input_seed is None:
            self.input_seed = (
                torch.zeros(1, 4, 5120, device=input_ids.device, dtype=torch.bfloat16),
                torch.zeros(1, 4, device=input_ids.device, dtype=torch.float32),
            )
        return self.input_seed

    def from_input_ids(self, positions, input_ids, engram):
        program = self.program()
        if (not self.native_input_enabled or input_ids.numel() != 1 or not _native_input_precision_compatible(program)):
            raise ValueError("Native input capture requires enabled ordinary BF16 C1 PP0 decode")
        if self.segmented_prefix_enabled:
            from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
            variant = self.variants.get(self._input_key())
            if variant is not None and variant in _native_entries:
                self.begin_segmented_from_input_ids(positions, input_ids)
                return self.finish_segmented(positions, input_ids, engram)
        return self(*self._input_seed(input_ids), positions, input_ids, engram, native_input=True)

    def _input_key(self, search=None):
        if search is None:
            search = getattr(self.program(), "search_length", 512)
        return (1, "input") if search <= 512 else (1, "input", search)

    def input_variant_ready(self, search):
        """Return whether a complete C1 input plan exists for ``search``.

        V2 starts the embedding/Engram-independent prefix before the next
        scheduler turn enters ``_forward``.  At a paged-attention bucket
        boundary, the program still names the previous bucket at that point.
        Check the next bucket explicitly so a segmented prefix can never be
        started on one recipe and finished on another.  The first token in a
        new bucket will capture the complete native recipe; later tokens can
        resume segmented replay.
        """
        from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
        variant = self.variants.get(self._input_key(int(search)))
        return variant is not None and variant in _native_entries

    def _complete_input_variant(self):
        variant = self.variants.get(self._input_key())
        if variant is None:
            raise RuntimeError("Segmented replay requires the warmed complete PP0 input graph")
        return variant

    @trace_phase
    def begin_segmented_from_input_ids(self, positions, input_ids):
        program = self.program()
        if (not self.segmented_prefix_enabled or input_ids.numel() != 1
                or not _native_input_precision_compatible(program)):
            raise ValueError("V2 segmented prefix requires enabled ordinary BF16 C1 PP0 decode")
        from vllm_gaudi.ops.tp2_prepared_plan import begin_segmented_native_decoder
        variant = self._complete_input_variant()
        hidden, pre_mix = self._input_seed(input_ids)
        roots = dict(hidden_states=hidden,
                     pre_mix=pre_mix,
                     positions=positions,
                     input_ids=input_ids,
                     attention_inputs=variant.engram,
                     metadata=variant.metadata,
                     state_generation=(program.generation, program.precision_fingerprint),
                     state_tensors=variant.states)
        begin_segmented_native_decoder(variant, **roots)

    @trace_phase
    def finish_segmented(self, positions, input_ids, engram):
        if not self.segmented_prefix_enabled or input_ids.numel() != 1:
            raise ValueError("V2 segmented suffix requires an active C1 PP0 prefix")
        from vllm_gaudi.ops.tp2_prepared_plan import finish_segmented_native_decoder
        variant = self._complete_input_variant()
        program = self.program()
        hidden, pre_mix = self._input_seed(input_ids)
        roots = dict(hidden_states=hidden,
                     pre_mix=pre_mix,
                     positions=positions,
                     input_ids=input_ids,
                     attention_inputs=engram,
                     metadata=variant.metadata,
                     state_generation=(program.generation, program.precision_fingerprint),
                     state_tensors=variant.states)
        return finish_segmented_native_decoder(variant, **roots)

    @trace_phase
    def __call__(self,
                 hidden,
                 pre_mix,
                 positions,
                 input_ids,
                 engram,
                 pp_wire=None,
                 fused_text_io=False,
                 native_input=False):
        tokens = input_ids.numel()
        program = self.program()
        if not 1 <= tokens <= 6:
            raise ValueError("V4.1 replay shape must be between C1 and C6")
        if not program.dspark and tokens != 1:
            raise ValueError("V4.1 replay shape must be C1 when DSpark is disabled")
        if native_input and (not self.native_input_enabled or not _native_input_precision_compatible(program)
                             or tokens != 1):
            raise ValueError("Native input capture requires enabled ordinary BF16 C1 PP0 decode")
        search = getattr(program, "search_length", 512)
        key = (((tokens, "input") if search <= 512 else
                (tokens, "input",
                 search)) if native_input else tokens if search <= 512 and not fused_text_io and pp_wire is None else
               (tokens, search, fused_text_io))
        if key not in self.variants:
            self.variants[key] = StageVariant(program,
                                              hidden,
                                              pre_mix,
                                              positions,
                                              input_ids,
                                              engram,
                                              pp_wire,
                                              fused_text_io,
                                              native_input=native_input)
        return self.variants[key](hidden, pre_mix, positions, input_ids, engram, pp_wire, fused_text_io, native_input)

    def require_ready(self, tokens, search=512):
        from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
        from vllm_gaudi import envs
        program = self.program()
        native_input = (self.native_input_enabled and tokens == 1 and _native_input_precision_compatible(program))
        fused = (program.pp_rank == 0 and envs.VLLM_HPU_DSV41_FUSED_STAGE_IO)
        key = (((tokens, "input") if search <= 512 else
                (tokens, "input", search)) if native_input else tokens if search <= 512 and not fused else
               (tokens, search, fused))
        variant = self.variants.get(key)
        if variant is None or variant not in _native_entries:
            raise RuntimeError("V4.1 warmup did not capture its complete native stage; serving cannot start")

    def close(self):
        from vllm_gaudi.ops.tp2_prepared_plan import invalidate_prepared_group_plans
        for variant in self.variants.values():
            invalidate_prepared_group_plans(owner=variant, reason="stage_close")
        self.variants.clear()
        self.input_seed = None


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
