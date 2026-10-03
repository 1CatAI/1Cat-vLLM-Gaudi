# SPDX-License-Identifier: Apache-2.0
"""V4.1 stage binding for the maintained native compute/HCL replay plan."""

from dataclasses import dataclass, replace
import weakref

import torch

from vllm_gaudi.ops.deepseek_v41_diagnostics import trace_phase
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology


def _native_input_precision_compatible(program):
    return not program.dspark and not (program.fp8_decode and not getattr(program, "expert_n256", False))


def stage_collectives(tp_rank, native, tp_size=2, *, native_fp32_gather=False):
    from vllm.distributed import tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce
    from vllm_gaudi import envs

    maximum = 64 * 5120 if envs.VLLM_HPU_DSV41_BATCH_DECODE else 32768

    def reduce(value, *, ready_outputs=()):
        if native and value.dtype == torch.bfloat16 and value.numel() <= maximum:
            flat = value.reshape(1, -1).contiguous()
            if tp_size == 2:
                peer = (
                    torch.ops.vllm_gaudi.tp2_exchange_peer_scheduled(flat, list(ready_outputs))
                    if ready_outputs
                    else torch.ops.vllm_gaudi.tp2_exchange_peer(flat)
                )
                return (flat + peer).reshape(value.shape)
            shards = (
                torch.ops.vllm_gaudi.tp_peer_allgather_scheduled(flat, tp_size, list(ready_outputs))
                if ready_outputs
                else torch.ops.vllm_gaudi.tp_peer_allgather(flat, tp_size)
            )
            shards = shards.reshape(tp_size, flat.numel())
            reduced = shards[0].float()
            for rank in range(1, tp_size):
                reduced = reduced + shards[rank].float()
            return reduced.to(value.dtype).reshape(value.shape)
        return tensor_model_parallel_all_reduce(value)

    def gather(value, dim):
        axis = dim % value.ndim if -value.ndim <= dim < value.ndim else dim
        native_dtype = value.dtype == torch.bfloat16 or (native_fp32_gather and value.dtype == torch.float32)
        wire_elements = value.numel() * (value.element_size() // 2)
        if native and axis == 1 and native_dtype and wire_elements <= maximum:
            # Exact bytes on the established BF16 peer wire. Reinterpretation
            # preserves FP32 argmax scores and token IDs, including NaN payloads.
            flat = value.reshape(1, -1).contiguous()
            if value.dtype == torch.float32:
                flat = flat.view(torch.bfloat16)
            # The native command path transfers 128 BF16 elements per unit.
            if wire_elements % 128:
                flat = torch.nn.functional.pad(flat, (0, -wire_elements % 128))
            if tp_size == 2:
                peer = torch.ops.vllm_gaudi.tp2_exchange_peer(flat)[:, :wire_elements]
                if value.dtype == torch.float32:
                    peer = peer.view(value.dtype)
                peer = peer.reshape(value.shape)
                first, second = (value, peer) if tp_rank == 0 else (peer, value)
                return torch.cat((first, second), dim=axis)
            shards = torch.ops.vllm_gaudi.tp_peer_allgather(flat, tp_size).reshape(tp_size, flat.numel())
            return torch.cat(
                tuple((shards[rank, :wire_elements].view(value.dtype)
                       if value.dtype == torch.float32 else shards[rank, :wire_elements]).reshape(value.shape)
                      for rank in range(tp_size)), dim=axis
            )
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
        "swa",
        "main",
        "decoded_swa",
        "decoded_main",
        "decoded_index_hot",
        "index",
        "indices",
        "candidate_pool",
        "kv_history",
        "score_history",
        "block_table",
    }
    active_decoded = None
    if getattr(program, "length", 512) > 512 and getattr(program, "search_length", 512) > 512:
        # Hot paged variants also consume decoded mirrors. Bind only the
        # mirrors selected by PagedAttention's capacity check: the long-context
        # packed variant has no producer/consumer for inactive allocations.
        active_decoded = set()
        shared = program.shared
        if getattr(shared, "decoded_kv_state", False):
            for cache in shared.sources.values():
                value = getattr(cache, "decoded_main", None)
                if (
                    cache.ratio in (1, 2)
                    and value is not None
                    and program.search_length // cache.ratio <= value.shape[0]
                ):
                    active_decoded.add(id(value))
            if active_decoded:
                active_decoded.add(id(shared.decoded_swa))
    from vllm_gaudi.ops.deepseek_v41_indexer import INDEX_MME_HOT_TOKENS

    if getattr(program, "search_length", 512) != INDEX_MME_HOT_TOKENS:
        # The capacity-independent packed recipe neither reads nor updates the
        # bounded mirror.  Do not bind inactive state into its native plan.
        mutable.discard("decoded_index_hot")
    from vllm_gaudi.ops.deepseek_v41_index_mirror import index_mirror_execution_mode

    if index_mirror_execution_mode(program):
        mutable.add("index_mirror")
    return tuple(
        value
        for name, value in program.named_buffers()
        if not name.startswith("draft.")
        and ".batch_state." not in name
        and name.rsplit(".", 1)[-1] in mutable
        and (
            active_decoded is None
            or name.rsplit(".", 1)[-1] not in ("decoded_swa", "decoded_main")
            or id(value) in active_decoded
        )
    )


def capture_engram_inputs(engram, *, direct=False, device_layer1=False, local_heads=12):
    if not direct:
        return tuple(value.clone() for value in engram)
    if len(engram) != 2:
        raise ValueError("Direct Engram capture requires both layer inputs")
    if device_layer1:
        first, late = engram
        if (
            local_heads < 1
            or first.dtype != torch.bfloat16
            or tuple(first.shape) != (1, local_heads, 256)
            or not first.is_contiguous()
            or late.dtype != torch.uint8
            or tuple(late.shape) != (1, local_heads, 264)
            or not late.is_contiguous()
            or first.device != late.device
        ):
            raise ValueError("Device Engram capture requires fixed BF16 layer-1 and packed U8 layer-14 inputs")
        return tuple(engram)
    first = engram[0]
    offset = 0
    for value in engram:
        if (
            value.dtype != torch.uint8
            or value.ndim != 3
            or value.shape[0] != 1
            or not value.is_contiguous()
            or value.device != first.device
            or value.shape[-1] != first.shape[-1]
            or value.untyped_storage()._cdata != first.untyped_storage()._cdata
            or value.storage_offset() != offset
            or value.numel() == 0
        ):
            raise ValueError("Direct Engram inputs must be ordered views of one complete C1 packet")
        offset += value.numel()
    if first.untyped_storage().nbytes() != offset:
        raise ValueError("Direct Engram capture requires the complete packet allocation")
    return tuple(engram)


class StageVariant(torch.nn.Module):
    def __init__(
        self,
        program,
        hidden,
        pre_mix,
        positions,
        input_ids,
        engram,
        pp_wire=None,
        fused_text_io=False,
        *,
        native_input=False,
        replay_tail=False,
    ):
        super().__init__()
        self.program = program
        self.native_input = native_input
        self.tail_enabled = (replay_tail and native_input and getattr(program, "is_last_stage", False)
                             and not program.dspark)
        self.tail_values = None
        layers = len(program.layers)
        if layers % 4:
            raise ValueError("Native V4.1 stage requires complete four-layer groups")
        name = (
            "deepseek_v41_pp0_input"
            if native_input
            else "deepseek_v41_pp0"
            if program.pp_rank == 0
            else "deepseek_v41_pp1"
        )
        engram_collectives = sum(getattr(layer, "layer", -1) in (1, 14) for layer in program.layers)
        self.adapter = DecoderTopology(name, (4,) * (layers // 4), 2, False, engram_collectives + int(native_input))
        if getattr(program, "decode_merge_mhc_partitions", False):
            self.adapter = replace(self.adapter, require_independent_overlap=False)
        if program.length > 512:
            extra = sum(
                2
                for layer in program.layers
                if layer.attention.owns_index and layer.attention.search_length // layer.attention.ratio > 512
            )
            self.adapter = replace(self.adapter, extra_collectives=self.adapter.extra_collectives + extra)
        from vllm_gaudi.models.deepseek_v41_program import CompiledStage

        self.wire_input = pp_wire is not None
        self.fused_text_io = fused_text_io
        if fused_text_io:
            self.adapter = replace(self.adapter, extra_collectives=self.adapter.extra_collectives + 1)
        if self.tail_enabled:
            self.adapter = replace(self.adapter, extra_collectives=self.adapter.extra_collectives + 1)
        self.compiled = CompiledStage(
            program, native=True, pp_wire_input=self.wire_input, fused_text_io=fused_text_io,
            native_input=native_input, replay_tail=self.tail_enabled,
        )
        self.fixed = tuple(
            value.clone() if value is not None else None for value in (hidden, pre_mix, positions, input_ids)
        )
        from vllm_gaudi import envs

        # The PP receive tensor has a persistent allocation. Native input
        # dependencies wait for its producer and register its last consumer.
        self.pp_wire = (pp_wire if envs.VLLM_HPU_DSV41_DIRECT_PP_WIRE else pp_wire.clone()) if self.wire_input else None
        self.direct_engram = bool(engram) and native_input and envs.VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT
        self.device_engram = bool(engram) and native_input and envs.VLLM_HPU_DSV41_V2_DEVICE_ENGRAM
        self.engram = capture_engram_inputs(
            engram,
            direct=self.direct_engram,
            device_layer1=self.device_engram,
            local_heads=24 // getattr(program, "tensor_parallel_size", 2),
        )
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
    def forward(
        self, hidden, pre_mix, positions, input_ids, engram, pp_wire=None, fused_text_io=False, native_input=False
    ):
        from vllm_gaudi.ops.tp2_prepared_plan import (
            collect_prepared_group_replays,
            record_native_decoder_outputs,
            replay_native_decoder,
        )

        if (
            (pp_wire is not None) != self.wire_input
            or fused_text_io != self.fused_text_io
            or native_input != self.native_input
        ):
            raise RuntimeError("V4.1 stage input contract changed without preparing its variant")
        roots = dict(
            hidden_states=hidden,
            pre_mix=pre_mix,
            positions=positions,
            input_ids=input_ids,
            pp_wire=pp_wire,
            attention_inputs=engram,
            metadata=self.metadata,
            state_generation=(self.program.generation, self.program.precision_fingerprint),
            state_tensors=self.states,
        )
        outputs = replay_native_decoder(self, **roots)
        if outputs is not None:
            return self.publish_outputs(outputs)
        for destination, source in zip(self.fixed, (hidden, pre_mix, positions, input_ids), strict=True):
            if destination is not None:
                destination.copy_(source)
        if self.wire_input and self.pp_wire is not pp_wire:
            self.pp_wire.copy_(pp_wire)
        if not self.direct_engram:
            for destination, source in zip(self.engram, engram, strict=True):
                destination.copy_(source)
        fixed_hidden, fixed_pre, fixed_positions, fixed_ids = self.fixed
        fixed_roots = dict(
            roots,
            hidden_states=fixed_hidden,
            pre_mix=fixed_pre,
            positions=fixed_positions,
            input_ids=fixed_ids,
            attention_inputs=self.engram,
            pp_wire=self.pp_wire,
        )
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
        with collect_prepared_group_replays(
            owner=self, adapter=self.adapter, snapshot=self.snapshot, **fixed_roots
        ) as context:
            for index, chunk in enumerate(self.compiled.chunks):
                context["group_index"] = index
                values = chunk(fixed_hidden, fixed_pre, fixed_positions, fixed_ids, self.engram)
                fixed_hidden, fixed_pre, aux = values[:3]
            outputs = values if self.tail_enabled else (fixed_hidden, fixed_pre, aux)
            record_native_decoder_outputs(*outputs)
        self.warm_calls += 1
        return self.publish_outputs(outputs)

    def publish_outputs(self, outputs):
        self.tail_values = outputs[3:] if self.tail_enabled else None
        return outputs[:3]


class StageReplay:
    def __init__(self, program, *, greedy_tail=False):
        from vllm_gaudi import envs

        self.program = weakref.ref(program)
        self.variants = {}
        self.native_input_enabled = envs.VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH and program.pp_rank == 0
        self.segmented_prefix_enabled = envs.VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX and program.pp_rank == 0
        if self.segmented_prefix_enabled and not self.native_input_enabled:
            raise ValueError("V2 segmented prefix requires native PP0 input replay")
        if envs.VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT and (
            not _native_input_precision_compatible(program) or not envs.VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH
        ):
            raise ValueError("Direct Engram capture requires ordinary BF16 native input replay")
        self.input_seed = None
        self.latest_tail = None
        self.greedy_tail_enabled = bool(greedy_tail)

    def _tail_values(self, hidden):
        """Return outputs owned by this completed C1 hidden allocation."""
        if self.latest_tail is None:
            return None
        generation, source, values = self.latest_tail
        program = self.program()
        if generation != program.generation or hidden.dtype != source.dtype or hidden.shape != source.shape:
            return None
        if hidden.data_ptr() != source.data_ptr():
            return None
        return values

    def greedy_tail_token(self, hidden):
        values = self._tail_values(hidden)
        if values is None:
            return None
        return values[3] if getattr(self.program(), "device_sampling", False) else values[0]

    def sampling_tail_values(self, hidden):
        return self._tail_values(hidden) if getattr(self.program(), "device_sampling", False) else None

    def tail_local_logits(self, hidden):
        values = self._tail_values(hidden)
        return values[1] if values is not None and len(values) > 1 else None

    def _publish_tail(self, variant, outputs):
        if variant.tail_values is None:
            self.latest_tail = None
        else:
            self.latest_tail = self.program().generation, outputs[0], variant.tail_values
        return outputs

    def _input_seed(self, input_ids):
        if self.input_seed is None:
            self.input_seed = (
                torch.zeros(1, 4, 5120, device=input_ids.device, dtype=torch.bfloat16),
                torch.zeros(1, 4, device=input_ids.device, dtype=torch.float32),
            )
        return self.input_seed

    def from_input_ids(self, positions, input_ids, engram):
        program = self.program()
        if not self.native_input_enabled or input_ids.numel() != 1 or not _native_input_precision_compatible(program):
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
        key = (1, "input") if search <= 512 else (1, "input", search)
        return self._with_index_mirror(key, search)

    def _with_index_mirror(self, key, search):
        from vllm_gaudi.ops.deepseek_v41_index_mirror import index_mirror_execution_mode

        mode = index_mirror_execution_mode(self.program(), search)
        key = key if mode is None else (key, "index_mirror", mode)
        bound = getattr(self.program(), "decode_token_bound", None) if search <= 32768 else None
        # The score/selection recipes have a fixed visible row extent. A
        # different extent must select its own captured writer/reader graph.
        return key if bound is None else (key, "visible_prefix", bound)

    def input_variant_ready(self, search):
        """Return whether a complete C1 input plan exists for ``search``.

        V2 starts the embedding/Engram-independent prefix before the next
        scheduler turn enters ``_forward``.  At a paged-attention bucket
        boundary, the program still names the previous bucket at that point.
        This is a readiness query, not a binding change. The caller must also
        verify that the next bucket equals the currently bound bucket before
        starting an early prefix. At a transition, _forward binds the new
        bucket before entering the complete native replay.
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
        if (
            not self.segmented_prefix_enabled
            or input_ids.numel() != 1
            or not _native_input_precision_compatible(program)
        ):
            raise ValueError("V2 segmented prefix requires enabled ordinary BF16 C1 PP0 decode")
        from vllm_gaudi.ops.tp2_prepared_plan import begin_segmented_native_decoder

        variant = self._complete_input_variant()
        hidden, pre_mix = self._input_seed(input_ids)
        roots = dict(
            hidden_states=hidden,
            pre_mix=pre_mix,
            positions=positions,
            input_ids=input_ids,
            attention_inputs=variant.engram,
            metadata=variant.metadata,
            state_generation=(program.generation, program.precision_fingerprint),
            state_tensors=variant.states,
        )
        begin_segmented_native_decoder(variant, **roots)

    @trace_phase
    def finish_segmented(self, positions, input_ids, engram):
        if not self.segmented_prefix_enabled or input_ids.numel() != 1:
            raise ValueError("V2 segmented suffix requires an active C1 PP0 prefix")
        from vllm_gaudi.ops.tp2_prepared_plan import finish_segmented_native_decoder

        variant = self._complete_input_variant()
        program = self.program()
        hidden, pre_mix = self._input_seed(input_ids)
        roots = dict(
            hidden_states=hidden,
            pre_mix=pre_mix,
            positions=positions,
            input_ids=input_ids,
            attention_inputs=engram,
            metadata=variant.metadata,
            state_generation=(program.generation, program.precision_fingerprint),
            state_tensors=variant.states,
        )
        outputs = variant.publish_outputs(finish_segmented_native_decoder(variant, **roots))
        return self._publish_tail(variant, outputs)

    @trace_phase
    def __call__(
        self, hidden, pre_mix, positions, input_ids, engram, pp_wire=None, fused_text_io=False, native_input=False
    ):
        tokens = input_ids.numel()
        program = self.program()
        if not 1 <= tokens <= 6:
            raise ValueError("V4.1 replay shape must be between C1 and C6")
        if not program.dspark and tokens != 1:
            raise ValueError("V4.1 replay shape must be C1 when DSpark is disabled")
        if native_input and (
            not self.native_input_enabled or not _native_input_precision_compatible(program) or tokens != 1
        ):
            raise ValueError("Native input capture requires enabled ordinary BF16 C1 PP0 decode")
        search = getattr(program, "search_length", 512)
        key = (
            ((tokens, "input") if search <= 512 else (tokens, "input", search))
            if native_input
            else tokens
            if search <= 512 and not fused_text_io and pp_wire is None
            else (tokens, search, fused_text_io)
        )
        key = self._with_index_mirror(key, search)
        if key not in self.variants:
            self.variants[key] = StageVariant(
                program,
                hidden,
                pre_mix,
                positions,
                input_ids,
                engram,
                pp_wire,
                fused_text_io,
                native_input=native_input,
                replay_tail=self.greedy_tail_enabled,
            )
        variant = self.variants[key]
        outputs = variant(hidden, pre_mix, positions, input_ids, engram, pp_wire, fused_text_io, native_input)
        return self._publish_tail(variant, outputs)

    def require_ready(self, tokens, search=512):
        from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
        from vllm_gaudi import envs

        program = self.program()
        native_input = self.native_input_enabled and tokens == 1 and _native_input_precision_compatible(program)
        fused = program.pp_rank == 0 and envs.VLLM_HPU_DSV41_FUSED_STAGE_IO
        key = (
            ((tokens, "input") if search <= 512 else (tokens, "input", search))
            if native_input
            else tokens
            if search <= 512 and not fused
            else (tokens, search, fused)
        )
        key = self._with_index_mirror(key, search)
        variant = self.variants.get(key)
        if variant is None or variant not in _native_entries:
            raise RuntimeError("V4.1 warmup did not capture its complete native stage; serving cannot start")

    def close(self):
        from vllm_gaudi.ops.tp2_prepared_plan import invalidate_prepared_group_plans

        for variant in self.variants.values():
            invalidate_prepared_group_plans(owner=variant, reason="stage_close")
        self.variants.clear()
        self.input_seed = None
        self.latest_tail = None


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
