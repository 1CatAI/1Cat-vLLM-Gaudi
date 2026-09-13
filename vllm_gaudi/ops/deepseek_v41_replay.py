# SPDX-License-Identifier: Apache-2.0
"""V4.1 stage binding for the maintained native compute/HCL replay plan."""

from dataclasses import dataclass
import weakref

import torch

from vllm_gaudi.ops.deepseek_v41_diagnostics import trace_phase

from vllm_gaudi.ops.tp2_model_adapter import DEEPSEEK_V41_PP0, DEEPSEEK_V41_PP1


def stage_collectives(tp_rank, native):
    from vllm.distributed import tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce

    def reduce(value):
        if native and value.dtype == torch.bfloat16 and value.numel() <= 32768:
            flat = value.reshape(1, -1).contiguous()
            peer = torch.ops.vllm_gaudi.tp2_exchange_peer(flat)
            return (flat + peer).reshape(value.shape)
        return tensor_model_parallel_all_reduce(value)

    def gather(value, dim):
        if native and dim == 1 and value.dtype == torch.bfloat16 and value.numel() <= 32768:
            peer = torch.ops.vllm_gaudi.tp2_exchange_peer(value.reshape(1, -1).contiguous()).reshape(value.shape)
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
        "score_history"
    }
    return tuple(value for name, value in program.named_buffers()
                 if not name.startswith("draft.") and name.rsplit(".", 1)[-1] in mutable)


class StageVariant(torch.nn.Module):

    def __init__(self, program, hidden, pre_mix, positions, input_ids, engram):
        super().__init__()
        self.program = program
        self.adapter = DEEPSEEK_V41_PP0 if program.pp_rank == 0 else DEEPSEEK_V41_PP1
        from vllm_gaudi.models.deepseek_v41_program import CompiledStage
        self.compiled = CompiledStage(program, native=True)
        self.fixed = tuple(value.clone() for value in (hidden, pre_mix, positions, input_ids))
        self.engram = tuple(value.clone() for value in engram)
        self.states = stage_state_tensors(program)
        self.metadata = _Metadata()
        self.capture_bytes = 0
        self.warm_calls = 0

    def snapshot(self):
        snapshot = _Snapshot(self.states)
        self.capture_bytes = snapshot.bytes
        return snapshot

    @trace_phase
    def forward(self, hidden, pre_mix, positions, input_ids, engram):
        from vllm_gaudi.ops.tp2_prepared_plan import (
            collect_prepared_group_replays,
            record_native_decoder_outputs,
            replay_native_decoder,
        )
        roots = dict(hidden_states=hidden,
                     pre_mix=pre_mix,
                     positions=positions,
                     input_ids=input_ids,
                     attention_inputs=engram,
                     metadata=self.metadata,
                     state_generation=(self.program.generation, self.program.precision_fingerprint),
                     state_tensors=self.states)
        outputs = replay_native_decoder(self, **roots)
        if outputs is not None:
            return outputs
        for destination, source in zip(self.fixed, (hidden, pre_mix, positions, input_ids), strict=True):
            destination.copy_(source)
        for destination, source in zip(self.engram, engram, strict=True):
            destination.copy_(source)
        fixed_hidden, fixed_pre, fixed_positions, fixed_ids = self.fixed
        fixed_roots = dict(roots,
                           hidden_states=fixed_hidden,
                           pre_mix=fixed_pre,
                           positions=fixed_positions,
                           input_ids=fixed_ids,
                           attention_inputs=self.engram)
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

    @trace_phase
    def __call__(self, hidden, pre_mix, positions, input_ids, engram):
        tokens = input_ids.numel()
        if tokens not in ((1, 6) if self.program().dspark else (1, )):
            raise ValueError("V4.1 replay shape must match C1 decode or enabled C6 DSpark verification")
        if tokens not in self.variants:
            self.variants[tokens] = StageVariant(self.program(), hidden, pre_mix, positions, input_ids, engram)
        return self.variants[tokens](hidden, pre_mix, positions, input_ids, engram)

    def require_ready(self, tokens):
        from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
        variant = self.variants.get(tokens)
        if variant is None or variant not in _native_entries:
            raise RuntimeError("V4.1 warmup did not capture its complete native stage; serving cannot start")

    def close(self):
        from vllm_gaudi.ops.tp2_prepared_plan import invalidate_prepared_group_plans
        invalidate_prepared_group_plans()
        self.variants.clear()
