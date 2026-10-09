# SPDX-License-Identifier: Apache-2.0
"""Join existing Target groups and the existing sampled draft into one replay.

This private capability changes ownership/submission only. All Target, MTP,
sampling and communication math remains in the common compiled entries.
Serving admission requires the actual producer/consumer native A/B gate.
"""
from types import SimpleNamespace

import torch

from vllm_gaudi.ops.deepseek_v41_replay import _PagedSnapshot, _Snapshot
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
from vllm_gaudi.ops.tp2_prepared_plan import (
    collect_prepared_group_replays,
    invalidate_prepared_group_plans,
    record_native_decoder_outputs,
    replay_native_decoder,
)


class _ScopedSnapshots:
    def __init__(self, target, protocol):
        self.target, self.protocol = target, protocol
        self.bytes = target.bytes + protocol.bytes

    def restore(self):
        self.target.restore()
        self.protocol.restore()


class JoinedTargetDraftProtocol(torch.nn.Module):
    """Reuse C1/C6 stage groups and the official p/q protocol without host glue.

    The Target must own final collapse and all three MTP context producers.
    This is equally valid for either sampling TP group; no rank-count branch
    or second implementation of the model/peer executor is introduced.
    """

    def __init__(self, target, protocol):
        super().__init__()
        if not protocol.sampled or protocol.closed or protocol.fixed is None:
            raise ValueError("Join requires the prepared official sampled protocol")
        if (not getattr(target.program, "is_last_stage", False)
                or not target.compiled.groups[-1].final or target.tail_enabled):
            raise ValueError("Joined Target must own final hidden and MTP context")
        if target.native_input or target.wire_input:
            raise ValueError("First capability uses the existing residual/pre input owner")
        if target.fixed[2].shape != (6,) or target.fixed[3].shape != (6,):
            raise ValueError("Join requires native C6 Target inputs")
        self.target = target
        self.protocol = protocol
        self.fixed = tuple(value.clone() if value is not None else None for value in target.fixed)
        self.engram = tuple(value.clone() for value in target.engram)
        # The protocol's hidden/auxiliary are produced by Target, rather than
        # staged inputs. Retain only its seven changing external tensors.
        source = protocol.fixed
        self.fixed_control = tuple(value.clone() for value in (source[1], source[2], *source[5:]))
        self.metadata = SimpleNamespace(native_completion=None)
        self.adapter = DecoderTopology(
            "deepseek_v41_joint_sampled",
            (*target.adapter.group_layers, *protocol.adapter.group_layers),
            target.adapter.reductions_per_layer,
            target.adapter.external_prefix,
            target.adapter.extra_collectives + protocol.adapter.collectives
            - sum(protocol.adapter.group_layers) * target.adapter.reductions_per_layer,
        )
        self.generation = self._generation()
        self.states = self._states()
        self.warm_calls = 0
        self.closed = False

    def _generation(self):
        generation = self.protocol.generation
        return (self.target.program.generation, self.target.program.precision_fingerprint,
                generation() if callable(generation) else generation)

    def _states(self):
        states = [*self.target.states, *self.protocol.mutable_states(), self.fixed_control[5]]
        frame = self.protocol.repair_frame
        if frame is not None:
            states.extend(frame.protocol_states(exact_repair=self.protocol.full and not self.protocol.full_main))
        for name in self.protocol.debug_state_names:
            value = getattr(self.protocol.draft, name, None)
            if value is not None:
                states.append(value)
        return tuple({id(value): value for value in states}.values())

    def snapshot(self):
        target = (
            _PagedSnapshot(self.target.program, self.fixed[2], self.target.states)
            if self.target.program.length > 512 else _Snapshot(self.target.states)
        )
        # Do not clone a full Target KV allocation. Only the protocol's small
        # MTP/journal write set is snapshotted in addition to paged Target rows.
        target_ids = {id(value) for value in self.target.states}
        protocol = _Snapshot(tuple(value for value in self.states if id(value) not in target_ids))
        return _ScopedSnapshots(target, protocol)

    def _roots(self, hidden, pre, positions, ids, engram, control):
        return dict(
            hidden_states=hidden, pre_mix=pre, positions=positions, input_ids=ids,
            pp_wire=None, attention_inputs=(*engram, *control), metadata=self.metadata,
            state_generation=self.generation, state_tensors=self.states,
        )

    def forward(self, hidden, pre, positions, ids, engram, proposed, control, q, parameters, seed, counter, offsets):
        if self.closed or self._generation() != self.generation:
            raise RuntimeError("Joined plan must retire before generation/weight replacement")
        changing = proposed, control, q, parameters, seed, counter, offsets
        if len(engram) != len(self.engram):
            raise ValueError("Prepare a new joined owner after Engram input layout changes")
        for old, new in zip((*self.fixed, *self.engram, *self.fixed_control),
                            (hidden, pre, positions, ids, *engram, *changing), strict=True):
            if old is None or new is None:
                if old is not new:
                    raise ValueError("Joined input optionality changed")
            elif old.shape != new.shape or old.dtype != new.dtype or old.device != new.device:
                raise ValueError("Joined input shape/dtype/device changed")
        self.states = self._states()
        roots = self._roots(hidden, pre, positions, ids, engram, changing)
        outputs = replay_native_decoder(self, **roots)
        if outputs is not None:
            return outputs
        for destination, source in zip((*self.fixed, *self.engram, *self.fixed_control),
                                       (hidden, pre, positions, ids, *engram, *changing), strict=True):
            if destination is not None:
                destination.copy_(source)
        fixed_hidden, fixed_pre, fixed_positions, fixed_ids = self.fixed
        fixed_roots = self._roots(*self.fixed, self.engram, self.fixed_control)
        with collect_prepared_group_replays(
            owner=self, adapter=self.adapter, snapshot=self.snapshot, **fixed_roots
        ) as context:
            for index, chunk in enumerate(self.target.compiled.chunks):
                context["group_index"] = index
                values = chunk(fixed_hidden, fixed_pre, fixed_positions, fixed_ids, self.engram)
                fixed_hidden, fixed_pre, auxiliary = values[:3]
            if auxiliary is None or auxiliary.shape != (6, 15360):
                raise RuntimeError("Joined Target omitted real layer37/38/39 MTP context producers")
            context["group_index"] = len(self.target.compiled.chunks)
            proposed, control, q, parameters, seed, counter, offsets = self.fixed_control
            sampled = self.protocol.compiled(
                fixed_hidden, proposed, control, auxiliary, fixed_positions,
                q, parameters, seed, counter, offsets,
            )
            outputs = fixed_hidden, fixed_pre, auxiliary, *sampled
            record_native_decoder_outputs(*outputs)
        self.warm_calls += 1
        return outputs

    def prepare(self, *inputs):
        initial = self.snapshot()
        try:
            for _ in range(3):
                initial.restore()
                self(*inputs)
                torch.hpu.synchronize()
            self.require_ready()
        finally:
            initial.restore()
            torch.hpu.synchronize()

    def require_ready(self):
        from vllm_gaudi.ops.tp2_prepared_plan import _native_entries

        if self.closed or self not in _native_entries:
            raise RuntimeError("Complete joined Target/draft native graph was not captured")

    def close(self):
        if not self.closed:
            invalidate_prepared_group_plans(owner=self, reason="joined_sampled_close")
            self.closed = True
