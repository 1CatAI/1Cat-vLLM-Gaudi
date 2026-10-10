# SPDX-License-Identifier: Apache-2.0
"""C6 verification/draft plan using the shared prepared decoder executor.

This plan is not selected by serving until its complete consumer chain is
qualified. Target math and the speculative prefix protocol remain owned by
PreparedDraft; no graph or communication implementation is duplicated here.
"""
from itertools import count
from types import FunctionType, MethodType, SimpleNamespace

import torch

from vllm_gaudi.ops.deepseek_v41_replay import _Snapshot
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
from vllm_gaudi.ops.tp2_prepared_plan import (
    collect_prepared_group_replays,
    invalidate_prepared_group_plans,
    record_native_decoder_outputs,
    replay_native_decoder,
)

_entries = count()


class NativeDraftProtocol(torch.nn.Module):
    """Keep weights and MTP state fixed while updating five input tensors."""

    def __init__(self, draft, *, generation, sampled=False, repair_frame=None, full=False, full_main=False,
                 known_stochastic=False, input_publication=None, state_publication=None):
        super().__init__()
        if draft.tensor_parallel_size not in (2, 4) or len(draft.layers) != 3:
            raise ValueError("Native speculative control requires the sampling TP group and three MTP layers")
        self.draft = draft
        self.generation = generation
        self.sampled = sampled
        self.full = full
        self.full_main = full_main
        self.known_stochastic = known_stochastic
        if input_publication is not None and not (sampled and full_main):
            raise ValueError("Next-round publication belongs to the sampled main plan, never exact repair")
        self.input_publication = input_publication
        if state_publication is not None and not sampled:
            raise ValueError("Sampling state publication belongs only to the sampled protocol")
        self.state_publication = state_publication
        if full_main and not (full and sampled):
            raise ValueError("Full-probability main plan requires official full sampled inputs")
        if full and not sampled:
            raise ValueError("Full native repair requires official sampling")
        if repair_frame is not None and not sampled:
            raise ValueError("Official-sampling repair belongs only to the sampled native plan")
        self.repair_frame = repair_frame
        self.debug_state_names = ("sampling_normalized_debug", "sampling_base_logits_debug", "sampling_logits_debug")
        # One target argmax, one draft embedding, six MTP reductions, and
        # five Markov embedding/argmax pairs. The existing peer wire already
        # transports FP32 scores bit-for-bit inside vocab_parallel_argmax.
        # Sampled control: target packet + three rejection exchanges,
        # embedding + six MTP reductions, then five embedding/packet pairs.
        collectives = 21 if sampled else 18
        from vllm_gaudi import envs

        # Large full-vocabulary logits retain the existing production
        # AllGather. This DSpark-only candidate changes neither the small
        # peer exchanges nor the shared replay/communication implementation.
        self.full_hccl = full and envs.VLLM_HPU_DSV41_DSPARK_FULL_HCCL_MAIN
        # The same exact transport also serves a journal-backed repair.
        # _new_entry selects verify_and_propose_sampled_full for that case,
        # which restores the saved state and forces the full distribution.
        if self.full_hccl and not full_main and repair_frame is None:
            raise ValueError("Full HCCL repair requires its saved request journal")
        if full and not self.full_hccl:
            local_vocab = draft.output_head.weight.shape[0]
            target_columns = 16384 // 6 // 64 * 64
            target_transfers = (local_vocab + target_columns - 1) // target_columns
            draft_transfers = (local_vocab + 16383) // 16384
            collectives += target_transfers - 1 + 5 * (draft_transfers - 1)
        self.adapter = DecoderTopology("deepseek_v41_dspark_control", (1,), 0, False, collectives)
        self.states = tuple(layer.attention.swa for layer in draft.layers)
        self.fixed = None
        self.metadata = None
        self.compiled = None
        self.captured_generation = None
        self.closed = False

    def _new_entry(self):
        method = (self.draft.verify_and_propose_sampled_full if self.full and not self.full_main else
                  self.draft.verify_and_propose_sampled_bound if self.sampled else self.draft.verify_and_propose)
        function = method.__func__
        name = f"{function.__name__}_native_control_{next(_entries)}"
        entry = FunctionType(function.__code__.replace(co_name=name), function.__globals__, name,
                             function.__defaults__, function.__closure__)
        entry.__kwdefaults__, entry.__module__ = dict(function.__kwdefaults__ or {}), function.__module__
        if self.sampled:
            entry.__kwdefaults__["repair_frame"] = self.repair_frame
            entry.__kwdefaults__["known_stochastic"] = self.known_stochastic
            if self.full_main:
                entry.__kwdefaults__["full"] = True
        bound = MethodType(entry, self.draft)
        if self.input_publication is None and self.state_publication is None:
            return torch.compile(bound, backend="hpu_backend", fullgraph=True, dynamic=False)
        publication = self.input_publication
        state_publication = self.state_publication

        def publish_inputs(*inputs):
            values = bound(*inputs)
            if state_publication is not None:
                state_publication(values)
            if publication is not None:
                publication(values[0], inputs[2])
            return values

        return torch.compile(publish_inputs, backend="hpu_backend", fullgraph=True, dynamic=False)

    @staticmethod
    def _validate(hidden, proposed, control, auxiliary, positions):
        if (hidden.ndim != 2 or hidden.shape[0] != 6 or auxiliary.ndim != 2 or auxiliary.shape[0] != 6
                or proposed.shape != (5,) or control.shape != (7,) or positions.shape != (6,)):
            raise ValueError("Native speculative control requires C6 hidden/auxiliary, C5 proposals and seven controls")
        tensors = (hidden, proposed, control, auxiliary, positions)
        if any(value.device != hidden.device or value.requires_grad or not value.is_contiguous() for value in tensors):
            raise ValueError("Native speculative control inputs must be contiguous matching inference tensors")

    @staticmethod
    def _outputs(outputs):
        # The common executor reserves output slot one for a residual.
        return outputs[0], *outputs[2:]

    def mutable_states(self):
        """Allow a derived DSpark plan to publish its extra device state."""
        states = tuple(layer.attention.swa for layer in self.draft.layers)
        if self.input_publication is not None:
            states += tuple(self.input_publication.buffers())
        if self.state_publication is not None:
            states += tuple(self.state_publication.buffers())
        return states

    def capture_snapshot(self):
        if self.repair_frame is not None:
            return self.repair_frame.capture_snapshot(self.states, exact_repair=self.full and not self.full_main)
        return _Snapshot(self.states)

    def forward(self, hidden, proposed, control, auxiliary, positions, *sampling):
        if self.closed:
            raise RuntimeError("A retired speculative control plan cannot be replayed")
        self._validate(hidden, proposed, control, auxiliary, positions)
        if self.sampled:
            if len(sampling) != 5:
                raise ValueError("Sampled native control requires actual q, parameters, seed, counter and offsets")
            q, parameters, seed, counter, offsets = sampling
            if (q.shape != (5, self.draft.output_head.weight.shape[0]) or parameters.shape != (11, 3)
                    or seed.shape != (1,) or counter.shape != (1,) or offsets.shape != (11,)):
                raise ValueError("Sampled native control input shapes differ from the request-owned protocol")
            if any(value.device != hidden.device or not value.is_contiguous() for value in sampling):
                raise ValueError("Sampled native control inputs must be contiguous on the sampling device")
        elif sampling:
            raise ValueError("Greedy native control does not consume sampling state")
        generation = self.generation() if callable(self.generation) else self.generation
        if self.fixed is not None and generation != self.captured_generation:
            invalidate_prepared_group_plans(owner=self, reason="draft_protocol_generation")
            self.compiled = self.fixed = self.metadata = None
        self.states = self.mutable_states()
        for name in self.debug_state_names:
            value = getattr(self.draft, name, None)
            if value is not None:
                self.states += (value,)
        if self.repair_frame is not None:
            self.states += self.repair_frame.protocol_states(exact_repair=self.full and not self.full_main)
            self.states = tuple({id(value): value for value in self.states}.values())
        if self.sampled and self.fixed is not None:
            self.states += (self.fixed[8],)
        metadata = SimpleNamespace(control=control, auxiliary=auxiliary, native_completion=None)
        if self.sampled:
            for name, value in zip(("proposal", "parameters", "seed", "counter", "offsets"), sampling, strict=True):
                setattr(metadata, name, value)
        roots = dict(hidden_states=hidden, input_ids=proposed, positions=positions, metadata=metadata,
                     state_generation=generation, state_tensors=self.states)
        outputs = replay_native_decoder(self, **roots)
        if outputs is not None:
            return self._outputs(outputs)
        if self.fixed is None:
            self.fixed = tuple(value.clone() for value in (hidden, proposed, control, auxiliary, positions, *sampling))
            self.compiled = self._new_entry()
            self.captured_generation = generation
        else:
            for destination, source in zip(self.fixed, (hidden, proposed, control, auxiliary, positions, *sampling),
                                           strict=True):
                if destination.shape != source.shape or destination.dtype != source.dtype:
                    raise RuntimeError("Prepare a new speculative control owner after an input contract change")
                destination.copy_(source)
        fixed_hidden, fixed_proposed, fixed_control, fixed_auxiliary, fixed_positions = self.fixed[:5]
        self.metadata = SimpleNamespace(control=fixed_control, auxiliary=fixed_auxiliary, native_completion=None)
        if self.sampled:
            self.states = (*self.states, self.fixed[8])
            self.states = tuple({id(value): value for value in self.states}.values())
            for name, value in zip(("proposal", "parameters", "seed", "counter", "offsets"), self.fixed[5:],
                                   strict=True):
                setattr(self.metadata, name, value)
        fixed_roots = dict(roots, hidden_states=fixed_hidden, input_ids=fixed_proposed, positions=fixed_positions,
                           metadata=self.metadata, state_tensors=self.states)
        with collect_prepared_group_replays(owner=self, adapter=self.adapter,
                                           snapshot=self.capture_snapshot, **fixed_roots) as context:
            context["group_index"] = 0
            values = self.compiled(*self.fixed)
            record_native_decoder_outputs(values[0], None, *values[1:])
        return values

    def close(self):
        if not self.closed:
            invalidate_prepared_group_plans(owner=self, reason="draft_protocol_close")
            self.compiled = self.fixed = self.metadata = None
            self.closed = True

    def prepare(self, *args):
        """Discover cold recipes, then capture the complete warmed control plan."""
        # The capture invokes publication too. Preserve its request cursor
        # and input staging allocations during cold discovery and warmup.
        states = tuple({id(value): value for value in (*self.states, *self.mutable_states())}.values())
        initial = (self.repair_frame.capture_snapshot(states, exact_repair=self.full and not self.full_main)
                   if self.repair_frame is not None else _Snapshot(states))
        try:
            # Cold recipe discovery may flush an incomplete prefix. As for
            # StageReplay, capture requires a subsequent complete invocation.
            for _ in range(2):
                initial.restore()
                self(*args)
                torch.hpu.synchronize()
            self.require_ready()
        finally:
            initial.restore()
            torch.hpu.synchronize()

    def prepare_repair(self, *args, journal_positions):
        """Initialize and capture the exact serving repair sequence together."""
        if not (self.sampled and self.full and not self.full_main and self.repair_frame is not None):
            raise ValueError("Repair preparation needs the saved sampled request frame")
        self.repair_frame.capture_inputs(*args)
        self.repair_frame.journal(journal_positions)
        # Surface a standalone output-binding defect at its real producer,
        # before subsequent clone/copy operations obscure the failing node.
        torch.hpu.synchronize()
        self.prepare(*args)

    def require_ready(self):
        """Reject an ordinary compiled fallback before any scored invocation."""
        from vllm_gaudi.ops.tp2_prepared_plan import _native_entries

        if self.closed or self not in _native_entries:
            raise RuntimeError("Speculative control did not capture its complete native plan")
        from vllm_gaudi import envs
        if self.sampled and envs.VLLM_HPU_DSV41_DSPARK_BATCH_INPUT_STAGING:
            graph = _native_entries[self][0]
            if not hasattr(graph, "enable_batched_input_staging"):
                raise RuntimeError("DSpark batch input staging requires its isolated capability bridge")
            graph.enable_batched_input_staging()
