# SPDX-License-Identifier: Apache-2.0
"""DSpark sampled continuation through the first real next-input consumer.

A capability plan, not a serving default. Exact full p/q, RNG advancement,
request state publication and cursor advance share one native submission.
The shared replay and peer implementations are unchanged.
"""
from types import FunctionType, MethodType

import torch

from vllm_gaudi.ops.deepseek_v41_draft_replay import NativeDraftProtocol, _entries
from vllm_gaudi.ops.deepseek_v41_round_inputs import advance_round
from vllm_gaudi.ops.deepseek_v41_verify import STATUS
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology


class NativeSampledRoundProtocol(NativeDraftProtocol):
    """Same sampled protocol, with device state and next embedding consumer."""

    def __init__(self, draft, embedding, *, generation, histories):
        super().__init__(draft, generation=generation, sampled=True, full=True, full_main=True)
        if histories.shape != (7, 3) or histories.dtype != torch.int32:
            raise ValueError("Continuation needs the actual seven committed-prefix histories")
        self.embedding = embedding
        self.register_buffer("round_histories", histories, False)
        device = histories.device
        self.register_buffer("round_ids", torch.empty(6, dtype=torch.int32, device=device), False)
        self.register_buffer("round_positions", torch.empty(6, dtype=torch.int32, device=device), False)
        self.register_buffer("round_control", torch.empty(12, dtype=torch.int64, device=device), False)
        self.register_buffer("round_history", torch.empty(3, dtype=torch.int32, device=device), False)
        self.register_buffer("round_proposal", torch.empty((5, draft.output_head.weight.shape[0]),
                                                          dtype=torch.float32, device=device), False)
        self.register_buffer("round_counter", torch.empty(1, dtype=torch.int64, device=device), False)
        self.register_buffer("round_valid", torch.empty(1, dtype=torch.int32, device=device), False)
        # Same full protocol plus the actual next-input TP embedding reduction.
        self.adapter = DecoderTopology("deepseek_v41_dspark_round_input", (1,), 0, False,
                                       self.adapter.collectives + 1)
        self.states = self.mutable_states()

    def mutable_states(self):
        return (*super().mutable_states(), self.round_histories, self.round_ids, self.round_positions,
                self.round_control, self.round_history, self.round_proposal, self.round_counter, self.round_valid)

    def _combined_forward(self, hidden, proposed, control, auxiliary, positions, q, parameters, seed, counter, offsets):
        values = self.draft.verify_and_propose_sampled_bound(
            hidden, proposed, control, auxiliary, positions, q, parameters, seed, counter, offsets, full=True)
        record, _, probability, _, advanced_counter = values
        self.round_proposal.copy_(probability)
        self.round_counter.copy_(advanced_counter)
        self.round_valid.copy_((record[STATUS:STATUS + 1] == 0).to(torch.int32))
        advance_round(record, control, self.round_histories, self.round_ids, self.round_positions,
                      self.round_control, self.round_history)
        # Ending at the real consumer prevents moving an unmeasured wait into
        # the following Target from appearing as a continuation speedup.
        next_embedding = self.embedding(self.round_ids.long())
        return (*values, *next_embedding, self.round_ids, self.round_positions, self.round_control,
                self.round_history, self.round_proposal, self.round_counter, self.round_valid)

    def _new_entry(self):
        function = self._combined_forward.__func__
        name = f"{function.__name__}_native_round_{next(_entries)}"
        entry = FunctionType(function.__code__.replace(co_name=name), function.__globals__, name,
                             function.__defaults__, function.__closure__)
        entry.__module__ = function.__module__
        return torch.compile(MethodType(entry, self), backend="hpu_backend", fullgraph=True, dynamic=False)
