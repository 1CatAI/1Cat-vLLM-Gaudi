# SPDX-License-Identifier: Apache-2.0
"""Native C5 producer on the common executor; official sampling stays outside."""

from types import FunctionType, MethodType

import torch

from vllm_gaudi.ops.deepseek_v41_draft_replay import NativeDraftProtocol, _entries
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology


def _body(draft, hidden, proposed, control, auxiliary, positions):
    return draft.forward_local(proposed, positions)


class NativeDraftBody(NativeDraftProtocol):
    """Reuse state/input ownership without capturing a large vocab collective.

    C5 embedding plus three Attention/FFN reduction pairs use seven existing
    peer exchanges. This is TP-parametric and changes no transport or C1 plan.
    The returned local logits feed the unchanged official Markov p/q sampler.
    """

    def __init__(self, draft, *, generation):
        super().__init__(draft, generation=generation)
        self.adapter = DecoderTopology("deepseek_v41_dspark_c5_body", (1,), 0, False, 7)
        self.debug_state_names = ("sampling_normalized_debug", "sampling_base_logits_debug")

    def _new_entry(self):
        name = f"native_c5_body_{next(_entries)}"
        entry = FunctionType(_body.__code__.replace(co_name=name), _body.__globals__, name)
        from vllm_gaudi.compilation.deepseek_v41_frontend_cache import cached_tensor_entry

        cached = cached_tensor_entry(entry, self.fixed, {}, owner=self.draft)
        if cached is not None:
            return cached
        return torch.compile(MethodType(entry, self.draft), backend="hpu_backend", fullgraph=True, dynamic=False)

    @staticmethod
    def _validate(hidden, proposed, control, auxiliary, positions):
        if proposed.shape != (1,) or positions.shape != (5,):
            raise ValueError("Native draft body requires one actual anchor and five device positions")
        if any(value.device != proposed.device or not value.is_contiguous() or value.requires_grad
               for value in (hidden, proposed, control, auxiliary, positions)):
            raise ValueError("Draft body inputs must be contiguous matching inference tensors")

    def forward(self, first_token, positions):
        # Unused protocol roots retain the common ownership interface. Only
        # the anchor and positions used by the recipes receive fixed bindings.
        return super().forward(first_token, first_token, first_token, first_token, positions)
