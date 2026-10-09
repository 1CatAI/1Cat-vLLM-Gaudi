# SPDX-License-Identifier: Apache-2.0
"""Request-owned sampling state writes inside the native DSpark tail."""
import torch


class SamplingStatePublication(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.register_buffer("proposal", state.proposal, persistent=False)
        self.register_buffer("counter", state.counter, persistent=False)
        self.register_buffer("valid", state.proposal_valid, persistent=False)
        if (self.proposal.ndim != 2 or self.proposal.shape[0] != 5
                or self.proposal.dtype != torch.float32 or self.counter.shape != (1,)
                or self.counter.dtype != torch.int32 or self.valid.shape != (1,)
                or self.valid.dtype != torch.bool):
            raise ValueError("Sampling publication requires the request's original probability/RNG allocations")

    def forward(self, values):
        from vllm_gaudi.ops.deepseek_v41_verify import STATUS

        record, _, proposal, _, counter = values
        self.proposal.copy_(proposal)
        self.counter.copy_(counter)
        self.valid.copy_(record[STATUS:STATUS + 1] == 0)

