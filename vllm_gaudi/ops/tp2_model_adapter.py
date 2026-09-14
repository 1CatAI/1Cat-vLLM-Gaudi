# SPDX-License-Identifier: Apache-2.0
"""Model contracts for the common prepared decoder executor."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderTopology:
    name: str
    group_layers: tuple[int, ...]
    reductions_per_layer: int
    external_prefix: bool
    extra_collectives: int = 0

    @property
    def groups(self):
        return len(self.group_layers)

    @property
    def reductions(self):
        return sum(self.group_layers) * self.reductions_per_layer

    @property
    def collectives(self):
        return self.reductions + self.extra_collectives


QWEN3_NEXT = DecoderTopology("qwen3_next", (8,) * 8, 2, True)
DEEPSEEK_V4 = DecoderTopology("deepseek_v4", (8, 8, 8, 8, 8, 3), 2, False)
DEEPSEEK_V41_PP0 = DecoderTopology("deepseek_v41_pp0", (4,) * 5, 2, False, 2)
DEEPSEEK_V41_PP0_INPUT = DecoderTopology("deepseek_v41_pp0_input", (4,) * 5, 2, False, 3)
DEEPSEEK_V41_PP1 = DecoderTopology("deepseek_v41_pp1", (4,) * 5, 2, False)
