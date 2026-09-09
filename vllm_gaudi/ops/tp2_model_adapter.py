# SPDX-License-Identifier: Apache-2.0
"""Model contracts for the common prepared decoder executor."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderTopology:
    name: str
    group_layers: tuple[int, ...]
    reductions_per_layer: int
    external_prefix: bool

    @property
    def groups(self):
        return len(self.group_layers)

    @property
    def reductions(self):
        return sum(self.group_layers) * self.reductions_per_layer


QWEN3_NEXT = DecoderTopology("qwen3_next", (8,) * 8, 2, True)
DEEPSEEK_V4 = DecoderTopology("deepseek_v4", (8, 8, 8, 8, 8, 3), 2, False)
