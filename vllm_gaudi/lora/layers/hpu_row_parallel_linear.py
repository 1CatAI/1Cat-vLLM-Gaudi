# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU-specific LoRA layer for RowParallelLinear.

This module registers HPU-aware LoRA wrappers that can handle 
HPURowParallelLinear instances created via OOT registration.
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.lora.layers.row_parallel_linear import (
    RowParallelLinearWithLoRA,
    RowParallelLinearWithShardedLoRA,
)
from vllm.lora.layers.utils import (
    _fully_sharded_can_replace,
    _not_fully_sharded_can_replace,
)
from vllm.lora import utils as lora_utils
from vllm.model_executor.layers.linear import RowParallelLinear


class HPURowParallelLinearWithLoRA(RowParallelLinearWithLoRA):
    """LoRA wrapper that can handle HPURowParallelLinear instances."""

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        # Accept both RowParallelLinear and HPURowParallelLinear
        return isinstance(source_layer, RowParallelLinear)


class HPURowParallelLinearWithShardedLoRA(RowParallelLinearWithShardedLoRA):
    """Sharded LoRA wrapper that can handle HPURowParallelLinear instances."""

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        # Use isinstance to accept HPURowParallelLinear (subclass of RowParallelLinear)
        return isinstance(source_layer, RowParallelLinear)


def register_hpu_lora_layers():
    """Replace upstream wrappers before LoRA model creation, preserving priority."""
    replacements = {
        RowParallelLinearWithLoRA: HPURowParallelLinearWithLoRA,
        RowParallelLinearWithShardedLoRA: HPURowParallelLinearWithShardedLoRA,
    }
    # Other HPU wrappers may already have converted the registry to a tuple.
    registry = lora_utils._all_lora_classes
    updated = list(dict.fromkeys(replacements.get(cls, cls) for cls in registry))
    for replacement in replacements.values():
        if replacement not in updated:
            updated.append(replacement)
    lora_utils._all_lora_classes = type(registry)(updated)
