# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned blocks for the bounded CSA2/DSpark stage state.

Each 512-token request owns one complete state block. Arrays retain their
actual dtype/shape and separate allocations, including compressor scratch
and candidate slots; they are not represented as BF16 attention vectors.
"""

from dataclasses import dataclass
import math

import torch

from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry


@dataclass(frozen=True, kw_only=True)
class V41StateSpec(KVCacheSpec):
    state_shape: tuple[int, ...]
    state_dtype: torch.dtype

    @property
    def prefix_cacheable(self):
        return False

    @property
    def num_heads(self):
        return 1

    @property
    def tokens_per_state(self):
        return self.block_size

    @property
    def state_content_size_bytes(self):
        return math.prod(self.state_shape) * self.state_dtype.itemsize

    @property
    def page_size_bytes(self):
        return self.state_content_size_bytes

    def max_memory_usage_bytes(self, vllm_config):
        if self.block_size != 512 or vllm_config.model_config.max_model_len > self.block_size:
            raise ValueError("V4.1 bounded state requires a complete 512-token request block")
        return self.page_size_bytes

    def copy_with_new_block_size(self, block_size):
        if block_size != self.block_size:
            raise ValueError("V4.1 bounded state cannot be split into token pages")
        return self


def register_state_spec(vllm_config=None):
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    KVCacheSpecRegistry._ensure_registered(vllm_config)
    KVCacheSpecRegistry.register(V41StateSpec, FullAttentionManager)


class StageStateBlocks:
    def __init__(self, program):
        self.program = program
        mutable = {"swa", "main", "index", "indices", "candidate_pool", "kv_history", "score_history"}
        self.bindings, self.specs, self.allocations = {}, {}, {}
        for module_name, module in program.named_modules():
            for name, value in module.named_buffers(recurse=False):
                if name not in mutable:
                    continue
                key = f"dsv41.pp{program.pp_rank}.{module_name}.{name}"
                self.bindings[key] = module, name
                self.specs[key] = V41StateSpec(block_size=512, state_shape=tuple(value.shape), state_dtype=value.dtype)
        self.active, self.blocks = None, 0

    def allocate(self, blocks, device):
        if blocks < 1:
            raise ValueError("V4.1 state allocator requires at least one whole request block")
        if self.program.replay_owner is not None:
            self.program.replay_owner.close()
        self.allocations = {name: torch.zeros((blocks, *spec.state_shape), dtype=spec.state_dtype, device=device)
                            for name, spec in self.specs.items()}
        self.blocks, self.active = blocks, None
        self.bind(0)

    def bind(self, block):
        if not 0 <= block < self.blocks:
            raise ValueError("Scheduler block ID exceeds allocated V4.1 state capacity")
        if block == self.active:
            return
        if self.program.replay_owner is not None:
            self.program.replay_owner.close()
        for key, (module, name) in self.bindings.items():
            setattr(module, name, self.allocations[key][block])
        self.active = block
        self.program.generation += 1

    def clear(self):
        if self.active is None:
            raise RuntimeError("Cannot reset V4.1 state before binding a request block")
        for module, name in self.bindings.values():
            value = getattr(module, name)
            value.fill_(-1 if name in ("indices", "candidate_pool") else 0)

    @property
    def allocated_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.allocations.values())
