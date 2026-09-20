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
    paged: bool = False

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
        if self.paged:
            return math.ceil(vllm_config.model_config.max_model_len / self.block_size) * self.page_size_bytes
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
        mutable = {
            "swa", "main", "decoded_swa", "decoded_main", "decoded_index_hot", "index", "indices",
            "candidate_pool", "kv_history", "score_history"
        }
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
        self.allocations = {
            name: torch.zeros((blocks, *spec.state_shape), dtype=spec.state_dtype, device=device)
            for name, spec in self.specs.items()
        }
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


class PagedStageState:
    """Shared compressed KV pool; fixed-address working state survives replay.

Only the small SWA/compressor working set is saved when scheduling another
request. Compressed history stays in its scheduler-owned HPU pages.
"""

    def __init__(self, program):
        from vllm_gaudi.ops.deepseek_v41_paged_attention import PAGE_TOKENS
        self.program = program
        self.bindings, self.specs, self.allocations = {}, {}, {}
        for source, cache in program.shared.sources.items():
            for name, width in (("main", 288), ("index", 68)):
                key = f"dsv41.pp{program.pp_rank}.source{source}.{name}"
                self.bindings[key] = cache, name
                self.specs[key] = V41StateSpec(block_size=PAGE_TOKENS,
                                               state_shape=(PAGE_TOKENS // cache.ratio, width),
                                               state_dtype=torch.uint8,
                                               paged=True)
        self.working = {
            name: value
            for name, value in program.named_buffers()
            if name.rsplit(".", 1)[-1] in ("swa", "decoded_swa", "decoded_main", "decoded_index_hot",
                                           "kv_history", "score_history", "indices", "candidate_pool")
        }
        self.saved, self.active, self.blocks = {}, None, 2
        # The scheduler page table is small but long-lived.  Rebuilding a
        # pageable CPU tensor and copying it on every C1 transaction forces
        # the bridge to allocate an implicit staging buffer while captured
        # recipe buffers are live.  Apart from wasting one H2D submission per
        # token, that can enter Synapse defragmentation with buffers in use.
        # Keep one explicitly pinned source and only publish changed tables.
        self.block_table_host = torch.empty(program.shared.block_table.shape,
                                            dtype=torch.int32,
                                            device="cpu").pin_memory("hpu")
        self.block_table_host_values = self.block_table_host.numpy()
        self.published_block_ids = None
        self.identity_block_table_resident = False

    def allocate(self, blocks, device):
        if blocks < 2:
            raise ValueError("Paged V4.1 state requires a null page and at least one live page")
        if self.program.replay_owner is not None:
            self.program.replay_owner.close()
        self.allocations = {
            key: torch.zeros((blocks, *spec.state_shape), dtype=spec.state_dtype, device=device)
            for key, spec in self.specs.items()
        }
        self.blocks = blocks
        self.bind(0)

    def bind(self, block):
        # Pool replacement is confined to initialization, before recipe capture.
        del block
        for key, (module, name) in self.bindings.items():
            setattr(module, name, self.allocations[key].flatten(0, 1))
        # With max_num_seqs=1 the scheduler hands the sole request the first
        # N physical pages.  Install that complete mapping before any recipe
        # captures the destination.  Every growing prefix can then be consumed
        # without mutating a graph-owned tensor at request time.
        values = self.block_table_host_values
        values[:] = range(1, len(values) + 1)
        self.program.shared.block_table.copy_(self.block_table_host, non_blocking=True)
        self.published_block_ids = None
        self.identity_block_table_resident = True
        self.program.generation += 1

    def _publish_block_table(self, block_ids):
        block_ids = tuple(block_ids)
        if block_ids == self.published_block_ids:
            return
        identity = all(block == index for index, block in enumerate(block_ids, 1))
        if identity and self.identity_block_table_resident:
            self.published_block_ids = block_ids
            return
        values = self.block_table_host_values
        if identity:
            values[:] = range(1, len(values) + 1)
        else:
            values.fill(0)
            values[:len(block_ids)] = block_ids
        # Pinned source plus a persistent destination gives Synapse a true
        # asynchronous DMA and requires no per-request device allocation.  A
        # non-identity remap is rare with single-request serving; wait for the
        # prior captured consumer before mutating its persistent input.
        if self.program.shared.block_table.device.type == "hpu":
            torch.hpu.synchronize()
        self.program.shared.block_table.copy_(self.block_table_host, non_blocking=True)
        self.published_block_ids = block_ids
        self.identity_block_table_resident = identity

    def activate(self, request_id, block_ids, *, reset=False):
        if not block_ids or any(not 0 < block < self.blocks for block in block_ids):
            raise RuntimeError("V4.1 request has invalid scheduler-owned compressed KV pages")
        if len(block_ids) > self.program.shared.block_table.numel():
            raise RuntimeError("V4.1 request exceeds the configured context page table")
        if self.active != request_id:
            if self.active is not None:
                if self.active not in self.saved:
                    self.saved[self.active] = {key: value.clone() for key, value in self.working.items()}
                else:
                    for key, value in self.working.items():
                        self.saved[self.active][key].copy_(value)
            if request_id in self.saved and not reset:
                for key, value in self.working.items():
                    value.copy_(self.saved[request_id][key])
            else:
                self.clear()
            self.active = request_id
        elif reset:
            self.clear()
        self._publish_block_table(block_ids)

    def release(self, request_id):
        self.saved.pop(request_id, None)
        if self.active == request_id:
            self.active = None

    def clear(self):
        for name, value in self.working.items():
            value.fill_(-1 if name.rsplit(".", 1)[-1] in ("indices", "candidate_pool") else 0)

    @property
    def allocated_bytes(self):
        return (sum(value.numel() * value.element_size()
                    for value in self.allocations.values()) + sum(value.numel() * value.element_size()
                                                                  for state in self.saved.values()
                                                                  for value in state.values()))
