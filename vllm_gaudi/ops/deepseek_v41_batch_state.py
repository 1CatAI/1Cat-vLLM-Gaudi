# SPDX-License-Identifier: Apache-2.0
"""Fixed-address request-slot state for multi-request V4.1 execution."""

from functools import lru_cache

import torch

from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots


def _decode_rows(packed, rows, *, width, group):
    from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4
    if packed.device.type == "hpu":
        from vllm_gaudi.ops.deepseek_v41_batch_attention import read_state_rows
        values = read_state_rows(packed, rows, torch.zeros_like(rows))
    else:
        values = packed.index_select(0, rows.clamp_min(0).long())
        values = values.masked_fill((rows < 0)[:, None], 0)
    return unpack_fp4(values, width, group)


@lru_cache(maxsize=2)
def _decoded_row_reader(width, group):

    def read(packed, rows):
        return _decode_rows(packed, rows, width=width, group=group)

    return torch.compile(read, backend="hpu_backend", fullgraph=True, dynamic=False)


class BatchLayerState(torch.nn.Module):

    def __init__(self, capacity, device, *, compressor):
        super().__init__()
        self.register_buffer("swa", torch.zeros(capacity * 256, 528, dtype=torch.uint8, device=device), False)
        if compressor:
            self.register_buffer("kv_history", torch.zeros(capacity * 8, 512, dtype=torch.float32, device=device),
                                 False)
            self.register_buffer("score_history", torch.zeros_like(self.kv_history), False)

    def clear_slot(self, slot):
        self.swa[slot * 256:(slot + 1) * 256].zero_()
        if hasattr(self, "kv_history"):
            self.kv_history[slot * 8:(slot + 1) * 8].zero_()
            self.score_history[slot * 8:(slot + 1) * 8].zero_()


class BatchStageState:
    """One persistent bank per stage; scheduler pages still own compressed KV.

    An in-flight batch owns its candidate scratch until its completion event.
    This initial interface permits one batch per stage. A second PP microbatch
    needs another scratch owner, not a second copy of all request histories.
    """

    def __init__(self, program, capacity):
        self.program, self.capacity = program, capacity
        self.slots = RequestSlots(capacity)
        shared = program.shared
        device = shared.block_table.device
        self.layers = {}
        # Native B1 replay binds these original addresses. The same request
        # slots remain authoritative whenever the scheduler selects B2+.
        self.single_bindings = [(shared, "block_table", shared.block_table)]
        for layer in program.layers:
            attention = layer.attention
            for name in ("swa", "kv_history", "score_history"):
                if hasattr(attention, name):
                    self.single_bindings.append((attention, name, getattr(attention, name)))
            state = BatchLayerState(capacity, device, compressor=attention.owns_kv and attention.ratio == 2)
            attention.batch_state = state
            if not attention.ratio:
                attention.register_buffer("batch_main_unused", torch.zeros(1, 288, dtype=torch.uint8, device=device),
                                          False)
            self.layers[layer.layer] = state
        self.pages = torch.zeros(capacity, shared.block_table.numel(), dtype=torch.int32, device=device)
        self.page_host = torch.zeros_like(self.pages, device="cpu").pin_memory("hpu")
        self.page_versions = {}
        self.indices = {name: torch.full((capacity, 512), -1, dtype=torch.int32, device=device) for name in shared.topk}
        self.candidates = torch.full((capacity, 2048), -1, dtype=torch.int32, device=device)
        self.pending = None
        self.generation = 0
        self.single_owner = None
        self.single_page_version = None

    def acquire(self, request_id):
        existed = request_id in self.slots.owners
        slot = self.slots.acquire(request_id)
        if not existed:
            for state in self.layers.values():
                state.clear_slot(slot.index)
            self.pages[slot.index].zero_()
            self.page_host[slot.index].zero_()
            self.page_versions.pop(slot.index, None)
        return slot

    def publish_pages(self, slot, block_ids, pool_blocks):
        self.slots.validate(slot)
        values = tuple(block_ids)
        if not values or len(values) > self.pages.shape[1] or any(not 0 < block < pool_blocks for block in values):
            raise ValueError("Batch request page table exceeds scheduler allocation")
        identity = (slot.generation, values)
        if self.page_versions.get(slot.index) == identity:
            return
        if self.pending is not None:
            raise RuntimeError("Cannot replace batch page metadata while a consumer is in flight")
        self.page_host[slot.index].zero_()
        self.page_host[slot.index, :len(values)] = torch.tensor(values, dtype=torch.int32)
        self.pages[slot.index].copy_(self.page_host[slot.index], non_blocking=True)
        self.page_versions[slot.index] = identity

    def bind_prefill(self, slot):
        """Alias the selected slot; never save/restore a whole working set."""
        self.slots.validate(slot)
        if self.pending is not None:
            raise RuntimeError("Prefill cannot replace the owner of in-flight batch scratch")
        self.leave_single()
        self.program.shared.block_table = self.pages[slot.index]
        for layer in self.program.layers:
            state = self.layers[layer.layer]
            layer.attention.swa = state.swa[slot.index * 256:(slot.index + 1) * 256]
            if hasattr(state, "kv_history"):
                layer.attention.kv_history = state.kv_history[slot.index * 8:(slot.index + 1) * 8]
                layer.attention.score_history = state.score_history[slot.index * 8:(slot.index + 1) * 8]

    def restore_single_bindings(self):
        for module, name, value in self.single_bindings:
            setattr(module, name, value)

    def _single_rows(self, slot):
        for layer in self.program.layers:
            state = self.layers[layer.layer]
            yield layer.attention.swa, state.swa[slot.index * 256:(slot.index + 1) * 256]
            for name in ("kv_history", "score_history"):
                if hasattr(state, name):
                    yield getattr(layer.attention, name), getattr(state, name)[slot.index * 8:(slot.index + 1) * 8]

    def leave_single(self):
        """Publish a completed B1 owner before another path consumes its slot."""
        if self.single_owner is None:
            return
        self.slots.validate(self.single_owner)
        # V2 consumes its token completion before changing scheduler owners.
        # This transition also retires all native state writers before copying
        # their small packed working set; continuous B1 steps never wait here.
        if self.pages.device.type == "hpu":
            torch.hpu.synchronize()
        for working, stored in self._single_rows(self.single_owner):
            stored.copy_(working)
        self.single_owner = None
        self.single_page_version = None

    def bind_single(self, slot, position):
        """Select native B1 with the same packed KV and slot lifetime as B2+."""
        self.slots.validate(slot)
        if self.pending is not None:
            raise RuntimeError("B1 cannot replace in-flight batch scratch")
        changed_owner = self.single_owner != slot
        if changed_owner:
            self.leave_single()
            self.restore_single_bindings()
            for working, stored in self._single_rows(slot):
                working.copy_(stored)
            self._restore_decoded(slot, position)
            self.single_owner = slot
        version = self.page_versions.get(slot.index)
        if version is None:
            raise RuntimeError("B1 requires the current scheduler page mapping")
        if self.single_page_version != (slot, version):
            self.program.shared.block_table.copy_(self.pages[slot.index])
            self.single_page_version = slot, version
        if changed_owner and self.pages.device.type == "hpu":
            # Finish the handoff before native replay consumes the reconstructed
            # mirrors. Continuous B1 retains the owner and has no fence here.
            torch.hpu.synchronize()

    def _restore_decoded(self, slot, position):
        """Rebuild bounded mirrors from canonical packed state on transitions.

        Mirrors are scratch for one selected B1 owner, never a second copy of
        every request's long context. B2+ continues to update scheduler pages.
        """
        from vllm_gaudi.ops.deepseek_v41_indexer import INDEX_MME_HOT_TOKENS
        from vllm_gaudi.ops.deepseek_v41_math import unpack_swa
        shared = self.program.shared
        if hasattr(shared, "decoded_swa"):
            shared.decoded_swa.zero_()
            for layer in self.program.layers:
                offset = layer.attention.decoded_swa_offset
                shared.decoded_swa[offset:offset + 256].copy_(unpack_swa(layer.attention.swa))
        # Selection is recomputed by its owning layer before reuse layers.
        for selection in shared.topk.values():
            selection.indices.fill_(-1)
        if shared.candidate_pool is not None:
            shared.candidate_pool.fill_(-1)
        block_ids = self.page_versions[slot.index][1]
        for cache in shared.sources.values():
            for name, packed, width, group in (("decoded_main", cache.main, 512, 16), ("decoded_index_hot", cache.index,
                                                                                       128, 32)):
                destination = getattr(cache, name, None)
                if destination is None:
                    continue
                destination.zero_()
                if position > INDEX_MME_HOT_TOKENS:
                    continue
                count = min(position // cache.ratio, destination.shape[0])
                page_rows = 128 // cache.ratio
                # A fixed tile bounds compiler work and avoids a new graph for
                # each surviving request's arbitrary context length.
                for start in range(0, count, 64):
                    end = min(start + 64, destination.shape[0])
                    logical = range(start, end)
                    physical = [
                        block_ids[row // page_rows] * page_rows + row % page_rows if row < count else -1
                        for row in logical
                    ]
                    rows = torch.tensor(physical, dtype=torch.int32, device=self.pages.device)
                    read = (_decoded_row_reader(width, group) if self.pages.device.type == "hpu" else lambda packed,
                            rows, width=width, group=group: _decode_rows(packed, rows, width=width, group=group))
                    destination[start:end].copy_(read(packed, rows))

    def begin(self, slots):
        if self.pending is not None:
            raise RuntimeError("Batch scratch still has a previous consumer")
        self.leave_single()
        slots = tuple(slots)
        if len({slot.index for slot in slots}) != len(slots):
            raise ValueError("A batch cannot alias writable request slots")
        for slot in slots:
            self.slots.validate(slot)
        self.generation += 1
        self.pending = (self.generation, slots)
        return self.generation

    def finish(self, generation, consumer_done):
        if self.pending is None or self.pending[0] != generation:
            raise RuntimeError("Batch completion belongs to another generation")
        # The owner must finish before another batch overwrites candidate
        # scratch or pinned page staging. This is a stage-consumer event.
        consumer_done.synchronize()
        for slot in self.pending[1]:
            self.slots.submitted(slot, consumer_done)
        self.pending = None

    def release(self, request_id):
        slot = self.slots.owners.get(request_id)
        if slot is not None:
            if self.single_owner == slot:
                self.leave_single()
            self.slots.release(slot)

    @property
    def allocated_bytes(self):
        arrays = [self.pages, self.candidates, *self.indices.values()]
        arrays.extend(value for state in self.layers.values() for value in state.buffers())
        return sum(value.numel() * value.element_size() for value in arrays)
