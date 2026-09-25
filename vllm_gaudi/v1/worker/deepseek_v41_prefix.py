# SPDX-License-Identifier: Apache-2.0
"""Auxiliary prefix checkpoints in the ordinary request-slot worker path."""

import torch

from vllm_gaudi.ops.deepseek_v41_prefix_state import BatchPrefixStore, PrefixStateTicket


def split_at_checkpoint(chunks, start, boundary):
    result = []
    for offset, values in chunks:
        cut = boundary - start - offset
        if 0 < cut < len(values):
            result.extend(((offset, values[:cut]), (offset + cut, values[cut:])))
        else:
            result.append((offset, values))
    return result


class PrefixCheckpoints:
    """Copy only at an exact committed boundary; never replay the cached tokens."""

    def __init__(self, runner):
        self.runner = runner
        self.store = BatchPrefixStore(min(128, 2 * runner.vllm_config.scheduler_config.max_num_seqs))
        self.histories = {}
        self.operations = None
        self.captured = set()

    @staticmethod
    def _done():
        event = torch.hpu.Event()
        event.record()
        event.synchronize()
        return event

    def begin(self, operations):
        if self.operations is not None:
            raise RuntimeError("Previous auxiliary prefix transaction has not retired")
        self.operations = operations
        self.captured = set()
        if operations is None:
            return
        runner, bank = self.runner, self.runner.model.batch_state
        # Prefix admission is outside steady decode. Drain the actual native
        # streams before touching a recycled slot or starting a source copy.
        runner.pp.drain()
        torch.hpu.synchronize()
        for request_id, descriptor in operations.restores.items():
            request = runner.requests[request_id]
            if request.num_computed_tokens != descriptor.num_tokens or len(descriptor.block_ids) != 1:
                raise RuntimeError("Auxiliary restore differs from scheduler admission")
            if tuple(request.block_ids[0][:descriptor.num_tokens // 128]) != descriptor.block_ids[0]:
                raise RuntimeError("Auxiliary restore changed its scheduler-owned pages")
            ticket = PrefixStateTicket(descriptor.slot, descriptor.generation)
            history = self.histories.get(ticket)
            if runner.model.engram_host is not None and (history is None or history[0] != descriptor.num_tokens):
                raise RuntimeError("Auxiliary restore lacks matching committed Engram history")
            slot = bank.acquire(request_id)
            bank.publish_pages(slot, request.block_ids[0], runner.state.blocks)
            done = self.store.restore(ticket,
                                      bank,
                                      slot,
                                      descriptor.num_tokens,
                                      descriptor.block_hash,
                                      self._done(),
                                      record_done=self._done)
            done.synchronize()
            if runner.model.engram_host is not None:
                runner.model.engram_host.restore_checkpoint(request_id, history)
            runner.audit["prefix_restores"] = runner.audit.get("prefix_restores", 0) + 1

    def chunks(self, request_id, start, chunks):
        descriptor = self.operations.captures.get(request_id) if self.operations is not None else None
        return chunks if descriptor is None else split_at_checkpoint(chunks, start, descriptor.num_tokens)

    def capture_at(self, request_id, position):
        if self.operations is None or request_id in self.captured:
            return
        descriptor = self.operations.captures.get(request_id)
        if descriptor is None or descriptor.num_tokens != position:
            return
        runner, bank = self.runner, self.runner.model.batch_state
        slot = bank.slots.owners[request_id]
        if tuple(runner.requests[request_id].block_ids[0][:position // 128]) != descriptor.block_ids[0]:
            raise RuntimeError("Auxiliary capture differs from scheduler-owned pages")
        runner.pp.drain()
        torch.hpu.synchronize()
        host = runner.model.engram_host
        history = host.snapshot_prefix(request_id) if host is not None else None
        if history is not None and history[0] != position:
            raise RuntimeError("Engram history did not commit the exact checkpoint boundary")
        ticket = PrefixStateTicket(descriptor.slot, descriptor.generation)
        self.store.capture(ticket, bank, slot, position, descriptor.block_hash, self._done(), record_done=self._done)
        self.store.publish(ticket)
        self.histories = {key: value for key, value in self.histories.items() if key.index != ticket.index}
        if history is not None:
            self.histories[ticket] = history
        self.captured.add(request_id)
        runner.audit["prefix_captures"] = runner.audit.get("prefix_captures", 0) + 1

    def finish(self):
        operations = self.operations
        if operations is None:
            return []
        if self.captured != operations.captures.keys():
            raise RuntimeError("Scheduled auxiliary checkpoint boundary was not executed")
        from vllm.distributed import get_world_group
        from vllm.v1.core.auxiliary_prefix_cache import AuxiliaryPrefixAck
        group = get_world_group()
        local = [AuxiliaryPrefixAck(descriptor.ticket, group.rank) for descriptor in operations.captures.values()]
        gathered = [None] * group.world_size
        # One host collective per checkpoint transaction, never per decode
        # token. PP1/TP0 returns the all-rank record in normal worker output.
        torch.distributed.all_gather_object(gathered, local, group=group.cpu_group)
        self.operations = None
        return [ack for rank in gathered for ack in rank]

    def close(self):
        self.store.close()
        self.histories.clear()
