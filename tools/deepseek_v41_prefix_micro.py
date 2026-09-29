# SPDX-License-Identifier: Apache-2.0
"""Checkpoint restoration followed by the real compiled TP decoder consumer."""

from types import SimpleNamespace

import torch

from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
from vllm_gaudi.ops.deepseek_v41_prefix_state import BatchPrefixStore, PrefixStateTicket


def check_prefix_consumers(native, inputs, layers, buffers, restore, capacity, rank):
    hidden, _, _, _, _, slots, pages, _, _, _, _ = inputs
    source_indices = slots.cpu().tolist()
    target_indices = [i for i in range(capacity) if i not in source_indices][:len(source_indices)]
    page_ids = pages.cpu().tolist()
    bank = SimpleNamespace(capacity=capacity,
                           pending=None,
                           slots=RequestSlots(capacity),
                           page_versions={},
                           layers={layer.layer: layer.attention.batch_state
                                   for layer in layers},
                           program=SimpleNamespace(pp_rank=layers[0].layer // 20,
                                                   tp_rank=rank,
                                                   generation=1,
                                                   precision_fingerprint="unchanged-prepared-micro-weights"))
    owners = [bank.slots.acquire(f"fixture-{i}") for i in range(capacity)]
    for row, (source, target) in enumerate(zip(source_indices, target_indices, strict=True)):
        for index in (source, target):
            bank.page_versions[index] = owners[index].generation, tuple(page_ids[row][:16])

    def record():
        done = torch.hpu.Event()
        done.record()
        return done

    def drained():
        torch.hpu.synchronize()
        return record()

    def values(indices):
        return [
            value[index * rows:(index + 1) * rows].cpu().clone() for index in indices for state in bank.layers.values()
            for name, value in state.named_buffers() for rows in (256 if name == "swa" else 8, )
        ]

    for _ in range(3):
        restore()
        native(*inputs)
        torch.hpu.synchronize()
    native.require_ready()
    restore()
    producer = drained()
    producer.synchronize()
    store = BatchPrefixStore(len(source_indices))
    tickets = [PrefixStateTicket(row, 1) for row in range(len(source_indices))]
    for row, source in enumerate(source_indices):
        store.capture(tickets[row], bank, owners[source], 1920, f"fixture-{row}".encode(), producer, record_done=record)
    torch.hpu.synchronize()
    for ticket in tickets:
        store.publish(ticket)
    checks = []
    for change in range(3):
        if change:
            hidden.copy_(torch.randn_like(hidden))
        restore()
        slots.copy_(torch.tensor(source_indices, dtype=torch.int32))
        reference = tuple(value.cpu().clone() for value in native(*inputs))
        expected = values(source_indices)
        expected_cache = [value.cpu().clone() for value in buffers[-2:]]
        restore()
        # The original owner and the destination have both been overwritten.
        # Only the published snapshot may recover the required recurrent state.
        for state in bank.layers.values():
            for index in source_indices + target_indices:
                state.clear_slot(index)
        neighbours = [i for i in range(capacity) if i not in source_indices + target_indices]
        untouched = values(neighbours)
        idle = drained()
        idle.synchronize()
        for row, target in enumerate(target_indices):
            done = store.restore(tickets[row],
                                 bank,
                                 owners[target],
                                 1920,
                                 f"fixture-{row}".encode(),
                                 idle,
                                 record_done=record)
            done.synchronize()
        slots.copy_(torch.tensor(target_indices, dtype=torch.int32))
        actual = tuple(value.cpu().clone() for value in native(*inputs))
        check = dict(
            change=change,
            outputs=[torch.equal(a, b) for a, b in zip(reference, actual, strict=True)],
            recurrent_state=[torch.equal(a, b) for a, b in zip(expected, values(target_indices), strict=True)],
            compressed_pages=[torch.equal(a, b.cpu()) for a, b in zip(expected_cache, buffers[-2:], strict=True)],
            neighbours=[torch.equal(a, b) for a, b in zip(untouched, values(neighbours), strict=True)])
        checks.append(check)
        if not all(check["outputs"] + check["recurrent_state"] + check["compressed_pages"] + check["neighbours"]):
            raise RuntimeError(f"Restored prefix changed its real TP consumer: {check}")
    allocated = store.allocated_bytes
    store.close()
    return dict(scope="HPU checkpoint copy, slot remap and real four-layer TP native replay; no throughput claim",
                batch=len(source_indices),
                tokens=1920,
                checks=checks,
                checkpoint_bytes=allocated,
                source_slots=source_indices,
                restored_slots=target_indices)
