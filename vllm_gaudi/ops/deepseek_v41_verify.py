# SPDX-License-Identifier: Apache-2.0
"""Fixed C6 DSpark device control and single-consumer completion ownership."""

from dataclasses import dataclass
from typing import Callable

import torch

from vllm.v1.outputs import AsyncModelRunnerOutput

# Wire layout: generation, committed, output_count, draft_count, output[6],
# draft[5], status.  PP0 copies only counts and final output ids to the host.
RECORD_SIZE = 16
OUTPUT_START, DRAFT_START, STATUS = 4, 10, 15


def vocab_parallel_argmax(local_logits, tp_rank, all_gather):
    """Return global greedy ids from contiguous vocab shards.

    ``all_gather`` is injected by the model's TP runtime so the helper stays
    usable by both the ordinary HCCL path and the prepared native replay.
    Communication is four FP32 scalars per row (value and global id on each
    of two ranks), independent of vocabulary size.
    """
    local_values, local_indices = local_logits.max(dim=-1)
    vocab_start = tp_rank * local_logits.shape[-1]
    global_indices = local_indices.to(torch.int64) + vocab_start
    pair = torch.stack((local_values.float(), global_indices.float()), dim=-1)
    # The dedicated TP2 exchange path accepts BF16 only.  Reinterpret the
    # exact FP32 pair as four BF16 words so the tiny argmax reduction uses the
    # same device exchange as the decoder, rather than launching a generic
    # FP32 all-gather for every target/draft row.  This is a bit-preserving
    # transport conversion; the values are reconstructed before argmax.
    if pair.device.type == "hpu":
        packed = pair.contiguous().view(torch.bfloat16)
        gathered = all_gather(packed, dim=1)
        gathered = gathered.contiguous().view(torch.float32)
        gathered = gathered.reshape(local_logits.shape[0], -1, 2)
    else:
        gathered = all_gather(pair, dim=-1)
        gathered = gathered.reshape(local_logits.shape[0], -1, 2)
    rank_index = gathered[..., 0].argmax(dim=-1, keepdim=True)
    return gathered[..., 1].gather(-1, rank_index).squeeze(-1).to(torch.int64)


def verify_control_from_target(target, proposed, metadata):
    """Apply the C6/C5 verification contract to already-reduced target ids.

    Keeping the control part separate lets a vocab-parallel producer reduce
    its local argmax to six token ids without materialising a full-vocabulary
    logits tensor.  The original :func:`verify_control` remains the reference
    path for callers that already own full logits.

    Metadata is [generation, valid targets, valid proposals, remaining output
    budget, absolute start of these target rows, context limit, need_sample].
    Invalid lanes are masked; no data-dependent host branch or slice occurs.
    """
    generation, count, proposal_count, remaining, start, limit, sample = metadata.unbind()
    lane = torch.arange(5, dtype=torch.int64, device=target.device)
    mismatch = (lane < proposal_count) & (target[:5] != proposed)
    # min(first mismatch, proposal_count) is the longest accepted prefix.
    accepted = torch.where(mismatch, lane, proposal_count).amin().reshape(1)
    has_proposals = proposal_count > 0
    committed = torch.where(has_proposals, accepted, count.reshape(1) - 1) + 1
    output_count = torch.where(sample.bool(), torch.where(has_proposals, committed, 1), 0)
    out_lane = torch.arange(6, dtype=torch.int64, device=target.device)
    last_index = (count - 1).clamp(0, 5).reshape(1)
    candidates = torch.where(has_proposals, target, target.gather(0, last_index).expand(6))
    output = torch.where(out_lane < output_count, candidates, -1)
    anchor_index = (output_count - 1).clamp(0, 5).long()
    anchor = output.gather(0, anchor_index).clamp_min(0)
    draft_enabled = (sample.bool() & (remaining - output_count >= 6) & (start + committed + 6 <= limit))
    valid = ((generation > 0) & (count >= 1) & (count <= 6)
             & (proposal_count >= 0) & (proposal_count <= 5)
             & ((proposal_count == 0) | (proposal_count + 1 == count))
             & (committed >= 1) & (committed <= count))
    status = (~valid).long().reshape(1)
    return target, output, committed, output_count, anchor, draft_enabled, status


def verify_control(logits, proposed, metadata):
    """Tensor equivalent of greedy_verify, with fixed C6/C5 input capacity."""
    return verify_control_from_target(logits.argmax(-1), proposed, metadata)


def pack_record(metadata, committed, output_count, output, draft_ids, draft_enabled, status):
    draft_count = torch.where(draft_enabled, 5, 0).long().reshape(1)
    return torch.cat((metadata[:1], committed.long(), output_count.long(), draft_count, output.long(),
                      torch.where(draft_enabled, draft_ids, -1).long(), status))


def encode_record_wire(record):
    """Encode the integer commit record for the BF16-only direct PP wire.

    The direct HCL primitive is a SUM exchange. Reinterpreting arbitrary
    token IDs as BF16 would allow the reduction to flush subnormals, round
    large IDs, or canonicalize NaNs. Four exact byte values per signed
    32-bit field keep every wire element in [0, 255], where adding the
    persistent zero source is exact. V4.1 control fields are bounded to the
    signed 32-bit range; the int64 record remains the host-facing contract.
    """
    value = record.to(torch.int32).reshape(-1, 1)
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=record.device)
    encoded = torch.bitwise_and(torch.bitwise_right_shift(value, shifts), 255)
    return encoded.to(torch.bfloat16).reshape(-1)


def decode_record_wire(wire):
    """Reconstruct the signed int64 commit record from byte-valued BF16."""
    encoded = wire.to(torch.int32).reshape(-1, 4).to(torch.int64)
    multipliers = torch.tensor([1, 1 << 8, 1 << 16, 1 << 24], dtype=torch.int64, device=wire.device)
    unsigned = (encoded * multipliers).sum(-1)
    return torch.where(unsigned >= (1 << 31), unsigned - (1 << 32), unsigned)


def validate_commit_wire(wire, expected_generation, max_committed):
    """Validate a direct PP wire after exact byte reconstruction."""
    return validate_commit(decode_record_wire(wire), expected_generation, max_committed)


def pp_commit_send(wire):
    """Experimental functional PP exchange; native capture is unqualified."""
    return torch.ops.vllm_gaudi.pp_exchange_peer_graph(wire)


def pp_commit_receive(zero, expected_generation, max_committed):
    """Receive and validate a PP commit in one fixed-shape graph."""
    wire = torch.ops.vllm_gaudi.pp_exchange_peer_graph(zero)
    return validate_commit_wire(wire, expected_generation, max_committed)


def validate_commit(record, expected_generation, max_committed):
    """Return PP0's minimal validated handoff (count plus output ids)."""
    valid = ((record[:1] == expected_generation) & (record[STATUS:STATUS + 1] == 0)
             & (record[1:2] >= 1) & (record[1:2] <= max_committed)
             & (record[2:3] >= 0) & (record[2:3] <= 6)
             & (record[3:4] >= 0) & (record[3:4] <= 5))
    committed = torch.where(valid, record[1:2], -1)
    output_count = torch.where(valid, record[2:3], -1)
    output = torch.where(valid.expand(6), record[OUTPUT_START:DRAFT_START], -1)
    return torch.cat((committed, output_count, output))


@dataclass(frozen=True)
class DeviceVerifyResult:
    slot: int
    generation: int
    record: torch.Tensor
    wire: torch.Tensor | None
    host: torch.Tensor
    completion: object
    max_committed: int
    # Commit uses an immutable per-slot snapshot.  ``record``/``wire`` remain
    # writable until the draft graph finishes so PP1 can publish draft ids to
    # the scheduler without racing the PP0 collective source.
    commit_record: torch.Tensor | None = None
    commit_wire: torch.Tensor | None = None


@dataclass(frozen=True)
class DevicePPCommit:
    """The asynchronous PP commit handoff owned by PP0.

    The collective is deliberately kept separate from the host consume.  A
    caller can enqueue the broadcast and return to the executor while the
    HPU performs the transfer.  PPBuffers.consume_device_commit is the only
    point that waits for the collective and publishes scheduler-visible ids.
    """

    work: object
    generation: int
    max_committed: int
    # When the direct path is graph compiled, the receive graph already owns
    # the HCL exchange and validator.  ``device_result`` is its persistent
    # eight-value output and is consumed without a second collective launch.
    device_result: torch.Tensor | None = None


class AsyncDeviceOutput(AsyncModelRunnerOutput):
    """Single-consumer wrapper for a device result with an explicit callback."""

    def __init__(self, result, consume: Callable):
        self.result, self.consume = result, consume
        self.consumed = False

    def get_output(self):
        if self.consumed:
            raise RuntimeError("device async output may only be consumed once")
        self.consumed = True
        with torch.profiler.record_function("v41::verify_and_commit::pp_commit_consume"):
            return self.consume(self.result)


class VerifyRing:
    """Persistent device/pinned buffers retained until the final consumer."""

    def __init__(self, device, *, last_rank, size=2):
        self.device, self.last_rank = torch.device(device), last_rank
        self.records = [torch.empty(RECORD_SIZE, dtype=torch.int64, device=device) for _ in range(size)]
        self.wires = ([torch.empty(RECORD_SIZE * 4, dtype=torch.bfloat16, device=device)
                       for _ in range(size)] if last_rank else [None] * size)
        self.commit_records = ([torch.empty(RECORD_SIZE, dtype=torch.int64, device=device)
                                for _ in range(size)] if last_rank else [None] * size)
        self.commit_wires = ([torch.empty(RECORD_SIZE * 4, dtype=torch.bfloat16, device=device)
                              for _ in range(size)] if last_rank else [None] * size)
        self.host = [
            torch.empty(RECORD_SIZE if last_rank else 8, dtype=torch.int64, device="cpu").pin_memory("hpu")
            for _ in range(size)
        ]
        self.events = [torch.hpu.Event() for _ in range(size)]
        self.generations, self.consumed = [0] * size, [0] * size
        self.generation, self.closed = 0, False
        self.expected = torch.empty(1, dtype=torch.int64, device=device)
        self.limit = torch.empty(1, dtype=torch.int64, device=device)
        self._validate = torch.compile(validate_commit, backend="hpu_backend", fullgraph=True, dynamic=False)

    def acquire(self, max_committed, *, generation=None):
        if self.closed:
            raise RuntimeError("DSpark verify ring is closed")
        next_generation = self.generation + 1 if generation is None else generation
        if next_generation <= self.generation:
            raise RuntimeError("DSpark verify generation must advance")
        # PP commits also occur for non-sampling prefill chunks.  Use the PP
        # transaction generation when supplied, so the record stays aligned
        # with PP0 even if no device result was acquired for those chunks.
        self.generation = next_generation
        slot = (self.generation - 1) % len(self.records)
        if self.generations[slot] != self.consumed[slot]:
            raise RuntimeError("DSpark verify slot still has an unconsumed result")
        self.generations[slot] = self.generation
        return DeviceVerifyResult(slot, self.generation, self.records[slot], self.wires[slot], self.host[slot],
                                  self.events[slot], max_committed, self.commit_records[slot], self.commit_wires[slot])

    def stage(self, ticket):
        self._check(ticket)
        if self.last_rank:
            source = ticket.record
        else:
            self.expected.fill_(ticket.generation)
            self.limit.fill_(ticket.max_committed)
            source = self._validate(ticket.record, self.expected, self.limit)
        ticket.host.copy_(source, non_blocking=True)
        ticket.completion.record(torch.hpu.current_stream())

    def _check(self, ticket):
        if (self.closed or self.generations[ticket.slot] != ticket.generation
                or self.consumed[ticket.slot] == ticket.generation):
            raise RuntimeError("Stale or already consumed DSpark verify result")

    def consume(self, ticket):
        self._check(ticket)
        ticket.completion.synchronize()
        values = ticket.host.tolist()  # pinned CPU data; never a device read
        if self.last_rank:
            generation, committed, output_count, draft_count = values[:4]
            if (generation != ticket.generation or values[STATUS] != 0 or not 1 <= committed <= ticket.max_committed
                    or not 0 <= output_count <= 6 or not 0 <= draft_count <= 5):
                raise RuntimeError("Invalid DSpark device verify generation or counts")
            output, draft = values[4:4 + output_count], values[10:10 + draft_count]
        else:
            committed, output_count = values[:2]
            output, draft = values[2:2 + output_count], []
            if not 1 <= committed <= ticket.max_committed or not 0 <= output_count <= 6:
                raise RuntimeError("Stale or invalid PP device verify commit")
        return committed, output, draft

    def release(self, ticket):
        self._check(ticket)
        self.consumed[ticket.slot] = ticket.generation

    def close(self):
        if self.closed:
            return
        for slot, generation in enumerate(self.generations):
            if generation != self.consumed[slot]:
                self.events[slot].synchronize()
        self.closed = True


class AsyncDSparkOutput(AsyncModelRunnerOutput):
    """The executor consumes this once before it updates the scheduler.

    HPU events stay in the owning worker.  vLLM's multiprocess response path
    already resolves AsyncModelRunnerOutput before serializing its result;
    its uniprocess future resolves at EngineCore's final result consumption.
    """

    def __init__(self, ring: VerifyRing, ticket: DeviceVerifyResult, finish: Callable, after_release=None):
        self.ring, self.ticket, self.finish = ring, ticket, finish
        self.after_release = after_release
        self.consumed = False

    def get_output(self):
        if self.consumed:
            raise RuntimeError("DSpark async output may only be consumed once")
        self.consumed = True
        # Do not retry or switch paths if completion/commit fails after writes.
        with torch.profiler.record_function("v41::verify_and_commit::final_consume"):
            result = self.finish(*self.ring.consume(self.ticket))
        self.ring.release(self.ticket)
        if self.after_release is not None:
            self.after_release(result)
        return result
