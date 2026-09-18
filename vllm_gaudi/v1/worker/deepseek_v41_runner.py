# SPDX-License-Identifier: Apache-2.0
"""Bounded V4.1 runner with scheduler-owned CSA2 state and PP verify commits.

The token/accepted-prefix contract follows vLLM #53577 (d2b1b735).
Transport uses persistent HPU buffers and HCCL, with one outstanding request.
"""

from dataclasses import dataclass, field
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
import json
import os
import time

import torch
import torch.distributed as dist

from vllm.distributed import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import (AsyncModelRunnerOutput, DraftTokenIds, EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput)

from vllm_gaudi import envs
from vllm_gaudi.extension.logger import logger as init_logger
from vllm_gaudi.extension.profiler import HabanaHighLevelProfiler
from vllm_gaudi.ops.deepseek_v41_state import PagedStageState, StageStateBlocks, register_state_spec
from vllm_gaudi.ops.deepseek_v41_verify import (
    AsyncDeviceOutput,
    AsyncDSparkOutput,
    DevicePPCommit,
    RECORD_SIZE,
    VerifyRing,
    validate_commit,
    validate_commit_wire,
)

logger = init_logger()

VERIFY_METADATA_SIZE = 7
VERIFY_PROPOSED_SIZE = 5
VERIFY_CONTROL_SIZE = VERIFY_METADATA_SIZE + VERIFY_PROPOSED_SIZE
PREFILL_BLOCK_TOKENS = 8192
# Grouped prefill reuses decoded expert weights across the scheduler chunk.
# Occupancy buckets bound temporary memory without reducing scheduler admission.
# C6 remains DSpark-only and C1 decode is unchanged.
PREFILL_MAX_INFLIGHT_BLOCKS = 1


def profile_phase(name):
    """Annotate host phases only during an explicitly requested acquisition."""

    def decorate(function):

        @wraps(function)
        def wrapped(self, *args, **kwargs):
            if not getattr(self, "trace_enabled", False):
                return function(self, *args, **kwargs)
            label = f"v41::{name}::PP{self.model.pp_rank}"
            if name == "target":
                label += f"::{'decode' if kwargs['decode'] else 'prefill'}::C{len(args[1])}"
            elif name == "insert_context":
                label += f"::C{args[1].numel()}"
            with torch.profiler.record_function(label):
                return function(self, *args, **kwargs)

        return wrapped

    return decorate


@dataclass
class RequestState:
    req_id: str
    prompt: list[int]
    mm_features: list
    sampling_params: object
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int = 0
    output: list[int] = field(default_factory=list)

    @property
    def tokens(self):
        return self.prompt + self.output

    def reconcile(self, output_count, new_tokens, all_tokens=None):
        if all_tokens is not None:
            self.output = list(all_tokens[len(self.prompt):len(self.prompt) + output_count])
        elif output_count > len(self.output):
            missing = output_count - len(self.output)
            if len(new_tokens) < missing:
                raise RuntimeError("PP scheduler did not supply the missing committed token prefix")
            self.output.extend(new_tokens[-missing:])
        else:
            self.output = self.output[:output_count]


def greedy_verify(target_ids, proposed_ids):
    """Return accepted drafts followed by the target's correction/bonus."""
    if len(target_ids) != len(proposed_ids) + 1:
        raise ValueError("DSpark target verification requires anchor plus every proposed token")
    accepted = 0
    for predicted, proposed in zip(target_ids, proposed_ids):
        if predicted != proposed:
            break
        accepted += 1
    return target_ids[:accepted + 1], accepted


def target_chunks(tokens, block_tokens=PREFILL_BLOCK_TOKENS):
    """Cover a scheduler prefill transaction with bounded device blocks.

    ``max_num_batched_tokens`` remains the scheduler admission limit.  This
    internal tiling only bounds the per-stage working set; C6 is reserved for
    DSpark anchor-plus-draft execution and is never used as an ordinary
    prefill geometry.
    """
    if block_tokens < 1 or block_tokens > PREFILL_BLOCK_TOKENS:
        raise ValueError("V4.1 prefill blocks must be no larger than max_num_batched_tokens=8192")
    offset = 0
    while offset < len(tokens):
        size = min(block_tokens, len(tokens) - offset)
        yield offset, tokens[offset:offset + size]
        offset += size


def target_search_length(start, count, maximum):
    """Choose one CSA2 search bucket for a complete scheduler transaction."""
    if start < 0 or count < 1 or start + count > maximum:
        raise ValueError("V4.1 search bucket is outside the configured context")
    return min(maximum, max(512, 1 << (start + count - 1).bit_length()))


def _exchange_payload_views(exchange_wire, capacity, hidden_slots=4, hidden_width=5120):
    """Return views for one native-BF16 stage-boundary payload.

    The hidden state stays BF16. ``pre_mix`` is transported as four numeric
    byte-valued BF16 lanes per FP32 value because the direct HCL primitive is
    a SUM exchange; raw FP32 bits would not survive adding a zero source.
    The returned ``pre`` tensor is a persistent logical FP32 buffer, while
    its encoded lanes remain in the tail of ``exchange_wire``.
    """
    hidden_elements = hidden_slots * hidden_width
    pre_elements = hidden_slots * 4
    expected = (capacity, hidden_elements + pre_elements)
    if exchange_wire.dtype != torch.bfloat16 or tuple(exchange_wire.shape) != expected:
        raise ValueError(f"invalid V4.1 PP BF16 exchange wire: dtype={exchange_wire.dtype}, "
                         f"shape={tuple(exchange_wire.shape)} != {expected}")
    hidden = exchange_wire[:, :hidden_elements].reshape(capacity, hidden_slots, hidden_width)
    pre = torch.empty((capacity, hidden_slots), dtype=torch.float32, device=exchange_wire.device)
    return hidden, pre


def _encode_exchange_pre(pre, wire):
    """Encode FP32 mHC state into exact byte-valued BF16 lanes."""
    if pre.ndim != 2 or wire.ndim != 3 or wire.shape[:2] != pre.shape or wire.shape[-1] != 4:
        raise ValueError(f"invalid pre_mix wire shapes: pre={tuple(pre.shape)} wire={tuple(wire.shape)}")
    value = pre.contiguous().view(torch.int32).reshape(-1, 1)
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=pre.device)
    encoded = torch.bitwise_and(torch.bitwise_right_shift(value, shifts), 255)
    wire.copy_(encoded.to(torch.bfloat16).reshape_as(wire))


def _decode_exchange_pre(wire, pre):
    """Decode byte-valued BF16 lanes back into the original FP32 bits."""
    if pre.ndim != 2 or wire.ndim != 3 or wire.shape[:2] != pre.shape or wire.shape[-1] != 4:
        raise ValueError(f"invalid pre_mix wire shapes: pre={tuple(pre.shape)} wire={tuple(wire.shape)}")
    encoded = wire.to(torch.int32).reshape(-1, 4).to(torch.int64)
    multipliers = torch.tensor([1, 1 << 8, 1 << 16, 1 << 24], dtype=torch.int64, device=wire.device)
    unsigned = (encoded * multipliers).sum(-1)
    signed = torch.where(unsigned >= (1 << 31), unsigned - (1 << 32), unsigned)
    pre.copy_(signed.to(torch.int32).view(torch.float32).reshape_as(pre))


class PPBuffers:

    def __init__(self, device, capacity=6, *, dspark=True, device_commit=None):
        self.group = get_pp_group()
        self.dspark = bool(dspark)
        if not self.dspark:
            self.device_commit_enabled = (envs.VLLM_HPU_DSV41_DEVICE_COMMIT
                                          if device_commit is None else bool(device_commit))
            self.device_commit = self.device_commit_enabled
            if (self.device_commit_enabled and not envs.VLLM_HPU_DSV41_GRAPH_REPLAY):
                raise ValueError("Device completion requires ordinary V4.1 native decode")
            self.hidden = torch.empty(capacity, 4, 5120, dtype=torch.bfloat16, device=device)
            self.pre = torch.empty(capacity, 4, dtype=torch.float32, device=device)
            self.commit = torch.empty(4, dtype=torch.int32, device=device)
            # Prompt completion is four host-generated integers.  Keep that
            # control record on the PP gloo group: lowering either pageable or
            # pinned H2D ``copy_`` after the large prefill graph can make the
            # eager Bridge create an invalid reinterpret-cast (dtype 524288).
            # Decode still uses ``commit`` and the native device-completion
            # path, so this does not add a host hand-off to steady-state C1.
            self.commit_host = torch.empty(4, dtype=torch.int32, device="cpu")
            self.commit_token = self.commit[3:4]
            self.commit_row = self.commit.view(1, 4)
            if self.device_commit_enabled:
                self.commit.zero_()
            self.generation = 0
            self.pending = []
            self.sends = self.receives = self.commits = 0
            self.packed = None
            if (envs.VLLM_HPU_DSV41_NATIVE_PP_COPY
                    and (not envs.VLLM_HPU_DSV41_PACKED_PP or not envs.VLLM_HPU_DSV41_GRAPH_REPLAY)):
                raise ValueError("Native PP copy requires ordinary packed C1 graph replay")
            if envs.VLLM_HPU_DSV41_PACKED_PP:
                from vllm_gaudi.ops.deepseek_v41_pp import PackedC1Buffers
                self.packed = PackedC1Buffers(device, native_copy=envs.VLLM_HPU_DSV41_NATIVE_PP_COPY)
            return
        if envs.VLLM_HPU_DSV41_MHC_SCHEDULE and not envs.VLLM_HPU_DSV41_GRAPH_REPLAY:
            raise RuntimeError("mHC scheduling requires V4.1 native graph replay")
        if envs.VLLM_HPU_DSV41_DIRECT_PP_WIRE and not (envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE
                                                       and envs.VLLM_HPU_DSV41_GRAPH_REPLAY):
            raise RuntimeError("Direct PP wire requires native stage replay and direct PP exchange")
        # A stage boundary used to enqueue one transfer for hidden and a
        # second transfer for pre_mix. Keep aligned typed views in one BF16
        # payload so each boundary has one queue entry and one DMA ownership
        # interval. pre_mix is transferred as raw bytes.
        hidden_slots, hidden_width = 4, 5120
        hidden_elements = hidden_slots * hidden_width
        pre_elements = hidden_slots * 4
        # The wire has one flat BF16 payload per token.  The previous layout
        # added a second ``4`` dimension to a byte tensor, replicating the
        # whole 4x hidden state payload.  Apart from being a 4x DMA
        # over-transfer, that shape could not be copied from the
        # [C,4,5120] stage boundary.
        # Keep the payload flat and use BF16 so HCCL/HCL does not insert a
        # uint8 compatibility conversion.
        self.exchange_wire = torch.empty((capacity, hidden_elements + pre_elements),
                                         dtype=torch.bfloat16,
                                         device=device)
        self.hidden, self.pre = _exchange_payload_views(self.exchange_wire, capacity, hidden_slots, hidden_width)
        self.pre_wire = self.exchange_wire[:, hidden_elements:].reshape(capacity, hidden_slots, 4)
        # The direct peer API is implemented by the version-locked TP2 HCL
        # primitive, whose reduction operator is SUM.  PP0 therefore needs a
        # receive sink and PP1 needs a *zero* source: using one mutable peer
        # buffer for both would feed the previous token back into the next
        # exchange and silently accumulate stage-boundary state.
        self.exchange_peer = torch.empty_like(self.exchange_wire)
        self.exchange_zero = torch.zeros_like(self.exchange_wire)
        # generation, committed input count, output count, draft count,
        # six output tokens and five draft tokens.
        # Legacy host commit remains available behind the feature flag.  The
        # device path broadcasts a fixed record and validates counts plus final
        # output ids on PP0, without copying draft ids or the complete record.
        self.commit = torch.empty(15, dtype=torch.int64, device=device)
        self.device_commit = torch.empty(RECORD_SIZE, dtype=torch.int64, device=device)
        # The direct PP commit exchange is BF16/SUM-only. Transport four
        # exact byte-valued BF16 lanes per int32 control field; arbitrary
        # int64 bit patterns must never be numerically reduced as BF16.
        self.commit_wire = torch.empty(RECORD_SIZE * 4, dtype=torch.bfloat16, device=device)
        self.commit_peer = torch.empty_like(self.commit_wire)
        self.commit_zero = torch.zeros_like(self.commit_wire)
        self.device_committed = torch.empty(8, dtype=torch.int64, device=device)
        self.device_expected = torch.empty(1, dtype=torch.int64, device=device)
        self.device_limit = torch.empty(1, dtype=torch.int64, device=device)
        self.device_host = torch.empty(8, dtype=torch.int64, device="cpu").pin_memory("hpu")
        self.device_event = torch.hpu.Event()
        self.device_validate = (torch.compile(validate_commit, backend="hpu_backend", fullgraph=True, dynamic=False)
                                if envs.VLLM_HPU_DSV41_DEVICE_VERIFY else None)
        self.device_validate_wire = (torch.compile(
            validate_commit_wire, backend="hpu_backend", fullgraph=True,
            dynamic=False) if envs.VLLM_HPU_DSV41_DEVICE_VERIFY and envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE else None)
        # The direct PP commit is compiled after the HPU bridge has been
        # registered in ``load_model``.  Keeping the exchange and wire
        # validator in one fixed graph makes the reverse handoff visible to
        # Synapse instead of leaving it as an eager operation between the
        # decoder and draft graphs.
        self.commit_send_graph = None
        self.commit_receive_graph = None
        self.generation = 0
        self.commit_consumed_generation = 0
        self.pending = []
        # The PP commit collective is independent of the host result read.
        # Keep its Work object alive until the async result is consumed (or a
        # later exchange/shutdown drains it).
        self.commit_work = None
        # HCCL enqueues the broadcast on the active HPU stream.  Recording a
        # stream event immediately after that enqueue gives the validator a
        # device-side dependency, so PP0 does not have to block the host on
        # Work.wait() before launching its tiny validation graph.
        self.commit_event = torch.hpu.Event()
        # PP commit/verification must not inherit the long target graph queue.
        # Keep its HCCL, validator and eight-value staging on a private HPU
        # stream.  The target stream is joined with a per-transaction input
        # event on PP1; PP0 has no local producer dependency.  This lets the
        # two stages submit the small commit exchange while their compute
        # streams drain independently.
        self.commit_stream = (torch.hpu.Stream() if
                              (envs.VLLM_HPU_DSV41_DEVICE_VERIFY and envs.VLLM_HPU_DSV41_INLINE_PP_COMMIT) else None)
        self.commit_record_events = {}
        # PP1 can overlap the next stage-boundary exchange with a prior commit
        # broadcast.  Keep the work handle by source-ring address so a record
        # is waited only when that ring slot is about to be reused.
        self.commit_records = {}
        self.sends, self.receives, self.commits = 0, 0, 0

    def prepare_commit_graph(self):
        """Compile the fixed direct PP commit exchange once per worker."""
        if (not self.dspark or not envs.VLLM_HPU_DSV41_DEVICE_VERIFY or not envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE
                or not envs.VLLM_HPU_DSV41_COMPILED_PP_COMMIT):
            return
        from vllm_gaudi.ops.deepseek_v41_verify import pp_commit_receive, pp_commit_send
        if self.group.is_last_rank:
            self.commit_send_graph = torch.compile(pp_commit_send, backend="hpu_backend", fullgraph=True, dynamic=False)
        else:
            self.commit_receive_graph = torch.compile(pp_commit_receive,
                                                      backend="hpu_backend",
                                                      fullgraph=True,
                                                      dynamic=False)

    def drain(self, *, include_commit=True):
        if not self.dspark:
            for work in self.pending:
                work.wait()
            self.pending.clear()
            return
        if include_commit and self.commit_work is not None:
            self.commit_work.wait()
            self.commit_work = None
            self.commit_records.clear()
        for work in self.pending:
            work.wait()
        self.pending.clear()
        if include_commit:
            for event in getattr(self, "commit_record_events", {}).values():
                event.synchronize()
            getattr(self, "commit_record_events", {}).clear()

    def wait_record(self, record):
        """Wait for a PP1 source slot only immediately before it is reused."""
        if not self.dspark:
            return
        if record is None or not hasattr(record, "data_ptr"):
            return
        work = self.commit_records.pop(int(record.data_ptr()), None)
        if work is not None:
            work.wait()
            if self.commit_work is work:
                self.commit_work = None
        event = getattr(self, "commit_record_events", {}).pop(int(record.data_ptr()), None)
        if event is not None:
            event.synchronize()

    def exchange(self, values, count, *, native_wire=False, decode=False):
        if not self.dspark:
            return self._exchange_ordinary(values, count, decode=decode)
        # PP0 reuses its receive/commit buffers only after its prior result was
        # consumed.  PP1's source records are ring-owned and can overlap this
        # exchange; wait for them at ring-slot reuse instead.
        self.drain(include_commit=self.group.is_first_rank)
        if self.group.is_first_rank:
            transfer_wire = self.exchange_wire[:count]
            if "pp_wire" in values.tensors:
                wire = values["pp_wire"]
                if (wire.shape != self.exchange_wire[:count].shape or wire.dtype != torch.bfloat16
                        or not wire.is_contiguous()):
                    raise RuntimeError("Compiled PP output violates the contiguous BF16 wire contract")
                if envs.VLLM_HPU_DSV41_DIRECT_PP_WIRE and envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE:
                    # Native stage outputs are held through communication
                    # completion by the Bridge's resource owner.
                    transfer_wire = wire
                else:
                    transfer_wire.copy_(wire)
            else:
                self.hidden[:count].copy_(values["hidden_states"])
                self.pre[:count].copy_(values["pre_mix"])
                _encode_exchange_pre(self.pre[:count], self.pre_wire[:count])
            if envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE:
                # Importing the TP2 module registers the custom op.  The
                # model loader normally does this during graph replay setup;
                # keep the direct-boundary contract explicit for callers
                # constructing PPBuffers in isolation.
                from vllm_gaudi.distributed import tp2_fused_ar_norm  # noqa: F401
                torch.ops.vllm_gaudi.pp_exchange_peer(transfer_wire, self.exchange_peer[:count])
            else:
                self.pending.append(
                    dist.isend(self.exchange_wire[:count], dst=self.group.ranks[1], group=self.group.device_group))
            self.sends += 1
            return None
        if envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE:
            from vllm_gaudi.distributed import tp2_fused_ar_norm  # noqa: F401
            # The direct API is collective on the two-rank PP communicator;
            # receive PP0's wire in place and contribute the persistent zero
            # source to its SUM reduction.
            torch.ops.vllm_gaudi.pp_exchange_peer(self.exchange_zero[:count], self.exchange_wire[:count])
        else:
            self.pending.append(
                dist.irecv(self.exchange_wire[:count], src=self.group.ranks[0], group=self.group.device_group))
            self.drain(include_commit=self.group.is_first_rank)
        self.receives += 1
        if native_wire:
            return IntermediateTensors({"pp_wire": self.exchange_wire[:count]})
        _decode_exchange_pre(self.pre_wire[:count], self.pre[:count])
        return IntermediateTensors({"hidden_states": self.hidden[:count], "pre_mix": self.pre[:count]})

    def _exchange_ordinary(self, values, count, *, decode=False):
        self.drain()
        if self.packed is not None and decode and count == 1:
            packet, received = self.packed.acquire()
            if self.group.is_first_rank:
                self.packed.pack(values)
                self.pending.append(dist.isend(packet, dst=self.group.ranks[1], group=self.group.device_group))
                self.sends += 1
                return None
            self.pending.append(dist.irecv(packet, src=self.group.ranks[0], group=self.group.device_group))
            self.drain()
            self.receives += 1
            return IntermediateTensors(received)
        if self.group.is_first_rank:
            self.hidden[:count].copy_(values["hidden_states"])
            self.pre[:count].copy_(values["pre_mix"])
            for value in (self.hidden[:count], self.pre[:count]):
                self.pending.append(dist.isend(value, dst=self.group.ranks[1], group=self.group.device_group))
            self.sends += 2
            return None
        for value in (self.hidden[:count], self.pre[:count]):
            self.pending.append(dist.irecv(value, src=self.group.ranks[0], group=self.group.device_group))
        self.drain()
        self.receives += 2
        return IntermediateTensors({"hidden_states": self.hidden[:count], "pre_mix": self.pre[:count]})

    def finish(self, committed=None, output=(), draft=()):
        if not self.dspark:
            raise RuntimeError("DSpark verify commit is disabled in ordinary C1 execution")
        self.drain(include_commit=self.group.is_first_rank)
        self.generation += 1
        if self.group.is_last_rank:
            values = [self.generation, committed, len(output), len(draft), *output]
            values += [-1] * (10 - len(values))
            values += list(draft) + [-1] * (5 - len(draft))
            self.commit.copy_(torch.tensor(values, dtype=torch.int64, device="cpu"))
        self.group.broadcast(self.commit, src=1)
        record = self.commit.cpu().tolist()
        if record[0] != self.generation or not 0 <= record[2] <= 6 or not 0 <= record[3] <= 5:
            raise RuntimeError("Stale or invalid PP verify commit generation")
        self.commit_consumed_generation = self.generation
        self.commits += 1
        return record[1], record[4:4 + record[2]], record[10:10 + record[3]]

    def complete_packet(self):
        if not self.dspark and self.packed is not None:
            self.packed.complete()

    def finish_single(self, consumed=None, token=None):
        if self.dspark:
            raise RuntimeError("Ordinary token completion cannot commit DSpark verification")
        self.drain()
        self.generation += 1
        if self.group.is_last_rank:
            record = [self.generation, consumed, int(token is not None), -1 if token is None else token]
            self.commit_host.copy_(torch.tensor(record, dtype=torch.int32, device="cpu"))
        # This path runs once at the prompt/decode boundary.  A CPU collective
        # is both smaller and more reliable than routing a 16-byte host record
        # through HPU eager lowering while the prefill recipes still own their
        # workspaces.  ``src`` is the global rank of local PP rank 1.
        dist.broadcast(self.commit_host, src=self.group.ranks[1], group=self.group.cpu_group)
        record = self.commit_host.tolist()
        if record[0] != self.generation or record[2] not in (0, 1):
            raise RuntimeError("Stale or invalid PP ordinary-token completion")
        self.commits += 1
        self.complete_packet()
        return record[1], [record[3]] if record[2] else []

    def finish_single_device(self):
        enabled = getattr(self, "device_commit_enabled", getattr(self, "device_commit", False))
        if self.dspark or not enabled:
            raise RuntimeError("Device completion is not enabled for ordinary C1")
        from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime
        self.drain()
        self.generation += 1
        self.group.broadcast(self.commit, src=1)
        bridge, _, _ = _resolve_runtime()
        host, done = bridge.copy_integer_record_to_host(self.commit_row)
        done.synchronize()
        record = host[0].tolist()
        if record[:3] != [self.generation, 1, 1] or record[3] < 0:
            raise RuntimeError("Stale or invalid device C1 completion")
        self.commits += 1
        self.complete_packet()
        return 1, [record[3]]

    def finish_device(self, record, max_committed, wire_record=None):
        """Enqueue a device record handoff without a host synchronization.

        PP0 used to validate, copy eight int64 values to pinned memory and
        synchronize an HPU event inside this method.  That made the whole PP
        commit appear as a CPU gap.  The fixed record uses either the direct
        BF16 peer exchange or the asynchronous broadcast compatibility path;
        PP0 validates and reads it only from the returned async result at the
        scheduler's actual consume point.
        """
        if (self.group.is_first_rank and self.generation != self.commit_consumed_generation):
            raise RuntimeError("PP device commit still has an unconsumed result")
        # PP0 must not reuse its receive/validation buffer before the prior
        # commit has been consumed.  PP1's source is a verify-ring slot,
        # however, and ``wait_record`` below already waits exactly when that
        # slot is about to be overwritten.  Draining PP1 commits here would
        # serialize the next target graph behind the previous 128-byte
        # broadcast and recreate the host gap visible in the C6 trace.
        self.drain(include_commit=self.group.is_first_rank)
        self.generation += 1
        # PP1's record already lives in the persistent verify-ring slot.  Use
        # that tensor as the broadcast source instead of copying it into a
        # second device buffer first.  PP0 receives into ``device_commit`` so
        # the validator keeps a stable address.  This removes one device-local
        # copy and its queue dependency from every C6 transaction; the wire
        # layout and generation contract are unchanged.
        wire = self.device_commit
        direct = envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE
        if direct:
            # The direct exchange is BF16-only. Use the graph-produced
            # byte-valued BF16 record; numeric SUM with the persistent zero
            # source is then exact for every control field.
            if self.group.is_last_rank:
                if record is None or record.numel() != RECORD_SIZE:
                    raise RuntimeError("PP1 did not produce a complete device verify record")
                if record.dtype != self.device_commit.dtype or record.device != self.device_commit.device:
                    raise RuntimeError("PP1 device verify record has an incompatible wire type")
                if (wire_record is None or wire_record.dtype != torch.bfloat16 or wire_record.numel() != RECORD_SIZE * 4
                        or wire_record.device != self.commit_wire.device):
                    raise RuntimeError("PP1 did not produce a complete BF16 commit wire")
                wire = wire_record
                peer = self.commit_peer
            else:
                wire = self.commit_zero
                peer = self.commit_wire
        elif self.group.is_last_rank:
            if record is None or record.numel() != RECORD_SIZE:
                raise RuntimeError("PP1 did not produce a complete device verify record")
            if record.dtype != self.device_commit.dtype or record.device != self.device_commit.device:
                raise RuntimeError("PP1 device verify record has an incompatible wire type")
            wire = record
        # The result is written by the verify graph on the normal compute
        # stream.  A new event per ring slot/transaction prevents re-recording
        # a still-consumed event while the next C6 is being prepared.
        commit_stream = getattr(self, "commit_stream", None)
        input_event = None
        if self.group.is_last_rank and commit_stream is not None:
            input_event = torch.hpu.Event()
            input_event.record(torch.hpu.current_stream())
        stream_context = (torch.hpu.stream(commit_stream) if commit_stream is not None else nullcontext())
        device_result = None
        timing = getattr(self, "timing", None)
        if timing:
            timing.host("commit_enqueue_start")
        with stream_context:
            if timing:
                timing.device("commit_stream_before_wait")
            if input_event is not None:
                commit_stream.wait_event(input_event)
            if timing:
                timing.device("commit_inputs_ready")
            with torch.profiler.record_function("v41::pp_commit::device_broadcast"):
                if direct:
                    from vllm_gaudi.distributed import tp2_fused_ar_norm  # noqa: F401
                    if envs.VLLM_HPU_DSV41_COMPILED_PP_COMMIT:
                        if self.group.is_first_rank:
                            if self.commit_receive_graph is None:
                                raise RuntimeError("PP receive graph was not prepared")
                            # These guards and their consumer must share the
                            # commit stream. The graph owns the only exchange
                            # for this generation; never precede it with an
                            # eager exchange on just one of the two ranks.
                            self.device_expected.fill_(self.generation)
                            self.device_limit.fill_(max_committed)
                            device_result = self.commit_receive_graph(wire, self.device_expected, self.device_limit)
                        else:
                            if self.commit_send_graph is None:
                                raise RuntimeError("PP send graph was not prepared")
                            self.commit_send_graph(wire)
                    else:
                        torch.ops.vllm_gaudi.pp_exchange_peer(wire, peer)
                    self.commit_work = None
                else:
                    self.commit_work = dist.broadcast(
                        wire,
                        src=self.group.ranks[1],
                        group=self.group.device_group,
                        async_op=True,
                    )
                    if self.group.is_first_rank and hasattr(self, "commit_event"):
                        self.commit_event.record(self.commit_stream)
                    if self.group.is_last_rank:
                        self.commit_records[int(record.data_ptr())] = self.commit_work
            if timing:
                timing.device("commit_transfer_done")
            # Direct HCL has no Work object.  Retain a device completion event
            # until the source ring slot is reused so it cannot be overwritten
            # while the collective still reads it.
            if self.group.is_last_rank and commit_stream is not None:
                done = torch.hpu.Event()
                done.record(commit_stream)
                self.commit_record_events[int(record.data_ptr())] = done
        if timing:
            timing.host("commit_enqueue_done")
        if self.group.is_first_rank:
            result = DevicePPCommit(self.commit_work, self.generation, max_committed, device_result)
            self.commits += 1
            return result
        self.commits += 1
        return None

    def consume_device_commit(self, result):
        """Resolve a PP0 commit at the final consumer."""
        timing = getattr(self, "timing", None)
        if timing:
            timing.host("consume_start")
        if not isinstance(result, DevicePPCommit):
            raise RuntimeError("Invalid PP device commit result")
        if (result.generation != self.generation or result.generation == self.commit_consumed_generation):
            raise RuntimeError("Stale PP device verify generation")
        if self.device_validate is None:
            raise RuntimeError("Device verify validator was not compiled")
        commit_stream = getattr(self, "commit_stream", None)
        stream_context = (torch.hpu.stream(commit_stream) if commit_stream is not None else nullcontext())
        with stream_context:
            with torch.profiler.record_function("v41::pp_commit::validate"):
                # The broadcast and validator share the active HPU stream.  The
                # event is therefore a device dependency and does not turn the
                # PP commit into a host-side queue drain.  Keep Work.wait as a
                # compatibility fallback for test doubles and older runtimes
                # which cannot expose an HPU stream event.
                stream = torch.hpu.current_stream()
                if result.work is None:
                    # Direct PP exchange and validation are both enqueued on
                    # the private commit stream; the operation itself is the
                    # dependency.
                    pass
                elif hasattr(self, "commit_event") and hasattr(stream, "wait_event"):
                    stream.wait_event(self.commit_event)
                else:
                    result.work.wait()
                if self.commit_work is result.work:
                    self.commit_work = None
                validator = self.device_validate_wire if result.work is None else self.device_validate
                if timing:
                    timing.host("validator_submit_start")
                    timing.device("validator_start")
                if result.device_result is not None:
                    # ``finish_device`` already submitted the fused receive
                    # and validation graph. Reuse its output instead of
                    # launching a second validator graph here.
                    self.device_committed.copy_(result.device_result)
                elif result.work is None and validator is None:
                    raise RuntimeError("Direct PP commit validator was not compiled")
                elif validator is not None:
                    self.device_expected.fill_(result.generation)
                    self.device_limit.fill_(result.max_committed)
                    source = self.commit_wire if result.work is None else self.device_commit
                    self.device_committed.copy_(validator(source, self.device_expected, self.device_limit))
                if timing:
                    timing.device("validator_done")
                    timing.host("validator_submit_done")
                    timing.host("d2h_submit_start")
                self.device_host.copy_(self.device_committed, non_blocking=True)
                if timing:
                    timing.host("d2h_submit_done")
                    timing.device("d2h_done")
                    timing.host("event_record_start")
                self.device_event.record(commit_stream or torch.hpu.current_stream())
                if timing:
                    timing.host("event_record_done")
            if timing:
                timing.synchronize(self.device_event)
            else:
                self.device_event.synchronize()
        with torch.profiler.record_function("v41::pp_commit::host_consume"):
            if timing:
                timing.host("tolist_start")
            values = self.device_host.tolist()
            if timing:
                timing.host("tolist_done")
        committed, output_count = int(values[0]), int(values[1])
        if committed < 1 or not 0 <= output_count <= 6:
            raise RuntimeError("Invalid or stale PP device verify commit")
        self.commit_consumed_generation = result.generation
        if timing:
            timing.host("consume_done")
        return committed, values[2:2 + output_count]


class V41ModelRunner:
    _PAD_BLOCK_ID, _PAD_SLOT_ID = 0, 0
    v2_completion = False

    def __init__(self, vllm_config, is_driver_worker=False):
        del is_driver_worker
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = vllm_config.device_config.device
        self.requests, self.encoder_cache = {}, {}
        self.model = self.state = None
        self.kv_caches, self.graphed_buckets = [], set()
        self.pending = self.draft_token_ids = None
        self.use_dspark = envs.VLLM_HPU_DSV41_DSPARK
        self._token_copy = self._next_input = None
        self.active_request = None
        self.trace_enabled = False
        # Generic multimodal batching uses the platform's pageable path;
        # Engram owns separate buffers pinned explicitly through the HPU API.
        self.pin_memory = False
        self.profiler = HabanaHighLevelProfiler()
        self.model_memory_usage = self.mem_margin = 0
        self.serving_workspace_reserve = (3 << 30) if self.model_config.max_model_len > 512 else 0
        self.pp = PPBuffers(self.device,
                            capacity=PREFILL_BLOCK_TOKENS,
                            dspark=self.use_dspark,
                            device_commit=False if self.v2_completion else None)
        self.input_ids = torch.empty(PREFILL_BLOCK_TOKENS, dtype=torch.int64, device=self.device)
        self.positions = torch.empty(PREFILL_BLOCK_TOKENS, dtype=torch.int32, device=self.device)
        # Decode/DSpark reuse exact Tensor objects. Large-M prompt buckets are
        # created on demand, avoiding sixteen thousand permanent Python Tensor
        # wrappers merely to represent all possible lengths through C8192.
        self.input_views = {count: self.input_ids[:count] for count in range(1, 129)}
        self.position_views = {count: self.positions[:count] for count in range(1, 129)}
        self.direct_token_ids = (envs.VLLM_HPU_DSV41_DIRECT_TOKEN_IDS and not self.use_dspark)
        self.decode_ids = (torch.empty(1, dtype=torch.int32, device=self.device) if self.direct_token_ids else None)
        self.position_bank = None
        if envs.VLLM_HPU_DSV41_FIXED_POSITIONS and not self.use_dspark:
            from vllm_gaudi.ops.deepseek_v41_inputs import PositionBank
            self.position_bank = PositionBank(self.model_config.max_model_len, PREFILL_BLOCK_TOKENS, self.device)
        self.input_staging = None
        self.batched_input_staging = envs.VLLM_HPU_DSV41_BATCHED_INPUT_STAGING
        if self.batched_input_staging and not envs.VLLM_HPU_DSV41_FUSED_STAGE_IO:
            raise RuntimeError("Batched input staging requires fused stage I/O")
        if envs.VLLM_HPU_DSV41_FUSED_STAGE_IO:
            self.input_staging = torch.empty((2, 7), dtype=torch.int64, device="cpu").pin_memory("hpu")
            self.input_staging_values = self.input_staging.numpy()
            self.input_control = torch.empty(7, dtype=torch.int64, device=self.device)
            self.input_dma_events = [torch.hpu.Event(), torch.hpu.Event()]
            self.input_dma_pending = [False, False]
            self.input_generation = 0
            offsets = torch.arange(6, device=self.device, dtype=torch.int32)
            self.prepare_positions = torch.compile(lambda control: control[6].to(torch.int32) + offsets,
                                                   backend="hpu_backend",
                                                   fullgraph=True,
                                                   dynamic=False)
        self.verify_ring = None
        self.verify_records = {}
        self.round_records = []
        self.round_context = None
        self.round_timing_enabled = envs.VLLM_HPU_DSV41_ROUND_TIMING
        if self.round_timing_enabled and not os.environ.get("DSV41_RUN_EVIDENCE"):
            raise RuntimeError("V4.1 round timing requires an evidence directory")
        self.verify_timing = None
        self.verify_and_propose = None
        self.verify_prefix = None
        self.draft_from_prefix = None
        # Metadata and draft ids are one fixed control payload.  Keeping views
        # for the compiled entry preserves its interface while turning two
        # independent tiny H2D copies into one 96-byte transfer.
        self.verify_control = torch.empty(VERIFY_CONTROL_SIZE, dtype=torch.int64, device=self.device)
        self.verify_metadata = self.verify_control[:VERIFY_METADATA_SIZE]
        self.verify_proposed = self.verify_control[VERIFY_METADATA_SIZE:]
        self.verify_positions = torch.empty(6, dtype=torch.int32, device=self.device)
        self.verify_offsets = torch.arange(6, dtype=torch.int32, device=self.device)
        # These are the only host-to-device control inputs in the steady
        # device-verify path.  Keep them pinned so the copy is an actual async
        # DMA instead of an implicit pageable staging/synchronization.
        self.verify_control_host = torch.empty(VERIFY_CONTROL_SIZE, dtype=torch.int64, device="cpu").pin_memory("hpu")
        self.verify_metadata_host = self.verify_control_host[:VERIFY_METADATA_SIZE]
        self.verify_proposed_host = self.verify_control_host[VERIFY_METADATA_SIZE:]
        self.verify_hidden = torch.empty((6, getattr(self.model_config.hf_config, "hidden_size", 5120)),
                                         dtype=torch.bfloat16,
                                         device=self.device)
        self.verify_aux = torch.empty((6, 3 * getattr(self.model_config.hf_config, "hidden_size", 5120)),
                                      dtype=torch.bfloat16,
                                      device=self.device)
        self.audit = {
            "target_steps": 0,
            "target_tokens": 0,
            "draft_steps": 0,
            "accepted_drafts": 0,
            "rejected_drafts": 0,
            "requests": 0,
            "prefill_steps": 0,
            "decode_steps": 0,
            "image_encodes": 0
        }

    def load_model(self):
        before = torch.hpu.memory_allocated()
        if envs.VLLM_HPU_DSV41_GRAPH_REPLAY:
            from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
            initialize_tp2_fused_ar_norm_runtime()
        # Compile the PP commit graph only after the bridge has registered its
        # custom op.  This keeps normal/non-direct paths unchanged.
        self.pp.prepare_commit_graph()
        self.model = get_model(vllm_config=self.vllm_config)
        state_type = PagedStageState if self.model_config.max_model_len > 512 else StageStateBlocks
        self.state = state_type(self.model.program)
        register_state_spec(self.vllm_config)
        self.model_memory_usage = torch.hpu.memory_allocated() - before
        if self.pp.group.is_last_rank and envs.VLLM_HPU_DSV41_DSPARK:
            draft = self.model.program.draft
            self.insert_context = torch.compile(draft.insert_context,
                                                backend="hpu_backend",
                                                fullgraph=True,
                                                dynamic=False)
            self.run_draft = torch.compile(draft, backend="hpu_backend", fullgraph=True, dynamic=False)
            self.sample_draft = torch.compile(draft.sample_greedy, backend="hpu_backend", fullgraph=True, dynamic=False)
            if envs.VLLM_HPU_DSV41_DEVICE_VERIFY:
                # Prefix verification and draft control have separate compile
                # entries.  The former can publish PP commit as soon as the
                # accepted prefix is known; the latter runs concurrently on
                # the normal compute stream and fills the scheduler record.
                self.verify_prefix = torch.compile(draft.verify_prefix,
                                                   backend="hpu_backend",
                                                   fullgraph=True,
                                                   dynamic=False)
                self.draft_from_prefix = torch.compile(draft.draft_from_prefix,
                                                       backend="hpu_backend",
                                                       fullgraph=True,
                                                       dynamic=False)
                self.verify_ring = VerifyRing(self.device, last_rank=True)
        elif envs.VLLM_HPU_DSV41_DEVICE_VERIFY and envs.VLLM_HPU_DSV41_DSPARK:
            self.verify_ring = VerifyRing(self.device, last_rank=False)
        elif self.pp.group.is_last_rank:
            sampler = self.model.program.sample_greedy_token if self.v2_completion else self.model.program.sample_greedy
            self.sample_target = torch.compile(sampler, backend="hpu_backend", fullgraph=True, dynamic=False)
            if self.pp.device_commit_enabled:
                self.sample_target_commit = torch.compile(self.model.program.sample_greedy_commit,
                                                          backend="hpu_backend",
                                                          fullgraph=True,
                                                          dynamic=False)
        logger.info("V4.1 PP%d prepared weights loaded; allocated %d bytes", self.model.pp_rank,
                    self.model_memory_usage)

    def get_model(self):
        return self.model

    def get_supported_tasks(self):
        return ("generate", )

    def reset_encoder_cache(self):
        self.encoder_cache.clear()

    def get_kv_cache_spec(self):
        logger.info("V4.1 PP%d exposes %d scheduler state arrays (%d bytes/page)", self.model.pp_rank,
                    len(self.state.specs), sum(spec.page_size_bytes for spec in self.state.specs.values()))
        return self.state.specs

    def uses_framework_kv_cache_layout(self, name):
        return name in self.state.specs

    def allocate_framework_kv_cache_layer(self, spec, num_blocks):
        return torch.zeros((num_blocks, *spec.state_shape), device=self.device, dtype=spec.state_dtype)

    def bind_framework_kv_caches(self, caches, runner_caches):
        logger.info("V4.1 PP%d binding profile state", self.model.pp_rank)
        if set(caches) != set(self.state.specs):
            raise ValueError("Scheduler did not allocate every V4.1 state array")
        self.state.allocations = caches
        counts = {value.shape[0] for value in caches.values()}
        if len(counts) != 1:
            raise ValueError("V4.1 state arrays must share scheduler block ownership")
        self.state.blocks, self.state.active = counts.pop(), None
        self.state.bind(0)
        runner_caches.extend((value, ) for value in caches.values())

    def initialize_kv_cache(self, config):
        names = {name for group in config.kv_cache_groups for name in group.layer_names}
        if names != set(self.state.specs):
            raise ValueError("Scheduler cache group does not match this PP stage's complete state")
        self.state.allocate(config.num_blocks, self.device)
        self.state.bind(1)
        self.state.clear()
        self.kv_caches = [(value, ) for value in self.state.allocations.values()]
        self.kv_cache_config = config

    def _bind_request(self, request):
        if isinstance(self.state, PagedStageState):
            if len(request.block_ids) != 1:
                raise RuntimeError("V4.1 compressed state expects one scheduler page group")
            reset = request.num_computed_tokens == 0
            self.state.activate(request.req_id, request.block_ids[0], reset=reset)
            if self.model.engram_host is not None:
                self.model.engram_host.activate(request.req_id, reset=reset)
            if reset:
                self.audit["requests"] += 1
            self.active_request = request.req_id
            return
        if len(request.block_ids) != 1 or len(request.block_ids[0]) != 1:
            raise RuntimeError("V4.1 expects one scheduler-owned whole-request state block")
        block = request.block_ids[0][0]
        if block == 0:
            raise RuntimeError("The scheduler assigned its null block to a live V4.1 request")
        self.state.bind(block)
        if self.active_request != request.req_id:
            if request.num_computed_tokens:
                raise RuntimeError("A V4.1 request resumed without its bound CSA2 and Engram state")
            self.state.clear()
            self.active_request = request.req_id
            self.audit["requests"] += 1

    def _update(self, scheduled):
        for req_id in scheduled.finished_req_ids:
            if self.verify_timing:
                self.verify_timing.flush()
            records = self.verify_records.pop(req_id, None)
            if records:
                logger.info("V4.1 device verify transactions PP%d TP%d: %s", self.model.pp_rank, self.model.tp_rank,
                            json.dumps({
                                "request_id": req_id,
                                "units": "ms",
                                "transactions": records
                            }))
            self.requests.pop(req_id, None)
            if isinstance(self.state, PagedStageState):
                self.state.release(req_id)
                if self.model.engram_host is not None:
                    self.model.engram_host.release_request(req_id)
            if self.active_request == req_id:
                self.active_request = None
        for mm_hash in scheduled.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)
        for new in scheduled.scheduled_new_reqs:
            if new.prompt_token_ids is None or new.prompt_embeds is not None or new.lora_request is not None:
                raise ValueError("V4.1 prepared execution requires processor-owned token/image inputs")
            self._validate_sampling(new.sampling_params)
            self.requests[new.req_id] = RequestState(new.req_id, list(new.prompt_token_ids), new.mm_features,
                                                     new.sampling_params, new.block_ids, new.num_computed_tokens)
        cached = scheduled.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            request = self.requests[req_id]
            new_blocks = cached.new_block_ids[index]
            if req_id in cached.resumed_req_ids:
                request.block_ids = new_blocks
                self.active_request = None
            elif new_blocks is not None:
                request.block_ids = tuple(old + new for old, new in zip(request.block_ids, new_blocks, strict=True))
            request.num_computed_tokens = cached.num_computed_tokens[index]
            new_tokens = cached.new_token_ids[index] if cached.new_token_ids else []
            request.reconcile(cached.num_output_tokens[index], new_tokens, cached.all_token_ids.get(req_id))

    @staticmethod
    def _validate_sampling(params):
        from vllm_gaudi.ops.deepseek_v41_config import validate_sampling
        validate_sampling(params)

    def _image_embeddings(self, request, start, count):
        if not request.mm_features or not self.pp.group.is_first_rank:
            return None
        self._execute_mm_encoder(request)
        values = self.model.embed_input_ids(self.input_ids[:count])
        for feature in request.mm_features:
            position = feature.mm_position
            low, high = max(start, position.offset), min(start + count, position.offset + position.length)
            if low >= high:
                continue
            source = self.encoder_cache[feature.identifier][low - position.offset:high - position.offset]
            if position.is_embed is None:
                values[low - start:high - start].copy_(source)
            else:
                mask = position.is_embed[low - position.offset:high - position.offset].to(self.device)
                destination = values[low - start:high - start]
                destination.copy_(torch.where(mask.unsqueeze(-1), source, destination))
        return values

    @profile_phase("vision")
    def _execute_mm_encoder(self, request):
        # Reuse upstream batching and output validation. HPU consumes static
        # placeholder indices instead of a dynamic device boolean scatter.
        from vllm.multimodal.utils import group_and_batch_mm_kwargs
        from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs
        from vllm_gaudi.ops.deepseek_v41_vision import scatter_image_embeddings
        features = [feature for feature in request.mm_features if feature.identifier not in self.encoder_cache]
        kwargs = [(feature.modality, feature.data) for feature in features]
        outputs = []
        for _, count, batch in group_and_batch_mm_kwargs(kwargs, device=self.device, pin_memory=self.pin_memory):
            encoded = self.model.embed_multimodal(**batch)
            sanity_check_mm_encoder_outputs(encoded, expected_num_items=count)
            outputs.extend(encoded)
            self.audit["image_encodes"] += count
        for feature, output in zip(features, outputs, strict=True):
            self.encoder_cache[feature.identifier] = scatter_image_embeddings(output, feature.mm_position.is_embed)

    @profile_phase("target")
    def _forward(self, request_id, tokens, start, *, decode, reset=False, request=None, search_length=None):
        count = len(tokens)
        # Avoid evaluating a fallback slice when a test or a C1-only runner
        # intentionally owns just the persistent captured view.
        ids = self.input_views[count] if count in self.input_views else self.input_ids[:count]
        positions = self.position_views[count] if count in self.position_views else self.positions[:count]
        # A scheduler may feed a normal prompt one token at a time.  The
        # resulting model transaction has exactly the same C1 tensor geometry
        # and cache writes as decode; only sampling/commit semantics remain
        # prefill semantics.  Bind that token to the captured C1 input tensor
        # instead of compiling an unqualified ordinary C1 stage on the first
        # public chat request.
        c1_replay = (getattr(self.model, "native", False) and count == 1 and start + count <= 1024)
        graph_c1 = decode or c1_replay
        if self.direct_token_ids and graph_c1:
            if count != 1:
                raise ValueError("Direct V4.1 token binding requires a C1 transaction")
            ids = self.decode_ids
        program = getattr(self.model, "program", None)
        if program is not None and program.length > 512:
            search = (target_search_length(start, count, program.length)
                      if search_length is None else int(search_length))
            if search < start + count or search > program.length:
                raise RuntimeError("V4.1 transaction search bucket does not cover its input")
            program.search_length = search
            for layer in program.layers:
                attention = layer.attention
                if hasattr(attention, "set_search_length"):
                    attention.set_search_length(search)
                else:
                    attention.search_length = search
        # The fused seven-value control packet belongs to C1/C6 replay.  A
        # normal prefill block has independent C128-capable input buffers and
        # must not be truncated through that DSpark-sized packet.
        staged_input = (getattr(self, "input_staging", None) is not None and count <= 6
                        and (graph_c1 or self.use_dspark))
        if staged_input:
            slot = self.input_generation % 2
            if self.input_dma_pending[slot]:
                self.input_dma_events[slot].synchronize()
            self.input_staging_values[slot, :count] = tokens
            self.input_staging_values[slot, count:6] = 0
            self.input_staging_values[slot, 6] = start
            self.input_control.copy_(self.input_staging[slot], non_blocking=True)
            self.input_dma_events[slot].record()
            self.input_dma_pending[slot] = True
            self.input_generation += 1
            positions = self.prepare_positions(self.input_control)
            if self.batched_input_staging:
                ids = self.input_control[:count]
                if self.use_dspark:
                    self.positions[:count].copy_(positions[:count])
                    positions = self.position_views.get(count, self.positions[:count])
                else:
                    positions = positions[:count]
            else:
                self.input_ids[:count].copy_(self.input_control[:count])
                self.positions[:count].copy_(positions[:count])
                ids = self.input_views.get(count, self.input_ids[:count])
                positions = self.position_views.get(count, self.positions[:count])
        else:
            if (not self.use_dspark and decode and count == 1 and self._next_input is not None
                    and self._next_input[:2] == (request_id, start)):
                if self.direct_token_ids:
                    ids = self._next_input[2]
                else:
                    ids.copy_(self._next_input[2])
            else:
                ids.copy_(torch.tensor(tokens, dtype=ids.dtype, device="cpu"))
            if self.position_bank is not None and graph_c1:
                positions = self.position_bank.view(start, count)
            else:
                positions.copy_(torch.arange(start, start + count, dtype=torch.int32, device="cpu"))
        self._round_phase("inputs_staged_ns")
        # Decode always uses a bucket-specific native plan, including paged
        # CSA2 buckets above 1024. Prompt C1 capture remains bounded to the
        # qualified prefix range so a one-token prefill tail cannot create a
        # long-search graph variant. StageReplay keys plans by search bucket.
        use_replay = (getattr(self.model, "native", False) and
                      (decode or (start + count <= 1024 and (c1_replay or (self.use_dspark and request is not None)))))
        self.model.prepare_step(request_id, tokens, is_decode=graph_c1, reset=reset, use_replay=use_replay)
        self._round_phase("engram_prepared_ns")
        timing = getattr(self, "verify_timing", None)
        if self.pp.group.is_first_rank:
            embeddings = self._image_embeddings(request, start, count) if request is not None else None
            if timing:
                timing.device("stage_model_start")
            value = self.model(ids, positions, inputs_embeds=embeddings)
            self._round_phase("stage_submitted_ns")
            if timing:
                timing.device("stage_target_done")
            self.pp.exchange(value, count, decode=graph_c1)
            self._round_phase("pp_exchanged_ns")
            output = None
        else:
            value = self.pp.exchange(None, count, native_wire=use_replay and self.use_dspark, decode=graph_c1)
            self._round_phase("pp_exchanged_ns")
            if timing:
                timing.device("stage_model_start")
            output = self.model(ids, positions, intermediate_tensors=value)
            self._round_phase("stage_submitted_ns")
            if timing:
                timing.device("stage_target_done")
        if timing:
            timing.host("target_submit_done")
        self.audit["target_steps"] += 1
        self.audit["target_tokens"] += count
        if request is not None:
            self.audit["decode_steps" if decode else "prefill_steps"] += 1
        return output

    @profile_phase("insert_context")
    def _insert(self, aux, positions):
        if envs.VLLM_HPU_DSV41_DSPARK and self.pp.group.is_last_rank:
            self.insert_context(aux, positions)

    @profile_phase("draft_propose")
    def _propose(self, anchor, position, *, diagnostic=False):
        first = torch.tensor([anchor], device=self.device, dtype=torch.int64)
        positions = torch.arange(position, position + 5, device=self.device, dtype=torch.int32)
        hidden, logits = self.run_draft(first, positions)
        if diagnostic:
            logger.info("V4.1 PP%d draft forward submitted", self.model.pp_rank)
        token_ids, confidence = self.sample_draft(first, hidden, logits)
        self.last_draft_confidence = confidence
        self.audit["draft_steps"] += 1
        if diagnostic:
            logger.info("V4.1 PP%d draft sampling submitted; waiting for host tokens", self.model.pp_rank)
        result = token_ids.cpu().tolist()
        if diagnostic:
            logger.info("V4.1 PP%d draft host tokens complete", self.model.pp_rank)
        return result

    def _use_direct_verify_inputs(self, hidden, count):
        return (count == 6 and hidden.shape == self.verify_hidden.shape and self.model.last_aux is not None
                and self.model.last_aux.shape[0] == 6 and self.positions.shape[0] >= 6)

    def _finish_request_device(self, request, start, count, last_count, proposed, target_hidden):
        """Run one fixed C6 verify transaction and defer its sole host read."""
        started = time.perf_counter_ns()
        timing = getattr(self, "verify_timing", None)
        if timing:
            timing.host("verify_start", started)
        phase_ms = {}
        ticket = None
        if self.pp.group.is_last_rank:
            if (self.verify_ring is None or self.verify_prefix is None or self.draft_from_prefix is None):
                raise RuntimeError("Device DSpark verify was not warmed and compiled")
            ticket = self.verify_ring.acquire(last_count, generation=self.pp.generation + 1)
            # A previous PP broadcast may still be consuming the alternate
            # ring slot.  Waiting here (before this graph writes the slot) lets
            # the current target/verify work overlap that transfer instead of
            # forcing every token to drain it at the stage boundary.
            self.pp.wait_record(ticket.record)
            if target_hidden is None or target_hidden.ndim != 2:
                raise RuntimeError("DSpark verify requires final target hidden states on PP1")
            if target_hidden.shape[0] > 6:
                raise RuntimeError("DSpark verify target bucket exceeds C6")
            # Keep all six rows and all five proposal ids at stable addresses.
            if self.verify_hidden.shape[1] != target_hidden.shape[1]:
                raise RuntimeError("DSpark target hidden width changed after preparation")
            # Native replay already returns the C6 output and auxiliary state
            # into persistent bindings.  Consume those addresses directly in
            # the common C6 case; the staging buffers remain for C1/tail
            # shapes and for non-replay producers whose addresses are not
            # guaranteed to survive the control graph.
            direct_c6 = self._use_direct_verify_inputs(target_hidden, last_count)
            verify_hidden = target_hidden if direct_c6 else self.verify_hidden
            verify_aux = self.model.last_aux if direct_c6 else self.verify_aux
            verify_positions = self.positions if direct_c6 else self.verify_positions
            if not direct_c6:
                self.verify_hidden.zero_()
                self.verify_hidden[:target_hidden.shape[0]].copy_(target_hidden)
            # Fill the pinned control staging in one host copy.  Indexed
            # scalar writes used to emit an individual aten::to/copy_ pair;
            # the first of those writes could wait behind the active HPU
            # pipeline for tens of milliseconds.  A single fixed-size CPU
            # tensor write keeps the graph input stable and removes that
            # hidden synchronization from the verify transaction.
            values = [
                ticket.generation,
                last_count,
                len(proposed),
                request.sampling_params.max_tokens - len(request.output),
                start + count - last_count,
                self.model_config.max_model_len,
                1,
                *proposed,
            ]
            values += [-1] * (VERIFY_CONTROL_SIZE - len(values))
            self.verify_control_host.copy_(torch.tensor(values, dtype=torch.int64, device="cpu"))
            metadata = self.verify_metadata
            # Submit metadata and proposal ids together.  They are both
            # immutable inputs for this invocation and their views retain the
            # addresses captured by the verify graph.
            self.verify_control.copy_(self.verify_control_host, non_blocking=True)
            if self.model.last_aux is None or self.model.last_aux.shape[0] < last_count:
                raise RuntimeError("DSpark verify is missing target auxiliary states")
            if not direct_c6 and self.verify_aux.shape[1] != self.model.last_aux.shape[1]:
                self.verify_aux = torch.empty((6, self.model.last_aux.shape[1]),
                                              dtype=self.model.last_aux.dtype,
                                              device=self.device)
            if not direct_c6:
                self.verify_aux.zero_()
                self.verify_aux[:last_count].copy_(self.model.last_aux[:last_count])
                self.verify_positions.copy_(self.positions[0].to(torch.int32) + self.verify_offsets)
            if timing:
                timing.host("prefix_submit_start")
                timing.device("prefix_start")
            (_, prefix_output, prefix_committed, prefix_output_count, anchor, draft_enabled, status, commit_record,
             commit_wire) = self.verify_prefix(verify_hidden, self.verify_proposed, metadata, verify_aux,
                                               verify_positions)
            if timing:
                timing.device("prefix_done")
                timing.host("prefix_submit_done")
            # Snapshot only the commit-visible fields before the draft graph.
            # The ring record remains writable so draft ids can be published
            # after the PP exchange without racing its source buffer.
            ticket.record.copy_(commit_record)
            if ticket.wire is not None:
                ticket.wire.copy_(commit_wire)
            if ticket.commit_record is not None:
                ticket.commit_record.copy_(commit_record)
            if ticket.commit_wire is not None:
                ticket.commit_wire.copy_(commit_wire)
            if timing:
                timing.device("commit_source_ready")
        phase_started = time.perf_counter_ns()
        commit_record = None if ticket is None else (
            ticket.commit_record if ticket.commit_record is not None else ticket.record)
        commit_wire = None if ticket is None else (
            ticket.commit_wire if ticket.commit_wire is not None else ticket.wire)
        pp_result = self.pp.finish_device(commit_record, last_count, commit_wire)
        phase_ms["pp_finish_device_ms"] = (time.perf_counter_ns() - phase_started) / 1e6
        if self.pp.group.is_last_rank:
            # Continue draft control on the normal compute stream only after
            # the commit source has been snapshotted. PP0 can now validate the
            # prefix while these three layers execute on PP1.
            record, wire_record, confidence = self.draft_from_prefix(metadata, verify_positions, prefix_output,
                                                                     prefix_committed, prefix_output_count, anchor,
                                                                     draft_enabled, status)
            ticket.record.copy_(record)
            if ticket.wire is not None:
                ticket.wire.copy_(wire_record)
            self.last_draft_confidence = confidence
            if timing:
                timing.device("draft_done")
        if self.pp.group.is_first_rank and envs.VLLM_HPU_DSV41_INLINE_PP_COMMIT:
            # Engram receives the validated count; PP0 also retains final token
            # ids because its scheduler cache does not resend that full prefix.
            # PP0 has no scheduler-visible sampled output.  Consume its small
            # control result before returning from this worker, so the async
            # executor handoff cannot add an unowned 100+ ms gap to the C6
            # verify transaction.  The only device wait remains the final
            # pinned scalar read in consume_device_commit.
            phase_started = time.perf_counter_ns()
            committed_value, output = self.pp.consume_device_commit(pp_result)
            phase_ms["pp_consume_device_ms"] = (time.perf_counter_ns() - phase_started) / 1e6
            if timing:
                timing.host("engram_complete_start")
            self.model.complete_step_device(committed_value)
            request.output.extend(output)
            if timing:
                timing.host("engram_complete_done")
                timing.finish(committed_value, len(output))
            self.verify_records.setdefault(request.req_id, []).append({
                "target_count":
                last_count,
                "proposed_count":
                len(proposed),
                "committed":
                committed_value,
                "output_count":
                len(output),
                "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                **phase_ms, "inline_pp_commit":
                True
            })
            self._record_round_completion(request, proposed, committed_value, output)
            self.pending = None
            return None
        if self.pp.group.is_first_rank:
            # Compatibility mode retains the asynchronous handoff until a
            # candidate proves the inline PP0 consume is safe and faster.
            def consume_pp(result):
                phase_started = time.perf_counter_ns()
                committed, output = self.pp.consume_device_commit(result)
                phase_ms["pp_consume_device_ms"] = (time.perf_counter_ns() - phase_started) / 1e6
                self.model.complete_step_device(committed)
                request.output.extend(output)
                if timing:
                    timing.finish(committed, len(output))
                self.verify_records.setdefault(request.req_id, []).append({
                    "target_count":
                    last_count,
                    "proposed_count":
                    len(proposed),
                    "committed":
                    committed,
                    "output_count":
                    len(output),
                    "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                    **phase_ms
                })
                self._record_round_completion(request, proposed, committed, output)
                self.pending = None
                return None

            return AsyncDeviceOutput(pp_result, consume_pp)

        assert ticket is not None
        self.verify_ring.stage(ticket)

        def consume(committed_value, output, draft):
            if timing:
                timing.host("final_host_consume")
            accepted = committed_value - 1 if proposed else 0
            self.audit["accepted_drafts"] += accepted
            self.audit["rejected_drafts"] += len(proposed) - accepted
            self.model.complete_step_device(committed_value)
            request.output.extend(output)
            self.draft_token_ids = DraftTokenIds([request.req_id], [draft])
            if timing:
                timing.finish(committed_value, len(output))
            self.pending = None
            result = ModelRunnerOutput(req_ids=[request.req_id],
                                       req_id_to_index={request.req_id: 0},
                                       sampled_token_ids=[output])
            self.verify_records.setdefault(request.req_id, []).append({
                "target_count":
                last_count,
                "proposed_count":
                len(proposed),
                "committed":
                committed_value,
                "output_count":
                len(output),
                "draft_count":
                len(draft),
                "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                **phase_ms
            })
            result.execution_rounds = self._record_round_completion(request, proposed, committed_value, output)
            return result

        return AsyncDSparkOutput(self.verify_ring, ticket, consume,
                                 self._record_round_released if self.round_timing_enabled else None)

    @torch.inference_mode()
    def execute_model(self, scheduled):
        if self.pending is not None:
            raise RuntimeError("Previous V4.1 execution has not completed sampling/verify")
        self._update(scheduled)
        if not scheduled.num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT
        outputs, drafts, ids, async_result = [], [], [], None
        execution_rounds = []
        batch_size = len(scheduled.num_scheduled_tokens)
        for req_id, count in scheduled.num_scheduled_tokens.items():
            self._execute_request(scheduled, req_id, count)
            result = self._finish_request()
            ids.append(req_id)
            if isinstance(result, AsyncModelRunnerOutput) and batch_size > 1:
                # Request switches must consume the previous Engram generation
                # before activating another state block.  The single-request
                # path lets the executor perform this final consumption.
                result = result.get_output()
            if isinstance(result, AsyncModelRunnerOutput):
                if async_result is not None:
                    raise RuntimeError("Device DSpark verify currently allows one in-flight request")
                async_result = result
            elif result is not None:
                outputs.extend(result.sampled_token_ids)
                execution_rounds.extend(getattr(result, "execution_rounds", None) or ())
                if self.draft_token_ids is not None:
                    drafts.extend(self.draft_token_ids.draft_token_ids)
        if async_result is None and self.pp.group.is_last_rank:
            self.draft_token_ids = DraftTokenIds(ids, drafts)
        if async_result is not None:
            self.batch_result = async_result
        elif self.pp.group.is_last_rank:
            self.batch_result = ModelRunnerOutput(req_ids=ids,
                                                  req_id_to_index={
                                                      key: index
                                                      for index, key in enumerate(ids)
                                                  },
                                                  sampled_token_ids=outputs)
            if execution_rounds:
                self.batch_result.execution_rounds = execution_rounds
        else:
            self.batch_result = None
        self.pending = "batch_ready"
        return None

    def _execute_request(self, scheduled, req_id, count):
        if self.round_timing_enabled:
            self.round_context = dict(request_id=req_id,
                                      generation=self.pp.generation + 1,
                                      target_count=count,
                                      start_ns=time.perf_counter_ns())
        request = self.requests[req_id]
        self._bind_request(request)
        start = request.num_computed_tokens
        proposed = scheduled.scheduled_spec_decode_tokens.get(req_id, [])
        if proposed and not self.use_dspark:
            raise RuntimeError("Speculative tokens reached a non-speculative V4.1 runner")
        tokens = request.tokens[start:start + count - len(proposed)] + proposed
        if len(tokens) != count or start + count > self.model_config.max_model_len:
            raise RuntimeError("Scheduled V4.1 inputs do not match the committed prefix and context budget")
        decode = start >= len(request.prompt)
        valid_decode = 1 <= count <= 6 if self.use_dspark else count == 1
        if decode and not valid_decode:
            raise RuntimeError(f"Unexpected V4.1 decode shape: tokens={count}, drafts={len(proposed)}")
        if self.verify_timing:
            self.verify_timing.begin(req_id, self.pp.generation + 1, count, len(proposed))
        # Match vLLM chunked-prefill semantics: one scheduler transaction is a
        # real large-M model invocation up to max_num_batched_tokens. Internal
        # C1/C6 decode tiling would reread expert weights for every prompt row.
        chunks = [(0, tokens)] if decode else target_chunks(tokens, PREFILL_BLOCK_TOKENS)
        program = getattr(self.model, "program", None)
        transaction_search = (target_search_length(start, count, program.length)
                              if not decode and program is not None and program.length > 512 else None)
        for block_index, (offset, chunk) in enumerate(chunks):
            hidden = self._forward(req_id,
                                   chunk,
                                   start + offset,
                                   decode=decode,
                                   reset=start + offset == 0,
                                   request=request,
                                   search_length=transaction_search)
            if offset + len(chunk) < count:
                self._insert(self.model.last_aux, self.positions[:len(chunk)])
                self.model.complete_step(len(chunk))
                # More than one scheduler transaction may only remain in
                # flight after its state and PP generation have committed.
                if (block_index + 1) % PREFILL_MAX_INFLIGHT_BLOCKS == 0:
                    self.pp.drain()
                    torch.hpu.synchronize()
        need_sample = start + count >= len(request.tokens)
        if self.pp.group.is_last_rank and need_sample:
            if not self.use_dspark:
                if decode and self.pp.device_commit_enabled:
                    sample_input = self.sample_target_commit(hidden[-1:], self.pp.commit)
                else:
                    sample_input = self.sample_target(hidden[-1:])
                if self.model.native and not (decode and (self.pp.device_commit_enabled or self.v2_completion)):
                    from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime
                    bridge, _, _ = _resolve_runtime()
                    self._token_copy = bridge.copy_sampled_tokens_to_host(sample_input)
            else:
                # Device verification owns projection, argmax and the small
                # TP candidate exchange, so it consumes target hidden state.
                sample_input = (hidden if envs.VLLM_HPU_DSV41_DEVICE_VERIFY else self.model.compute_logits(hidden))
        else:
            sample_input = None
        self.pending = (request, start, count, len(chunk), proposed, need_sample, sample_input)
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output=None):
        if grammar_output is not None:
            raise ValueError("V4.1 prepared greedy verification does not support grammar sampling")
        if self.pending == "batch_ready":
            result, self.batch_result = self.batch_result, None
            if isinstance(result, AsyncModelRunnerOutput):
                # The executor owns this HPU event until get_output().  Do not
                # clear pending before its callback commits request state.
                # Multiprocess execution serializes only PP1/TP0's output
                # rank.  PP0 never serializes a result at all, so it must
                # consume its local commit here as well; otherwise its
                # Engram ticket and pending request would remain in-flight.
                if (not self.pp.group.is_last_rank or getattr(self.model, "tp_rank", 0) != 0):
                    return result.get_output()
                return result
            self.pending = None
            return result
        return self._finish_request()

    @torch.inference_mode()
    @profile_phase("verify_and_commit")
    def _finish_request(self):
        if self.pending is None:
            return EMPTY_MODEL_RUNNER_OUTPUT
        request, start, count, last_count, proposed, need_sample, sample_input = self.pending
        if not self.use_dspark:
            return self._sample_single()
        if (envs.VLLM_HPU_DSV41_DEVICE_VERIFY and envs.VLLM_HPU_DSV41_DSPARK and need_sample
                and (self.verify_prefix is not None or self.pp.group.is_first_rank)):
            return self._finish_request_device(request, start, count, last_count, proposed, sample_input)
        output, draft, committed = [], [], last_count
        if self.pp.group.is_last_rank:
            if need_sample:
                target = sample_input.argmax(-1).cpu().tolist()
                if proposed:
                    output, accepted = greedy_verify(target, proposed)
                    committed = accepted + 1
                    self.audit["accepted_drafts"] += accepted
                    self.audit["rejected_drafts"] += len(proposed) - accepted
                else:
                    output = target[-1:]
            self._insert(self.model.last_aux[:committed], self.positions[:committed])
            next_position = start + count - last_count + committed
            remaining = request.sampling_params.max_tokens - len(request.output) - len(output)
            if (output and envs.VLLM_HPU_DSV41_DSPARK and remaining >= 6
                    and next_position + 6 <= self.model_config.max_model_len):
                draft = self._propose(output[-1], next_position)
        committed, output, draft = self.pp.finish(committed, output, draft)
        if not 1 <= committed <= last_count:
            raise RuntimeError("PP accepted prefix exceeds the target's pending input transaction")
        self.model.complete_step(committed)
        request.output.extend(output)
        self.draft_token_ids = DraftTokenIds([request.req_id], [draft])
        self.pending = None
        if not self.pp.group.is_last_rank:
            return None
        return ModelRunnerOutput(req_ids=[request.req_id],
                                 req_id_to_index={request.req_id: 0},
                                 sampled_token_ids=[output])

    def _sample_single(self):
        request, start, count, last_count, proposed, need_sample, selected = self.pending
        if proposed:
            raise RuntimeError("Ordinary sampling cannot consume a draft prefix")
        device_commit_enabled = getattr(self.pp, "device_commit_enabled", getattr(self.pp, "device_commit", False))
        device_commit = (device_commit_enabled and need_sample and start >= len(request.prompt))
        token = None
        if self.pp.group.is_last_rank and need_sample and not device_commit:
            if self._token_copy is not None:
                host, done = self._token_copy
                done.synchronize()
                token = int(host[0, 0])
            else:
                token = int(selected.cpu()[0, 0])
        consumed, output = (self.pp.finish_single_device() if device_commit else self.pp.finish_single(
            last_count, token))
        if consumed != last_count:
            raise RuntimeError("Ordinary PP completion did not consume the complete input chunk")
        self.model.complete_step(consumed)
        request.output.extend(output)
        # CPU completion publishes commit_host only. Its device commit tensor
        # is uninitialized (or belongs to an older generation); binding that
        # tensor here feeds stale IDs into otherwise correct native replay.
        # Only device completion owns a current device token. CPU completion
        # uses the scheduler/request token through the normal input staging.
        self._next_input = ((request.req_id, start + count, self.pp.commit_token) if output and device_commit else None)
        self._token_copy = None
        self.pending = self.draft_token_ids = None
        if not self.pp.group.is_last_rank:
            return None
        return ModelRunnerOutput(req_ids=[request.req_id],
                                 req_id_to_index={request.req_id: 0},
                                 sampled_token_ids=[output])

    def take_draft_token_ids(self):
        value, self.draft_token_ids = self.draft_token_ids, None
        return value if self.pp.group.is_last_rank else None

    @torch.inference_mode()
    def _dummy_run(self, tokens, *, native=False, start_position=0):
        logger.info("V4.1 PP%d C%d warmup target start (native=%s, preceding steps=%d)", self.model.pp_rank, tokens,
                    bool(native), self.audit["target_steps"])
        self.state.clear()
        hidden = self._forward("__v41_warmup__", [1 + index for index in range(tokens)],
                               start_position,
                               decode=native and (self.use_dspark or tokens == 1),
                               reset=True)
        from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats
        stats = prepared_group_stats()
        logger.info("V4.1 PP%d target submitted (native graphs=%d, replays=%d, entries=%d)", self.model.pp_rank,
                    stats["native_graphs"], stats["native_replays"], stats["native_entry_replays"])
        if self.pp.group.is_last_rank:
            if (envs.VLLM_HPU_DSV41_DEVICE_VERIFY and self.verify_prefix is not None and envs.VLLM_HPU_DSV41_DSPARK):
                # Compile and exercise the fixed control graph during warmup;
                # this invocation is discarded with the warmup state.
                self.verify_hidden.zero_()
                self.verify_hidden[:tokens].copy_(hidden)
                if self.verify_aux.shape[1] != self.model.last_aux.shape[1]:
                    self.verify_aux = torch.empty((6, self.model.last_aux.shape[1]),
                                                  dtype=self.model.last_aux.dtype,
                                                  device=self.device)
                self.verify_aux.zero_()
                self.verify_aux[:tokens].copy_(self.model.last_aux[:tokens])
                self.verify_positions.copy_(self.positions[0].to(torch.int32) + self.verify_offsets)
                self.verify_proposed.fill_(-1)
                self.verify_control_host.fill_(0)
                self.verify_metadata_host.copy_(
                    torch.as_tensor((1, tokens, 0, 64, start_position, self.model_config.max_model_len, 1),
                                    dtype=torch.int64))
                self.verify_control.copy_(self.verify_control_host, non_blocking=True)
                # Exercise the same producer tensors as serving. Padded
                # buffers have different alias/inference contracts from
                # native C6 outputs and do not warm the direct control path.
                direct_c6 = self._use_direct_verify_inputs(hidden, tokens)
                verify_hidden = hidden if direct_c6 else self.verify_hidden
                verify_aux = self.model.last_aux if direct_c6 else self.verify_aux
                verify_positions = self.positions if direct_c6 else self.verify_positions
                prefix = self.verify_prefix(verify_hidden, self.verify_proposed, self.verify_metadata, verify_aux,
                                            verify_positions)
                self.draft_from_prefix(self.verify_metadata, verify_positions, prefix[1], prefix[2], prefix[3],
                                       prefix[4], prefix[5], prefix[6])
            else:
                if self.use_dspark:
                    self.model.compute_logits(hidden)
                    self._insert(self.model.last_aux, self.positions[:tokens])
                    self._propose(1, start_position + tokens, diagnostic=True)
                else:
                    self.sample_target(hidden[-1:])
        self.pp.drain()
        self.model.complete_step(tokens)
        torch.hpu.synchronize()
        self.pp.complete_packet()
        # Capture iterations must finish on both stages before either stage
        # starts another iteration. The real request path has a verify commit;
        # warmup uses the CPU group so it does not queue an unmatched HPU receive.
        self.pp.group.barrier()
        logger.info("V4.1 PP%d C%d warmup complete", self.model.pp_rank, tokens)

    def profile_run(self, initialize_only=False):
        del initialize_only
        tokens = 6 if self.use_dspark else 1
        logger.info("V4.1 PP%d starting C%d memory profile", self.model.pp_rank, tokens)
        self._dummy_run(tokens)
        logger.info("V4.1 PP%d completed C%d memory profile", self.model.pp_rank, tokens)

    @torch.inference_mode()
    def warmup_model(self):
        # Ordinary serving keeps the qualified C1 native replay for decode.
        # C6 is a DSpark-only anchor-plus-draft geometry. Prompt C128 recipes
        # are compiled by real prefill qualification and persisted in cache.
        for count in ((1, 6) if self.use_dspark else (1, )):
            for _ in range(4 if self.model.native else 1):
                self._dummy_run(count, native=self.model.native)
            if self.model.native:
                self.model.program.replay_owner.require_ready(count)
            self.graphed_buckets.add(count)
        if self.use_dspark and isinstance(self.state, PagedStageState):
            # Exercise the first real indexer/head exchange before advertising
            # API readiness. This is one boundary warmup, not a length sweep.
            for _ in range(4 if self.model.native else 1):
                self._dummy_run(6, native=self.model.native, start_position=512)
            if self.model.native:
                self.model.program.replay_owner.require_ready(6, search=1024)
        if envs.VLLM_HPU_DSV41_DSPARK and self.pp.group.is_last_rank:
            # A verify may commit any prefix, not just the C1/C6 target buckets.
            # Compile those real shapes before serving; the common state clear
            # below also clears the draft KV writes produced by this warmup.
            for count in (2, 3, 4, 5):
                self._insert(self.model.last_aux[:count], self.positions[:count])
            torch.hpu.synchronize()
        self.pp.group.barrier()
        self.state.clear()
        self.active_request = None
        if envs.VLLM_HPU_DSV41_VERIFY_TIMING:
            if (not envs.VLLM_HPU_DSV41_DEVICE_VERIFY or not envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE
                    or not envs.VLLM_HPU_DSV41_INLINE_PP_COMMIT or envs.VLLM_HPU_DSV41_COMPILED_PP_COMMIT):
                raise RuntimeError("Verify timing requires the qualified eager direct commit path")
            from vllm_gaudi.ops.deepseek_v41_verify_timing import VerifyPhaseTiming
            from pathlib import Path
            self.verify_timing = VerifyPhaseTiming(
                Path(os.environ["DSV41_RUN_EVIDENCE"]) / "verify-phases", self.model.pp_rank * 2 + self.model.tp_rank)
            self.pp.timing = self.verify_timing
            self.verify_timing.calibrate()

    def close(self):
        if isinstance(getattr(self, "batch_result", None), AsyncModelRunnerOutput):
            self.batch_result.get_output()
        self.pp.drain()
        if self.verify_ring is not None:
            self.verify_ring.close()
        if self.verify_timing:
            self.verify_timing.flush()
        try:
            self._flush_round_timing()
        finally:
            if self.model is not None:
                self.model.close()

    shutdown_inc = close

    def _flush_round_timing(self):
        if not self.round_timing_enabled:
            return
        directory = Path(os.environ["DSV41_RUN_EVIDENCE"]) / "round-timing"
        directory.mkdir(exist_ok=True)
        rank = self.model.pp_rank * 2 + self.model.tp_rank
        (directory / f"rank{rank}.json"
         ).write_text(json.dumps({
             "rank": rank,
             "clock": "perf_counter_ns",
             "records": self.round_records
         }) + "\n")

    def _round_phase(self, name):
        if getattr(self, "round_context", None) is not None:
            self.round_context[name] = time.perf_counter_ns()

    def _record_round_released(self, result):
        keys = result.execution_rounds
        row = self.round_records[-1]
        if keys != [(row["request_id"], row["generation"], row["target_count"])]:
            raise RuntimeError("Released DSpark ring does not own the completed round")
        row["end_ns"] = time.perf_counter_ns()
        row["ring_released"] = True

    def _record_round_completion(self, request, proposed, committed, output):
        if not self.round_timing_enabled:
            return None
        context = self.round_context
        if context is None or context["request_id"] != request.req_id or context["generation"] != self.pp.generation:
            raise RuntimeError("V4.1 round completion does not match its input generation")
        if len(self.round_records) >= 65536:
            raise RuntimeError("V4.1 round timing capacity exceeded")
        row = dict(context,
                   end_ns=time.perf_counter_ns(),
                   proposed_count=len(proposed),
                   committed=committed,
                   output_count=len(output),
                   output=list(output))
        self.round_records.append(row)
        self.round_context = None
        return [(row["request_id"], row["generation"], row["target_count"])]
