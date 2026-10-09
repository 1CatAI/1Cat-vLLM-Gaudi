# SPDX-License-Identifier: Apache-2.0
"""Bounded V4.1 runner with scheduler-owned CSA2 state and PP verify commits.

The token/accepted-prefix contract follows vLLM #53577 (d2b1b735).
Transport uses persistent HPU buffers and HCCL, with one outstanding request.
"""

from dataclasses import dataclass, field
from contextlib import nullcontext
from functools import partial, wraps
from pathlib import Path
from types import FunctionType, MethodType
import json
import os
import time

import torch
import torch.distributed as dist

from vllm.distributed import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from vllm_gaudi import envs
from vllm_gaudi.extension.logger import logger as init_logger
from vllm_gaudi.extension.profiler import HabanaHighLevelProfiler
from vllm_gaudi.ops.deepseek_v41_config import decode_source_prefix_bound, select_native_warmup_geometries
from vllm_gaudi.ops.deepseek_v41_indexer import INDEX_MME_HOT_TOKENS
from vllm_gaudi.ops.deepseek_v41_diagnostics import trace_phase
from vllm_gaudi.ops.deepseek_v41_state import PagedStageState, StageStateBlocks, register_state_spec
from vllm_gaudi.ops.deepseek_v41_verify import (
    AsyncDeviceOutput,
    AsyncDSparkOutput,
    DevicePPCommit,
    RECORD_SIZE,
    STATUS,
    VerifyRing,
    validate_commit,
    validate_commit_wire,
)

from vllm_gaudi.ops.deepseek_v41_prefill_capacity import (
    DEFAULT_PREFILL_TOKENS,
    MAX_PREFILL_TOKENS,
    PREFILL_COMPUTE_BUCKETS,  # noqa: F401 - compatibility export for existing callers.
    prefill_capacity,
    prefill_compute_buckets,
)

logger = init_logger()

VERIFY_METADATA_SIZE = 7
VERIFY_PROPOSED_SIZE = 5
VERIFY_CONTROL_SIZE = VERIFY_METADATA_SIZE + VERIFY_PROPOSED_SIZE
PREFILL_BLOCK_TOKENS = DEFAULT_PREFILL_TOKENS


class DraftContextPlans:
    """Keep prompt and accepted-prefix shapes in separate compile entries."""

    def __init__(self, insert_context):
        self.insert_context = insert_context
        self.plans = {}

    def __call__(self, aux, positions):
        signature = tuple(
            (tuple(value.shape), tuple(value.stride()), value.dtype, value.device) for value in (aux, positions))
        plan = self.plans.get(signature)
        if plan is None:
            function = self.insert_context.__func__
            entry = FunctionType(function.__code__.replace(co_name=f"draft_insert_{len(self.plans)}"),
                                 function.__globals__, function.__name__, function.__defaults__, function.__closure__)
            entry.__kwdefaults__ = function.__kwdefaults__
            bound = MethodType(entry, self.insert_context.__self__)
            plan = torch.compile(bound, backend="hpu_backend", fullgraph=True, dynamic=False)
            self.plans[signature] = plan
        return plan(aux, positions)


# Scheduler admission and device execution have different jobs.  vLLM may
# admit a complete C8192 transaction, while the model executes it with this
# finite set of exact (unpadded) shapes.  Powers down to C128 keep large-M
# weight reuse; a sub-C128 tail reuses the already qualified C1 replay.  This
# prevents request lengths from creating new Synapse recipes after the 1M KV
# pool is resident without introducing padding writes into KV/Engram state.
# TP4 large-M prefill uses the prepared N256 hybrid expert plans. Expert
# workspaces remain bounded while the route/output contract covers the whole
# configured chunk. C6 decode and C1 sampling retain their existing contracts.
PREFILL_WAVEFRONT_SLOTS = 2


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
            elif name == "verify_and_commit" and self.pending is not None:
                request, start, count, _, _, need_sample, _ = self.pending
                phase = "decode" if start >= len(request.prompt) else "prefill"
                label += f"::{phase}::P{start}::C{count}::emit{int(need_sample)}"
            elif name == "insert_context":
                label += f"::C{args[1].numel()}"
            if os.getenv("VLLM_HPU_DSV41_RAW_TRACE", "0") == "1":
                from vllm_gaudi.ops.deepseek_v41_native_trace import scope

                with scope(label):
                    return function(self, *args, **kwargs)
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
    recompute_until: int = 0
    speculative_sampling: object | None = None

    @property
    def tokens(self):
        return self.prompt + self.output

    @property
    def token_count(self):
        return len(self.prompt) + len(self.output)

    def token_at(self, position):
        """Read one committed input without copying the complete prefix."""
        if position < 0:
            position += self.token_count
        if not 0 <= position < self.token_count:
            raise IndexError("Request token position is outside its committed prefix")
        prompt_count = len(self.prompt)
        return self.prompt[position] if position < prompt_count else self.output[position - prompt_count]

    @property
    def decode_start(self):
        # A preempted request must replay its generated prefix as prefill too.
        return max(len(self.prompt), self.recompute_until)
    def token_slice(self, start, stop):
        """Select the scheduled prefix without materializing the full history."""
        if not 0 <= start <= stop:
            raise ValueError("Scheduled token range must be nonnegative and ordered")
        boundary = len(self.prompt)
        if start >= boundary:
            return self.output[start - boundary : stop - boundary]
        if stop <= boundary:
            return self.prompt[start:stop]
        return self.prompt[start:] + self.output[: stop - boundary]

    def reconcile(self, output_count, new_tokens, all_tokens=None):
        if all_tokens is not None:
            self.output = list(all_tokens[len(self.prompt) : len(self.prompt) + output_count])
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
    return target_ids[: accepted + 1], accepted


def target_chunks(tokens, block_tokens=PREFILL_BLOCK_TOKENS):
    """Cover a scheduler prefill transaction with bounded device blocks.

    ``max_num_batched_tokens`` remains the scheduler admission limit.  This
    internal tiling only bounds the per-stage working set; C6 is reserved for
    DSpark anchor-plus-draft execution and is never used as an ordinary
    prefill geometry.
    """
    if block_tokens < 1 or block_tokens > MAX_PREFILL_TOKENS:
        raise ValueError("V4.1 prefill blocks must be no larger than 16384 tokens")
    buckets = tuple(size for size in prefill_compute_buckets(block_tokens) if size <= block_tokens)
    offset = 0
    while offset < len(tokens):
        remaining = len(tokens) - offset
        size = next((candidate for candidate in buckets if candidate <= remaining), 1)
        yield offset, tokens[offset : offset + size]
        offset += size


def target_search_length(start, count, maximum):
    """Choose one CSA2 search bucket for a complete scheduler transaction."""
    if start < 0 or count < 1 or start + count > maximum:
        raise ValueError("V4.1 search bucket is outside the configured context")
    return min(maximum, max(512, 1 << (start + count - 1).bit_length()))


def runtime_search_length(start, count, maximum):
    """Select a prewarmed search bucket covering the complete visible prefix."""
    if start < 0 or count < 1 or start + count > maximum:
        raise ValueError("V4.1 runtime index range is outside the configured context")
    # Keep the first 512 visible rows on the exact static candidate path.  The
    # runtime indexer is needed for a larger prefix, but paying its score /
    # threshold / emit chain while the static compressed offsets are complete
    # was the main difference between the historical C1 replay and the current
    # 1M-capacity build.  The next geometry is prewarmed below, so crossing
    # this boundary cannot trigger a compile during a request.
    if start + count <= 512:
        return min(maximum, 512)
    hot = min(maximum, INDEX_MME_HOT_TOKENS)
    if start + count <= hot:
        return hot
    # Both TP geometries reuse the bounded derived key cache and MME scorer.
    # Geometric buckets bound visible work and finite replay geometries.
    # Configured-capacity mirrors serve long buckets in native source windows.
    bounded = min(maximum, 32768)
    if start + count <= bounded:
        return bounded
    return min(maximum, 1 << (start + count - 1).bit_length())


def prefill_search_length(start, count, maximum, *, reuse_index_keys=False):
    """Use a bounded shared-key geometry without changing decode capture.

    The hot geometry is shared with decode. Complete prompt geometries avoid
    scoring unused source rows; longer buckets use the same bounded source
    windows. Query positions still mask every future row.
    """
    search = runtime_search_length(start, count, maximum)
    if reuse_index_keys and INDEX_MME_HOT_TOKENS < start + count <= 32768:
        # Complete prompt buckets avoid scanning unused source tiles while
        # decode retains its separately captured geometry. These are work
        # bounds, not admission cutoffs for the shared attention algorithm.
        return min(maximum, 16384 if start + count <= 16384 else 32768)
    if not reuse_index_keys and start + count > INDEX_MME_HOT_TOKENS:
        return maximum
    return search


def decode_search_warmups(maximum, *, runtime_indexer=False):
    """Yield one valid position per decode graph geometry."""
    if runtime_indexer:
        hot = min(maximum, INDEX_MME_HOT_TOKENS)
        yield 0, min(maximum, 512)
        if hot > 512:
            yield 512, hot
        if hot < maximum:
            bounded = min(maximum, 32768)
            # Native recipes specialize their visible rows in 4K buckets.
            # Prepare each finite geometry before accepting decode requests.
            for start in range(hot, bounded, 4096):
                yield start, bounded
            while bounded < maximum:
                search = runtime_search_length(bounded, 1, maximum)
                from vllm_gaudi.ops.deepseek_v41_config import decode_source_window_quantum

                for start in range(bounded, search, decode_source_window_quantum(search)):
                    yield start, search
                bounded = search
        return
    start = 0
    while start < maximum:
        search = target_search_length(start, 1, maximum)
        yield start, search
        start = search


def prefill_search_warmups(maximum, capacity, *, reuse_index_keys=True):
    """Cover reachable finite tile/search pairs, including interior tails.

    A scheduler transaction supplies one search geometry to all its internal
    tiles. Short tiles can therefore execute in any later search bucket;
    warming each tile only at position zero leaves those real tails cold.
    """
    buckets = [512, INDEX_MME_HOT_TOKENS, 16384, 32768]
    while buckets[-1] < maximum:
        buckets.append(buckets[-1] * 2)
    seen = set()
    for search in dict.fromkeys(min(maximum, size) for size in buckets):
        if search < 1:
            continue
        floor = 0 if search <= 512 else 512 if search <= INDEX_MME_HOT_TOKENS else (
            INDEX_MME_HOT_TOKENS if search <= 16384 else search // 2
        )
        for count in prefill_compute_buckets(capacity):
            if count > search:
                continue
            start = max(0, floor + 1 - count)
            actual = prefill_search_length(start, count, maximum, reuse_index_keys=reuse_index_keys)
            if (count, actual) not in seen:
                seen.add((count, actual))
                yield start, count


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
        raise ValueError(
            f"invalid V4.1 PP BF16 exchange wire: dtype={exchange_wire.dtype}, "
            f"shape={tuple(exchange_wire.shape)} != {expected}"
        )
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
        self.single_stage = getattr(self.group, "is_first_rank", False) and getattr(self.group, "is_last_rank", False)
        if self.single_stage:
            # No pipeline peer consumes hidden/pre buffers in TP4 x PP1.
            # Avoid reserving a full prompt-sized transport allocation.
            capacity = 0
        if not self.dspark:
            self.device_commit_enabled = (
                envs.VLLM_HPU_DSV41_DEVICE_COMMIT if device_commit is None else bool(device_commit)
            )
            self.device_commit = self.device_commit_enabled
            if self.device_commit_enabled and not envs.VLLM_HPU_DSV41_GRAPH_REPLAY:
                raise ValueError("Device completion requires ordinary V4.1 native decode")
            self.prefill_wavefront = bool(envs.VLLM_HPU_DSV41_PREFILL_PP_WAVEFRONT)
            slot_count = PREFILL_WAVEFRONT_SLOTS if self.prefill_wavefront else 1
            # Large prompt chunks use a two-slot ring.  PP0 may produce block
            # n+1 while PP1 consumes block n, but a slot cannot be overwritten
            # until its transfer (PP0) or real model consumer (PP1) completes.
            # The ring adds one bounded C8192 boundary allocation (~320 MiB)
            # per rank and does not duplicate weights or KV state.
            self.prefill_hidden = torch.empty(slot_count, capacity, 4, 5120, dtype=torch.bfloat16, device=device)
            self.prefill_pre = torch.empty(slot_count, capacity, 4, dtype=torch.float32, device=device)
            self.hidden = self.prefill_hidden[0]
            self.pre = self.prefill_pre[0]
            self.prefill_slot_count = slot_count
            self.prefill_generation = 0
            self.prefill_active_slot = None
            self.prefill_send_works = [[] for _ in range(slot_count)]
            self.prefill_consumer_events = (
                [torch.hpu.Event() for _ in range(slot_count)] if self.prefill_wavefront else [None]
            )
            self.prefill_consumer_pending = [False] * slot_count
            self.prefill_packed_consumer = None
            self.prefill_packed_consumer_pending = False
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
            if envs.VLLM_HPU_DSV41_NATIVE_PP_COPY and (
                not envs.VLLM_HPU_DSV41_PACKED_PP or not envs.VLLM_HPU_DSV41_GRAPH_REPLAY
            ):
                raise ValueError("Native PP copy requires ordinary packed C1 graph replay")
            if envs.VLLM_HPU_DSV41_PACKED_PP:
                from vllm_gaudi.ops.deepseek_v41_pp import PackedC1Buffers

                self.packed = PackedC1Buffers(device, native_copy=envs.VLLM_HPU_DSV41_NATIVE_PP_COPY)
            return
        if envs.VLLM_HPU_DSV41_MHC_SCHEDULE and not envs.VLLM_HPU_DSV41_GRAPH_REPLAY:
            raise RuntimeError("mHC scheduling requires V4.1 native graph replay")
        if envs.VLLM_HPU_DSV41_DIRECT_PP_WIRE and not (
            envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE and envs.VLLM_HPU_DSV41_GRAPH_REPLAY
        ):
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
        self.exchange_wire = torch.empty(
            (capacity, hidden_elements + pre_elements), dtype=torch.bfloat16, device=device
        )
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
        self.device_validate = (
            torch.compile(validate_commit, backend="hpu_backend", fullgraph=True, dynamic=False)
            if envs.VLLM_HPU_DSV41_DEVICE_VERIFY
            else None
        )
        self.device_validate_wire = (
            torch.compile(validate_commit_wire, backend="hpu_backend", fullgraph=True, dynamic=False)
            if envs.VLLM_HPU_DSV41_DEVICE_VERIFY and envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE
            else None
        )
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
        self.commit_stream = (torch.hpu.Stream() if not self.single_stage and
                              (envs.VLLM_HPU_DSV41_DEVICE_VERIFY and envs.VLLM_HPU_DSV41_INLINE_PP_COMMIT) else None)
        self.commit_record_events = {}
        # PP1 can overlap the next stage-boundary exchange with a prior commit
        # broadcast.  Keep the work handle by source-ring address so a record
        # is waited only when that ring slot is about to be reused.
        self.commit_records = {}
        self.sends, self.receives, self.commits = 0, 0, 0

    def prepare_commit_graph(self):
        """Compile the fixed direct PP commit exchange once per worker."""
        if (self.single_stage or not self.dspark or not envs.VLLM_HPU_DSV41_DEVICE_VERIFY
                or not envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE or not envs.VLLM_HPU_DSV41_COMPILED_PP_COMMIT):
            return
        from vllm_gaudi.ops.deepseek_v41_verify import pp_commit_receive, pp_commit_send

        if self.group.is_last_rank:
            self.commit_send_graph = torch.compile(pp_commit_send, backend="hpu_backend", fullgraph=True, dynamic=False)
        else:
            self.commit_receive_graph = torch.compile(
                pp_commit_receive, backend="hpu_backend", fullgraph=True, dynamic=False
            )

    def drain(self, *, include_commit=True):
        if not self.dspark:
            for slot in range(getattr(self, "prefill_slot_count", 0)):
                self._retire_prefill_send_slot(slot)
            for work in self.pending:
                work.wait()
            self.pending.clear()
            for slot, pending in enumerate(getattr(self, "prefill_consumer_pending", ())):
                if pending:
                    self.prefill_consumer_events[slot].synchronize()
                    self.prefill_consumer_pending[slot] = False
            if getattr(self, "prefill_packed_consumer_pending", False):
                self.prefill_packed_consumer.synchronize()
                self.prefill_packed_consumer_pending = False
            self.prefill_active_slot = None
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
        if self.single_stage:
            raise RuntimeError("Single-stage V4.1 has no pipeline exchange")
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
                if (
                    wire.shape != self.exchange_wire[:count].shape
                    or wire.dtype != torch.bfloat16
                    or not wire.is_contiguous()
                ):
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
                    dist.isend(self.exchange_wire[:count], dst=self.group.ranks[1], group=self.group.device_group)
                )
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
                dist.irecv(self.exchange_wire[:count], src=self.group.ranks[0], group=self.group.device_group)
            )
            self.drain(include_commit=self.group.is_first_rank)
        self.receives += 1
        if native_wire:
            return IntermediateTensors({"pp_wire": self.exchange_wire[:count]})
        _decode_exchange_pre(self.pre_wire[:count], self.pre[:count])
        return IntermediateTensors({"hidden_states": self.hidden[:count], "pre_mix": self.pre[:count]})

    def _exchange_ordinary(self, values, count, *, decode=False):
        if self.packed is not None and decode and count == 1:
            self.drain()
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
        if not self.prefill_wavefront:
            self.drain()
        slot = self.prefill_generation % self.prefill_slot_count
        hidden = self.prefill_hidden[slot]
        pre = self.prefill_pre[slot]
        if self.group.is_first_rank:
            self._retire_prefill_send_slot(slot)
            hidden[:count].copy_(values["hidden_states"])
            pre[:count].copy_(values["pre_mix"])
            works = [
                dist.isend(value, dst=self.group.ranks[1], group=self.group.device_group)
                for value in (hidden[:count], pre[:count])
            ]
            if self.prefill_wavefront:
                self.prefill_send_works[slot] = works
            else:
                self.pending.extend(works)
            self.sends += 2
            self.prefill_generation += 1
            return None
        if self.prefill_consumer_pending[slot]:
            self.prefill_consumer_events[slot].synchronize()
            self.prefill_consumer_pending[slot] = False
        works = [
            dist.irecv(value, src=self.group.ranks[0], group=self.group.device_group)
            for value in (hidden[:count], pre[:count])
        ]
        for work in works:
            work.wait()
        self.receives += 2
        self.prefill_active_slot = slot
        self.prefill_generation += 1
        return IntermediateTensors({"hidden_states": hidden[:count], "pre_mix": pre[:count]})

    def _retire_prefill_send_slot(self, slot):
        works = self.prefill_send_works[slot]
        for work in works:
            work.wait()
        works.clear()

    def mark_prefill_consumed(self):
        """Publish the last real reader of the active PP receive slot."""
        if self.dspark or self.group.is_first_rank:
            return
        if self.packed is not None and self.packed.active is not None:
            # A prompt tail can use C1 packet geometry without sampling a
            # token. Its CPU commit must still wait for the PP1 model reader
            # before releasing that packet or submitting native batch replay.
            if self.prefill_packed_consumer is None:
                self.prefill_packed_consumer = torch.hpu.Event()
            self.prefill_packed_consumer.record(torch.hpu.current_stream())
            self.prefill_packed_consumer_pending = True
            return
        if not self.prefill_wavefront:
            return
        slot = self.prefill_active_slot
        if slot is None:
            return
        self.prefill_consumer_events[slot].record(torch.hpu.current_stream())
        self.prefill_consumer_pending[slot] = True
        self.prefill_active_slot = None

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
        if not self.single_stage:
            self.group.broadcast(self.commit, src=len(self.group.ranks) - 1)
        record = self.commit.cpu().tolist()
        if record[0] != self.generation or not 0 <= record[2] <= 6 or not 0 <= record[3] <= 5:
            raise RuntimeError("Stale or invalid PP verify commit generation")
        self.commit_consumed_generation = self.generation
        self.commits += 1
        return record[1], record[4 : 4 + record[2]], record[10 : 10 + record[3]]

    def complete_packet(self):
        if not self.dspark and self.packed is not None:
            self.packed.complete()

    def finish_single(self, consumed=None, token=None):
        if self.dspark:
            raise RuntimeError("Ordinary token completion cannot commit DSpark verification")
        self.drain()
        self.generation += 1
        if self.group.is_first_rank and self.group.is_last_rank:
            self.commits += 1
            return consumed, [] if token is None else [token]
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
        if self.single_stage:
            if record is None or record.numel() != RECORD_SIZE or record.dtype != self.device_commit.dtype:
                raise RuntimeError("Sampling stage did not produce a complete device verify record")
            self.generation += 1
            self.commits += 1
            return None
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
                if (
                    wire_record is None
                    or wire_record.dtype != torch.bfloat16
                    or wire_record.numel() != RECORD_SIZE * 4
                    or wire_record.device != self.commit_wire.device
                ):
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
        stream_context = torch.hpu.stream(commit_stream) if commit_stream is not None else nullcontext()
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
        if result.generation != self.generation or result.generation == self.commit_consumed_generation:
            raise RuntimeError("Stale PP device verify generation")
        if self.device_validate is None:
            raise RuntimeError("Device verify validator was not compiled")
        commit_stream = getattr(self, "commit_stream", None)
        stream_context = torch.hpu.stream(commit_stream) if commit_stream is not None else nullcontext()
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
        return committed, values[2 : 2 + output_count]


class V41ModelRunner:
    fatal_execution_errors = True
    _PAD_BLOCK_ID, _PAD_SLOT_ID = 0, 0
    v2_completion = False

    def __init__(self, vllm_config, is_driver_worker=False):
        del is_driver_worker
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = vllm_config.device_config.device
        self.requests, self.encoder_cache = {}, {}
        self.model = self.state = None
        self.request_batches = None
        self.prefix_checkpoints = None
        self.kv_caches, self.graphed_buckets = [], set()
        self.pending = self.draft_token_ids = None
        self.use_dspark = envs.VLLM_HPU_DSV41_DSPARK
        self._token_copy = self._next_input = None
        self.tp4_token_readback = None
        self.active_request = None
        self.trace_enabled = False
        # Generic multimodal batching uses the platform's pageable path;
        # Engram owns separate buffers pinned explicitly through the HPU API.
        self.pin_memory = False
        self.profiler = HabanaHighLevelProfiler()
        self.model_memory_usage = self.mem_margin = 0
        self.serving_workspace_reserve = (3 << 30) if self.model_config.max_model_len > 512 else 0
        self.prefill_capacity = prefill_capacity(
            vllm_config.scheduler_config.max_num_batched_tokens, vllm_config.parallel_config.tensor_parallel_size
        )
        self.pp = PPBuffers(
            self.device,
            capacity=self.prefill_capacity,
            dspark=self.use_dspark,
            device_commit=False if self.v2_completion else None,
        )
        self.input_ids = torch.empty(self.prefill_capacity, dtype=torch.int64, device=self.device)
        self.positions = torch.empty(self.prefill_capacity, dtype=torch.int32, device=self.device)
        # Decode/DSpark reuse exact Tensor objects. Large-M prompt buckets are
        # created on demand, avoiding sixteen thousand permanent Python Tensor
        # wrappers merely to represent all possible lengths through C8192.
        self.input_views = {count: self.input_ids[:count] for count in range(1, 129)}
        self.position_views = {count: self.positions[:count] for count in range(1, 129)}
        self.direct_token_ids = envs.VLLM_HPU_DSV41_DIRECT_TOKEN_IDS and not self.use_dspark
        self.decode_ids = torch.empty(1, dtype=torch.int32, device=self.device) if self.direct_token_ids else None
        self.position_bank = None
        if envs.VLLM_HPU_DSV41_FIXED_POSITIONS and not self.use_dspark:
            from vllm_gaudi.ops.deepseek_v41_inputs import PositionBank

            self.position_bank = PositionBank(self.model_config.max_model_len, self.prefill_capacity, self.device)
        self.input_staging = None
        self.tp4_control_inputs = {}
        self._decode_geometry_key = None
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
            self.prepare_positions = torch.compile(
                lambda control: control[6].to(torch.int32) + offsets,
                backend="hpu_backend",
                fullgraph=True,
                dynamic=False,
            )
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
        self.sampled_verify_prefix = None
        self.sampled_draft_from_prefix = None
        self.native_draft_protocol = None
        self.native_draft_body = None
        self.sampled_native_protocols = {}
        self.sampled_native_full_protocols = {}
        self.sampled_round_frames = {}
        self.device_round_inputs = self.device_round_engram = None
        self.device_round_queue = None
        self.device_round_tail = []
        self.device_round_lookahead = 2 if envs.VLLM_HPU_DSV41_DSPARK_DEEP_QUEUE else 1
        self.device_round_current = None
        self.device_round_eligible = False
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
        self.verify_hidden = torch.empty(
            (6, getattr(self.model_config.hf_config, "hidden_size", 5120)), dtype=torch.bfloat16, device=self.device
        )
        self.verify_aux = torch.empty(
            (6, 3 * getattr(self.model_config.hf_config, "hidden_size", 5120)), dtype=torch.bfloat16, device=self.device
        )
        self.audit = {
            "target_steps": 0,
            "target_tokens": 0,
            "draft_steps": 0,
            "accepted_drafts": 0,
            "rejected_drafts": 0,
            "requests": 0,
            "prefill_steps": 0,
            "decode_steps": 0,
            "image_encodes": 0,
        }

    def load_model(self):
        before = torch.hpu.memory_allocated()
        if envs.VLLM_HPU_DSV41_GRAPH_REPLAY:
            from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime

            initialize_tp2_fused_ar_norm_runtime()
        # Compile the PP commit graph only after the bridge has registered its
        # custom op.  This keeps normal/non-direct paths unchanged.
        self.pp.prepare_commit_graph()
        # Grouped prefill has only eight compute-body shapes once route gather
        # is separated from the MME chain. Prepare those shapes before model
        # loading so their first use is outside the serving request.
        if envs.VLLM_HPU_DSV41_PREFILL_GROUPED and envs.VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES:
            from vllm_gaudi.ops.deepseek_v41_grouped_prefill import prepare_device_grouped_prefill_recipes

            prepare_device_grouped_prefill_recipes(
                normal_scales=True,
                tensor_parallel_size=self.vllm_config.parallel_config.tensor_parallel_size,
                prefill_tokens=self.prefill_capacity,
            )
        self.model = get_model(vllm_config=self.vllm_config)
        if getattr(self.model, "tensor_parallel_size", 2) == 4 and not self.use_dspark:
            from vllm_gaudi.ops.deepseek_v41_completion import prepare_decode_control

            self.tp4_control_inputs, self.tp4_token_readback = prepare_decode_control(
                self.input_views, self.position_views
            )
            if self.position_bank is not None:
                self.position_bank.prepare_compiled_copies({count: self.position_views[count] for count in range(1, 7)})
        state_type = PagedStageState if self.model_config.max_model_len > 512 else StageStateBlocks
        self.state = (
            state_type(self.model.program, auxiliary_prefix=self.vllm_config.cache_config.enable_prefix_caching)
            if state_type is PagedStageState
            else state_type(self.model.program)
        )
        if envs.VLLM_HPU_DSV41_BATCH_DECODE:
            self.model.initialize_request_batches(self.vllm_config.scheduler_config.max_num_seqs)
            from vllm_gaudi.v1.worker.deepseek_v41_batch_runner import BatchExecution

            self.request_batches = BatchExecution(self, self.vllm_config.scheduler_config.max_num_seqs)
        if self.vllm_config.cache_config.enable_prefix_caching:
            from vllm_gaudi.v1.worker.deepseek_v41_prefix import PrefixCheckpoints

            if self.request_batches is None:
                self.model.initialize_request_state(self.vllm_config.scheduler_config.max_num_seqs)
            self.prefix_checkpoints = PrefixCheckpoints(self)
        register_state_spec(self.vllm_config)
        self.model_memory_usage = torch.hpu.memory_allocated() - before
        if self.pp.group.is_last_rank and envs.VLLM_HPU_DSV41_DSPARK:
            draft = self.model.program.draft
            self.insert_context = DraftContextPlans(draft.insert_context)
            self.run_draft = torch.compile(draft, backend="hpu_backend", fullgraph=True, dynamic=False)
            self.sample_draft = torch.compile(draft.sample_greedy, backend="hpu_backend", fullgraph=True, dynamic=False)
            if envs.VLLM_HPU_DSV41_DEVICE_VERIFY:
                # Prefix verification and draft control have separate compile
                # entries.  The former can publish PP commit as soon as the
                # accepted prefix is known; the latter runs concurrently on
                # the normal compute stream and fills the scheduler record.
                self.verify_prefix = torch.compile(
                    draft.verify_prefix, backend="hpu_backend", fullgraph=True, dynamic=False
                )
                self.draft_from_prefix = torch.compile(
                    draft.draft_from_prefix, backend="hpu_backend", fullgraph=True, dynamic=False
                )
                # These entries are used only by positive-temperature requests.
                # The generic/greedy entries retain their original dispatch.
                stochastic_only = envs.VLLM_HPU_DSV41_DSPARK_STOCHASTIC_ONLY
                self.sampled_verify_prefix = torch.compile(
                    partial(draft.verify_sampled_prefix_full, known_stochastic=stochastic_only),
                    backend="hpu_backend", fullgraph=True, dynamic=False)
                self.sampled_draft_from_prefix = torch.compile(
                    partial(draft.draft_sampled_from_prefix, known_stochastic=stochastic_only),
                    backend="hpu_backend", fullgraph=True, dynamic=False)
                self.sampled_propose = torch.compile(
                    partial(draft.propose_sampled_local, known_stochastic=stochastic_only),
                    backend="hpu_backend", fullgraph=True, dynamic=False)
                if envs.VLLM_HPU_DSV41_DSPARK_NATIVE_DRAFT_BODY:
                    if not self.model.native:
                        raise RuntimeError("Native C5 body requires the common native stage executor")
                    from vllm_gaudi.ops.deepseek_v41_draft_body_replay import NativeDraftBody
                    from vllm_gaudi.ops.deepseek_v41_verify import pack_record, encode_record_wire

                    # Target search-bucket generations do not replace these
                    # immutable MTP weights/ring allocations. Track the actual
                    # draft ring identities instead of recapturing per bucket.
                    self.native_draft_body = NativeDraftBody(
                        draft, generation=lambda: tuple(id(layer.attention.swa) for layer in draft.layers))

                    def body_inputs(anchor, positions, committed):
                        return anchor.reshape(1), positions[0].int() + committed.int() + draft.offsets

                    def body_sample(anchor, hidden, logits, controls):
                        return draft.sample_proposal_local(
                            anchor, hidden, logits, controls, full=True, force_legacy=True,
                            known_stochastic=stochastic_only)

                    def body_publish(metadata, output, committed, count, ids, enabled, status):
                        record = pack_record(metadata, committed, count, output, ids, enabled, status)
                        return record, encode_record_wire(record)

                    self.native_body_inputs = torch.compile(
                        body_inputs, backend="hpu_backend", fullgraph=True, dynamic=False)
                    self.native_body_sample = torch.compile(
                        body_sample, backend="hpu_backend", fullgraph=True, dynamic=False)
                    self.native_body_publish = torch.compile(
                        body_publish, backend="hpu_backend", fullgraph=True, dynamic=False)
                    self.sampled_draft_from_prefix = self._sampled_draft_body_from_prefix
                from vllm_gaudi.ops.deepseek_v41_speculative_sampling import speculative_sampling_draws

                self.draw_speculative = torch.compile(
                    speculative_sampling_draws, backend="hpu_backend", fullgraph=True, dynamic=False)
                from vllm_gaudi.ops.deepseek_v41_sampling import sample_probabilities

                self.sampled_prompt_target = torch.compile(
                    partial(sample_probabilities, filtered=True), backend="hpu_backend", fullgraph=True, dynamic=False)
                native_readback = native_record_readback = None
                if envs.VLLM_HPU_DSV41_DEVICE_ROUNDS:
                    from vllm_gaudi.ops.deepseek_v41_completion import resolve_device_runtime
                    from vllm_gaudi.ops.deepseek_v41_device_engram import DeviceEngramRounds
                    from vllm_gaudi.ops.deepseek_v41_round_inputs import DeviceRoundInputs

                    if not self.pp.single_stage or not self.model.native:
                        raise RuntimeError("Device rounds require one native stage owning input and sampling")
                    bridge, _ = resolve_device_runtime(self.model.tensor_parallel_size)
                    native_readback = bridge.copy_integer_record_to_host
                    if envs.VLLM_HPU_DSV41_DSPARK_RECORD_READBACK:
                        if getattr(bridge, "dspark_record_readback_version", None) != 1:
                            raise RuntimeError("Bulk speculative readback requires its private bridge ABI")
                        native_record_readback = bridge.copy_dspark_record_to_host
                    self.device_round_inputs = DeviceRoundInputs(
                        self.input_views[6], self.position_views[6],
                        frames=4 if self.device_round_lookahead == 2 else 2)
                    self.device_round_engram = DeviceEngramRounds(
                        self.model.engram_host, self.input_views[6], self.device_round_inputs.history)
                    if envs.VLLM_HPU_DSV41_NATIVE_DRAFT_PROTOCOL:
                        from vllm_gaudi.ops.deepseek_v41_draft_replay import NativeDraftProtocol

                        self.native_draft_protocol = NativeDraftProtocol(
                            draft, generation=lambda: self.model.program.generation)
                self.verify_ring = VerifyRing(self.device, last_rank=True, native_readback=native_readback,
                                              native_record_readback=native_record_readback,
                                              size=4 if self.device_round_lookahead == 2 else 2)
        elif envs.VLLM_HPU_DSV41_DEVICE_VERIFY and envs.VLLM_HPU_DSV41_DSPARK:
            self.verify_ring = VerifyRing(self.device, last_rank=False)
        elif self.pp.group.is_last_rank:
            sampler = self.model.program.sample_greedy_token if self.v2_completion else self.model.program.sample_greedy
            sampler_backend = "hpu_backend"
            if getattr(self.model, "tensor_parallel_size", 2) == 4:
                from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend

                sampler_backend = make_backend()
            self.sampling_backend = sampler_backend
            self.stochastic_samplers = {}
            self.sampling_buffers = {}
            self.sample_local_head = torch.compile(
                self.model.program._head_projection, backend=sampler_backend, fullgraph=True, dynamic=False
            )
            from vllm_gaudi.ops.deepseek_v41_sampling import commit_sampled_token

            self.sample_commit = torch.compile(
                commit_sampled_token,
                backend=sampler_backend, fullgraph=True, dynamic=False
            )
            self.sample_target = torch.compile(sampler, backend=sampler_backend, fullgraph=True, dynamic=False)
            if self.pp.device_commit_enabled:
                self.sample_target_commit = torch.compile(
                    self.model.program.sample_greedy_commit, backend="hpu_backend", fullgraph=True, dynamic=False
                )
        logger.info(
            "V4.1 PP%d prepared weights loaded; allocated %d bytes", self.model.pp_rank, self.model_memory_usage
        )

    def _prepare_device_sampling_request(self, request):
        program = getattr(self.model, "program", None)
        if not getattr(program, "device_sampling", False):
            return
        if not getattr(self, "v2_completion", False):
            raise ValueError("Device sampling requires the asynchronous C1 completion owner")
        params = request.sampling_params
        owner = (id(request), request.req_id, params.temperature, params.top_p, params.top_k, params.seed)
        if getattr(self, "_device_sampling_owner", None) == owner:
            return
        import hashlib

        seed = (int(params.seed) if params.seed is not None else
                int.from_bytes(hashlib.blake2b(request.req_id.encode(), digest_size=4).digest(), "little"))
        seed &= 0xffffffff
        seed = seed - 2**32 if seed >= 2**31 else seed
        # Request admission is the sole host-to-device parameter upload. The
        # replay derives the draw ordinal from its device position input;
        # neither pinned staging nor an HPU event is created in the token loop.
        program.sampling_params.copy_(torch.tensor([[params.temperature, params.top_p, params.top_k]]))
        program.sampling_seed.copy_(torch.tensor([seed], dtype=torch.int32))
        program.sampling_counter.copy_(torch.tensor([len(request.output)], dtype=torch.int32))
        # Synthetic startup requests have no decode boundary. Their origin is
        # zero; a real request replaces it once at admission.
        prompt_length = len(request.prompt) if hasattr(request, "prompt") else getattr(request, "decode_start", 1)
        program.sampling_origin.copy_(torch.tensor([prompt_length - 1], dtype=torch.int32))
        if not hasattr(self, "device_sampling_stats"):
            self.device_sampling_stats = {}
        self.device_sampling_stats.setdefault(request.req_id, dict(bounded_steps=0, fallbacks=0))
        self._device_sampling_owner = owner

    def _sample_full_local(self, local, controls, *, filtered):
        from functools import partial
        from vllm_gaudi.ops.deepseek_v41_sampling import sample_probabilities

        logits = self.model.program.all_gather(local, dim=-1)
        key = local.shape[0], filtered
        if key not in self.stochastic_samplers:
            self.stochastic_samplers[key] = torch.compile(
                partial(sample_probabilities, filtered=filtered), backend=self.sampling_backend,
                fullgraph=True, dynamic=False
            )
        return self.stochastic_samplers[key](logits, controls)

    @torch.inference_mode()
    def _repair_device_sample(self, payload, destination):
        _, local, controls, _ = payload[:4]
        owner = self._device_sampling_owner
        selected = self._sample_full_local(local, controls, filtered=owner[3] < 1 or owner[4] > 0)
        destination.copy_(selected)
        host, done = self.tp4_token_readback(destination)
        done.synchronize()
        self.audit["device_sampling_fallbacks"] = self.audit.get("device_sampling_fallbacks", 0) + 1
        self.device_sampling_stats[self._device_sampling_owner[1]]["fallbacks"] += 1
        return int(host[0, 0])

    @trace_phase
    def _sample_requests(self, hidden, requests, *, replay=None):
        from functools import partial
        from vllm_gaudi.ops.deepseek_v41_sampling import request_uniform, sample_probabilities

        batch = hidden.shape[0]
        filtered = any(req.sampling_params.top_p < 1 or req.sampling_params.top_k > 0 for req in requests)
        if getattr(self.model.program, "device_sampling", False) and batch == 1:
            self._prepare_device_sampling_request(requests[0])
            values = replay.sampling_tail_values(hidden) if replay is not None else None
            if values is not None:
                self.device_sampling_stats[requests[0].req_id]["bounded_steps"] += 1
                self._device_sampling_payload = values
                return values[3]
            from vllm_gaudi.ops.deepseek_v41_sampling import device_sampling_draw

            if not hasattr(self, "device_sampling_draw"):
                self.device_sampling_draw = torch.compile(
                    device_sampling_draw, backend=self.sampling_backend, fullgraph=True, dynamic=False
                )
            program = self.model.program
            controls = self.device_sampling_draw(
                program.sampling_params, program.sampling_seed, program.sampling_counter
            )
            return self._sample_full_local(self.sample_local_head(hidden), controls, filtered=filtered)
        if batch not in self.sampling_buffers:
            host = torch.zeros(batch, 4, dtype=torch.float32).pin_memory("hpu")
            device = torch.empty_like(host, device=self.device)
            self.sampling_buffers[batch] = host, device
        host, controls = self.sampling_buffers[batch]
        # A previous asynchronous transfer may still read the pinned buffer.
        # Retain ownership through the transfer event before filling it again.
        pending = getattr(self, "sampling_copy_events", {}).get(batch)
        if pending is not None:
            pending.synchronize()
        host[:, 0] = 0
        host[:, 1] = 1
        host[:, 2] = 0.5
        host[:, 3] = -1
        for row, req in enumerate(requests):
            params = req.sampling_params
            host[row, 0] = params.temperature
            host[row, 1] = params.top_p
            host[row, 2] = request_uniform(req.req_id, params.seed, len(req.output))
            host[row, 3] = params.top_k
        controls.copy_(host, non_blocking=True)
        event = torch.hpu.Event()
        event.record()
        if not hasattr(self, "sampling_copy_events"):
            self.sampling_copy_events = {}
        self.sampling_copy_events[batch] = event
        local = replay.tail_local_logits(hidden) if replay is not None else None
        if local is None:
            local = self.sample_local_head(hidden)
        logits = self.model.program.all_gather(local, dim=-1)
        key = batch, filtered
        if key not in self.stochastic_samplers:
            self.stochastic_samplers[key] = torch.compile(
                partial(sample_probabilities, filtered=filtered), backend=self.sampling_backend,
                fullgraph=True, dynamic=False
            )
        return self.stochastic_samplers[key](logits, controls)

    def get_model(self):
        return self.model

    def get_supported_tasks(self):
        return ("generate",)

    def reset_encoder_cache(self):
        self.encoder_cache.clear()

    def get_kv_cache_spec(self):
        logger.info(
            "V4.1 PP%d exposes %d scheduler state arrays (%d bytes/page)",
            self.model.pp_rank,
            len(self.state.specs),
            sum(spec.page_size_bytes for spec in self.state.specs.values()),
        )
        return self.state.specs

    def uses_framework_kv_cache_layout(self, name):
        return name in self.state.specs

    def allocate_framework_kv_cache_layer(self, spec, num_blocks):
        return torch.zeros((num_blocks, *spec.state_shape), device=self.device, dtype=spec.state_dtype)

    def profile_kv_cache_blocks(self):
        """Return the temporary page count needed by pre-KV recipe warmup.

        One null page is followed by identity-mapped live pages.  The second
        warmup geometry starts at the runtime-indexer's cold boundary, so it
        exercises the capacity-independent recipe used by long conversations.
        These pages exist only during ``determine_available_memory``.
        """
        if not isinstance(self.state, PagedStageState):
            return 1
        from vllm_gaudi.ops.deepseek_v41_paged_attention import PAGE_TOKENS

        compute_tokens = max(prefill_compute_buckets(self.prefill_capacity), default=1)
        last_position = min(self.model_config.max_model_len, INDEX_MME_HOT_TOKENS + compute_tokens)
        return 1 + (last_position + PAGE_TOKENS - 1) // PAGE_TOKENS

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
        runner_caches.extend((value,) for value in caches.values())

    def initialize_kv_cache(self, config):
        names = {name for group in config.kv_cache_groups for name in group.layer_names}
        if names != set(self.state.specs):
            raise ValueError("Scheduler cache group does not match this PP stage's complete state")
        self.state.allocate(config.num_blocks, self.device)
        self.state.bind(1)
        self.state.clear()
        self.kv_caches = [(value,) for value in self.state.allocations.values()]
        self.kv_cache_config = config

    @property
    def request_slots_enabled(self):
        return (getattr(self, "request_batches", None) is not None
                or getattr(self, "prefix_checkpoints", None) is not None)

    def _bind_request(self, request):
        if self.request_slots_enabled:
            bank = self.model.batch_state
            owner = bank.acquire(request.req_id)
            bank.publish_pages(owner, request.block_ids[0], self.state.blocks)
            if request.num_computed_tokens >= request.decode_start:
                bank.bind_single(owner, request.num_computed_tokens)
            else:
                bank.bind_prefill(owner)
            if self.model.engram_host is not None:
                self.model.engram_host.activate(request.req_id, reset=request.num_computed_tokens == 0)
            if request.num_computed_tokens == request.decode_start and envs.VLLM_HPU_DSV41_STATE_AUDIT_DIR:
                from vllm_gaudi.ops.deepseek_v41_state_audit import save_single_handoff

                host = self.model.engram_host
                save_single_handoff(
                    bank,
                    owner,
                    request.num_computed_tokens,
                    request.req_id,
                    envs.VLLM_HPU_DSV41_STATE_AUDIT_DIR,
                    engram_history=host.history.history.tolist() if host is not None else None,
                )
            self.active_request = request.req_id
            return
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

    def _release_batch_state(self, req_id):
        self.model.batch_state.release(req_id)
        if isinstance(self.state, PagedStageState):
            self.state.release(req_id)
        if self.model.engram_host is not None:
            self.model.engram_host.release_request(req_id)
        if self.active_request == req_id:
            self.active_request = None

    @trace_phase
    def _update(self, scheduled):
        queued = getattr(self, "device_round_queue", None)
        if queued is not None and queued[0] in (
            set(scheduled.finished_req_ids) | set(getattr(scheduled, "preempted_req_ids", None) or ())
        ):
            self._discard_device_round()
        if self.request_slots_enabled:
            for req_id in getattr(scheduled, "preempted_req_ids", None) or ():
                self._release_batch_state(req_id)
                self.audit["batch_preemptions"] = self.audit.get("batch_preemptions", 0) + 1
        for req_id in scheduled.finished_req_ids:
            sampling = getattr(self, "device_sampling_stats", {}).pop(req_id, None)
            if sampling is not None:
                logger.info("V4.1 sampling completion TP%d: %s", self.model.tp_rank,
                            json.dumps(dict(request_id=req_id, **sampling)))
            if self.verify_timing:
                self.verify_timing.flush()
            records = self.verify_records.pop(req_id, None)
            if records:
                logger.info(
                    "V4.1 device verify transactions PP%d TP%d: %s",
                    self.model.pp_rank,
                    self.model.tp_rank,
                    json.dumps({"request_id": req_id, "units": "ms", "transactions": records}),
                )
            self.requests.pop(req_id, None)
            if self.request_slots_enabled:
                self._release_batch_state(req_id)
            elif isinstance(self.state, PagedStageState):
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
            self.requests[new.req_id] = RequestState(
                new.req_id,
                list(new.prompt_token_ids),
                new.mm_features,
                new.sampling_params,
                new.block_ids,
                new.num_computed_tokens,
            )
            prefix = getattr(new, "prefill_token_ids", None)
            if self.request_slots_enabled and prefix is not None:
                request = self.requests[new.req_id]
                restores = getattr(scheduled, "auxiliary_prefix_operations", None)
                checkpoint = restores.restores.get(new.req_id) if restores is not None else None
                cached_prefix = checkpoint is not None and checkpoint.num_tokens == new.num_computed_tokens
                if prefix[: len(request.prompt)] != request.prompt or (new.num_computed_tokens and not cached_prefix):
                    raise ValueError("Request batch recomputation needs a full committed prefix from position zero")
                request.output = list(prefix[len(request.prompt) :])
                request.recompute_until = len(prefix)
        cached = scheduled.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            request = self.requests[req_id]
            new_blocks = cached.new_block_ids[index]
            if req_id in cached.resumed_req_ids:
                if self.request_slots_enabled:
                    operations = getattr(scheduled, "auxiliary_prefix_operations", None)
                    checkpoint = operations.restores.get(req_id) if operations is not None else None
                    cached_prefix = (
                        checkpoint is not None and checkpoint.num_tokens == cached.num_computed_tokens[index]
                    )
                    if (cached.num_computed_tokens[index] and not cached_prefix) or new_blocks is None:
                        raise ValueError("Request batch resumption requires recomputation and new scheduler pages")
                    # Also handles forced reset without a preceding preemption
                    # notification: stale SWA/Engram owners cannot survive it.
                    self._release_batch_state(req_id)
                request.block_ids = new_blocks
                self.active_request = None
            elif new_blocks is not None:
                request.block_ids = tuple(old + new for old, new in zip(request.block_ids, new_blocks, strict=True))
            request.num_computed_tokens = cached.num_computed_tokens[index]
            new_tokens = cached.new_token_ids[index] if cached.new_token_ids else []
            request.reconcile(cached.num_output_tokens[index], new_tokens, cached.all_token_ids.get(req_id))
            if self.request_slots_enabled and req_id in cached.resumed_req_ids:
                request.recompute_until = len(request.tokens)

    @staticmethod
    def _validate_sampling(params):
        from vllm_gaudi.ops.deepseek_v41_config import validate_sampling

        validate_sampling(params)

    def _image_embeddings(self, request, start, count, input_ids):
        if not request.mm_features or not self.pp.group.is_first_rank:
            return None
        active = [
            feature
            for feature in request.mm_features
            if feature.mm_position.offset < start + count
            and start < feature.mm_position.offset + feature.mm_position.length
        ]
        if not active:
            return None
        self._execute_mm_encoder(request)
        # Device continuation may bind a sampled tensor instead of the host
        # staging buffer. Preserve those actual IDs for non-image rows.
        values = self.model.embed_input_ids(input_ids)
        for feature in active:
            position = feature.mm_position
            low, high = max(start, position.offset), min(start + count, position.offset + position.length)
            if low >= high:
                continue
            source = self.encoder_cache[feature.identifier][low - position.offset : high - position.offset]
            if position.is_embed is None:
                values[low - start : high - start].copy_(source)
            else:
                mask = position.is_embed[low - position.offset : high - position.offset].to(self.device)
                destination = values[low - start : high - start]
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
        device_position = getattr(self, "_next_position", None)
        continuing_position = (decode and count == 1 and not reset and device_position is not None
                               and device_position[:2] == (request_id, start))
        if continuing_position:
            positions = device_position[2]
        # A scheduler may feed a normal prompt one token at a time.  The
        # resulting model transaction has exactly the same C1 tensor geometry
        # and cache writes as decode; only sampling/commit semantics remain
        # prefill semantics.  Bind that token to the captured C1 input tensor
        # instead of compiling an unqualified ordinary C1 stage on the first
        # public chat request.
        program = getattr(self.model, "program", None)
        c1_replay = (
            getattr(self.model, "native", False)
            and count == 1
            and (getattr(program, "runtime_indexer", False) or start + count <= 1024)
        )
        graph_c1 = decode or c1_replay
        if self.request_slots_enabled and request is not None and not decode and c1_replay:
            # Scalar prompt tails own the same native addresses as decode.
            # Publish the prefill slot before replay; larger prompt tiles keep
            # their slot aliases. Forcing this tail through ordinary compiled
            # groups instead creates a new graph inside each cold request.
            bank = self.model.batch_state
            bank.bind_single(bank.acquire(request_id), start)
        if self.direct_token_ids and graph_c1:
            if count != 1:
                raise ValueError("Direct V4.1 token binding requires a C1 transaction")
            ids = self.decode_ids
        if program is not None and program.length > 512:
            if search_length is not None:
                search = int(search_length)
            elif not getattr(program, "runtime_indexer", False):
                search = target_search_length(start, count, program.length)
            elif graph_c1:
                search = runtime_search_length(start, count, program.length)
            else:
                search = prefill_search_length(
                    start, count, program.length, reuse_index_keys=envs.VLLM_HPU_DSV41_PREFILL_REINDEX_REUSE
                )
            if search < start + count or search > program.length:
                raise RuntimeError("V4.1 transaction search bucket does not cover its input")
            decode_bound = (
                decode_source_prefix_bound(
                    start + count,
                    search,
                    getattr(program, "tensor_parallel_size", 2),
                    runtime_indexer=getattr(program, "runtime_indexer", False),
                )
                if graph_c1
                else None
            )
            geometry_key = (id(program), getattr(program, "generation", None), search, decode_bound)
            reuse_geometry = graph_c1 and getattr(program, "tensor_parallel_size", 2) == 4
            if not reuse_geometry or geometry_key != getattr(self, "_decode_geometry_key", None):
                program.search_length = search
                program.decode_token_bound = decode_bound
                for layer in program.layers:
                    attention = layer.attention
                    # This is scheduler metadata, not a device tensor readback.
                    # Prefill changes the visible end on every transaction;
                    # decode reuses its static geometry until the bucket or
                    # physical-state generation changes.
                    attention.prefill_token_end = None if graph_c1 else start + count
                    if hasattr(attention, "set_search_length"):
                        attention.set_search_length(search)
                    else:
                        attention.search_length = search
                    if hasattr(attention, "set_decode_visible_tokens"):
                        attention.set_decode_visible_tokens(decode_bound)
                self._decode_geometry_key = geometry_key if reuse_geometry else None
            shared = getattr(program, "shared", None)
            mirror_capacity = getattr(shared, "index_mirror_tokens", 0)
            if mirror_capacity:
                # _bind_request has already installed canonical pages. Rebuild
                # only at an ownership/prefill transition, before any compiled
                # consumer; ordinary decode only maintains newly written rows.
                if (graph_c1 or (decode and self.use_dspark)) and count <= 6 and 512 < search <= mirror_capacity:
                    shared.prepare_index_mirror(start)
                else:
                    shared.invalidate_index_mirror()
        # The fused seven-value control packet belongs to C1/C6 replay.  A
        # normal prefill block has independent C128-capable input buffers and
        # must not be truncated through that DSpark-sized packet.
        staged_input = getattr(self, "input_staging", None) is not None and count <= 6 and (graph_c1 or self.use_dspark)
        tp4_control = (
            getattr(self, "tp4_control_inputs", {}).get(count)
            if graph_c1 and not (count == 1 and self.direct_token_ids)
            else None
        )
        if tp4_control is not None:
            if ids is not self.input_views[count] or positions is not self.position_views[count]:
                raise RuntimeError("Prepared TP4 control input was rebound without preparation")
            tp4_control.upload(tokens, start)
        elif staged_input:
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
            if (
                not self.use_dspark
                and decode
                and count == 1
                and self._next_input is not None
                and self._next_input[:2] == (request_id, start)
            ):
                if self.direct_token_ids:
                    ids = self._next_input[2]
                else:
                    ids.copy_(self._next_input[2])
            else:
                ids.copy_(torch.tensor(tokens, dtype=ids.dtype, device="cpu"))
            if continuing_position:
                # The previous native tail produced this row. Bind it as the
                # next fixed input; no external position-copy recipe is needed.
                pass
            elif self.position_bank is not None and graph_c1:
                if getattr(self.model, "tensor_parallel_size", 2) == 4:
                    if not self.model.decode_prefix_pending:
                        self.position_bank.copy_into(positions, start)
                else:
                    positions = self.position_bank.view(start, count)
            else:
                positions.copy_(torch.arange(start, start + count, dtype=torch.int32, device="cpu"))
        self._round_phase("inputs_staged_ns")
        # Slot-owned scalar C1 tails have already published their state to
        # the fixed replay addresses. They use the same warmed search buckets
        # as decode; the legacy prompt/DSpark path keeps its bounded capture.
        use_replay = (
            getattr(self.model, "native", False)
            or (self.v2_completion and getattr(self.model, "tensor_parallel_size", 2) == 4)
        ) and (
            (decode and 1 <= count <= (6 if self.use_dspark else 1))
            or (c1_replay and (self.request_slots_enabled or start + count <= 1024))
            or (start + count <= 1024 and self.use_dspark and request is not None)
        )
        if self.request_slots_enabled and request is not None and not decode and not c1_replay:
            # Prompt/tail transactions keep their request-slot aliases.
            # B1 decode binds the original working addresses before replay.
            use_replay = False
        self.model.prepare_step(request_id, tokens, is_decode=graph_c1, reset=reset, use_replay=use_replay)
        self._round_phase("engram_prepared_ns")
        timing = getattr(self, "verify_timing", None)
        if self.pp.group.is_first_rank:
            embeddings = self._image_embeddings(request, start, count, ids) if request is not None else None
            if timing:
                timing.device("stage_model_start")
            value = self.model(ids, positions, inputs_embeds=embeddings)
            self._round_phase("stage_submitted_ns")
            if timing:
                timing.device("stage_target_done")
            # TP4 is a single PP stage.  The model owns the final norm and
            # head on this rank, so keep its output instead of sending it to
            # the nonexistent PP1 peer.  TP2xPP2 retains the ordinary
            # boundary exchange below.
            if getattr(self.pp.group, "is_last_rank", False):
                output = value
            else:
                self.pp.exchange(value, count, decode=graph_c1)
                output = None
            self._round_phase("pp_exchanged_ns")
        else:
            value = self.pp.exchange(None, count, native_wire=use_replay and self.use_dspark, decode=graph_c1)
            self._round_phase("pp_exchanged_ns")
            if timing:
                timing.device("stage_model_start")
            output = self.model(ids, positions, intermediate_tensors=value)
            if not decode:
                self.pp.mark_prefill_consumed()
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
        positions = self.position_views.get(6)
        return (count == 6 and hidden.shape == self.verify_hidden.shape and self.model.last_aux is not None
                and self.model.last_aux.shape[0] == 6 and positions is not None and positions.shape == (6, ))

    def _finish_request_device(self, request, start, count, last_count, proposed, target_hidden):
        """Run one fixed C6 verify transaction and defer its sole host read."""
        started = time.perf_counter_ns()
        timing = getattr(self, "verify_timing", None)
        if timing:
            timing.host("verify_start", started)
        phase_ms = {}
        ticket = None
        if self.pp.group.is_last_rank:
            if self.verify_ring is None or self.verify_prefix is None or self.draft_from_prefix is None:
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
            verify_positions = self.position_views[6] if direct_c6 else self.verify_positions
            if not direct_c6:
                self.verify_hidden.zero_()
                self.verify_hidden[: target_hidden.shape[0]].copy_(target_hidden)
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
                self.verify_aux = torch.empty(
                    (6, self.model.last_aux.shape[1]), dtype=self.model.last_aux.dtype, device=self.device
                )
            if not direct_c6:
                self.verify_aux.zero_()
                self.verify_aux[:last_count].copy_(self.model.last_aux[:last_count])
                self.verify_positions.copy_(self.positions[0].to(torch.int32) + self.verify_offsets)
            if timing:
                timing.host("prefix_submit_start")
                timing.device("prefix_start")
            sampled = request.sampling_params.temperature != 0
            if sampled:
                state = self._request_speculative_sampling(request)
                draft_controls, acceptance, correction, target_controls = self.draw_speculative(
                    state.parameters, state.seed, state.counter, state.offsets)
                (
                    prefix_output, prefix_committed, prefix_output_count, anchor, draft_enabled, status,
                    commit_record, commit_wire,
                ) = self.sampled_verify_prefix(
                    verify_hidden, self.verify_proposed, state.proposal, metadata, verify_aux, verify_positions,
                    target_controls, acceptance, correction)
            else:
                (
                    _, prefix_output, prefix_committed, prefix_output_count, anchor, draft_enabled, status,
                    commit_record, commit_wire,
                ) = self.verify_prefix(verify_hidden, self.verify_proposed, metadata, verify_aux, verify_positions)
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
        commit_record = (
            None if ticket is None else (ticket.commit_record if ticket.commit_record is not None else ticket.record)
        )
        commit_wire = (
            None if ticket is None else (ticket.commit_wire if ticket.commit_wire is not None else ticket.wire)
        )
        pp_result = self.pp.finish_device(commit_record, last_count, commit_wire)
        phase_ms["pp_finish_device_ms"] = (time.perf_counter_ns() - phase_started) / 1e6
        if self.pp.group.is_last_rank:
            # Continue draft control on the normal compute stream only after
            # the commit source has been snapshotted. PP0 can now validate the
            # prefix while these three layers execute on PP1.
            arguments = (metadata, verify_positions, prefix_output, prefix_committed, prefix_output_count,
                         anchor, draft_enabled, status)
            if sampled:
                record, wire_record, confidence, probability, covered = self.sampled_draft_from_prefix(
                    *arguments, draft_controls)
                state.proposal.copy_(probability)
                state.proposal_valid.copy_(covered.all().reshape(1) & draft_enabled & (status == 0))
            else:
                record, wire_record, confidence = self.draft_from_prefix(*arguments)
            ticket.record.copy_(record)
            if ticket.wire is not None:
                ticket.wire.copy_(wire_record)
            self.last_draft_confidence = confidence
            if timing:
                timing.device("draft_done")
        if (self.pp.group.is_first_rank and not self.pp.single_stage and envs.VLLM_HPU_DSV41_INLINE_PP_COMMIT):
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
            self.verify_records.setdefault(request.req_id, []).append(
                {
                    "target_count": last_count,
                    "proposed_count": len(proposed),
                    "committed": committed_value,
                    "output_count": len(output),
                    "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                    **phase_ms,
                    "inline_pp_commit": True,
                }
            )
            self._record_round_completion(request, proposed, committed_value, output)
            self.pending = None
            return None
        if self.pp.group.is_first_rank and not self.pp.single_stage:
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
                self.verify_records.setdefault(request.req_id, []).append(
                    {
                        "target_count": last_count,
                        "proposed_count": len(proposed),
                        "committed": committed,
                        "output_count": len(output),
                        "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                        **phase_ms,
                    }
                )
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
            result = ModelRunnerOutput(
                req_ids=[request.req_id], req_id_to_index={request.req_id: 0}, sampled_token_ids=[output]
            )
            self.verify_records.setdefault(request.req_id, []).append(
                {
                    "target_count": last_count,
                    "proposed_count": len(proposed),
                    "committed": committed_value,
                    "output_count": len(output),
                    "draft_count": len(draft),
                    "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                    **phase_ms,
                }
            )
            result.execution_rounds = self._record_round_completion(request, proposed, committed_value, output)
            return result

        return AsyncDSparkOutput(
            self.verify_ring, ticket, consume, self._record_round_released if self.round_timing_enabled else None
        )

    def _request_speculative_sampling(self, request):
        """Bind probability and RNG state to the scheduler's request owner."""
        from vllm_gaudi.ops.deepseek_v41_speculative_sampling import SpeculativeRequestSampling

        if request.speculative_sampling is None:
            import hashlib

            params = request.sampling_params
            seed = params.seed
            if seed is None:
                seed = int.from_bytes(hashlib.blake2b(request.req_id.encode(), digest_size=4).digest(), "little")
            request.speculative_sampling = SpeculativeRequestSampling(
                (params.temperature, params.top_p, params.top_k), seed,
                self.model.program.draft.output_head.weight.shape[0], self.device)
        return request.speculative_sampling

    @torch.inference_mode()
    @trace_phase
    def execute_model(self, scheduled):
        if self.pending is not None:
            raise RuntimeError("Previous V4.1 execution has not completed sampling/verify")
        self._update(scheduled)
        operations = getattr(scheduled, "auxiliary_prefix_operations", None)
        if self.prefix_checkpoints is not None:
            if operations is not None:
                self.model.batch_state.leave_single()
            self.prefix_checkpoints.begin(operations)
        elif operations is not None:
            raise RuntimeError("Auxiliary prefix operations reached a worker without checkpoint support")
        if not scheduled.num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT
        outputs, drafts, ids, async_result = [], [], [], None
        execution_rounds = []
        batch_size = len(scheduled.num_scheduled_tokens)
        batched_ids = set()
        if self.request_batches is not None:
            batch_requests = [
                self.requests[name]
                for name, count in scheduled.num_scheduled_tokens.items()
                if count == 1 and self.requests[name].num_computed_tokens >= self.requests[name].decode_start
            ]
            if batch_requests and (len(scheduled.num_scheduled_tokens) > 1 or operations is not None):
                self._next_input = None
                result = self.request_batches.execute(batch_requests)
                batched_ids = {request.req_id for request in batch_requests}
                ids.extend(request.req_id for request in batch_requests)
                if self.pp.group.is_last_rank:
                    outputs.extend(result.sampled_token_ids)
        for req_id, count in scheduled.num_scheduled_tokens.items():
            if req_id in batched_ids:
                continue
            request = self.requests.get(req_id) if self.request_batches is not None else None
            single_trace = (
                self.request_batches.trace_single(request)
                if self.request_batches is not None
                and request is not None
                and request.num_computed_tokens >= request.decode_start
                else nullcontext()
            )
            with single_trace:
                self._execute_request(scheduled, req_id, count)
                result = self._finish_request()
            if operations is not None and isinstance(result, AsyncModelRunnerOutput):
                # Admission/checkpoint transactions must retire the accepted
                # input before publishing auxiliary state or acknowledgments.
                # Steady speculative decode keeps its asynchronous output.
                result = result.get_output()
            if self.prefix_checkpoints is not None:
                self.prefix_checkpoints.capture_at(req_id, self.requests[req_id].num_computed_tokens + count)
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
        acknowledgments = self.prefix_checkpoints.finish() if self.prefix_checkpoints is not None else []
        if operations is not None and async_result is not None:
            raise RuntimeError("Auxiliary prefix completion must precede asynchronous decode")
        if async_result is None and self.pp.group.is_last_rank:
            self.draft_token_ids = DraftTokenIds(ids, drafts)
        if async_result is not None:
            self.batch_result = async_result
        elif self.pp.group.is_last_rank:
            self.batch_result = ModelRunnerOutput(
                req_ids=ids, req_id_to_index={key: index for index, key in enumerate(ids)}, sampled_token_ids=outputs
            )
            if acknowledgments:
                self.batch_result.auxiliary_prefix_acknowledgments = acknowledgments
            if execution_rounds:
                self.batch_result.execution_rounds = execution_rounds
        else:
            self.batch_result = None
        self.pending = "batch_ready"
        return None

    @trace_phase
    def _execute_request(self, scheduled, req_id, count):
        if self.round_timing_enabled:
            self.round_context = dict(
                request_id=req_id,
                generation=self.pp.generation + 1,
                target_count=count,
                start_ns=time.perf_counter_ns(),
            )
        request = self.requests[req_id]
        self._bind_request(request)
        self._prepare_device_sampling_request(request)
        self._device_sampling_payload = None
        start = request.num_computed_tokens
        proposed = scheduled.scheduled_spec_decode_tokens.get(req_id, [])
        if proposed and not self.use_dspark:
            raise RuntimeError("Speculative tokens reached a non-speculative V4.1 runner")
        tokens = request.token_slice(start, start + count - len(proposed)) + proposed
        if len(tokens) != count or start + count > self.model_config.max_model_len:
            raise RuntimeError("Scheduled V4.1 inputs do not match the committed prefix and context budget")
        decode = start >= request.decode_start
        self.device_round_eligible = bool(
            getattr(self, "device_round_inputs", None) is not None and decode and count == 6 and len(proposed) == 5
            and len(scheduled.num_scheduled_tokens) == 1 and not request.mm_features
        )
        if getattr(self, "device_round_queue", None) is not None:
            queued = self.device_round_queue
            if not self.device_round_eligible or queued[0] != req_id:
                self._discard_device_round()
            else:
                # Target, verification and draft are already enqueued, before
                # the preceding output was read by the scheduler. The host
                # supplies ownership/lifetime checks, never the next tokens.
                self.device_round_current = queued
                tail = getattr(self, "device_round_tail", [])
                self.device_round_queue = tail.pop(0) if tail else None
                self.round_context = queued[2]
                self._bind_request(request)
                self.pending = (request, start, count, count, proposed, True, None)
                return None
        self._bind_request(request)
        from vllm_gaudi.ops import deepseek_v41_prefill_event_trace as prefill_events

        tracing_prefill = not decode and prefill_events.begin(
            req_id, self.pp.generation + 1, count, self.model.pp_rank, self.model.tp_rank
        )
        valid_decode = 1 <= count <= 6 if self.use_dspark else count == 1
        if decode and not valid_decode:
            raise RuntimeError(f"Unexpected V4.1 decode shape: tokens={count}, drafts={len(proposed)}")
        if self.verify_timing:
            self.verify_timing.begin(req_id, self.pp.generation + 1, count, len(proposed))
        # Match vLLM chunked-prefill semantics: one scheduler transaction is a
        # real large-M model invocation up to max_num_batched_tokens. Internal
        # C1/C6 decode tiling would reread expert weights for every prompt row.
        chunks = [(0, tokens)] if decode else target_chunks(tokens, self.prefill_capacity)
        prefix_checkpoints = getattr(self, "prefix_checkpoints", None)
        if prefix_checkpoints is not None:
            chunks = prefix_checkpoints.chunks(
                req_id, start, chunks,
                inline_eligible=(
                    count > 256
                    and len(scheduled.num_scheduled_tokens) == 1
                    and not request.mm_features
                    and getattr(request.sampling_params, "prompt_logprobs", None) is None
                    and not self.use_dspark
                ),
            )
        program = getattr(self.model, "program", None)
        transaction_search = (
            (
                prefill_search_length(
                    start, count, program.length, reuse_index_keys=envs.VLLM_HPU_DSV41_PREFILL_REINDEX_REUSE
                )
                if getattr(program, "runtime_indexer", False)
                else target_search_length(start, count, program.length)
            )
            if not decode and program is not None and program.length > 512
            else None
        )
        for block_index, (offset, chunk) in enumerate(chunks):
            halo_mode = "full"
            if not decode:
                from vllm_gaudi.ops.deepseek_v41_decoder_halo import decoder_halo_mode

                tp4_halo = getattr(program, "tensor_parallel_size", 2) == 4
                halo_block = self.prefill_capacity if tp4_halo else PREFILL_BLOCK_TOKENS
                halo_mode = decoder_halo_mode(
                    start + offset,
                    len(chunk),
                    len(request.prompt),
                    eligible=(
                        envs.VLLM_HPU_DSV41_PREFILL_DECODER_HALO
                        and not self.use_dspark
                        and (not tp4_halo or self.prefill_capacity == 16384)
                        and (tp4_halo or (start == 0 and count == len(request.prompt)))
                        and max(prefill_compute_buckets(self.prefill_capacity), default=1) == halo_block
                        and len(scheduled.num_scheduled_tokens) == 1
                        and not request.mm_features
                        and getattr(request.sampling_params, "prompt_logprobs", None) is None
                    ),
                    block_tokens=halo_block,
                    allow_single_block=tp4_halo,
                )
            if program is not None:
                program.prefill_halo_mode = halo_mode
            if prefix_checkpoints is not None:
                prefix_checkpoints.activate_chunk(req_id, start + offset, len(chunk))
            try:
                with prefill_events.span("transaction_chunk", rows=len(chunk)):
                    hidden = self._forward(
                        req_id,
                        chunk,
                        start + offset,
                        decode=decode,
                        reset=start + offset == 0,
                        request=request,
                        search_length=transaction_search,
                    )
            finally:
                if program is not None:
                    program.prefill_halo_mode = "full"
                    shared = getattr(program, "shared", None)
                    if shared is not None:
                        shared.inline_prefix_capture = None
            if offset + len(chunk) < count:
                self._insert(self.model.last_aux, self.positions[: len(chunk)])
                self.model.complete_step(len(chunk))
                if prefix_checkpoints is not None:
                    prefix_checkpoints.capture_at(req_id, start + offset + len(chunk))
                # Keep the exact two-slot PP ring owned through its consumer.
                if not getattr(self.pp, "prefill_wavefront", False) or len(chunk) == 1:
                    self.pp.drain()
                    torch.hpu.synchronize()
                    # Exact prefill tails can reuse the single-token packet.
                    # Retire it after both transport and its consumer finish;
                    # the final block remains owned by normal sampling.
                    self.pp.complete_packet()
        need_sample = start + count >= len(request.prompt) + len(request.output)
        if self.pp.group.is_last_rank and need_sample:
            if not self.use_dspark:
                if request.sampling_params.temperature != 0:
                    replay = getattr(getattr(self.model, "program", None), "replay_owner", None) if decode else None
                    sample_input = self._sample_requests(hidden[-1:], (request,), replay=replay)
                    if decode and self.pp.device_commit_enabled:
                        sample_input = self.sample_commit(sample_input, self.pp.commit)
                elif decode and self.pp.device_commit_enabled:
                    sample_input = self.sample_target_commit(hidden[-1:], self.pp.commit)
                else:
                    replay = getattr(getattr(self.model, "program", None), "replay_owner", None) if decode else None
                    cached = replay.greedy_tail_token(hidden[-1:]) if replay is not None else None
                    sample_input = self.sample_target(hidden[-1:]) if cached is None else cached
                if self.model.native and not (decode and (self.pp.device_commit_enabled or self.v2_completion)):
                    from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime

                    bridge, _, _ = _resolve_runtime()
                    self._token_copy = bridge.copy_sampled_tokens_to_host(sample_input)
                elif getattr(self, "tp4_token_readback", None) is not None and not (decode and self.v2_completion):
                    self._token_copy = self.tp4_token_readback(sample_input)
            else:
                # Device verification owns projection, argmax and the small
                # TP candidate exchange, so it consumes target hidden state.
                if envs.VLLM_HPU_DSV41_DEVICE_VERIFY and len(chunk) <= 6:
                    sample_input = hidden
                else:
                    # A large prompt commits all its input rows and samples
                    # only its final row. It cannot enter the C6 prefix/ring
                    # protocol or materialize a full prompt vocabulary tensor.
                    sample_input = self.model.compute_logits(hidden if decode else hidden[-1:])
        else:
            sample_input = None
        self.pending = (request, start, count, len(chunk), proposed, need_sample, sample_input)
        if tracing_prefill:
            prefill_events.finish()
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
                if not self.pp.group.is_last_rank or getattr(self.model, "tp_rank", 0) != 0:
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
        if getattr(self, "device_round_eligible", False):
            return self._finish_device_round(request, start, count, proposed, sample_input)
        if (envs.VLLM_HPU_DSV41_DEVICE_VERIFY and envs.VLLM_HPU_DSV41_DSPARK and need_sample and last_count <= 6
                and (self.verify_prefix is not None or self.pp.group.is_first_rank)):
            return self._finish_request_device(request, start, count, last_count, proposed, sample_input)
        output, draft, committed = [], [], last_count
        if self.pp.group.is_last_rank:
            sampled = request.sampling_params.temperature != 0
            if sampled and need_sample:
                state = self._request_speculative_sampling(request)
                proposal_controls, _, _, target_controls = self.draw_speculative(
                    state.parameters, state.seed, state.counter, state.offsets)
            if need_sample:
                if sampled:
                    if proposed:
                        raise RuntimeError("Sampled proposal verification must enter the bounded C1-C6 protocol")
                    selected = self.sampled_prompt_target(sample_input[-1:], target_controls[-1:])
                    target = selected.reshape(-1).cpu().tolist()
                else:
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
            if (
                output
                and envs.VLLM_HPU_DSV41_DSPARK
                and remaining >= 6
                and next_position + 6 <= self.model_config.max_model_len
            ):
                if sampled:
                    first = torch.tensor([output[-1]], dtype=torch.int64, device=self.device)
                    positions = torch.arange(next_position, next_position + 5, dtype=torch.int32, device=self.device)
                    draft_ids, q, confidence, covered = self.sampled_propose(first, positions, proposal_controls)
                    state.proposal.copy_(q)
                    state.proposal_valid.copy_(covered.all().reshape(1))
                    self.last_draft_confidence = confidence
                    draft = draft_ids.cpu().tolist()
                    self.audit["draft_steps"] += 1
                else:
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
        return ModelRunnerOutput(
            req_ids=[request.req_id], req_id_to_index={request.req_id: 0}, sampled_token_ids=[output]
        )

    def _device_round_scope(self, name):
        if not getattr(self, "trace_enabled", False):
            return nullcontext()
        label = f"v41::device_round::{name}"
        if os.getenv("VLLM_HPU_DSV41_RAW_TRACE", "0") == "1":
            from vllm_gaudi.ops.deepseek_v41_native_trace import device_scope, scope

            if name == "target" and os.getenv("VLLM_HPU_DSV41_DSPARK_PHASE_EVENTS", "0") == "1":
                return device_scope(label)
            return scope(label)
        return torch.profiler.record_function(label)

    def _submit_round_verify(self, request, hidden, aux, context):
        control = self.device_round_inputs.control
        self.pp.generation += 1
        ticket = self.verify_ring.acquire(6, generation=self.pp.generation)
        parameters = getattr(request, "sampling_params", None)
        sampled = parameters is not None and parameters.temperature != 0
        if sampled:
            state = self._request_speculative_sampling(request)
            if envs.VLLM_HPU_DSV41_REQUEST_FIXTURE_DIR:
                from vllm_gaudi.ops.deepseek_v41_micro_fixtures import export_request_fixture

                export_request_fixture(self.model.program, hidden, aux, self.device_round_inputs, state,
                                       envs.VLLM_HPU_DSV41_REQUEST_FIXTURE_DIR, request)
            if envs.VLLM_HPU_DSV41_DSPARK_PROTOCOL_WRITEBACK and not request.req_id.startswith("__v41_"):
                self._prepare_sampled_native_protocols(hidden, aux, state)
            native = getattr(self, "sampled_native_protocols", {}).get(self.device_round_inputs.parity)
            if native is not None and not request.req_id.startswith("__v41_"):
                with self._device_round_scope("sampled_native_protocol"):
                    record, _, q, confidence, counter = native(
                        hidden, control[7:], control[:7], aux, self.position_views[6], state.proposal,
                        state.parameters, state.seed, state.counter, state.offsets)
                    if native.state_publication is None:
                        state.proposal.copy_(q)
                        state.counter.copy_(counter)
                        state.proposal_valid.copy_(record[STATUS:STATUS + 1] == 0)
                if native.repair_frame is not None:
                    self.sampled_round_frames[ticket.generation] = native.repair_frame
            else:
                record, confidence = self._sampled_round_reference(request, hidden, aux, control)
        elif getattr(self, "native_draft_protocol", None) is not None:
            with self._device_round_scope("verify_draft_native"):
                record, _, _, confidence = self.native_draft_protocol(
                    hidden, control[7:], control[:7], aux, self.position_views[6])
        else:
            with self._device_round_scope("verify"):
                _, output, committed, output_count, anchor, enabled, status, _, _ = self.verify_prefix(
                    hidden, control[7:], control[:7], aux, self.position_views[6])
            with self._device_round_scope("draft"):
                record, _, confidence = self.draft_from_prefix(
                    control[:7], self.position_views[6], output, committed, output_count, anchor, enabled, status)
        ticket.record.copy_(record)
        self.last_draft_confidence = confidence
        # Queue the output read before the next round, retaining producer
        # ordering without a frontend event or a control upload.
        with self._device_round_scope("readback_enqueue"):
            self.verify_ring.stage(ticket)
        if context is not None:
            context["device_round_pipeline"] = True
        return request.req_id, ticket, context

    def _sampled_draft_body_from_prefix(self, metadata, target_positions, output, committed, count,
                                      anchor, enabled, status, controls, *, prepare=False):
        first_token, positions = self.native_body_inputs(anchor, target_positions, committed)
        if prepare:
            self.native_draft_body.prepare(first_token, positions)
        else:
            self.native_draft_body.require_ready()
        hidden, logits = self.native_draft_body(first_token, positions)
        ids, q, confidence, covered = self.native_body_sample(first_token, hidden, logits, controls)
        record, wire = self.native_body_publish(metadata, output, committed, count, ids, enabled, status)
        return record, wire, confidence, q, covered

    def _sampled_round_reference(self, request, hidden, aux, control):
        """Original exact probability protocol, also used with the saved draw."""
        state = self._request_speculative_sampling(request)
        with self._device_round_scope("sampled_draws"):
            proposal_controls, acceptance, correction, target_controls = self.draw_speculative(
                state.parameters, state.seed, state.counter, state.offsets)
        with self._device_round_scope("sampled_verify"):
            output, committed, output_count, anchor, enabled, status, _, _ = self.sampled_verify_prefix(
                hidden, control[7:], state.proposal, control[:7], aux, self.position_views[6],
                target_controls, acceptance, correction)
        with self._device_round_scope("sampled_draft"):
            record, _, confidence, q, covered = self.sampled_draft_from_prefix(
                control[:7], self.position_views[6], output, committed, output_count, anchor, enabled, status,
                proposal_controls)
            state.proposal.copy_(q)
            state.proposal_valid.copy_(covered.all().reshape(1) & enabled & (status == 0))
        return record, confidence

    def _prepare_sampled_native_protocols(self, hidden, aux, state):
        if not envs.VLLM_HPU_DSV41_DSPARK_NATIVE_SAMPLED_PROTOCOL:
            return
        writeback = envs.VLLM_HPU_DSV41_DSPARK_PROTOCOL_WRITEBACK
        if self.sampled_native_protocols:
            if not writeback or getattr(self, "sampled_publication_owner", None) is state:
                return
            # A publication owns one request's allocations. Retire all old
            # captures at this request boundary; no per-round rebinding or
            # cross-request q/counter storage is allowed.
            if self.device_round_queue is not None or self.device_round_tail:
                raise RuntimeError("Drain the previous device request before replacing sampling publication")
            torch.hpu.synchronize()
            for plan in (*self.sampled_native_protocols.values(), *self.sampled_native_full_protocols.values()):
                plan.close()
            self.sampled_native_protocols.clear()
            self.sampled_native_full_protocols.clear()
            self.sampled_round_frames.clear()
        state_publication = None
        if writeback:
            from vllm_gaudi.ops.deepseek_v41_sampling_publication import SamplingStatePublication

            state_publication = SamplingStatePublication(state)
            self.sampled_publication_owner = state
        if not self.pp.single_stage or not self.model.native:
            raise RuntimeError("Sampled native continuation requires a stage owning input and sampling")
        from vllm_gaudi.ops.deepseek_v41_draft_replay import NativeDraftProtocol
        from vllm_gaudi.ops.deepseek_v41_round_repair import SampledRoundRepairFrame

        cursor, engram, program = self.device_round_inputs, self.device_round_engram, self.model.program
        prototype = (hidden, cursor.control[7:], cursor.control[:7], aux, cursor.positions, state.proposal,
                     state.parameters, state.seed, state.counter, state.offsets)
        for parity in range(len(cursor.controls)):
            publication = None
            if envs.VLLM_HPU_DSV41_DSPARK_ROUND_INPUT_PUBLICATION:
                from vllm_gaudi.ops.deepseek_v41_round_inputs import RoundInputPublication

                publication = RoundInputPublication(cursor, engram, parity)
            if (envs.VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_MAIN
                    and not envs.VLLM_HPU_DSV41_DSPARK_GLOBAL_BOUNDED_SAMPLING):
                if (not envs.VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_REPAIR
                        or envs.VLLM_HPU_DSV41_DSPARK_NUCLEUS_MASS):
                    raise RuntimeError("Exact full main requires full peer transport and the legacy official sampler")
                # Exact full-vocabulary p/q has no coverage repair. Keep the
                # whole verification/C5/Markov protocol in the same native
                # plan, without a speculative rollback journal on every round.
                plan = NativeDraftProtocol(program.draft, generation=lambda: program.generation,
                                           sampled=True, full=True, full_main=True,
                                           known_stochastic=envs.VLLM_HPU_DSV41_DSPARK_STOCHASTIC_ONLY,
                                           input_publication=publication, state_publication=state_publication)
                plan.prepare(*prototype)
                plan.require_ready()
                self.sampled_native_protocols[parity] = plan
                continue
            frame = SampledRoundRepairFrame(
                program, cursor, engram, prototype, parity,
                lookahead=getattr(self, "device_round_lookahead", 1))
            full_main = (envs.VLLM_HPU_DSV41_DSPARK_NUCLEUS_MASS
                         or envs.VLLM_HPU_DSV41_DSPARK_GLOBAL_BOUNDED_SAMPLING)
            if full_main and not envs.VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_REPAIR:
                raise RuntimeError("Nucleus main plan requires captured exact official repair")
            plan = NativeDraftProtocol(program.draft, generation=lambda: program.generation,
                                       sampled=True, repair_frame=frame, full=full_main, full_main=full_main,
                                       known_stochastic=envs.VLLM_HPU_DSV41_DSPARK_STOCHASTIC_ONLY,
                                       input_publication=publication, state_publication=state_publication)
            plan.prepare(*prototype)
            plan.require_ready()
            self.sampled_native_protocols[parity] = plan
            if envs.VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_REPAIR:
                # Cold capture must never restore uninitialized journal rows.
                full_plan = NativeDraftProtocol(program.draft, generation=lambda: program.generation,
                                                sampled=True, full=True, repair_frame=frame,
                                                known_stochastic=envs.VLLM_HPU_DSV41_DSPARK_STOCHASTIC_ONLY,
                                                state_publication=state_publication)
                full_plan.prepare_repair(*prototype, journal_positions=cursor.positions)
                full_plan.require_ready()
                self.sampled_native_full_protocols[parity] = full_plan

    def _queue_next_device_round(self, request, *, published=False):
        next_context = (dict(request_id=request.req_id, generation=self.pp.generation + 1,
                             target_count=6, start_ns=time.perf_counter_ns())
                        if self.round_timing_enabled else None)
        cursor, engram = self.device_round_inputs, self.device_round_engram
        with self._device_round_scope("engram"):
            rows = engram.prepare(request.req_id, cursor.ids, cursor.history, published=published)
        with self._device_round_scope("target"):
            next_hidden = self.model.forward_device_round(cursor.ids, cursor.positions, rows)
        for name, increment in (("target_steps", 1), ("target_tokens", 6), ("decode_steps", 1)):
            self.audit[name] = self.audit.get(name, 0) + increment
        if next_context is not None:
            next_context["stage_submitted_ns"] = time.perf_counter_ns()
        queued = self._submit_round_verify(request, next_hidden, self.model.last_aux, next_context)
        if self.device_round_queue is None:
            self.device_round_queue = queued
        else:
            self.device_round_tail.append(queued)

    @torch.inference_mode()
    def _repair_sampled_round(self, request, ticket):
        """Discard lookahead, restore the exact official draw, then requeue."""
        frame = self.sampled_round_frames.get(ticket.generation)
        if frame is None:
            raise RuntimeError("Exact full official protocol published an invalid distribution; "
                               "no bounded repair exists")
        queued = self.device_round_queue
        retained_count = 0
        if queued is not None:
            if queued[0] != request.req_id:
                raise RuntimeError("Official-sampling repair cannot discard another request's lookahead")
            discarded_rounds = [queued, *getattr(self, "device_round_tail", [])]
            retained_count = len(discarded_rounds)
            for discarded in discarded_rounds:
                if discarded[0] != request.req_id:
                    raise RuntimeError("Repair cannot discard another request's retained round")
                self.verify_ring.await_record(discarded[1])
                self.verify_ring.release(discarded[1])
                self.sampled_round_frames.pop(discarded[1].generation, None)
            self.device_round_queue = None
            self.device_round_tail = []
        # Both Target and draft writes are complete before restoring their
        # bounded write sets. Restore the original counter, not a fresh draw.
        if hasattr(frame, "coverage_flags"):
            target_failed, draft_failed = frame.coverage_flags.cpu().tolist()
            requests = self.audit.setdefault("sampled_coverage_by_request", {})
            counts = requests.setdefault(request.req_id, {"target_only": 0, "draft_only": 0, "both": 0,
                                                          "unclassified": 0})
            category = ("both" if target_failed and draft_failed else "target_only" if target_failed
                        else "draft_only" if draft_failed else "unclassified")
            counts[category] += 1
        native = getattr(self, "sampled_native_full_protocols", {}).get(frame.parity)
        if native is None:
            frame.journal.restore()
            frame.restore_protocol()
        cursor, engram = self.device_round_inputs, self.device_round_engram
        cursor.parity = frame.parity
        hidden, _, _, aux, _, proposal, _, _, counter, _ = frame.payload
        state = self._request_speculative_sampling(request)
        with self._device_round_scope("sampled_exact_repair"):
            if native is None:
                state.proposal.copy_(proposal)
                state.counter.copy_(counter)
                record, confidence = self._sampled_round_reference(request, hidden, aux, cursor.control)
            else:
                record, _, q, confidence, advanced_counter = native(*frame.payload)
                if native.state_publication is None:
                    state.proposal.copy_(q)
                    state.counter.copy_(advanced_counter)
                    state.proposal_valid.copy_(record[STATUS:STATUS + 1] == 0)
            ticket.record.copy_(record)
            self.last_draft_confidence = confidence
            self.verify_ring.stage(ticket)
        if queued is not None:
            # A discarded ticket's generation is never reused. The repaired
            # current record retains its original generation; only the next
            # cursor advances past the discarded unpublished transaction.
            next_generation = self.verify_ring.generation + 1
            while any((next_generation + offset - 1) % len(self.verify_ring.records) == ticket.slot
                      for offset in range(retained_count)):
                next_generation += 1
            self.pp.generation = next_generation - 1
            cursor.control[:1].fill_(self.pp.generation)
            cursor.next(ticket.record, engram.histories)
            self._queue_next_device_round(request)
            for _ in range(retained_count - 1):
                tail = getattr(self, "device_round_tail", [])
                latest = tail[-1] if tail else self.device_round_queue
                cursor.next(latest[1].record, engram.histories)
                self._queue_next_device_round(request)
        self.audit["sampled_exact_repairs"] = self.audit.get("sampled_exact_repairs", 0) + 1

    def _discard_device_round(self):
        queued = self.device_round_queue
        if queued is None:
            return
        # Cancellation, request reuse and a capacity boundary can discard one
        # extra round. Drain its writes before releasing scheduler pages.
        torch.hpu.synchronize()
        for discarded in [queued, *getattr(self, "device_round_tail", [])]:
            self.verify_ring.release(discarded[1])
            getattr(self, "sampled_round_frames", {}).pop(discarded[1].generation, None)
        self.device_round_inputs.retire(queued[0])
        self.device_round_engram.retire(queued[0])
        self.device_round_queue = None
        self.device_round_tail = []

    def _warm_device_round_inputs(self):
        """Warm both control frames and readback slots before API readiness."""
        if getattr(self, "device_round_inputs", None) is None:
            return
        from types import SimpleNamespace

        cursor, engram = self.device_round_inputs, self.device_round_engram
        from vllm.sampling_params import SamplingParams

        start = int(cursor.positions[0].cpu())
        seed_tokens = cursor.ids.cpu().tolist()
        for parameters in (None, SamplingParams(temperature=1., top_p=.95, seed=42)):
            owner = "__v41_device_round_warmup__" if parameters is None else "__v41_sampled_round_warmup__"
            request = SimpleNamespace(req_id=owner, sampling_params=parameters, speculative_sampling=None)
            tokens = seed_tokens
            if parameters is not None:
                state = self._request_speculative_sampling(request)
                controls, _, _, _ = self.draw_speculative(state.parameters, state.seed, state.counter, state.offsets)
                first = torch.tensor([tokens[0]], dtype=torch.int64, device=self.device)
                positions = torch.arange(start + 1, start + 6, dtype=torch.int32, device=self.device)
                proposed, q, _, covered = self.sampled_propose(first, positions, controls)
                state.proposal.copy_(q)
                state.proposal_valid.copy_(covered.all().reshape(1))
                tokens = tokens[:1] + proposed.cpu().tolist()
            cursor.seed(owner, tokens, start, 64, self.model_config.max_model_len,
                        self.pp.generation + 1, [-1, -1, -1])
            previous = None
            for _ in range(4):
                rows = engram.prepare(owner, cursor.ids, cursor.history)
                hidden = self.model.forward_device_round(cursor.ids, cursor.positions, rows)
                if parameters is not None:
                    self._prepare_sampled_native_protocols(hidden, self.model.last_aux, state)
                if parameters is not None and envs.VLLM_HPU_DSV41_WARMUP_FIXTURE_DIR:
                    from vllm_gaudi.ops.deepseek_v41_micro_fixtures import export_warmup_fixture

                    export_warmup_fixture(self.model.program, hidden, cursor.ids, cursor.positions,
                                          envs.VLLM_HPU_DSV41_WARMUP_FIXTURE_DIR, proposal=state.proposal)
                queued = self._submit_round_verify(request, hidden, self.model.last_aux, None)
                # The real next consumer is queued before the previous result
                # is read, for both greedy and official sampled rounds.
                if previous is not None:
                    self.verify_ring.consume(previous[1])
                    self.verify_ring.release(previous[1])
                cursor.next(queued[1].record, engram.histories)
                previous = queued
            self.verify_ring.consume(previous[1])
            self.verify_ring.release(previous[1])
            torch.hpu.synchronize()
            cursor.retire(owner)
            engram.retire(owner)
        logger.info("V4.1 warmed device round controls, Engram, input producer and readback slots")

    def _finish_device_round(self, request, start, count, proposed, hidden):
        cursor, engram = self.device_round_inputs, self.device_round_engram
        queued = self.device_round_current
        initial = queued is None
        tokens = request.token_slice(start, start + 1) + proposed
        if initial:
            if cursor.owner is not None:
                raise RuntimeError("A device cursor survived its previous request transaction")
            history = self.model.engram_host.history
            lookback = [-1] * 3
            tail = history.history[-3:][::-1].tolist()
            lookback[:len(tail)] = tail
            cursor.seed(request.req_id, tokens, start, request.sampling_params.max_tokens - len(request.output),
                        self.model_config.max_model_len, self.pp.generation + 1, lookback)
            # The first C6 used the host reference inputs; build its device
            # prefix histories once. Later rounds consume these producers.
            engram.prepare(request.req_id, cursor.ids, cursor.history)
            queued = self._submit_round_verify(request, hidden, self.model.last_aux, self.round_context)
        self.device_round_current = None
        ticket, context = queued[1:]
        remaining = request.sampling_params.max_tokens - len(request.output)
        search = self.model.program.search_length
        # Scheduler pages are still authoritative. Do not execute ahead of an
        # unallocated page, a search-bucket transition or a short final tail.
        page_capacity = len(request.block_ids[0]) * self.vllm_config.cache_config.block_size
        visible_bound = self.model.program.decode_token_bound or search
        ahead = remaining >= 12 and start + 12 <= min(
            self.model_config.max_model_len, search, visible_bound, page_capacity)
        capacity = getattr(self, "device_round_lookahead", 1)
        bound = min(self.model_config.max_model_len, search, visible_bound, page_capacity)
        while ahead:
            retained = ([self.device_round_queue] if self.device_round_queue is not None else [])
            retained += getattr(self, "device_round_tail", [])
            if len(retained) >= capacity:
                break
            required = 6 * (len(retained) + 2)
            if remaining < required or start + required > bound:
                break
            predecessor = retained[-1] if retained else queued
            native = self.sampled_native_protocols.get(cursor.parity)
            published = (request.sampling_params.temperature != 0
                         and not request.req_id.startswith("__v41_")
                         and native is not None and native.input_publication is not None)
            with self._device_round_scope("advance"):
                cursor.next(predecessor[1].record, engram.histories, published=published)
            self._queue_next_device_round(request, published=published)
        ahead = self.device_round_queue is not None

        def consume(committed, output, draft):
            timing = getattr(self, "verify_timing", None)
            if timing is not None and timing.active is not None:
                if timing.active["generation"] != ticket.generation:
                    raise RuntimeError("Device round diagnostic belongs to another generation")
                timing.finish(committed, len(output))
            if initial:
                self.model.complete_step_device(committed)
            else:
                # The device history has already selected this prefix. Keep
                # the host mirror for cancellation, page boundaries and the
                # ordinary fallback; it does no lookup or device transfer.
                history = self.model.engram_host.history
                transaction = history.prepare(request.req_id, tokens)
                history.commit(transaction, committed)
            request.output.extend(output)
            self.audit["accepted_drafts"] += committed - 1
            self.audit["rejected_drafts"] += 6 - committed
            self.draft_token_ids = DraftTokenIds([request.req_id], [draft])
            self.round_context = context
            result = ModelRunnerOutput(req_ids=[request.req_id], req_id_to_index={request.req_id: 0},
                                       sampled_token_ids=[output])
            result.execution_rounds = self._record_round_completion(
                request, proposed, committed, output, generation=ticket.generation)
            self.pending = None
            if not ahead:
                cursor.retire(request.req_id)
                engram.retire(request.req_id)
            return result

        repair = ((lambda: self._repair_sampled_round(request, ticket))
                  if ticket.generation in getattr(self, "sampled_round_frames", {}) else None)

        def release(result):
            getattr(self, "sampled_round_frames", {}).pop(ticket.generation, None)
            if self.round_timing_enabled:
                self._record_round_released(result)

        return AsyncDSparkOutput(self.verify_ring, ticket, consume, release, repair=repair)

    def _sample_single(self):
        request, start, count, last_count, proposed, need_sample, selected = self.pending
        if proposed:
            raise RuntimeError("Ordinary sampling cannot consume a draft prefix")
        device_commit_enabled = getattr(self.pp, "device_commit_enabled", getattr(self.pp, "device_commit", False))
        device_commit = device_commit_enabled and need_sample and start >= request.decode_start
        token = None
        if self.pp.group.is_last_rank and need_sample and not device_commit:
            payload = getattr(self, "_device_sampling_payload", None)
            if payload is not None:
                host, done = self.tp4_token_readback(payload[0])
                done.synchronize()
                from vllm_gaudi.ops.deepseek_v41_sampling import unpack_sample_status

                token, covered = unpack_sample_status(host[0].tolist())
                if not covered:
                    token = self._repair_device_sample(payload, selected)
                self._device_sampling_payload = None
            elif self._token_copy is not None:
                host, done = self._token_copy
                done.synchronize()
                token = int(host[0, 0])
            else:
                token = int(selected.cpu()[0, 0])
        consumed, output = self.pp.finish_single_device() if device_commit else self.pp.finish_single(last_count, token)
        if consumed != last_count:
            raise RuntimeError("Ordinary PP completion did not consume the complete input chunk")
        self.model.complete_step(consumed)
        request.output.extend(output)
        # CPU completion publishes commit_host only. Its device commit tensor
        # is uninitialized (or belongs to an older generation); binding that
        # tensor here feeds stale IDs into otherwise correct native replay.
        # Only device completion owns a current device token. CPU completion
        # uses the scheduler/request token through the normal input staging.
        self._next_input = (request.req_id, start + count, self.pp.commit_token) if output and device_commit else None
        self._token_copy = None
        self.pending = self.draft_token_ids = None
        if not self.pp.group.is_last_rank:
            return None
        return ModelRunnerOutput(
            req_ids=[request.req_id], req_id_to_index={request.req_id: 0}, sampled_token_ids=[output]
        )

    def take_draft_token_ids(self):
        value, self.draft_token_ids = self.draft_token_ids, None
        return value if self.pp.group.is_last_rank else None

    @torch.inference_mode()
    def _dummy_run(self, tokens, *, native=False, start_position=0):
        logger.info(
            "V4.1 PP%d C%d warmup target start (native=%s, preceding steps=%d)",
            self.model.pp_rank,
            tokens,
            bool(native),
            self.audit["target_steps"],
        )
        self.state.clear()
        if (native and tokens == 1 and getattr(self.model.program, "device_sampling", False)
                and not getattr(self, "_device_sampling_completion_warmed", False)):
            from types import SimpleNamespace

            request = SimpleNamespace(req_id='__v41_sampling_warmup__', output=[],
                                      sampling_params=SimpleNamespace(temperature=1., top_p=1., top_k=-1, seed=42))
            self._prepare_device_sampling_request(request)
        hidden = self._forward(
            "__v41_warmup__",
            [1 + index for index in range(tokens)],
            start_position,
            decode=(native or getattr(self.model, "tensor_parallel_size", 2) == 4) and (self.use_dspark or tokens == 1),
            reset=True,
        )
        from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats

        stats = prepared_group_stats()
        logger.info(
            "V4.1 PP%d target submitted (native graphs=%d, replays=%d, entries=%d)",
            self.model.pp_rank,
            stats["native_graphs"],
            stats["native_replays"],
            stats["native_entry_replays"],
        )
        if self.pp.group.is_last_rank:
            if self.use_dspark and tokens > 6:
                self._insert(self.model.last_aux, self.positions[:tokens])
            elif (envs.VLLM_HPU_DSV41_DEVICE_VERIFY and self.verify_prefix is not None and envs.VLLM_HPU_DSV41_DSPARK):
                # Compile and exercise the fixed control graph during warmup;
                # this invocation is discarded with the warmup state.
                self.verify_hidden.zero_()
                self.verify_hidden[:tokens].copy_(hidden)
                if self.verify_aux.shape[1] != self.model.last_aux.shape[1]:
                    self.verify_aux = torch.empty(
                        (6, self.model.last_aux.shape[1]), dtype=self.model.last_aux.dtype, device=self.device
                    )
                self.verify_aux.zero_()
                self.verify_aux[:tokens].copy_(self.model.last_aux[:tokens])
                self.verify_positions.copy_(self.positions[0].to(torch.int32) + self.verify_offsets)
                self.verify_proposed.fill_(-1)
                self.verify_control_host.fill_(0)
                self.verify_metadata_host.copy_(
                    torch.as_tensor(
                        (1, tokens, 0, 64, start_position, self.model_config.max_model_len, 1), dtype=torch.int64
                    )
                )
                self.verify_control.copy_(self.verify_control_host, non_blocking=True)
                # Exercise the same producer tensors as serving. Padded
                # buffers have different alias/inference contracts from
                # native C6 outputs and do not warm the direct control path.
                direct_c6 = self._use_direct_verify_inputs(hidden, tokens)
                verify_hidden = hidden if direct_c6 else self.verify_hidden
                verify_aux = self.model.last_aux if direct_c6 else self.verify_aux
                verify_positions = self.position_views[6] if direct_c6 else self.verify_positions
                if self.native_draft_protocol is not None and direct_c6:
                    self.native_draft_protocol.prepare(verify_hidden, self.verify_proposed, self.verify_metadata,
                                                       verify_aux, verify_positions)
                else:
                    prefix = self.verify_prefix(verify_hidden, self.verify_proposed, self.verify_metadata, verify_aux,
                                                verify_positions)
                    self.draft_from_prefix(self.verify_metadata, verify_positions, prefix[1], prefix[2], prefix[3],
                                           prefix[4], prefix[5], prefix[6])
                sampled_aliases = getattr(self, "sampled_warm_aliases", set())
                if direct_c6 not in sampled_aliases:
                    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import SpeculativeRequestSampling

                    local_vocab = self.model.program.draft.output_head.weight.shape[0]
                    sampled_state = SpeculativeRequestSampling((1., .95, -1.), 42, local_vocab, self.device)
                    proposal_controls, acceptance, correction, target_controls = self.draw_speculative(
                        sampled_state.parameters, sampled_state.seed, sampled_state.counter, sampled_state.offsets)
                    prefix = self.sampled_verify_prefix(
                        verify_hidden, self.verify_proposed, sampled_state.proposal, self.verify_metadata,
                        verify_aux, verify_positions, target_controls, acceptance, correction)
                    warm_body = {"prepare": True} if self.native_draft_body is not None else {}
                    self.sampled_draft_from_prefix(
                        self.verify_metadata, verify_positions, *prefix[:6], proposal_controls, **warm_body)
                    sampled_aliases.add(direct_c6)
                    self.sampled_warm_aliases = sampled_aliases
            else:
                if self.use_dspark:
                    self.model.compute_logits(hidden)
                    self._insert(self.model.last_aux, self.positions[:tokens])
                    self._propose(1, start_position + tokens, diagnostic=True)
                else:
                    self.sample_target(hidden[-1:])
                    if (1, True) not in self.stochastic_samplers:
                        # Sampling is part of first-token latency. Warm the
                        # ordinary request path before readiness, independently
                        # of the discarded greedy startup tokens.
                        from types import SimpleNamespace

                        for top_p in (.95, 1.):
                            params = SimpleNamespace(temperature=1., top_p=top_p, top_k=-1, seed=42)
                            request = SimpleNamespace(req_id='__v41_sampling_warmup__',
                                                      sampling_params=params, output=[])
                            self._sample_requests(hidden[-1:], (request,))
                    if native and tokens == 1 and getattr(self.model.program, "device_sampling", False):
                        self._validate_device_sampling_warmup(hidden[-1:])
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
        if self.use_dspark:
            self.profile_phase_memory = []
            for phase in (("compile_warmup", "warmed_execution") if self.use_dspark else ("profile", )):
                if phase == "warmed_execution":
                    # Separate first-time compilation from serving allocations.
                    # Re-execute every qualified shape after compilation
                    # before measuring the persistent cache/working-state budget.
                    torch.hpu.synchronize()
                    torch.hpu.reset_peak_memory_stats()
                if isinstance(self.state, PagedStageState):
                    warmup_buckets = prefill_compute_buckets()
                    geometries = (0, min(INDEX_MME_HOT_TOKENS, self.model_config.max_model_len - 1))
                    for start_position in geometries:
                        for tokens in warmup_buckets:
                            if start_position + tokens > self.model_config.max_model_len:
                                continue
                            logger.info("V4.1 PP%d starting pre-KV C%d prefill recipe warmup at position %d",
                                        self.model.pp_rank, tokens, start_position)
                            self._dummy_run(tokens, start_position=start_position)
                            logger.info("V4.1 PP%d completed pre-KV C%d prefill recipe warmup at position %d",
                                        self.model.pp_rank, tokens, start_position)
                            if self.use_dspark:
                                self._record_profile_memory(phase, tokens, start_position)
                tokens = 6 if self.use_dspark else 1
                logger.info("V4.1 PP%d starting C%d memory profile", self.model.pp_rank, tokens)
                self._dummy_run(tokens)
                logger.info("V4.1 PP%d completed C%d memory profile", self.model.pp_rank, tokens)
                if self.use_dspark:
                    self._record_profile_memory(phase, tokens, 0)
            return
        self.profile_memory_steps = []

        def record_memory(tokens, position):
            # _dummy_run already drains the device and crosses the PP/TP
            # barrier. Read counters here without adding a synchronization or
            # resetting the cumulative peak used by cache admission.
            record = dict(
                tokens=tokens,
                position=position,
                allocated_bytes=torch.hpu.memory_allocated(),
                cumulative_peak_bytes=torch.hpu.max_memory_allocated(),
            )
            self.profile_memory_steps.append(record)
            logger.info("V4.1 initialization memory step: %s", record)

        record_memory(0, 0)
        if not self.use_dspark and isinstance(self.state, PagedStageState):
            cases = [(start, count) for start in (0, min(INDEX_MME_HOT_TOKENS,
                                                       self.model_config.max_model_len - 1))
                     for count in prefill_compute_buckets(self.prefill_capacity)
                     if start + count <= self.model_config.max_model_len]
            for start_position, tokens in cases:
                logger.info(
                    "V4.1 PP%d starting pre-KV C%d prefill recipe warmup at position %d",
                    self.model.pp_rank, tokens, start_position,
                )
                self._dummy_run(tokens, start_position=start_position)
                record_memory(tokens, start_position)
                logger.info(
                    "V4.1 PP%d completed pre-KV C%d prefill recipe warmup at position %d",
                    self.model.pp_rank, tokens, start_position,
                )
        tokens = 6 if self.use_dspark else 1
        logger.info("V4.1 PP%d starting C%d memory profile", self.model.pp_rank, tokens)
        self._dummy_run(tokens)
        record_memory(tokens, 0)
        logger.info("V4.1 PP%d completed C%d memory profile", self.model.pp_rank, tokens)

    def _record_profile_memory(self, phase, tokens, start_position):
        row = dict(phase=phase,
                   tokens=tokens,
                   start_position=start_position,
                   resident_bytes=torch.hpu.memory_allocated(),
                   peak_bytes=torch.hpu.max_memory_allocated())
        self.profile_phase_memory.append(row)
        logger.info("V4.1 PP%d TP%d profile phase memory: %s", self.model.pp_rank, self.model.tp_rank, row)

    @torch.inference_mode()
    def warmup_model(self):
        if self.request_batches is not None:
            self.request_batches.warmup()
            self.model.batch_state.restore_single_bindings()
        if getattr(self, "prefix_checkpoints", None) is not None:
            self.model.batch_state.warmup_single_handoff(self.state.blocks)
            # Profile-time pages are replaced by the scheduler pool before
            # this entry. Warm the actual serving tensor contracts, including
            # tail tiles in later search buckets, before freezing executors.
            for start, count in prefill_search_warmups(
                    self.model_config.max_model_len, self.prefill_capacity,
                    reuse_index_keys=envs.VLLM_HPU_DSV41_PREFILL_REINDEX_REUSE):
                logger.info("V4.1 PP%d warming serving C%d prefill at position %d",
                            self.model.pp_rank, count, start)
                self._dummy_run(count, start_position=start)
        if getattr(self, "prefix_checkpoints", None) is not None and self.prefill_capacity > 8192:
            from vllm_gaudi.ops.deepseek_v41_prefix_state import InlinePrefixCapture

            bank, program = self.model.batch_state, self.model.program
            count = max(prefill_compute_buckets(self.prefill_capacity))
            owner = "__v41_prefix_warmup__"
            slot = bank.acquire(owner)
            bank.publish_pages(slot, range(1, count // 128 + 1), self.state.blocks)
            bank.bind_prefill(slot)
            capture = InlinePrefixCapture(bank, slot, 0, count, count - 128)
            program.shared.inline_prefix_capture = capture
            program.prefill_halo_mode = "final" if program.stop == 40 else "full"
            try:
                self._dummy_run(count)
                capture.require_complete()
                bank.bind_single(slot, count)
                logger.info("V4.1 PP%d warmed slot-owned prefill and %d inline prefix states",
                            self.model.pp_rank, len(capture.tensors))
            finally:
                program.shared.inline_prefix_capture = None
                program.prefill_halo_mode = "full"
                bank.restore_single_bindings()
                bank.release(owner)
            self.state.clear()
        # Ordinary serving keeps the qualified C1 native replay for decode.
        # C6 is a DSpark-only anchor-plus-draft geometry. Prompt C128 recipes
        # are compiled by real prefill qualification and persisted in cache.
        # Fixed five-draft serving produces a C1 seed or C6 target. Accepted
        # prefixes still cover C1-C6 in the control graph below; they are not
        # separate target shapes. Retaining C2-C5 native target graphs here
        # exhausts the native HBM allocator before C6 can be prepared.
        for count in ((1, 6) if self.use_dspark else (1, )):
            runtime = count == 1 and getattr(self.model.program, "runtime_indexer", False)
            paged_dspark = self.use_dspark and isinstance(self.state, PagedStageState)
            geometries = (tuple(decode_search_warmups(self.model.program.length, runtime_indexer=runtime))
                          if runtime or paged_dspark else ((0, 512), ))
            additional = getattr(getattr(self, "vllm_config", None), "additional_config", {}) or {}
            geometries = select_native_warmup_geometries(geometries, additional)
            logger.info("V4.1 C%d native warmup searches: %s", count, [search for _, search in geometries])
            for start_position, search in geometries:
                for _ in range(4 if self.model.native else 1):
                    self._dummy_run(count, native=self.model.native, start_position=start_position)
                if self.model.native:
                    self.model.program.replay_owner.require_ready(count, search=search)
            self.graphed_buckets.add(count)
        if (
            not self.use_dspark
            and self.model.native
            and isinstance(self.state, PagedStageState)
            and not getattr(self.model.program, "runtime_indexer", False)
        ):
            # Prepare every reachable C1 search geometry before API readiness.
            # Otherwise a healthy stream pauses for compilation at each new
            # bucket, and already-warmed buckets remain untested at startup.
            for start, search in decode_search_warmups(
                self.model.program.length, runtime_indexer=getattr(self.model.program, "runtime_indexer", False)
            ):
                if start == 0:
                    continue
                logger.info("V4.1 PP%d preparing C1 search bucket %d", self.model.pp_rank, search)
                for _ in range(4):
                    self._dummy_run(1, native=True, start_position=start)
                self.model.program.replay_owner.require_ready(1, search=search)
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
        self._warm_device_round_inputs()
        self.pp.group.barrier()
        self.state.clear()
        self.active_request = None
        if getattr(self, "prefix_checkpoints", None) is not None:
            from vllm_gaudi.ops.deepseek_v41_prefill_regions import freeze_prefill_regions

            freeze_prefill_regions()
            # Inline capture preserves complete tiles; other checkpoint cuts
            # reuse the finite prefill buckets already warmed above.
        if envs.VLLM_HPU_DSV41_VERIFY_TIMING:
            direct_commit = (self.pp.single_stage
                             or (envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE and envs.VLLM_HPU_DSV41_INLINE_PP_COMMIT
                                 and not envs.VLLM_HPU_DSV41_COMPILED_PP_COMMIT))
            if not envs.VLLM_HPU_DSV41_DEVICE_VERIFY or not direct_commit:
                raise RuntimeError("Verify timing requires device verification and a direct commit path")
            from vllm_gaudi.ops.deepseek_v41_verify_timing import VerifyPhaseTiming
            from pathlib import Path

            self.verify_timing = VerifyPhaseTiming(
                Path(os.environ["DSV41_RUN_EVIDENCE"]) / "verify-phases",
                self.model.pp_rank * self.model.tensor_parallel_size + self.model.tp_rank)
            self.pp.timing = self.verify_timing
            self.verify_timing.calibrate()
            self.model.verify_timing = self.verify_timing

    def close(self):
        if isinstance(getattr(self, "batch_result", None), AsyncModelRunnerOutput):
            self.batch_result.get_output()
        self.pp.drain()
        self._discard_device_round()
        if getattr(self, "native_draft_protocol", None) is not None:
            self.native_draft_protocol.close()
        if getattr(self, "native_draft_body", None) is not None:
            self.native_draft_body.close()
        for plan in (*getattr(self, "sampled_native_protocols", {}).values(),
                     *getattr(self, "sampled_native_full_protocols", {}).values()):
            plan.close()
        if self.device_round_engram is not None:
            self.device_round_engram.close()
        if self.prefix_checkpoints is not None:
            self.prefix_checkpoints.close()
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
        rank = self.model.pp_rank * self.model.tensor_parallel_size + self.model.tp_rank
        (directory / f"rank{rank}.json").write_text(
            json.dumps({
                "rank": rank,
                "tensor_parallel_size": self.model.tensor_parallel_size,
                "sampling_owner": self.pp.group.is_last_rank,
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
        if row.get("device_round_pipeline"):
            row["enqueue_start_ns"] = row["start_ns"]
            previous = self.round_records[-2] if len(self.round_records) > 1 else None
            if previous is not None and previous["request_id"] == row["request_id"]:
                # Queue residence overlaps preceding rounds. Report delivered
                # completion periods, never sum overlapping queue latencies
                # into the complete-round throughput cost.
                row["start_ns"] = previous["end_ns"]
                row["timer_mode"] = "completion_period"

    def _record_round_completion(self, request, proposed, committed, output, *, generation=None):
        if not self.round_timing_enabled:
            return None
        context = self.round_context
        expected = self.pp.generation if generation is None else generation
        if context is None or context["request_id"] != request.req_id or context["generation"] != expected:
            raise RuntimeError("V4.1 round completion does not match its input generation")
        if len(self.round_records) >= 65536:
            raise RuntimeError("V4.1 round timing capacity exceeded")
        row = dict(
            context,
            end_ns=time.perf_counter_ns(),
            proposed_count=len(proposed),
            committed=committed,
            output_count=len(output),
            output=list(output),
        )
        self.round_records.append(row)
        self.round_context = None
        return [(row["request_id"], row["generation"], row["target_count"])]
