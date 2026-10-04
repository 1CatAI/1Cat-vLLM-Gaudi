# SPDX-License-Identifier: Apache-2.0
"""V4.1 adapter for vLLM V2 scheduling and asynchronous token ownership."""

from dataclasses import dataclass, field
import threading
import time

import torch

from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from vllm_gaudi import envs
from vllm_gaudi.ops.deepseek_v41_config import decode_source_prefix_bound, uses_v2, validate_v2
from vllm_gaudi.ops.deepseek_v41_native_trace import annotations_enabled, scope
from vllm_gaudi.ops.deepseek_v41_diagnostics import trace_phase
from vllm_gaudi.v1.worker.deepseek_v41_runner import (
    V41ModelRunner,
    logger,
    runtime_search_length,
    target_search_length,
)


def sampling_completion_helper_cpu():
    import os
    from vllm_gaudi.ops.deepseek_v4_config import parse_cpu_set

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    groups = os.environ.get("VLLM_HPU_DSV4_WORKER_HELPER_CPUS", "").split(";")
    return min(parse_cpu_set(groups[rank])) if 0 <= rank < len(groups) and groups[rank] else None


@dataclass(frozen=True)
class CompletionRecord:
    request_id: str
    generation: int
    start: int
    host: object
    done: object
    device_token: object
    fallback: object = None
    device_position: object = None
    _token_lock: object = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _token: int | None = field(default=None, init=False, repr=False, compare=False)

    def token(self) -> int:
        if annotations_enabled():
            with scope(f"v41::completion_consume::P{self.start}"):
                return self._consume_token()
        return self._consume_token()

    def _consume_token(self) -> int:
        with self._token_lock:
            if self._token is None:
                if annotations_enabled():
                    with scope(f"v41::completion_wait::P{self.start}"):
                        self.done.synchronize()
                    with scope(f"v41::completion_host_value::P{self.start}"):
                        values = self.host[0].tolist()
                else:
                    self.done.synchronize()
                    values = self.host[0].tolist()
                if self.fallback is not None:
                    from vllm_gaudi.ops.deepseek_v41_sampling import unpack_sample_status

                    token, covered = unpack_sample_status(values)
                    values = [token if covered else self.fallback()]
                elif len(values) != 1:
                    raise RuntimeError("Invalid V2 PP token completion")
                if values[0] < 0:
                    raise RuntimeError("Invalid V2 PP token completion")
                object.__setattr__(self, "_token", int(values[0]))
            return self._token


class V41AsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, record):
        self.record = record

    def get_output(self):
        if annotations_enabled():
            with scope(f"v41::async_output::P{self.record.start}"):
                return self._get_output()
        return self._get_output()

    def _get_output(self):
        token = self.record.token()
        return ModelRunnerOutput(
            req_ids=[self.record.request_id], req_id_to_index={self.record.request_id: 0}, sampled_token_ids=[[token]]
        )


class V41V2ModelRunner(V41ModelRunner):
    v2_completion = True

    def prepare_shutdown(self):
        """Join sampler repair while recipes and the communicator still live."""
        executor = getattr(self, "_sampling_completion_executor", None)
        if executor is None:
            return
        try:
            future = getattr(self, "_sampling_completion_future", None)
            if future is not None:
                future.result()
        finally:
            executor.shutdown(wait=True)
            self._sampling_completion_executor = None
            self._sampling_completion_future = None

    def close(self):
        self.prepare_shutdown()
        self._discard_device_step()
        super().close()

    shutdown_inc = close

    def __init__(self, vllm_config, is_driver_worker=False):
        if not uses_v2(vllm_config):
            raise ValueError("The V2 HPU adapter requires explicit VLLM_HPU_DSV41_V2 selection")
        validate_v2(vllm_config)
        super().__init__(vllm_config, is_driver_worker)
        self._completion = None
        self._input_committed = None
        self._prefix_started = None
        self._device_loop_enabled = envs.VLLM_HPU_DSV41_DEVICE_CLOSED_LOOP
        self._device_step = None
        self._device_loop_owner = None
        self._device_loop_position = None
        self._device_step_input = None
        self._sampling_prefix_handoff = envs.VLLM_HPU_DSV41_SAMPLING_PREFIX_HANDOFF
        # The device-token continuation owns one request generation.  A
        # multi-request scheduler step must use the ordinary synchronous
        # completion path so each request retires its PP packet before the
        # runner binds the next request slot.
        self._v2_async_step = False
        self._relay_token = torch.empty((1, 1), dtype=torch.int32, device=self.device)
        logger.info("V4.1 V2 HPU adapter: async output and device token continuation")

    @staticmethod
    def _identity(record):
        return record.request_id, record.generation, record.start

    def _commit_input(self, record):
        identity = self._identity(record)
        committed = self._input_committed
        if committed is not None and committed != identity:
            raise RuntimeError("V2 logical input commit belongs to another generation")
        if committed is None:
            if annotations_enabled():
                with scope(f"v41::continuation_retire::P{record.start}"):
                    self._complete_loop_input(record)
            else:
                self._complete_loop_input(record)
            self._input_committed = identity
            self.audit["v2_early_input_commits"] = self.audit.get("v2_early_input_commits", 0) + 1

    def _complete_loop_input(self, record):
        if getattr(self, "_device_step_input", None) == (record.request_id, record.start):
            # The scheduler has accepted this input. Only mirror its compressed
            # history; device producers already performed both row lookups.
            history = self.model.engram_host.history
            request = self.requests[record.request_id]
            batch = history.prepare_mirror(record.request_id, [request.tokens[record.start]])
            history.commit(batch, 1)
            self._device_step_input = None
        else:
            self.model.complete_step(1)

    def _device_ahead_authorized(self, request, start):
        program = self.model.program
        following = start + 1
        if request.req_id not in self.requests:
            return False
        maximum = request.sampling_params.max_tokens
        if maximum is not None and len(request.output) + 1 >= maximum:
            return False
        if (not getattr(self, "_device_loop_enabled", False) or getattr(request, "mm_features", ())
                or not getattr(program, "device_next_position", False)
                or not self.model.native or program.dspark or not self.pp.group.is_first_rank
                or not self.pp.group.is_last_rank or self._device_step is not None):
            return False
        # The scheduler owns allocation, including the next position's page.
        # Do not speculate into an unallocated page or an uncaptured geometry.
        if following >= self.model_config.max_model_len or len(request.block_ids) != 1:
            return False
        from vllm_gaudi.ops.deepseek_v41_paged_attention import PAGE_TOKENS

        if following >= len(request.block_ids[0]) * PAGE_TOKENS:
            return False
        search = runtime_search_length(following, 1, program.length)
        bound = decode_source_prefix_bound(following + 1, search, self.model.tensor_parallel_size,
                                          runtime_indexer=program.runtime_indexer)
        return (search == program.search_length and bound == program.decode_token_bound
                and program.replay_owner.input_variant_ready(search))

    def _discard_device_step(self):
        step, self._device_step = self._device_step, None
        if step is not None:
            step.drain()
            self.audit["device_loop_discards"] = self.audit.get("device_loop_discards", 0) + 1
        self._device_loop_position = None

    def _queue_device_step(self, request, start, frame):
        from vllm_gaudi.ops.deepseek_v41_device_loop import DeviceStep

        inputs = self.model.device_decode_inputs()
        owner = id(request), request.req_id
        rows = inputs.prepare(owner, frame[3].reshape(-1), frame[5])
        hidden = self.model.forward_device_input(frame[3], frame[4], rows)
        variant = self.model.program.replay_owner._complete_input_variant()
        self._device_step = DeviceStep(request.req_id, start + 1, hidden, variant.metadata.native_completion, frame)
        self._device_loop_position = start + 2
        self.audit["device_loop_queued"] = self.audit.get("device_loop_queued", 0) + 1

    @torch.inference_mode()
    def _repair_loop_sample(self, frame, destination):
        # Commands already submitted to the device are drained and discarded;
        # their outputs are never published. Repair uses the preserved draw.
        step = self._device_step
        if step is not None:
            step.drain()
            self._device_step = None
        token = self._repair_device_sample(frame[:5], destination)
        if step is not None:
            self._queue_device_step(self.requests[step.request_id], step.start - 1, frame)
            self.audit["device_loop_recomputes"] = self.audit.get("device_loop_recomputes", 0) + 1
        return token

    @trace_phase
    def _forward(self, request_id, tokens, start, **kwargs):
        step = getattr(self, "_device_step", None)
        if step is not None:
            if step.request_id == request_id and step.start == start and kwargs.get("decode"):
                self._device_step = None
                self._device_step_input = request_id, start
                self.audit["target_steps"] += 1
                self.audit["target_tokens"] += 1
                self.audit["decode_steps"] += 1
                self.audit["device_loop_consumed"] = self.audit.get("device_loop_consumed", 0) + 1
                return step.hidden
            self._discard_device_step()
        output = super()._forward(request_id, tokens, start, **kwargs)
        if getattr(self, "_device_loop_enabled", False):
            from vllm_gaudi.ops.deepseek_v41_device_loop import DeviceInputTransaction

            if isinstance(self.model.step_ticket, DeviceInputTransaction):
                self._device_loop_position = start + 1
        return output

    @staticmethod
    def _continuation_authorized(record, scheduled):
        if scheduled is None or record.request_id in scheduled.finished_req_ids:
            return False
        if getattr(scheduled, "auxiliary_prefix_operations", None) is not None:
            return False
        if record.request_id in (getattr(scheduled, "preempted_req_ids", None) or ()):
            return False
        tokens = scheduled.num_scheduled_tokens
        proposed = getattr(scheduled, "scheduled_spec_decode_tokens", {}).get(record.request_id, ())
        return len(tokens) == 1 and tokens.get(record.request_id) == 1 and not proposed

    def _prefix_authorized(self, record, scheduled):
        added_pages = None
        if not (envs.VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX and self.pp.group.is_first_rank
                and self._continuation_authorized(record, scheduled)):
            return False
        if getattr(self.model, "tensor_parallel_size", 2) == 4:
            cached = scheduled.scheduled_cached_reqs
            if record.request_id not in cached.req_ids or record.request_id in cached.resumed_req_ids:
                return False
            index = cached.req_ids.index(record.request_id)
            if cached.num_computed_tokens[index] != record.start + 1:
                self.audit["v2_prefix_state_transitions"] = self.audit.get("v2_prefix_state_transitions", 0) + 1
                return False
            added_pages = cached.new_block_ids[index]
        program = self.model.program
        next_search = (runtime_search_length(record.start + 1, 1, program.length) if getattr(
            program, "runtime_indexer", False) else target_search_length(record.start + 1, 1, program.length))
        next_bound = decode_source_prefix_bound(
            record.start + 2,
            next_search,
            getattr(self.model, "tensor_parallel_size", 2),
            runtime_indexer=getattr(program, "runtime_indexer", False),
        )
        if hasattr(program, "decode_token_bound") and next_bound != program.decode_token_bound:
            # The scheduler's next full entry must rebind this bounded
            # selection geometry before a prefix may consume the new token.
            self.audit["v2_prefix_visible_transitions"] = self.audit.get("v2_prefix_visible_transitions", 0) + 1
            return False
        shared = getattr(program, "shared", None)
        if 512 < next_search <= getattr(shared, "index_mirror_tokens", 0) and not shared.index_mirror_valid:
            # The normal entry restores this derived state from the currently
            # bound pages before the first segmented consumer can read it.
            self.audit["v2_prefix_index_restores"] = self.audit.get("v2_prefix_index_restores", 0) + 1
            return False
        ready = self.model.decode_prefix_ready(next_search)
        if not ready:
            self.audit["v2_prefix_bucket_captures"] = self.audit.get("v2_prefix_bucket_captures", 0) + 1
        if next_search != getattr(program, "search_length", min(512, program.length)):
            # _forward owns the attention and rotary bindings.  At a geometry
            # transition the complete native entry must switch them before a
            # segmented prefix can be launched safely.
            self.audit["v2_prefix_bucket_transitions"] = self.audit.get("v2_prefix_bucket_transitions", 0) + 1
            return False
        if ready and added_pages is not None:
            state = getattr(self, "state", None)
            if not hasattr(state, "append_single_pages") or getattr(self.model, "batch_state", None) is not None:
                self.audit["v2_prefix_state_transitions"] = self.audit.get("v2_prefix_state_transitions", 0) + 1
                return False
            request = self.requests[record.request_id]
            if (len(request.block_ids) != 1 or len(added_pages) != 1 or not state.append_single_pages(
                    record.request_id, request.block_ids[0], added_pages[0], self.state.blocks)):
                self.audit["v2_prefix_state_transitions"] = self.audit.get("v2_prefix_state_transitions", 0) + 1
                return False
            self.audit["v2_prefix_page_appends"] = self.audit.get("v2_prefix_page_appends", 0) + 1
        return ready

    def _consume_completion(self, scheduled=None):
        record = self._completion
        if (
            record is not None
            and getattr(self, "trace_enabled", False)
            and getattr(self.model, "tensor_parallel_size", 2) == 4
        ):
            with scope(f"v41::worker_commit::PP0::decode::P{record.start}::C1::emit1"):
                return self._consume_completion_impl(scheduled)
        return self._consume_completion_impl(scheduled)

    def _consume_completion_impl(self, scheduled=None):
        record = self._completion
        if record is None:
            return
        request = self.requests.get(record.request_id)
        if request is None or self.active_request != record.request_id:
            raise RuntimeError("V2 completion outlived its request generation")
        if record.generation != self.pp.generation or len(request.tokens) != record.start + 1:
            raise RuntimeError("V2 completion does not extend the worker's exact token prefix")
        if record.fallback is not None and getattr(self, "_certificate_before_staging", False):
            record.token()
        early = envs.VLLM_HPU_DSV41_V2_EARLY_INPUT_COMMIT
        identity = self._identity(record)
        if early:
            self._commit_input(record)
        if getattr(self, "_device_loop_enabled", False):
            authorized = False
        elif annotations_enabled():
            with scope(f"v41::continuation_authorize::P{record.start + 1}"):
                authorized = self._prefix_authorized(record, scheduled)
        else:
            authorized = self._prefix_authorized(record, scheduled)
        if authorized:
            if not early or self.position_bank is None:
                raise RuntimeError("V2 segmented prefix requires committed history and fixed positions")
            if self._prefix_started is not None and self._prefix_started != identity:
                raise RuntimeError("V2 prefix replay belongs to another completion generation")
            if self._prefix_started is None:
                if record.device_position is not None:
                    position = record.device_position
                elif getattr(self.model, "tensor_parallel_size", 2) == 4:
                    # A changing view offset would create a new compiled
                    # input contract every token. The fixed destination is
                    # ordered after its previous consumer by the device copy.
                    position = self.position_views[1]
                    if annotations_enabled():
                        with scope(f"v41::continuation_position::P{record.start + 1}"):
                            self.position_bank.copy_into(position, record.start + 1)
                    else:
                        self.position_bank.copy_into(position, record.start + 1)
                else:
                    position = self.position_bank.view(record.start + 1, 1)
                # Position staging and ownership checks do not consume the
                # sampled token. Keep them before the certificate wait so
                # they can overlap the previous device invocation. Resolve
                # every TP certificate before embedding/Engram/KV consume
                # the provisional candidate; rare repair still runs once.
                if record.fallback is not None:
                    record.token()
                if envs.VLLM_HPU_DSV41_V2_DEVICE_ENGRAM:
                    if annotations_enabled():
                        with scope(f"v41::continuation_engram::P{record.start + 1}"):
                            self.model.prepare_device_engram(record.request_id, record.device_token)
                    else:
                        self.model.prepare_device_engram(record.request_id, record.device_token)
                    self.audit["v2_device_engram_starts"] = self.audit.get("v2_device_engram_starts", 0) + 1
                self.model.begin_decode_prefix(record.device_token, position)
                if record.fallback is not None and getattr(self, "_sampling_prefix_handoff", False):
                    # A cached certificate no longer performs the original
                    # post-prefix blocking read. Yield the interpreter after
                    # enqueueing, before preparing the late-input suffix.
                    # This diagnostic candidate remains off until qualified.
                    time.sleep(0)
                self._prefix_started = identity
                self.audit["v2_prefix_starts"] = self.audit.get("v2_prefix_starts", 0) + 1
        token = record.token()
        self.pp.complete_packet()
        self.pp.commits += 1
        if not early:
            self._complete_loop_input(record)
        request.output.append(token)
        self._next_input = (request.req_id, record.start + 1, record.device_token)
        self._next_position = ((request.req_id, record.start + 1, record.device_position)
                               if record.device_position is not None else None)
        self._completion = None
        self._input_committed = None
        self.audit["v2_worker_commits"] = self.audit.get("v2_worker_commits", 0) + 1

    @torch.inference_mode()
    @trace_phase
    def execute_model(self, scheduled):
        if getattr(self, "trace_enabled", False):
            self._step_trace_start = time.perf_counter_ns(), time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        if (
            scheduled.num_scheduled_tokens
            or scheduled.finished_req_ids
            or getattr(scheduled, "preempted_req_ids", None)
        ):
            self._consume_completion(scheduled)
        self._v2_async_step = (
            len(scheduled.num_scheduled_tokens) == 1 and getattr(scheduled, "auxiliary_prefix_operations", None) is None
        )
        try:
            output = super().execute_model(scheduled)
        finally:
            self._v2_async_step = False
        if self._prefix_started is not None and self.pp.group.is_first_rank:
            if self.model.decode_prefix_pending:
                raise RuntimeError("Authorized V2 prefix was not consumed by its suffix")
            self._prefix_started = None
            self.audit["v2_prefix_finishes"] = self.audit.get("v2_prefix_finishes", 0) + 1
        return output

    @torch.inference_mode()
    @trace_phase
    def sample_tokens(self, grammar_output=None):
        if (
            getattr(self.model, "tensor_parallel_size", 2) == 4
            and self.pending == "batch_ready"
            and isinstance(self.batch_result, V41AsyncOutput)
        ):
            if grammar_output is not None:
                raise ValueError("V4.1 device continuation does not support grammar sampling")
            # Non-output ranks keep the completion record for their next
            # authorized step. Synchronously consuming it here would block
            # three TP4 workers before they can enqueue their next prefix.
            result, self.batch_result, self.pending = self.batch_result, None, None
            return result if self.model.tp_rank == 0 else None
        result = super().sample_tokens(grammar_output)
        if self.pending == "batch_ready":
            if self.batch_result is not None:
                raise RuntimeError("V2 sampling did not take the prepared batch result")
            # The V2 completion record, rather than the base batch marker,
            # owns the device event until the next scheduled or finish step.
            self.pending = None
        return result

    @trace_phase
    def _update(self, scheduled):
        if getattr(self, "_device_step", None) is not None:
            step = self._device_step
            if (not self._continuation_authorized(
                    step, scheduled)
                    or getattr(scheduled, "auxiliary_prefix_operations", None) is not None):
                self._discard_device_step()
        retiring = set(scheduled.finished_req_ids) | set(getattr(scheduled, "preempted_req_ids", ()) or ())
        owner = getattr(self, "_device_loop_owner", None)
        if owner is not None and owner[1] in retiring:
            inputs = self.model.device_decode_inputs()
            if inputs.owner is not None:
                inputs.retire(owner)
            self._device_loop_owner = self._device_loop_position = None
        request_batches = self.request_slots_enabled
        for new in scheduled.scheduled_new_reqs:
            operations = getattr(scheduled, "auxiliary_prefix_operations", None)
            checkpoint = operations.restores.get(new.req_id) if operations is not None else None
            cached_prefix = (
                request_batches and checkpoint is not None and checkpoint.num_tokens == new.num_computed_tokens
            )
            if new.num_computed_tokens and not cached_prefix:
                raise ValueError("V2 HPU request resumption requires full recomputation from position zero")
            if new.req_id in self.requests and new.req_id not in scheduled.finished_req_ids:
                raise ValueError("V2 HPU cannot replace a live request without a finish event")
            prefill = getattr(new, "prefill_token_ids", None)
            if prefill is not None and prefill != new.prompt_token_ids and not request_batches:
                raise ValueError("V2 HPU replay of a generated prefix is not qualified")
        cached = scheduled.scheduled_cached_reqs
        if cached.resumed_req_ids and not request_batches:
            raise ValueError("V2 HPU cached request resumption is not qualified")
        for index, req_id in enumerate(cached.req_ids):
            if cached.num_output_tokens[index] != len(self.requests[req_id].output):
                raise RuntimeError("V2 scheduler placeholder count disagrees with the worker's committed prefix")
        super()._update(scheduled)

    @trace_phase
    def _sample_single(self):
        if not getattr(self, "_v2_async_step", True):
            return super()._sample_single()
        request, start, count, last_count, proposed, need_sample, selected = self.pending
        if start < request.decode_start or not need_sample:
            return super()._sample_single()
        if proposed or count != 1 or last_count != 1 or self._completion is not None:
            raise RuntimeError("V2 token relay requires one unmatched ordinary decode completion")
        from vllm_gaudi.ops.deepseek_v41_completion import resolve_device_runtime

        self.pp.drain()
        self.pp.generation += 1
        token = selected if self.pp.group.is_last_rank else self._relay_token
        tp_size = getattr(self.model, "tensor_parallel_size", 2)
        if tp_size == 2:
            self.pp.group.broadcast(token, src=1)
        elif tp_size != 4 or not (self.pp.group.is_first_rank and self.pp.group.is_last_rank):
            raise RuntimeError("TP4 device continuation requires a single pipeline stage")
        payload = getattr(self, "_device_sampling_payload", None)
        frame = None
        ahead = payload is not None and self._device_ahead_authorized(request, start)
        if ahead:
            from vllm_gaudi.ops.deepseek_v41_device_loop import SamplingFrames

            # Commit the current input transaction before seeding its lookback.
            from types import SimpleNamespace

            current = SimpleNamespace(request_id=request.req_id, generation=self.pp.generation, start=start)
            self._commit_input(current)
            inputs = self.model.device_decode_inputs()
            owner = id(request), request.req_id
            if self._device_loop_owner != owner or self._device_loop_position != start + 1:
                import numpy as np

                if inputs.owner is not None and inputs.owner != owner:
                    inputs.retire(inputs.owner)

                history = self.model.engram_host.history.history[-3:][::-1]
                suffix = np.full(3, -1, dtype=np.int32)
                suffix[:len(history)] = history
                inputs.history_views[-1].copy_(torch.from_numpy(suffix))
                self._device_loop_owner = owner
            if not hasattr(self, "_sampling_frames"):
                self._sampling_frames = SamplingFrames(payload, inputs.history_views[-1])
            frame = self._sampling_frames.preserve(payload, inputs.history_views[-1])
            payload, token = frame[:5], frame[3]
        source = payload[0] if payload is not None else token
        if tp_size == 4:
            # load_model already verified the native ABI and retained this
            # producer. Resolving again would hash runtime libraries and scan
            # /proc/self/maps on every token, serializing the serving loop.
            host, done = self.tp4_token_readback(source)
        else:
            bridge, _ = resolve_device_runtime(tp_size)
            host, done = bridge.copy_sampled_tokens_to_host(source)
        if ahead:
            self._queue_device_step(request, start, frame)
        fallback = ((lambda: self._repair_loop_sample(frame, token)) if ahead else
                    (lambda: self._repair_device_sample(payload, token))) if payload is not None else None
        position = payload[4] if payload is not None and len(payload) == 5 else None
        record = CompletionRecord(request.req_id, self.pp.generation, start, host, done, token.view(1), fallback,
                                  position)
        self._device_sampling_payload = None
        if fallback is not None:
            # The non-output workers do not serialize AsyncOutput. Start
            # certificate consumption on every worker so rare full-vocabulary
            # repair can rendezvous before rank zero publishes a token.
            if not hasattr(self, "_sampling_completion_executor"):
                from concurrent.futures import ThreadPoolExecutor
                import os

                helper = sampling_completion_helper_cpu()

                def bind_completion_thread():
                    if helper is not None:
                        os.sched_setaffinity(0, {helper})

                self._sampling_completion_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="v41-sample", initializer=bind_completion_thread
                )
            self._sampling_completion_future = self._sampling_completion_executor.submit(record.token)
        self._completion = record
        self.pending = self.draft_token_ids = self._token_copy = None
        self.audit["v2_async_completions"] = self.audit.get("v2_async_completions", 0) + 1
        return V41AsyncOutput(record) if self.pp.group.is_last_rank else None

    @torch.inference_mode()
    def _validate_device_loop_repair(self, payload, expected):
        """Prove same-position replay overwrites a discarded ordinary step.

        This runs only at startup, with sparse canonical row snapshots. No
        snapshots, comparisons or host tensor reads enter the hot loop.
        """
        from types import SimpleNamespace
        from vllm_gaudi.ops.deepseek_v41_device_loop import bitwise_equal
        from vllm_gaudi.ops.deepseek_v41_replay import _InputFeedbackSnapshot, _PagedSnapshot

        program = self.model.program
        variant = program.replay_owner._complete_input_variant()
        inputs = self.model.device_decode_inputs()
        history = torch.full((3,), -1, dtype=torch.int32, device=self.device)
        frame = self._sampling_frames.preserve(payload, history)
        before = _InputFeedbackSnapshot(_PagedSnapshot(program, frame[4], variant.states), variant.fixed[2:4])
        saved_rows = tuple(value.clone() for value in (*inputs.rows, inputs.histories))
        request = SimpleNamespace(req_id="device-loop-startup-repair")
        owner = id(request), request.req_id
        self.requests[request.req_id] = request
        try:
            frame[3].copy_(expected)
            self._queue_device_step(request, 0, frame)
            self._device_step.drain()
            # Warmup snapshots are regular Torch copies, outside the native
            # completion's ownership. Drain them at this diagnostic boundary.
            torch.hpu.synchronize()
            reference_hidden = self._device_step.hidden.clone()
            reference = _PagedSnapshot(program, frame[4], variant.states)
            torch.hpu.synchronize()
            self._device_step = None
            before.restore()
            frame[3].copy_((expected + 1).remainder(129280))
            self._queue_device_step(request, 0, frame)
            result = self._repair_loop_sample(frame, frame[3])
            self._device_step.drain()
            torch.hpu.synchronize()
            if result != int(expected.cpu().reshape(-1)[0]):
                raise RuntimeError("Device loop repair changed the preserved inverse-CDF draw")
            pairs = [(self._device_step.hidden, reference_hidden)]
            pairs.extend(zip(reference.small.tensors, reference.small.saved, strict=True))
            pairs.extend((value.index_select(0, rows), saved) for value, rows, saved in reference.rows)
            # Read bytes only at startup. This also diagnoses the actual
            # changed rows without depending on an HPU equality reduction.
            host_pairs = [(value.detach().cpu(), saved.detach().cpu()) for value, saved in pairs]
            checks = [bitwise_equal(value, saved) for value, saved in host_pairs]
            if not all(checks):
                names = {id(value): name for name, value in program.named_buffers()}
                labels = ["hidden"] + [names.get(id(value), "small_state") for value in reference.small.tensors]
                labels += [names.get(id(value), "paged_state") for value, _, _ in reference.rows]
                logger.error("Device loop repair changed rollback bytes: %s",
                             [label for label, passed in zip(labels, checks, strict=True) if not passed])
                for label, passed, (value, saved) in zip(labels, checks, host_pairs, strict=True):
                    if not passed:
                        actual = value.contiguous().reshape(-1).view(torch.uint8)
                        expected_bytes = saved.contiguous().reshape(-1).view(torch.uint8)
                        changed = (actual != expected_bytes).nonzero().flatten()
                        first = changed[:16]
                        logger.error("Repair byte difference %s: shape=%s bytes=%d offsets=%s actual=%s expected=%s",
                                     label, tuple(value.shape), changed.numel(), first.tolist(),
                                     actual[first].tolist(), expected_bytes[first].tolist())
                raise RuntimeError("Discarded device step leaked state into same-position repair")
            if not bool(torch.isfinite(host_pairs[0][0]).all()):
                raise RuntimeError("Device loop repair produced a nonfinite hidden state")
            self.audit["device_loop_repair_warmup_passed"] = True
        finally:
            self._discard_device_step()
            before.restore()
            for destination, source in zip((*inputs.rows, inputs.histories), saved_rows, strict=True):
                destination.copy_(source)
            inputs.retire(owner)
            self.requests.pop(request.req_id)
            program.replay_owner.latest_tail = None

    def _validate_device_sampling_warmup(self, hidden):
        """Exercise the actual scalar copy, four-worker repair and completion owner."""
        if getattr(self, "_device_sampling_completion_warmed", False):
            return
        from types import SimpleNamespace

        payload = self.model.program.replay_owner.sampling_tail_values(hidden)
        if payload is None:
            raise RuntimeError("Device sampling warmup did not retain its native tail outputs")
        if getattr(self, "_device_loop_enabled", False) and not hasattr(self, "_sampling_frames"):
            from vllm_gaudi.ops.deepseek_v41_device_loop import SamplingFrames

            inputs = self.model.device_decode_inputs()
            saved = tuple(value.clone() for value in (*inputs.rows, inputs.histories))
            history = torch.full((3,), -1, dtype=torch.int32, device=self.device)
            self._sampling_frames = SamplingFrames(payload, history)
            for _ in range(2):
                self._sampling_frames.preserve(payload, history)
                inputs.prepare("startup", payload[3].reshape(-1), history)
            torch.hpu.synchronize()
            for destination, source in zip((*inputs.rows, inputs.histories), saved, strict=True):
                destination.copy_(source)
            inputs.retire("startup")
        owner = self._device_sampling_owner
        # The startup sampler leaves the unfiltered request parameters active.
        # This guarantees the candidate certificate requests the full fallback.
        if owner[2:5] != (1., 1., -1):
            raise RuntimeError("Device sampling fallback warmup requires unfiltered startup controls")
        previous = self.device_sampling_stats[owner[1]]["fallbacks"]
        request = SimpleNamespace(req_id=owner[1], decode_start=0)
        self.pending = request, 0, 1, 1, False, True, payload[3]
        self._device_sampling_payload = payload
        self._v2_async_step = True
        try:
            self._sample_single()
            result = self._completion.token()
            if self.device_sampling_stats[owner[1]]["fallbacks"] != previous + 1:
                raise RuntimeError("Device sampling warmup did not exercise the complete fallback")
            expected = self._sample_full_local(payload[1], payload[2], filtered=False)
            host, done = self.tp4_token_readback(expected)
            done.synchronize()
            if int(host[0, 0]) != result:
                raise RuntimeError("Device sampling fallback changed the startup inverse-CDF result")
            # The forced top_p=1 certificate warms the unfiltered repair.
            # A real nucleus miss also needs the filtered recipe with these
            # actual native-tail tensor layouts and the destination copy.
            # Discard this synthetic draw with the rest of startup state.
            filtered = self._sample_full_local(payload[1], payload[2], filtered=True)
            payload[3].copy_(filtered)
            host, done = self.tp4_token_readback(payload[3])
            done.synchronize()
            if getattr(self, "_device_loop_enabled", False):
                self._validate_device_loop_repair(payload, expected)
        finally:
            self._v2_async_step = False
            self._completion = None
            self._device_sampling_payload = None
            self.pending = None
        self._device_sampling_completion_warmed = True
        logger.info("V4.1 TP%d device sampling scalar completion and full fallback warmup passed", self.model.tp_rank)
