# SPDX-License-Identifier: Apache-2.0
"""V4.1 adapter for vLLM V2 scheduling and asynchronous token ownership."""

from dataclasses import dataclass, field
import threading

import torch

from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from vllm_gaudi import envs
from vllm_gaudi.ops.deepseek_v41_config import uses_v2, validate_v2
from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner, logger, target_search_length


@dataclass(frozen=True)
class CompletionRecord:
    request_id: str
    generation: int
    start: int
    host: object
    done: object
    device_token: object
    _token_lock: object = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _token: int | None = field(default=None, init=False, repr=False, compare=False)

    def token(self) -> int:
        with self._token_lock:
            if self._token is None:
                self.done.synchronize()
                values = self.host[0].tolist()
                if len(values) != 1 or values[0] < 0:
                    raise RuntimeError("Invalid V2 PP token completion")
                object.__setattr__(self, "_token", int(values[0]))
            return self._token


class V41AsyncOutput(AsyncModelRunnerOutput):

    def __init__(self, record):
        self.record = record

    def get_output(self):
        token = self.record.token()
        return ModelRunnerOutput(req_ids=[self.record.request_id],
                                 req_id_to_index={self.record.request_id: 0},
                                 sampled_token_ids=[[token]])


class V41V2ModelRunner(V41ModelRunner):
    v2_completion = True

    def __init__(self, vllm_config, is_driver_worker=False):
        if not uses_v2(vllm_config):
            raise ValueError("The V2 HPU adapter requires explicit VLLM_HPU_DSV41_V2 selection")
        validate_v2(vllm_config)
        super().__init__(vllm_config, is_driver_worker)
        self._completion = None
        self._input_committed = None
        self._prefix_started = None
        self._relay_token = torch.empty((1, 1), dtype=torch.int32, device=self.device)
        logger.info("V4.1 V2 HPU adapter: async output, device PP token relay, native continuation")

    @staticmethod
    def _identity(record):
        return record.request_id, record.generation, record.start

    def _commit_input(self, record):
        identity = self._identity(record)
        committed = self._input_committed
        if committed is not None and committed != identity:
            raise RuntimeError("V2 logical input commit belongs to another generation")
        if committed is None:
            self.model.complete_step(1)
            self._input_committed = identity
            self.audit["v2_early_input_commits"] = self.audit.get("v2_early_input_commits", 0) + 1

    @staticmethod
    def _continuation_authorized(record, scheduled):
        if scheduled is None or record.request_id in scheduled.finished_req_ids:
            return False
        tokens = scheduled.num_scheduled_tokens
        proposed = getattr(scheduled, "scheduled_spec_decode_tokens", {}).get(record.request_id, ())
        return len(tokens) == 1 and tokens.get(record.request_id) == 1 and not proposed

    def _prefix_authorized(self, record, scheduled):
        if not (envs.VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX and self.pp.group.is_first_rank
                and self._continuation_authorized(record, scheduled)):
            return False
        program = self.model.program
        next_search = (program.length if getattr(program, "runtime_indexer", False) else
                       target_search_length(record.start + 1, 1, program.length))
        ready = self.model.decode_prefix_ready(next_search)
        if not ready:
            self.audit["v2_prefix_bucket_captures"] = self.audit.get("v2_prefix_bucket_captures", 0) + 1
        if next_search != getattr(program, "search_length", min(512, program.length)):
            # _forward owns the attention/rotary bindings. Until it switches
            # buckets, begin_decode_prefix would start the previous variant,
            # even when the next variant has already been captured. Run this
            # boundary token through the complete native entry after binding;
            # subsequent tokens resume the early segmented prefix.
            self.audit["v2_prefix_bucket_transitions"] = self.audit.get("v2_prefix_bucket_transitions", 0) + 1
            return False
        return ready

    def _consume_completion(self, scheduled=None):
        record = self._completion
        if record is None:
            return
        request = self.requests.get(record.request_id)
        if request is None or self.active_request != record.request_id:
            raise RuntimeError("V2 completion outlived its request generation")
        if record.generation != self.pp.generation or len(request.tokens) != record.start + 1:
            raise RuntimeError("V2 completion does not extend the worker's exact token prefix")
        early = envs.VLLM_HPU_DSV41_V2_EARLY_INPUT_COMMIT
        identity = self._identity(record)
        if early:
            self._commit_input(record)
        if self._prefix_authorized(record, scheduled):
            if not early or self.position_bank is None:
                raise RuntimeError("V2 segmented prefix requires committed history and fixed positions")
            if self._prefix_started is not None and self._prefix_started != identity:
                raise RuntimeError("V2 prefix replay belongs to another completion generation")
            if self._prefix_started is None:
                position = self.position_bank.view(record.start + 1, 1)
                if envs.VLLM_HPU_DSV41_V2_DEVICE_ENGRAM:
                    self.model.prepare_device_engram(record.request_id, record.device_token)
                    self.audit["v2_device_engram_starts"] = self.audit.get("v2_device_engram_starts", 0) + 1
                self.model.begin_decode_prefix(record.device_token, position)
                self._prefix_started = identity
                self.audit["v2_prefix_starts"] = self.audit.get("v2_prefix_starts", 0) + 1
        token = record.token()
        self.pp.complete_packet()
        self.pp.commits += 1
        if not early:
            self.model.complete_step(1)
        request.output.append(token)
        self._next_input = (request.req_id, record.start + 1, record.device_token)
        self._completion = None
        self._input_committed = None
        self.audit["v2_worker_commits"] = self.audit.get("v2_worker_commits", 0) + 1

    @torch.inference_mode()
    def execute_model(self, scheduled):
        if scheduled.num_scheduled_tokens or scheduled.finished_req_ids:
            self._consume_completion(scheduled)
        output = super().execute_model(scheduled)
        if self._prefix_started is not None and self.pp.group.is_first_rank:
            if self.model.decode_prefix_pending:
                raise RuntimeError("Authorized V2 prefix was not consumed by its suffix")
            self._prefix_started = None
            self.audit["v2_prefix_finishes"] = self.audit.get("v2_prefix_finishes", 0) + 1
        return output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output=None):
        result = super().sample_tokens(grammar_output)
        if self.pending == "batch_ready":
            if self.batch_result is not None:
                raise RuntimeError("V2 sampling did not take the prepared batch result")
            # The V2 completion record, rather than the base batch marker,
            # owns the device event until the next scheduled or finish step.
            self.pending = None
        return result

    def _update(self, scheduled):
        for new in scheduled.scheduled_new_reqs:
            if new.num_computed_tokens:
                raise ValueError("V2 HPU request resumption requires full recomputation from position zero")
            if new.req_id in self.requests and new.req_id not in scheduled.finished_req_ids:
                raise ValueError("V2 HPU cannot replace a live request without a finish event")
            prefill = getattr(new, "prefill_token_ids", None)
            if prefill is not None and prefill != new.prompt_token_ids:
                raise ValueError("V2 HPU replay of a generated prefix is not qualified")
        cached = scheduled.scheduled_cached_reqs
        if cached.resumed_req_ids:
            raise ValueError("V2 HPU cached request resumption is not qualified")
        for index, req_id in enumerate(cached.req_ids):
            if cached.num_output_tokens[index] != len(self.requests[req_id].output):
                raise RuntimeError("V2 scheduler placeholder count disagrees with the worker's committed prefix")
        super()._update(scheduled)

    def _sample_single(self):
        request, start, count, last_count, proposed, need_sample, selected = self.pending
        if start < len(request.prompt) or not need_sample:
            return super()._sample_single()
        if proposed or count != 1 or last_count != 1 or self._completion is not None:
            raise RuntimeError("V2 token relay requires one unmatched ordinary decode completion")
        from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime
        self.pp.drain()
        self.pp.generation += 1
        token = selected if self.pp.group.is_last_rank else self._relay_token
        self.pp.group.broadcast(token, src=1)
        bridge, _, _ = _resolve_runtime()
        host, done = bridge.copy_sampled_tokens_to_host(token)
        record = CompletionRecord(request.req_id, self.pp.generation, start, host, done, token.view(1))
        self._completion = record
        self.pending = self.draft_token_ids = self._token_copy = None
        self.audit["v2_async_completions"] = self.audit.get("v2_async_completions", 0) + 1
        return V41AsyncOutput(record) if self.pp.group.is_last_rank else None
