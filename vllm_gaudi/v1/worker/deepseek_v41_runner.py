# SPDX-License-Identifier: Apache-2.0
"""Bounded V4.1 runner with scheduler-owned CSA2 state and PP verify commits.

The token/accepted-prefix contract follows vLLM #53577 (d2b1b735).
Transport uses persistent HPU buffers and HCCL, with one outstanding request.
"""

from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from vllm.distributed import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import DraftTokenIds, EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from vllm_gaudi import envs
from vllm_gaudi.extension.logger import logger as init_logger
from vllm_gaudi.extension.profiler import HabanaHighLevelProfiler
from vllm_gaudi.ops.deepseek_v41_state import StageStateBlocks, register_state_spec

logger = init_logger()


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


class PPBuffers:
    def __init__(self, device, capacity=6, *, dspark=True):
        self.group = get_pp_group()
        self.hidden = torch.empty(capacity, 4, 5120, dtype=torch.bfloat16, device=device)
        self.pre = torch.empty(capacity, 4, dtype=torch.float32, device=device)
        # generation, committed input count, output count, draft count,
        # six output tokens and five draft tokens.
        self.dspark = dspark
        self.device_commit = envs.VLLM_HPU_DSV41_DEVICE_COMMIT
        if self.device_commit and (dspark or not envs.VLLM_HPU_DSV41_GRAPH_REPLAY):
            raise ValueError("Device completion requires ordinary V4.1 native decode")
        self.commit = torch.empty(15 if dspark else 4, dtype=torch.int64 if dspark else torch.int32, device=device)
        self.commit_token = self.commit[3:4] if not dspark else None
        self.commit_row = self.commit.view(1, 4) if not dspark else None
        if self.device_commit:
            self.commit.zero_()
        self.generation = 0
        self.pending = []
        self.sends, self.receives, self.commits = 0, 0, 0
        self.packed = None
        if envs.VLLM_HPU_DSV41_NATIVE_PP_COPY and (dspark or not envs.VLLM_HPU_DSV41_PACKED_PP
                                                 or not envs.VLLM_HPU_DSV41_GRAPH_REPLAY):
            raise ValueError("Native PP copy requires ordinary packed C1 graph replay")
        if envs.VLLM_HPU_DSV41_PACKED_PP and not dspark:
            from vllm_gaudi.ops.deepseek_v41_pp import PackedC1Buffers
            self.packed = PackedC1Buffers(device, native_copy=envs.VLLM_HPU_DSV41_NATIVE_PP_COPY)

    def drain(self):
        for work in self.pending:
            work.wait()
        self.pending.clear()

    def exchange(self, values, count, *, decode=False):
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
        self.drain()
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
        self.commits += 1
        return record[1], record[4:4 + record[2]], record[10:10 + record[3]]

    def finish_single(self, consumed=None, token=None):
        if self.dspark:
            raise RuntimeError("Ordinary token completion cannot commit DSpark verification")
        self.drain()
        self.generation += 1
        if self.group.is_last_rank:
            record = [self.generation, consumed, int(token is not None), -1 if token is None else token]
            self.commit.copy_(torch.tensor(record, dtype=torch.int32, device="cpu"))
        self.group.broadcast(self.commit, src=1)
        record = self.commit.cpu().tolist()
        if record[0] != self.generation or record[2] not in (0, 1):
            raise RuntimeError("Stale or invalid PP ordinary-token completion")
        self.commits += 1
        self.complete_packet()
        return record[1], [record[3]] if record[2] else []

    def complete_packet(self):
        if self.packed is not None:
            self.packed.complete()

    def finish_single_device(self):
        if not self.device_commit or self.dspark:
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


class V41ModelRunner:
    _PAD_BLOCK_ID, _PAD_SLOT_ID = 0, 0

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
        # Generic multimodal batching uses the platform's pageable path;
        # Engram owns separate buffers pinned explicitly through the HPU API.
        self.pin_memory = False
        self.profiler = HabanaHighLevelProfiler()
        self.model_memory_usage = self.mem_margin = 0
        self.pp = PPBuffers(self.device, dspark=self.use_dspark)
        self.input_ids = torch.empty(6, dtype=torch.int64, device=self.device)
        self.positions = torch.empty(6, dtype=torch.int32, device=self.device)
        self.input_views = {count: self.input_ids[:count] for count in range(1, 7)}
        self.direct_token_ids = envs.VLLM_HPU_DSV41_DIRECT_TOKEN_IDS and not self.use_dspark
        self.decode_ids = torch.empty(1, dtype=torch.int32, device=self.device) if self.direct_token_ids else None
        self.position_views = {count: self.positions[:count] for count in range(1, 7)}
        self.position_bank = None
        if envs.VLLM_HPU_DSV41_FIXED_POSITIONS and not self.use_dspark:
            from vllm_gaudi.ops.deepseek_v41_inputs import PositionBank
            self.position_bank = PositionBank(self.model_config.max_model_len, 6, self.device)
        self.audit = {"target_steps": 0, "target_tokens": 0, "draft_steps": 0,
                      "accepted_drafts": 0, "rejected_drafts": 0, "requests": 0}

    def load_model(self):
        before = torch.hpu.memory_allocated()
        if envs.VLLM_HPU_DSV41_GRAPH_REPLAY:
            from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
            initialize_tp2_fused_ar_norm_runtime()
        self.model = get_model(vllm_config=self.vllm_config)
        self.state = StageStateBlocks(self.model.program)
        register_state_spec(self.vllm_config)
        self.model_memory_usage = torch.hpu.memory_allocated() - before
        if self.pp.group.is_last_rank and envs.VLLM_HPU_DSV41_DSPARK:
            draft = self.model.program.draft
            self.insert_context = torch.compile(draft.insert_context, backend="hpu_backend", fullgraph=True,
                                                dynamic=False)
            self.run_draft = torch.compile(draft, backend="hpu_backend", fullgraph=True, dynamic=False)
            self.sample_draft = torch.compile(draft.sample_greedy, backend="hpu_backend", fullgraph=True, dynamic=False)
        elif self.pp.group.is_last_rank:
            self.sample_target = torch.compile(self.model.program.sample_greedy, backend="hpu_backend",
                                               fullgraph=True, dynamic=False)
            if self.pp.device_commit:
                self.sample_target_commit = torch.compile(self.model.program.sample_greedy_commit,
                                                          backend="hpu_backend", fullgraph=True, dynamic=False)
        logger.info("V4.1 PP%d prepared weights loaded; allocated %d bytes",
                    self.model.pp_rank, self.model_memory_usage)

    def get_model(self):
        return self.model

    def get_supported_tasks(self):
        return ("generate",)

    def reset_encoder_cache(self):
        self.encoder_cache.clear()

    def get_kv_cache_spec(self):
        logger.info("V4.1 PP%d exposes %d scheduler state arrays (%d bytes/request)", self.model.pp_rank,
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

    def _bind_request(self, request):
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
            self.requests.pop(req_id, None)
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
        if params is None:
            raise ValueError("V4.1 runner requires sampling parameters")
        if (params.temperature != 0 or params.logprobs is not None or params.prompt_logprobs is not None
                or params.presence_penalty != 0 or params.frequency_penalty != 0 or params.repetition_penalty != 1
                or params.allowed_token_ids is not None or params.bad_words or params.logit_bias
                or params.structured_outputs is not None):
            raise ValueError("The initial V4.1 runner supports unmodified greedy sampling; unsupported options rejected")

    def _image_embeddings(self, request, start, count):
        if not request.mm_features or not self.pp.group.is_first_rank:
            return None
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
        HPUModelRunner._execute_mm_encoder(self, None, [request.req_id])
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

    def _forward(self, request_id, tokens, start, *, decode, reset=False, request=None):
        count = len(tokens)
        ids, positions = self.input_views[count], self.position_views[count]
        if self.direct_token_ids and decode:
            if count != 1:
                raise ValueError("Direct V4.1 token binding requires ordinary C1 decode")
            ids = self.decode_ids
        if (not self.use_dspark and decode and count == 1 and self._next_input is not None
                and self._next_input[:2] == (request_id, start)):
            if self.direct_token_ids:
                ids = self._next_input[2]
            else:
                ids.copy_(self._next_input[2])
        else:
            ids.copy_(torch.tensor(tokens, dtype=ids.dtype, device="cpu"))
        if self.position_bank is not None and decode:
            # Native fixed-input staging consumes this offset view directly.
            # A framework copy would lower the view before each decoder call.
            positions = self.position_bank.view(start, count)
        else:
            positions.copy_(torch.arange(start, start + count, dtype=torch.int32, device="cpu"))
        self.model.prepare_step(request_id, tokens, is_decode=decode, reset=reset)
        if self.pp.group.is_first_rank:
            embeddings = self._image_embeddings(request, start, count) if request is not None else None
            value = self.model(ids, positions, inputs_embeds=embeddings)
            self.pp.exchange(value, count, decode=decode)
            output = None
        else:
            value = self.pp.exchange(None, count, decode=decode)
            output = self.model(ids, positions, intermediate_tensors=value)
        self.audit["target_steps"] += 1
        self.audit["target_tokens"] += count
        return output

    def _insert(self, aux, positions):
        if envs.VLLM_HPU_DSV41_DSPARK and self.pp.group.is_last_rank:
            self.insert_context(aux, positions)

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

    @torch.inference_mode()
    def execute_model(self, scheduled):
        if self.pending is not None:
            raise RuntimeError("Previous V4.1 execution has not completed sampling/verify")
        self._update(scheduled)
        if not scheduled.num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT
        if len(scheduled.num_scheduled_tokens) != 1:
            raise ValueError("V4.1 initial prepared profile admits one request")
        req_id, count = next(iter(scheduled.num_scheduled_tokens.items()))
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
        if decode and count not in ((1, 6) if self.use_dspark else (1,)):
            raise RuntimeError("Unexpected partial DSpark verify; draft proposal must respect the request budget")
        for offset in range(0, count, 6):
            chunk = tokens[offset:offset + 6]
            hidden = self._forward(req_id, chunk, start + offset, decode=decode,
                                   reset=start + offset == 0, request=request)
            if offset + len(chunk) < count:
                self._insert(self.model.last_aux, self.positions[:len(chunk)])
                self.model.complete_step(len(chunk))
        need_sample = start + count >= len(request.tokens)
        logits = None
        if self.pp.group.is_last_rank and need_sample:
            if not self.use_dspark and decode and self.pp.device_commit:
                logits = self.sample_target_commit(hidden[-1:], self.pp.commit)
            else:
                logits = self.model.compute_logits(hidden) if self.use_dspark else self.sample_target(hidden[-1:])
            if not self.use_dspark and self.model.native and not (decode and self.pp.device_commit):
                from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime
                bridge, _, _ = _resolve_runtime()
                self._token_copy = bridge.copy_sampled_tokens_to_host(logits)
        self.pending = (request, start, count, len(chunk), proposed, need_sample, logits)
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output=None):
        if grammar_output is not None:
            raise ValueError("V4.1 prepared greedy verification does not support grammar sampling")
        if self.pending is None:
            return EMPTY_MODEL_RUNNER_OUTPUT
        if not self.use_dspark:
            return self._sample_single()
        request, start, count, last_count, proposed, need_sample, logits = self.pending
        output, draft, committed = [], [], last_count
        if self.pp.group.is_last_rank:
            if need_sample:
                target = logits.argmax(-1).cpu().tolist()
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
        return ModelRunnerOutput(req_ids=[request.req_id], req_id_to_index={request.req_id: 0},
                                 sampled_token_ids=[output])

    def _sample_single(self):
        request, start, count, last_count, proposed, need_sample, selected = self.pending
        if proposed:
            raise RuntimeError("Ordinary sampling cannot consume a draft prefix")
        device_commit = self.pp.device_commit and need_sample and start >= len(request.prompt)
        token = None
        if self.pp.group.is_last_rank and need_sample and not device_commit:
            if self._token_copy is not None:
                host, done = self._token_copy
                done.synchronize()
                token = int(host[0, 0])
            else:
                token = int(selected.cpu()[0, 0])
        consumed, output = (self.pp.finish_single_device() if device_commit else
                            self.pp.finish_single(last_count, token))
        if consumed != last_count:
            raise RuntimeError("Ordinary PP completion did not consume the complete input chunk")
        self.model.complete_step(consumed)
        request.output.extend(output)
        self._next_input = (request.req_id, start + count, self.pp.commit_token) if output else None
        self._token_copy = None
        self.pending = self.draft_token_ids = None
        if not self.pp.group.is_last_rank:
            return None
        return ModelRunnerOutput(req_ids=[request.req_id], req_id_to_index={request.req_id: 0},
                                 sampled_token_ids=[output])

    def take_draft_token_ids(self):
        value, self.draft_token_ids = self.draft_token_ids, None
        return value if self.pp.group.is_last_rank else None

    @torch.inference_mode()
    def _dummy_run(self, tokens, *, native=False):
        logger.info("V4.1 PP%d C%d warmup target start (native=%s, preceding steps=%d)",
                    self.model.pp_rank, tokens, bool(native), self.audit["target_steps"])
        self.state.clear()
        hidden = self._forward("__v41_warmup__", [1 + index for index in range(tokens)], 0,
                               decode=native, reset=True)
        from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats
        stats = prepared_group_stats()
        logger.info("V4.1 PP%d target submitted (native graphs=%d, replays=%d, entries=%d)",
                    self.model.pp_rank, stats["native_graphs"], stats["native_replays"],
                    stats["native_entry_replays"])
        if self.pp.group.is_last_rank:
            if self.use_dspark:
                self.model.compute_logits(hidden)
            else:
                self.sample_target(hidden[-1:])
            self._insert(self.model.last_aux, self.positions[:tokens])
            if envs.VLLM_HPU_DSV41_DSPARK:
                self._propose(1, tokens, diagnostic=True)
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
        logger.info("V4.1 PP%d starting C6 memory profile", self.model.pp_rank)
        self._dummy_run(6)
        logger.info("V4.1 PP%d completed C6 memory profile", self.model.pp_rank)

    def warmup_model(self):
        for count in ((1, 6) if self.use_dspark else (1,)):
            for _ in range(4 if self.model.native else 1):
                self._dummy_run(count, native=self.model.native)
            if self.model.native:
                self.model.program.replay_owner.require_ready(count)
            self.graphed_buckets.add(count)
        self.state.clear()
        self.active_request = None

    def close(self):
        self.pp.drain()
        if self.model is not None:
            self.model.close()

    shutdown_inc = close
