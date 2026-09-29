# SPDX-License-Identifier: Apache-2.0
"""Exercise real scheduling and output commits with one owned PP transaction.

The fixture follows vLLM's scheduler tests, but uses a local tiny config.
No model weights, hardware kernels or remote configuration are required.
"""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from transformers import OPTConfig

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler


@pytest.fixture
def scheduler_factory(tmp_path):
    OPTConfig(architectures=["OPTForCausalLM"],
              hidden_size=32,
              ffn_dim=64,
              num_hidden_layers=1,
              num_attention_heads=2,
              max_position_embeddings=8192).save_pretrained(tmp_path)

    def make(reentry):
        config = VllmConfig(model_config=ModelConfig(model=str(tmp_path), dtype="float16", max_model_len=8192),
                            parallel_config=ParallelConfig(),
                            cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
                            scheduler_config=SchedulerConfig(max_num_seqs=32,
                                                             max_num_batched_tokens=8192,
                                                             max_model_len=8192,
                                                             enable_chunked_prefill=True,
                                                             is_encoder_decoder=False,
                                                             async_scheduling=True,
                                                             watermark=0.0))
        config.cache_config.num_gpu_blocks = 10000
        register_all_kvcache_specs(config)
        cache = KVCacheConfig(num_blocks=10000,
                              kv_cache_tensors=[],
                              kv_cache_groups=[
                                  KVCacheGroupSpec(["layer"],
                                                   FullAttentionSpec(block_size=16,
                                                                     num_kv_heads=1,
                                                                     head_size=1,
                                                                     dtype=torch.float32))
                              ])
        scheduler = HPUAsyncScheduler(vllm_config=config,
                                      kv_cache_config=cache,
                                      block_size=16,
                                      log_stats=True,
                                      structured_output_manager=StructuredOutputManager(config))
        scheduler.pp_size = 2
        scheduler.use_pp = True
        scheduler.use_v2_model_runner = True
        # The fixture models the V4.1 completed-transaction contract while
        # using a tiny local config for the real scheduler's KV allocator.
        scheduler._completed_batch_reentry = reentry
        return scheduler

    return make


def create_requests(count):
    result = []
    for index in range(count):
        params = SamplingParams(temperature=0, max_tokens=64)
        params.update_from_generation_config({}, 50256)
        result.append(
            Request(request_id=str(index),
                    prompt_token_ids=[index + 1] * 2048,
                    sampling_params=params,
                    pooling_params=None))
    return result


def complete(scheduler, output):
    ids = list(output.num_scheduled_tokens)
    samples = [[100 + index] if not scheduler.requests[rid].is_prefill_chunk else [] for index, rid in enumerate(ids)]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(req_ids=ids, req_id_to_index={
            rid: i
            for i, rid in enumerate(ids)
        }, sampled_token_ids=samples))


def run_wave(scheduler, concurrency):
    requests = create_requests(concurrency)
    for request in requests:
        scheduler.add_request(request)
    buckets = Counter()
    for _ in range(160):
        if not scheduler.has_requests():
            break
        out = scheduler.schedule()
        # Only count windows with the whole wave still resident.
        if (all(request.num_computed_tokens >= request.num_prompt_tokens for request in requests)
                and all(request.status == RequestStatus.RUNNING for request in requests)):
            buckets[len(out.num_scheduled_tokens)] += 1
        complete(scheduler, out)
    assert not scheduler.has_requests()
    assert all(request.num_output_tokens == 64 for request in requests)
    return buckets


@pytest.mark.parametrize("concurrency", [1, 2, 4, 8, 16, 32])
def test_completed_transactions_rejoin_next_decode_batch(scheduler_factory, concurrency):
    buckets = run_wave(scheduler_factory(True), concurrency)
    assert buckets[concurrency] >= 50
    assert not buckets[0]


def test_upstream_pp_cadence_fragments_a_completed_c32_wave(scheduler_factory):
    before = run_wave(scheduler_factory(False), 32)
    after = run_wave(scheduler_factory(True), 32)
    assert before[16] >= 50 and before[32] == 0
    assert after[32] >= 50


def test_uncommitted_transaction_keeps_original_cadence(scheduler_factory):
    scheduler = scheduler_factory(True)
    request, = create_requests(1)
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert request.num_output_placeholders == 1
    eligible = request.next_decode_eligible_step
    assert eligible > scheduler.current_step
    # Do not complete the owned generation: a premature next schedule must
    # not reuse the request's writable state or consume its placeholder.
    premature = scheduler.schedule()
    assert request.request_id not in premature.num_scheduled_tokens
    complete(scheduler, output)
    assert request.num_output_placeholders == 0
    assert request.next_decode_eligible_step <= scheduler.current_step
    next_output = scheduler.schedule()
    assert next_output.num_scheduled_tokens == {request.request_id: 1}


def test_cancelled_request_cannot_be_readmitted_by_late_output(scheduler_factory):
    scheduler = scheduler_factory(True)
    request, = create_requests(1)
    scheduler.add_request(request)
    output = scheduler.schedule()
    eligible = request.next_decode_eligible_step
    scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(req_ids=[request.request_id],
                          req_id_to_index={request.request_id: 0},
                          sampled_token_ids=[[100]]))
    assert request.next_decode_eligible_step == eligible
    assert request.num_output_tokens == 0
    assert request.request_id not in scheduler.schedule().num_scheduled_tokens


@pytest.mark.parametrize("model,batch,v2,queue,spec,enabled", [
    ("deepseek_v41", "1", "1", 1, None, True),
    ("deepseek_v41", "1", "1", 3, None, False),
    ("deepseek_v41", "0", "1", 1, None, False),
    ("deepseek_v41", "1", "0", 1, None, False),
    ("deepseek_v41", "1", "1", 1, "dspark", False),
    ("qwen3", "1", "1", 1, None, False),
])
def test_automatic_selection_requires_completed_transaction_contract(monkeypatch, model, batch, v2, queue, spec,
                                                                     enabled):
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model)),
                             speculative_config=spec,
                             max_concurrent_batches=queue)
    monkeypatch.setattr(AsyncScheduler, "__init__", lambda self: setattr(self, "vllm_config", config))
    monkeypatch.setenv("VLLM_HPU_DSV41_BATCH_DECODE", batch)
    monkeypatch.setenv("VLLM_HPU_DSV41_V2", v2)
    assert HPUAsyncScheduler()._completed_batch_reentry is enabled
