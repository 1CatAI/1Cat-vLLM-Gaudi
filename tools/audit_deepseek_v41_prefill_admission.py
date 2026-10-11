# SPDX-License-Identifier: Apache-2.0
"""Replay prefill admission on the real scheduler without loading a model.

The tiny local model supplies scheduler configuration only. Request token IDs,
the DSpark slot property, scheduler and prepared chunk/halo functions are real.
No hardware timing or model correctness claim is made by this diagnostic.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-token-ids', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import OPTConfig
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig, SpeculativeConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler
    from vllm_gaudi.v1.worker.deepseek_v41_runner import target_chunks
    from vllm_gaudi.ops.deepseek_v41_decoder_halo import decoder_halo_mode
    from vllm_gaudi.ops.deepseek_v41_prefill_capacity import (
        prefill_capacity, prefill_target_tokens, reserve_dspark_input_slots,
    )

    tiny = args.output / 'scheduler-only-model'
    OPTConfig(architectures=['OPTForCausalLM'], hidden_size=32, ffn_dim=64,
              num_hidden_layers=1, num_attention_heads=2,
              max_position_embeddings=262144).save_pretrained(tiny)
    tokens = json.loads(args.prompt_token_ids.read_text())
    if not isinstance(tokens, list) or len(tokens) != 16384:
        raise ValueError('Use the saved exact 16K formal prompt token IDs')
    reports = []
    for label, dspark, budget in [('C1', False, 16384), ('DSpark-current', True, 16384),
                                  ('DSpark-separate-slot-budget', True, 16384)]:
        config = VllmConfig(
            model_config=ModelConfig(model=str(tiny), dtype='float16', max_model_len=262144),
            parallel_config=ParallelConfig(),
            cache_config=CacheConfig(block_size=128, enable_prefix_caching=False),
            scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=budget,
                                             max_model_len=262144, enable_chunked_prefill=True,
                                             is_encoder_decoder=False, async_scheduling=True, watermark=0.0))
        # Model validation is deliberately separate: the toy model cannot
        # draft. This uses the real validated profile's forced parallel mode
        # and exact SpeculativeConfig property in the scheduler constructor.
        if dspark:
            spec = object.__new__(SpeculativeConfig)
            for name, field in SpeculativeConfig.__pydantic_fields__.items():
                object.__setattr__(spec, name, field.get_default(call_default_factory=True))
            for name, value in dict(method='dspark', num_speculative_tokens=5, parallel_drafting=True,
                                    disable_eagle_block_drop=True).items():
                object.__setattr__(spec, name, value)
            config.speculative_config = spec
        if label == 'DSpark-separate-slot-budget':
            reserve_dspark_input_slots(config)
            # Engine/worker deserialization must not reserve the slots twice.
            reserve_dspark_input_slots(config)
        config.cache_config.num_gpu_blocks = 2056
        register_all_kvcache_specs(config)
        cache = KVCacheConfig(num_blocks=2056, kv_cache_tensors=[],
                              kv_cache_groups=[KVCacheGroupSpec(
                                  ['layer'], FullAttentionSpec(block_size=128, num_kv_heads=1,
                                                              head_size=1, dtype=torch.float32))])
        scheduler = HPUAsyncScheduler(vllm_config=config, kv_cache_config=cache, block_size=128,
                                      log_stats=False, structured_output_manager=StructuredOutputManager(config))
        params = SamplingParams(temperature=1., top_p=.95, seed=42, max_tokens=32768)
        params.update_from_generation_config({}, 50256)
        request = Request(request_id=label, prompt_token_ids=tokens, sampling_params=params, pooling_params=None)
        scheduler.add_request(request)
        rounds = []
        for _ in range(3):
            start = request.num_computed_tokens
            scheduled = scheduler.schedule()
            count = scheduled.num_scheduled_tokens[label]
            blocks = [dict(start=start + offset, rows=len(part),
                           halo=decoder_halo_mode(start + offset, len(part), len(tokens), eligible=True,
                                                  block_tokens=16384, allow_single_block=True))
                      for offset, part in target_chunks(tokens[start:start + count], 16384)]
            rounds.append(dict(start=start, scheduled=count, worker_calls=len(blocks),
                               scalar_calls=sum(b['rows'] == 1 for b in blocks),
                               large_shapes=[b['rows'] for b in blocks if b['rows'] > 1],
                               halo_modes={m: sum(b['halo'] == m for b in blocks)
                                           for m in ('full', 'final', 'prefix_only')}))
            if request.num_computed_tokens >= len(tokens):
                break
            scheduler.update_from_output(scheduled, ModelRunnerOutput(
                req_ids=[label], req_id_to_index={label: 0}, sampled_token_ids=[[]]))
        reports.append(dict(profile=label, admission_budget=config.scheduler_config.max_num_batched_tokens,
                            target_compute_budget=prefill_target_tokens(config.scheduler_config),
                            worker_prefill_capacity=prefill_capacity(prefill_target_tokens(config.scheduler_config), 4),
                            draft_slots=config.speculative_config.max_num_new_slots_for_drafting if dspark else 0,
                            schedule=rounds, worker_calls=sum(r['worker_calls'] for r in rounds),
                            scalar_calls=sum(r['scalar_calls'] for r in rounds)))
    result = dict(scope=__doc__, prompt_token_ids=str(args.prompt_token_ids), prompt_tokens=len(tokens),
                  hardware_runs=0, formal_gain_credit=False, profiles=reports)
    (args.output / 'admission.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
