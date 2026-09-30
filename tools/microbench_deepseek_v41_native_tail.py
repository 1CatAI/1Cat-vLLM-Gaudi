# SPDX-License-Identifier: Apache-2.0
"""Real vocabulary head -> distributed argmax -> next input, before stage integration."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--iterations', type=int, default=128)
    parser.add_argument('--split-baseline', action='store_true')
    parser.add_argument('--fixed-input', action='store_true')
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    for name in ('PT_HPU_RECIPE_CACHE_CONFIG', 'GRAPH_VISUALIZATION_DIR', 'HABANA_LOGS'):
        if '{rank}' in os.environ.get(name, ''):
            os.environ[name] = os.environ[name].replace('{rank}', str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared)
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from torch import nn
    import torch.distributed as dist
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.distributed import init_distributed_environment, initialize_model_parallel, tensor_model_parallel_all_gather
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives, _Snapshot, _Metadata
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_sampling import local_greedy_candidate, select_greedy_candidate
    from vllm_gaudi.models.deepseek_v41_program import PreparedInput, _compile_group
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.tp2_prepared_plan import (
        register_tp2_prepared_group_pass, collect_prepared_group_replays,
        record_native_decoder_outputs, replay_native_decoder, prepared_group_stats,
    )
    torch.set_num_threads(1)
    size = int(os.environ['WORLD_SIZE'])
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    init_distributed_environment(world_size=size, rank=rank, local_rank=rank,
                                 distributed_init_method='env://', backend='hccl')
    config = EngineArgs(model=str(args.prepared), dtype='bfloat16', tensor_parallel_size=size, pipeline_parallel_size=1,
                        load_format='dsv41_prepared', max_model_len=1048576, max_num_seqs=32,
                        max_num_batched_tokens=8192, block_size=128, enable_prefix_caching=False, async_scheduling=True).create_engine_config()
    out = Path(os.environ['DSV41_RUN_EVIDENCE']) / f'tail-rank{rank}.json'
    report = {'rank':rank, 'scope':'actual sharded BF16 head -> exact peer candidate -> next sharded embedding consumer',
              'formal_gain_credit':False, 'checks':[], 'timing':[]}

    def save():
        out.write_text(json.dumps(report, indent=2)+'\n')

    with set_current_vllm_config(config), torch.inference_mode():
        initialize_model_parallel(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
        initialize_tp2_fused_ar_norm_runtime()
        reduce, gather = stage_collectives(rank, True, size, native_fp32_gather=True)
        reduce(torch.ones(1, device='hpu', dtype=torch.bfloat16)).cpu()
        bind_worker_helpers(rank)
        register_tp2_prepared_group_pass()
        shard = PreparedV41Shard(args.prepared, 0, rank)
        head = shard.tensor('head.weight', 'hpu').bfloat16()
        embedding = nn.Module()
        embedding.register_buffer('weight', shard.tensor('embed.weight', 'hpu'))
        input_module = PreparedInput(embedding, rank, reduce)

        class Tail(nn.Module):
            def __init__(self, native):
                super().__init__()
                self.register_buffer('head', head)
                self.input = input_module
                self.gather = gather if native else tensor_model_parallel_all_gather

            def forward(self, value):
                logits = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(value.contiguous(), self.head)
                selected = select_greedy_candidate(self.gather(local_greedy_candidate(logits, rank), dim=-1)).int()
                residual, pre = self.input(selected.reshape(-1))
                return selected, residual, pre

        class NativePlan(nn.Module):
            def __init__(self, module, value, collectives=2):
                super().__init__()
                self.module = module
                self.compiled = _compile_group(module, native=False)
                self.register_buffer('fixed', value if args.fixed_input else value.clone())
                self.metadata = _Metadata()
                self.adapter = DecoderTopology('deepseek_v41_tail', (0,), 0, False, collectives)

            def forward(self, value):
                roots = dict(hidden_states=value, residual=None, positions=None, input_ids=None,
                             attention_inputs=(), metadata=self.metadata, state_generation=(1, 'tail'), state_tensors=())
                result = replay_native_decoder(self, **roots)
                if result is not None:
                    return result
                self.fixed.copy_(value)
                roots['hidden_states'] = self.fixed
                with collect_prepared_group_replays(owner=self, adapter=self.adapter,
                                                   snapshot=lambda:_Snapshot((self.fixed,)), **roots):
                    result = self.compiled(self.fixed)
                    record_native_decoder_outputs(*result)
                return result

        generator = torch.Generator().manual_seed(19371)
        inputs = [torch.randn(rows, 5120, generator=generator).bfloat16().to('hpu') for rows in (1,2,6)]
        old = torch.compile(Tail(False), backend='hpu_backend', fullgraph=True, dynamic=False)
        class Sampler(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('head', head)
            def forward(self, value):
                logits = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(value.contiguous(), self.head)
                return select_greedy_candidate(tensor_model_parallel_all_gather(
                    local_greedy_candidate(logits, rank), dim=-1)).int()
        sampler = torch.compile(Sampler(), backend='hpu_backend', fullgraph=True, dynamic=False)
        embed_plan = NativePlan(input_module, torch.ones(1, dtype=torch.int32, device='hpu'), collectives=1)
        def split_old(value):
            selected = sampler(value)
            residual, pre = embed_plan(selected.reshape(-1))
            return selected, residual, pre
        candidate = NativePlan(Tail(True), inputs[0])
        for value in inputs:
            expected = old(value)
            # Direct compiled native peers first prove C1/C2/C6 arithmetic.
            direct = torch.compile(Tail(True), backend='hpu_backend', fullgraph=True, dynamic=False)
            actual = direct(value)
            for a,b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
            report['checks'].append({'rows':value.shape[0], 'tokens_and_embedding_exact':True})
            save()
        for _ in range(4):
            candidate(inputs[0])[0].cpu()
        baseline = split_old if args.split_baseline else old
        for _ in range(4):baseline(inputs[0])[0].cpu()
        # Verify changing inputs through queued native reads and output consumers.
        expected_tokens, actual_tokens = [], []
        for _ in range(8):
            value = torch.randn(1, 5120, generator=generator).bfloat16().to('hpu')
            inputs[0].copy_(value)
            expected_tokens.append(sampler(inputs[0]).clone())
            actual_tokens.append(candidate(inputs[0])[0].clone())
        torch.testing.assert_close(torch.cat(actual_tokens).cpu(), torch.cat(expected_tokens).cpu(), rtol=0, atol=0)
        report['queued_changing_token_outputs_exact'] = True
        report['split_baseline'] = args.split_baseline
        report['fixed_input_alias'] = args.fixed_input
        before = prepared_group_stats()
        assert before['native_replays'] > 0, 'No actual native graph replay'
        report['native_before'] = before
        report['load_at_timing'] = subprocess.check_output(
            ['hl-smi','--query-aip=module_id,memory.used,utilization.aip','--format=csv,noheader,nounits'],text=True)
        seeds = [torch.randn(1, 5120, generator=generator).bfloat16().to('hpu') for _ in range(8)]
        for arm, fn in [('baseline',baseline),('candidate',candidate)]:
            samples = []
            for _ in range(3):
                for _ in range(16):fn(inputs[0])
                torch.hpu.synchronize()
                first,last = torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
                start = time.perf_counter()
                first.record()
                for i in range(args.iterations):
                    # Changing feedback values; required producer is included in both arms.
                    inputs[0].copy_(seeds[i % len(seeds)])
                    selected,residual,pre = fn(inputs[0])
                    inputs[0].copy_(residual[:,0,:])
                last.record();last.synchronize()
                samples.append({'device_ms':first.elapsed_time(last)/args.iterations,
                                'host_drained_ms':(time.perf_counter()-start)*1000/args.iterations})
            report['timing'].append({'arm':arm,'samples':samples,
                                    'median_device_ms':statistics.median(x['device_ms'] for x in samples),
                                    'median_host_drained_ms':statistics.median(x['host_drained_ms'] for x in samples)})
            save()
        report['native_after'] = prepared_group_stats()
        assert report['native_after']['native_captures']==before['native_captures'], 'Hot preparation'
        report['status']='complete_small_tail_chain_quality_and_real16_pending'
        save()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
