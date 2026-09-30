# SPDX-License-Identifier: Apache-2.0
"""Bounded TP4 query/gain producer -> collective -> native score consumer gate.

This synthetic component includes both local projection weights, exact FP4
query rounding, paired communication submission and the actual native MME/TPC score consumer.
It does not qualify model latency, changed selection semantics or the ledger.
"""
import json
import os
from pathlib import Path
import time


def main():
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    if '{rank}' in os.environ.get('PT_HPU_RECIPE_CACHE_CONFIG', ''):
        os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = os.environ['PT_HPU_RECIPE_CACHE_CONFIG'].replace('{rank}', str(rank))
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from habana_frameworks.torch.hpu.metrics import metric_global
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_index_exchange import gather_index_query_pair, prepare_index_gather_pair
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives

    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    bind_worker_helpers(rank)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://',
                                 local_rank=rank, backend='hccl')
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(rank=rank, status='preparing', scope=__doc__, checks=[], timings={},
                  new_full_model_requests=0, gain_ledger_credit=False,
                  baseline_reason='Reuse index-packet-85/micro02 separate timings; reconstruct untimed score oracles '
                                  'because the archive did not save score tensors. No baseline timing repeat.')

    def save():
        (root / f'pair-rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            _, gather = stage_collectives(rank, False)
            bridge = prepare_index_gather_pair()
            ops = torch.ops.custom_op
            gen = torch.Generator().manual_seed(8501)
            # Replicated inputs/cache; distinct per-rank immutable projection shards.
            query_weights = torch.randn(4, 1024, 1280, generator=gen).bfloat16() / 32
            gain_weights = torch.randn(4, 8, 5120, generator=gen).bfloat16() / 64
            keys = (torch.randn(10240, 128, generator=gen) / 8).bfloat16().to('hpu')
            qweight, gweight = query_weights[rank].to('hpu'), gain_weights[rank].to('hpu')
            rows = torch.arange(10240, dtype=torch.int32, device='hpu')

            class Chain(torch.nn.Module):
                def __init__(self, packed):
                    super().__init__()
                    self.packed = packed

                def forward(self, qr, value, qweight, gweight, keys, positions, rows):
                    query = F.linear(qr, qweight).reshape(qr.shape[0], 8, 128)
                    query = ops.custom_deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2(
                        query.reshape(-1, 128).contiguous()).reshape(qr.shape[0], 8, 128)
                    gains = F.linear(value, gweight) * (1 / 64)
                    if self.packed:
                        query, gains = gather_index_query_pair(query, gains)
                    else:
                        query, gains = gather(query, 1), gather(gains, 1)
                    return ops.custom_deepseek_v41_prefill_index_scores_gaudi2(
                        query.contiguous(), gains.contiguous(), keys, positions, rows, 2, 8)

            programs = [torch.compile(Chain(flag), backend='hpu_backend', fullgraph=True, dynamic=False)
                        for flag in (False, True)]
            banks, expected_b1 = {}, []
            for batch in (1, 2):
                fixtures = []
                for generation in range(2):
                    qr = (torch.randn(batch, 1280, generator=gen) / 8).bfloat16()
                    value = (torch.randn(batch, 5120, generator=gen) / 8).bfloat16()
                    positions = torch.arange(batch, dtype=torch.int32) + 16514 + generation
                    fixtures.append((qr.to('hpu'), value.to('hpu'), qweight, gweight, keys,
                                     positions.to('hpu'), rows))
                banks[batch] = fixtures
                for generation, fixture in enumerate(fixtures):
                    old, new = [program(*fixture).cpu() for program in programs]
                    assert torch.equal(old, new), (rank, batch, generation, 'score mismatch')
                    if batch == 1:
                        expected_b1.append(old)
                    report['checks'].append(dict(batch=batch, generation=generation, native_scores_exact=True))
                # Reorder and reuse the same slots after another generation.
                fixture = fixtures[0]
                reordered = (fixture[0].flip(0).contiguous(), fixture[1].flip(0).contiguous(), *fixture[2:5],
                             fixture[5].flip(0).contiguous(), fixture[6])
                old, new = [program(*reordered).cpu() for program in programs]
                assert torch.equal(old, new), (rank, batch, 'slot reorder')
                report['checks'].append(dict(batch=batch, reordered_slots_exact=True))
            save()
            for name, program in (('paired', programs[1]),):
                for step in range(32):
                    output = program(*banks[1][step % 2])
                torch.hpu.synchronize()
                dist.barrier()
                metric = metric_global('graph_compilation')
                before = dict(metric.stats())['TotalNumber']
                pair_before = bridge.tp4_index_gather_pair_launches()
                first, last = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                start = time.perf_counter_ns()
                first.record()
                for step in range(96):
                    output = program(*banks[1][step % 2])
                last.record()
                last.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1e6 / 96
                assert dict(metric.stats())['TotalNumber'] == before
                pair_delta = bridge.tp4_index_gather_pair_launches() - pair_before
                assert pair_delta == 96, (rank, pair_delta, 'paired path did not execute every step')
                assert torch.equal(output.cpu(), expected_b1[1]), (rank, name, 'final consumer')
                report['timings'][name] = dict(host_ms=elapsed, device_ms=first.elapsed_time(last) / 96,
                                               warm_steps=32, measured_steps=96, no_hot_compilation=True,
                                               paired_launches=pair_delta)
                save()
                print(json.dumps(dict(rank=rank, arm=name, **report['timings'][name])), flush=True)
            report['status'] = 'component_measured_exact'
            save()
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        save()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
