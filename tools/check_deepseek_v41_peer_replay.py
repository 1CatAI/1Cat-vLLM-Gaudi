# SPDX-License-Identifier: Apache-2.0
"""Four-rank exchange -> fixed-rank sum -> MME feedback, using shared replay."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    if 'PT_HPU_RECIPE_CACHE_CONFIG' in os.environ:
        os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = os.environ['PT_HPU_RECIPE_CACHE_CONFIG'].format(rank=rank)
    # This probe has no mHC branch. The real layer chain uses the default
    # dependency overlap; here both communication arms use the serial plan.
    os.environ['VLLM_HPU_DSV41_TP_MHC_OVERLAP'] = '0'
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import torch.distributed as dist
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives, _Snapshot
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.tp2_prepared_plan import (
        collect_prepared_group_replays, record_native_decoder_outputs, replay_native_decoder,
        prepared_group_stats, shutdown_prepared_group_plans,
    )
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(rank=rank, status='running', cases=[], mhc_overlap=False)
    path = root / f'peer-replay-rank{rank}.json'
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://', local_rank=rank,
                                 backend='hccl')
    config = EngineArgs(model=str(args.prepared), dtype='bfloat16', tensor_parallel_size=4,
                        pipeline_parallel_size=1, load_format='dsv41_prepared', max_model_len=1048576,
                        max_num_seqs=32, max_num_batched_tokens=8192, block_size=128,
                        enable_prefix_caching=False, async_scheduling=True).create_engine_config()
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            reduce, gather = stage_collectives(rank, True, tp_size=4)
            local_cpu = (torch.arange(5120).reshape(1, -1) % 17).bfloat16() * .125 + rank
            local = local_cpu.to('hpu')
            shards = torch.ops.vllm_gaudi.tp_peer_allgather(local, 4).cpu().reshape(4, 5120)
            wanted = torch.stack([(torch.arange(5120) % 17).bfloat16() * .125 + peer for peer in range(4)])
            assert torch.equal(shards, wanted)
            reduced = reduce(local).cpu()
            reference = wanted[0].float()
            for peer in range(1, 4):
                reference = reference + wanted[peer].float()
            assert torch.equal(reduced, reference.bfloat16().reshape(1, -1))
            small = torch.full((1, 8), rank + .25, dtype=torch.bfloat16, device='hpu')
            assert torch.equal(gather(small, 1).cpu(), torch.cat([
                torch.full((1, 8), peer + .25, dtype=torch.bfloat16) for peer in range(4)], 1))
            report['rank_order_gather_and_sum_exact'] = True
            torch.manual_seed(9030)
            left = (torch.randn(5120, 128) / 128).bfloat16().to('hpu')
            right = (torch.randn(128, 5120) / 64).bfloat16().to('hpu')
            owners = []
            for batch in (1, 2):
                seed = torch.full((batch, 5120), .125, dtype=torch.bfloat16, device='hpu')
                for label in ('native_hccl_reduce', 'native_peer_fixed_sum'):
                    owner = torch.nn.Module()
                    owners.append(owner)
                    metadata = SimpleNamespace(native_completion=None)
                    adapter = DecoderTopology('deepseek_v41_peer_probe', (4,), 2, False)
                    def body(value, label=label):
                        for _ in range(8):
                            produced = value * .03125 + (rank + 1) * .125
                            if label == 'native_peer_fixed_sum':
                                collected = reduce(produced)
                            else:
                                collected = torch.ops.vllm_gaudi.tp4_allreduce_plain(
                                    produced.reshape(1, -1)).reshape(produced.shape)
                            value = (collected @ left) @ right
                        return value, value.float().mean(-1)
                    compiled = torch.compile(body, backend='hpu_backend', fullgraph=True, dynamic=False)
                    fixed = seed.clone()
                    roots = dict(hidden_states=fixed, pre_mix=None, positions=None, input_ids=None,
                                 attention_inputs=(), metadata=metadata, state_generation=1, state_tensors=())
                    def call(value):
                        dynamic = dict(roots, hidden_states=value)
                        result = replay_native_decoder(owner, **dynamic)
                        if result is not None:
                            return result[0]
                        fixed.copy_(value)
                        with collect_prepared_group_replays(owner=owner, adapter=adapter,
                                                            snapshot=lambda: _Snapshot(()), **roots) as context:
                            context['group_index'] = 0
                            result = compiled(fixed)
                            record_native_decoder_outputs(*result)
                        return result[0]
                    value = seed
                    for _ in range(32):
                        value = call(value)
                    torch.hpu.synchronize()
                    stats_before = prepared_group_stats()
                    assert stats_before['native_graphs'] > 0
                    dist.barrier()
                    start = torch.hpu.Event(enable_timing=True)
                    end = torch.hpu.Event(enable_timing=True)
                    start.record()
                    begun = time.perf_counter_ns()
                    for _ in range(128):
                        value = call(value)
                    end.record()
                    end.synchronize()
                    host_ms = (time.perf_counter_ns() - begun) / 1e6 / 128
                    device_ms = start.elapsed_time(end) / 128
                    stats_after = prepared_group_stats()
                    actual = value.cpu()
                    all_ranks = [torch.empty_like(value) for _ in range(4)]
                    dist.all_gather(all_ranks, value)
                    assert all(torch.equal(actual, peer.cpu()) for peer in all_ranks)
                    assert torch.isfinite(actual).all()
                    case = dict(batch=batch, arm=label, collectives_per_step=8, host_ms=host_ms,
                                device_ms=device_ms, per_exchange_device_ms=device_ms / 8,
                                four_ranks_exact=True, finite=True, stats_before=stats_before,
                                stats_after=stats_after)
                    report['cases'].append(case)
                    path.write_text(json.dumps(report, indent=2) + '\n')
            report['status'] = 'passed'
            shutdown_prepared_group_plans()
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + '\n')
        if dist.is_initialized():
            dist.destroy_process_group()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
