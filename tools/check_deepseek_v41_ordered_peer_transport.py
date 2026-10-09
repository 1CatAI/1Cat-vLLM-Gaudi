# SPDX-License-Identifier: Apache-2.0
"""Actual C6 activation packets through native peer exchange and the C1 sum."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--mme-producer', action='store_true',
                        help='Diagnostic rank-dependent identity MME producers; no performance claim')
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
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_replay import _Snapshot
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.tp2_prepared_plan import (
        collect_prepared_group_replays, record_native_decoder_outputs, replay_native_decoder,
        prepared_group_stats, register_tp2_prepared_group_pass, shutdown_prepared_group_plans,
    )
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(rank=rank, status='running', cases=[], mhc_overlap=False)
    path = root / f'peer-replay-rank{rank}.json'
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://', local_rank=rank,
                                 backend='hccl')
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4))
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            register_tp2_prepared_group_pass()
            bind_worker_helpers(rank)
            from vllm_gaudi.ops.deepseek_v41_ordered_peer_sum import ordered_peer_sum, ordered_peer_sum_reference
            from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators
            load_native_operators(required=('custom_deepseek_v41_ordered_peer_sum_gaudi2',))
            fixtures = Path('/opt/ssd960/builds/dsv41-tp4-dspark-round-chain-v1'
                            '/production-c6-request-fixtures-02-with-history')
            inputs = []
            for case in range(3):
                data = torch.load(fixtures / f'rank{rank}/c6-{case}.pt', map_location='cpu', weights_only=True)
                if not data['request_context_qualified']:
                    raise ValueError('Actual request identity required')
                inputs.append(data['groups'][5]['residual'][:, rank, :].contiguous())
            owners, calls = [], []
            identity = torch.eye(5120, dtype=torch.bfloat16, device='hpu') if args.mme_producer else None
            for label in ('generic', 'c1_ordered'):
                owner = torch.nn.Module()
                owners.append(owner)
                metadata = SimpleNamespace(native_completion=None)
                adapter = DecoderTopology('deepseek_v41_peer_probe', (1,), 2, False)
                def body(value, label=label):
                    output = value
                    if identity is not None:
                        for _ in range(1 + rank % 3):
                            output = output @ identity
                    for _ in range(2):
                        peers = torch.ops.vllm_gaudi.tp_peer_allgather(output.reshape(1, -1).contiguous(), 4)
                        shards = peers.reshape(4, value.numel())
                        output = (ordered_peer_sum_reference(shards) if label == 'generic'
                                  else ordered_peer_sum(shards)).reshape(value.shape)
                    return output, output.float().mean(-1)
                compiled = torch.compile(body, backend='hpu_backend', fullgraph=True, dynamic=False)
                fixed = inputs[0].to('hpu')
                roots = dict(hidden_states=fixed, pre_mix=None, positions=None, input_ids=None,
                             attention_inputs=(), metadata=metadata, state_generation=1, state_tensors=())
                def call(value, owner=owner, fixed=fixed, roots=roots, adapter=adapter, compiled=compiled):
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
                for _ in range(8):
                    call(fixed)
                torch.hpu.synchronize()
                calls.append(call)
            for case, local in enumerate(inputs):
                expected_peers = [None] * 4
                dist.all_gather_object(expected_peers, local, group=get_tp_group().cpu_group)
                wanted = ordered_peer_sum_reference(torch.stack(expected_peers).reshape(4, 30720)).reshape(6, 5120)
                wanted = ordered_peer_sum_reference(wanted.reshape(1, -1).expand(4, -1).contiguous()).reshape(6, 5120)
                outcomes = []
                for label, call in zip(('generic', 'c1_ordered'), calls, strict=True):
                    dist.barrier()
                    actual = call(local.to('hpu')).cpu()
                    outcomes.append(dict(arm=label,
                                         exact=torch.equal(actual.view(torch.int16), wanted.view(torch.int16)),
                                         max_abs=float((actual.float()-wanted.float()).abs().max()),
                                         different_words=int((actual.view(torch.int16)!=wanted.view(torch.int16)).sum())))
                report['cases'].append(dict(case=case, shape=[4,6,5120], outcomes=outcomes))
                path.write_text(json.dumps(report, indent=2)+'\n')
            report.update(native_stats=prepared_group_stats(), performance_measured=False,
                          mme_producer=args.mme_producer,
                          scope='Native peer transport/consumer correctness; no model precision claim')
            if report['native_stats']['native_graphs'] < 2:
                raise AssertionError('Both arms must execute the native replay path')
            if any(not row['exact'] for case in report['cases'] for row in case['outcomes']):
                raise AssertionError('Native transport/consumer differs from the C1 fixed-rank FP32 sum')
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
