# SPDX-License-Identifier: Apache-2.0
"""Actual head -> official p -> rejection in the production native executor."""
import argparse
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = os.environ['PT_HPU_RECIPE_CACHE_CONFIG'].format(rank=rank)
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment, load_native_operators
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import torch.distributed as dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.models.deepseek_v41_program import output_head_projection
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_replay import _Snapshot, stage_collectives
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
        full_sampling_distribution_sharded, sample_speculative_prefix_sharded, speculative_sampling_draws,
    )
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.tp2_prepared_plan import (
        collect_prepared_group_replays, prepared_group_stats, record_native_decoder_outputs,
        register_tp2_prepared_group_pass, replay_native_decoder, shutdown_prepared_group_plans,
    )

    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    path = root/f'target-packet-rank{rank}.json'
    report = dict(rank=rank, status='starting', cases=[], rounds=[], performance_credit_ms=0)

    def save():
        path.write_text(json.dumps(report, indent=2)+'\n')

    save()
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
            load_native_operators()
            weight = PreparedV41Shard(args.prepared, 0, rank).tensor('head.weight', 'hpu')
            _, gather = stage_collectives(rank, True, 4)
            cases = []
            raw = []
            for source in sorted((args.fixtures/f'rank{rank}').glob('c6-*.pt'))[:3]:
                data = torch.load(source, map_location='cpu', weights_only=True)
                if not data['request_context_qualified'] or data['rank'] != rank:
                    raise ValueError('Actual rank-qualified requests required')
                raw.append(data)
                cases.append(tuple(data[k].to('hpu') for k in ('hidden', 'proposal', 'ids', 'sampling_parameters',
                                   'sampling_seed', 'sampling_counter', 'sampling_offsets')))
            owners, calls = [], []
            for arm in range(2):
                os.environ['VLLM_HPU_DSV41_DSPARK_TP_PACKET_TARGET'] = str(arm)
                owner = torch.nn.Module()
                owners.append(owner)
                fixed = tuple(t.clone() for t in cases[0])
                metadata = SimpleNamespace(native_completion=None)
                adapter = DecoderTopology('deepseek_v41_target_packet_probe', (1,), 0, False, 4)

                def body(hidden, q, ids, parameters, seed, counter, offsets):
                    _, acceptance, correction, controls = speculative_sampling_draws(
                        parameters, seed, counter.clone(), offsets)
                    logits = output_head_projection(hidden, weight, bf16=True)
                    _, p, covered = full_sampling_distribution_sharded(
                        logits, controls, tp_rank=rank, all_gather=gather, return_coverage=True,
                        known_stochastic=True)
                    output, committed, valid = sample_speculative_prefix_sharded(
                        p, q, ids[1:], acceptance, correction, rank, 4, gather)
                    return p, covered, output, committed, valid, logits

                compiled = torch.compile(body, backend='hpu_backend', fullgraph=True, dynamic=False)

                def call(values, fixed=fixed, owner=owner, metadata=metadata, adapter=adapter,
                         compiled=compiled):
                    roots = dict(hidden_states=values[0], pre_mix=values[1], positions=None, input_ids=values[2],
                                 attention_inputs=values[3:], metadata=metadata, state_generation=1,
                                 state_tensors=())
                    result = replay_native_decoder(owner, **roots)
                    if result is not None:
                        return result
                    for old, new in zip(fixed, values, strict=True):
                        old.copy_(new)
                    capture = dict(roots, hidden_states=fixed[0], pre_mix=fixed[1], input_ids=fixed[2],
                                   attention_inputs=fixed[3:])
                    with collect_prepared_group_replays(owner=owner, adapter=adapter,
                                                        snapshot=lambda: _Snapshot(()), **capture) as context:
                        context['group_index'] = 0
                        result = compiled(*fixed)
                        record_native_decoder_outputs(*result)
                    return result

                for _ in range(4):
                    call(cases[0])
                torch.hpu.synchronize()
                calls.append(call)
            report['status'] = 'checking'
            save()
            for case, values in enumerate(cases):
                results = []
                for call in calls:
                    results.append(tuple(t.cpu() for t in call(values)))
                torch.save(dict(outputs=results, proposal=raw[case]['proposal'], ids=raw[case]['ids'],
                                request_id=raw[case]['request_id']), root/f'target-packet-case{case}-rank{rank}.pt')
                report['cases'].append(dict(case=case, logits_exact=torch.equal(results[0][-1], results[1][-1]),
                    probability_max_abs=float((results[0][0]-results[1][0]).abs().max()),
                    parent_coverage=results[0][1].tolist(), candidate_coverage=results[1][1].tolist(),
                    output_exact=torch.equal(results[0][2], results[1][2]),
                    committed=[int(x[3].item()) for x in results]))
                from check_deepseek_v41_teacher_forced_acceptance import distribution

                across = [None]*4
                dist.all_gather_object(across, (results[0][-1], results[1][0], raw[case]['proposal']),
                                       group=get_tp_group().cpu_group)
                logits, candidate, proposal = [torch.cat([r[i] for r in across], -1) for i in range(3)]
                official = distribution(logits, 1., .95)
                covered = results[1][1].bool()
                max_error = float((candidate[covered].double()-official[covered]).abs().max()) if covered.any() else 0.
                repaired = torch.where(covered[:, None], candidate.double(), official)
                alpha_ref = torch.minimum(official[:5], proposal.double()).sum(-1)
                alpha_new = torch.minimum(repaired[:5], proposal.double()).sum(-1)
                report['cases'][-1].update(official_certified_p_max_abs=max_error,
                    teacher_forced_alpha_delta=(alpha_new-alpha_ref).tolist(),
                    alpha_reference=alpha_ref.tolist(), alpha_candidate=alpha_new.tolist())
                if max_error > 2e-6 or float((alpha_new-alpha_ref).mean()) < -2e-6:
                    raise AssertionError('Certified Target p exceeds official precision/acceptance allowance')
                save()
            report['native_stats'] = prepared_group_stats()
            if report['native_stats']['native_graphs'] < 2:
                raise AssertionError('Both arms require native replay')
            if not all(c['logits_exact'] for c in report['cases']):
                raise AssertionError('The unchanged production head changed logits')
            report['whole_distribution_gate_pending'] = False
            report['acceptance_scope'] = 'Three actual fixed prefixes; unchanged q; repaired uncertified p rows'
            for iteration in range(3):
                for arm, call in enumerate(calls):
                    times = []
                    for sample in range(6):
                        dist.barrier(group=get_tp_group().cpu_group)
                        begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                        begin.record()
                        call(cases[sample % 3])
                        end.record()
                        end.synchronize()
                        times.append(begin.elapsed_time(end))
                    report['rounds'].append(dict(iteration=iteration, arm=arm, device_ms=times,
                                                 median_device_ms=statistics.median(times)))
                    save()
            report['status'] = 'component_completed'
            shutdown_prepared_group_plans()
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
