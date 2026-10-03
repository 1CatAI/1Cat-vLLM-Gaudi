# SPDX-License-Identifier: Apache-2.0
"""Resident real-16 measurement control; model arithmetic remains in the shared implementation."""
import argparse
import copy
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from types import MethodType
import uuid


COMPILER_CANDIDATES = {
    'dense_fp8_unsliced': {'SRAM_SLICER_MAX_CAPACITY_BYTES': '0'},
    'dense_fp8_legacy_slicer': {'ENABLE_PIPELINE_MANAGEMENT': '0'},
}


@contextmanager
def compiler_settings(settings, library=None):
    """Supported Synapse settings apply only to cold candidate compilation.

    Restore them before timing: all arms run their own already captured plans.
    This is diagnostic configuration, not a promoted serving implementation.
    """
    if not settings:
        yield
        return
    if library is None:
        library = ctypes.CDLL('libSynapse.so')
        library.synConfigurationGet.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64]
        library.synConfigurationSet.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    previous = {}
    try:
        for key, value in settings.items():
            old = ctypes.create_string_buffer(256)
            status = library.synConfigurationGet(key.encode(), old, len(old))
            if status:
                raise RuntimeError(f'Synapse configuration read {key} failed: {status}')
            previous[key] = old.value
            status = library.synConfigurationSet(key.encode(), value.encode())
            if status:
                raise RuntimeError(f'Synapse configuration set {key} failed: {status}')
            check = ctypes.create_string_buffer(256)
            status = library.synConfigurationGet(key.encode(), check, len(check))
            if status or check.value.lower() != value.encode().lower():
                raise RuntimeError(f'Synapse did not apply {key}={value}: {status}, {check.value!r}')
        yield
    finally:
        for key, value in reversed(list(previous.items())):
            status = library.synConfigurationSet(key.encode(), value)
            if status:
                raise RuntimeError(f'Synapse configuration restore {key} failed: {status}')


def summarize(values):
    values = list(values)
    if len(values) < 200:
        raise ValueError('A measurement period requires at least 200 token intervals')
    q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    return dict(n=len(values), median_ms=statistics.median(values), q1_ms=q1, q3_ms=q3, iqr_ms=q3-q1,
                mean_ms=statistics.mean(values), minimum_ms=min(values), maximum_ms=max(values))


def compare_periods(periods):
    if [p['arm'] for p in periods] != ['A', 'B', 'A', 'B', 'A', 'B']:
        raise ValueError('The comparison order must be ABABAB')
    baseline = summarize(v for p in periods if p['arm'] == 'A' for v in p['token_intervals_ms'])
    candidate = summarize(v for p in periods if p['arm'] == 'B' for v in p['token_intervals_ms'])
    delta = baseline['median_ms'] - candidate['median_ms']
    threshold = 2 * baseline['iqr_ms']
    return dict(baseline=baseline, candidate=candidate, saving_ms=delta, validity_threshold_ms=threshold,
                effective=delta > threshold, slower=-delta > threshold,
                period_medians_ms=[summarize(p['token_intervals_ms'])['median_ms'] for p in periods],
                formal_gain_credit=False)


def clone_module(module, *, recursive=False):
    """Separate module ownership while retaining immutable tensor allocations."""
    result = copy.copy(module)
    result._parameters = dict(module._parameters)
    result._buffers = dict(module._buffers)
    result._modules = {
        name: clone_module(child, recursive=True) if recursive and child is not None else child
        for name, child in module._modules.items()}
    for name, value in module.__dict__.items():
        if isinstance(value, MethodType) and value.__self__ is module:
            setattr(result, name, MethodType(value.__func__, result))
    return result


def make_dense_stage(stage, shard, sidecar_path, config_path):
    import torch
    from vllm_gaudi.models.deepseek_v41_program import load_weight_tree
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import (
        DenseFP8Sidecar, INPUT_PROJECTIONS, precision_config, projection_prefix)

    result = clone_module(stage)
    result.weights = clone_module(stage.weights, recursive=True)
    config = precision_config(config_path)
    sidecar = DenseFP8Sidecar(sidecar_path, shard)
    names = {projection_prefix(block.layer, p) + 'weight' for block in stage.layers for p in INPUT_PROJECTIONS}
    specs = {name: spec for name, spec in shard.specs.items() if name in names}
    if set(specs) != names:
        raise ValueError('Input/shared FP8 candidate is missing projection ownership')
    load_weight_tree(shard, result.weights, 'hpu', specs, dense_sidecar=sidecar, dense_config=config)
    blocks = []
    for source in stage.layers:
        block = clone_module(source)
        block.weights = result.weights.layers.get_submodule(str(source.layer))
        block.attention = clone_module(source.attention)
        block.attention.weights = block.weights.attn
        block.attention.invalidate_qkv_input_weight()
        block.attention.prepare_qkv_input_weight()
        block.moe = clone_module(source.moe)
        block.moe.weights = block.weights.ffn
        block.moe.release_shared_gate_up_weight()
        block.moe.prepare_shared_gate_up_weight()
        blocks.append(block)
    result.layers = torch.nn.ModuleList(blocks)
    result.precision_fingerprint = (*stage.precision_fingerprint, 'input_shared_fp8', sidecar.fingerprint)
    return result


def graph_info(replay):
    from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
    graphs = [entry[0] for variant in replay.variants.values() if (entry := _native_entries.get(variant)) is not None]
    return dict(graphs=len(graphs), compute_calls=sum(g.segment_count() for g in graphs),
                native_collective_points=sum(g.collective_count() for g in graphs),
                head_collectives_outside_replay=0 if replay.greedy_tail_enabled else 1,
                joint_info=[list(g.joint_info()) for g in graphs])


def make_swa_stage(stage):
    """Keep candidate dispatch independent while retaining the same allocations."""
    import torch

    result = clone_module(stage)
    result.layers = torch.nn.ModuleList([clone_module(block) for block in stage.layers])
    for block in result.layers:
        block.attention = clone_module(block.attention)
        block.attention.decode_swa_packed = not block.attention.ratio
    return result


def make_attention_norm_stage(stage):
    """Fuse norm/quant after the packed SWA parent, without A/B flag aliasing."""
    result = make_swa_stage(stage)
    for block in result.layers:
        block.decode_attention_norm_quant = True
    return result


def make_handoff_stage(stage):
    result = make_attention_norm_stage(stage)
    for block in result.layers:
        block.decode_engram_update = True
        block.mhc_interlayer_collapse = True
        block.mhc_interlayer_bf16 = True
    return result


def device_load():
    result = subprocess.run(['hl-smi', '-Q', 'module_id,utilization.aip,memory.used', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        module, util, memory = line.split(',')
        rows.append(dict(module=int(module), utilization=int(util.strip().split()[0]),
                         memory_mib=int(memory.strip().split()[0])))
    return dict(time_ns=time.time_ns(), modules=rows)


def loading_modules(samples, own_modules=(0, 1, 4, 5), minimum_growth_mib=128):
    mappings = [{x['module']: x['memory_mib'] for x in s['modules']} for s in samples]
    return [module for module in mappings[0] if module not in own_modules
            and mappings[-1][module] - mappings[0][module] >= minimum_growth_mib
            and all(b[module] >= a[module] for a, b in zip(mappings, mappings[1:]))]



def settling_modules(samples, own_modules=(0, 1, 4, 5), maximum_variation_mib=16):
    """Driver pool resets can hide weight loading behind a temporary memory drop."""
    mappings = [{x['module']: x['memory_mib'] for x in sample['modules']} for sample in samples]
    return [module for module in mappings[0] if module not in own_modules
            and max(row[module] for row in mappings)-min(row[module] for row in mappings) >= maximum_variation_mib]

def wait_for_loading(directory, rank, dist):
    history = []
    own_modules = tuple(int(module) for module in os.environ.get('HABANA_VISIBLE_MODULES', '0,1,4,5').split(','))
    while True:
        decision = [None]
        if rank == 0:
            samples = []
            for _ in range(11):
                samples.append(device_load())
                time.sleep(1)
            growing = settling_modules(samples, own_modules=own_modules)
            history.append(dict(samples=samples, loading_modules=growing))
            decision[0] = not growing
            if growing:
                print(f'Resident A/B waiting for weight loading on modules {growing}', flush=True)
        dist.broadcast_object_list(decision, src=0)
        if decision[0]:
            if rank == 0:
                with (directory / 'competing-load.jsonl').open('a') as record:
                    for check in history:
                        record.write(json.dumps(check) + '\n')
            return


def serve(stage, baseline, shard, chain, report, args, preparation_counts):
    import torch.distributed as dist
    from vllm_gaudi.ops.deepseek_v41_replay import StageReplay

    rank = dist.get_rank()
    control = args.resident_control_dir.resolve()
    control.mkdir(parents=True, exist_ok=True)
    arms = {'baseline': baseline}
    stages = {'baseline': stage}
    deadline = None

    def arm(name):
        if name not in arms:
            if name in ('dense_fp8', 'dense_fp8_tail', 'dense_fp8_static_int32',
                        'dense_fp8_swa_packed', 'dense_fp8_swa_norm_quant', 'dense_fp8_swa_norm_handoff',
                        'long_index_packed', 'long_index_mme', 'long_index_paged_mme', *COMPILER_CANDIDATES):
                if 'dense_fp8' not in stages:
                    stages['dense_fp8'] = make_dense_stage(stage, shard, args.ab_dense_sidecar, args.ab_dense_config)
                program = (make_handoff_stage(stages['dense_fp8'])
                           if name in ('dense_fp8_swa_norm_handoff', 'long_index_packed',
                                       'long_index_mme', 'long_index_paged_mme')
                           else make_attention_norm_stage(stages['dense_fp8']) if name == 'dense_fp8_swa_norm_quant'
                           else make_swa_stage(stages['dense_fp8']) if name == 'dense_fp8_swa_packed'
                           else clone_module(stages['dense_fp8']))
            elif name == 'tail':
                program = clone_module(stage)
            else:
                raise ValueError(f'Unknown resident candidate {name}')
            program.decode_static_int32 = name == 'dense_fp8_static_int32'
            if name in ('long_index_packed', 'long_index_mme', 'long_index_paged_mme'):
                for block in program.layers:
                    block.attention.runtime_index_mme = name != 'long_index_packed'
                    if name == 'long_index_paged_mme':
                        block.attention.index_mirror_scores = False
            replay = StageReplay(program, greedy_tail=name in ('tail', 'dense_fp8_tail'))
            program.replay_owner = replay
            stages[name], arms[name] = program, replay
            # Preserve actual compiler output for kernel-count changes. This
            # public diagnostic setting does not acquire another trace.
            graph_directory = control / 'compiler-graphs' / name / f'rank{rank}'
            graph_directory.mkdir(parents=True, exist_ok=True)
            settings = dict(COMPILER_CANDIDATES.get(name, {}))
            if name not in ('long_index_packed', 'long_index_mme', 'long_index_paged_mme'):
                settings['DUMP_POST_GRAPHS'] = str(graph_directory)
            with compiler_settings(settings):
                chain(True, 2, measure=False, engine=replay, warm_steps=6)
            if not replay.input_variant_ready(program.search_length):
                raise RuntimeError('Candidate did not capture its own complete native input replay')
        return arms[name]

    if rank == 0:
        ready = dict(pid=os.getpid(), source=os.environ['DSV41_RUN_EVIDENCE'],
                     candidates=['dense_fp8', 'tail', 'dense_fp8_tail', 'dense_fp8_static_int32',
                                 'dense_fp8_swa_packed', 'dense_fp8_swa_norm_quant', 'dense_fp8_swa_norm_handoff',
                                 'long_index_packed', 'long_index_mme', 'long_index_paged_mme', *COMPILER_CANDIDATES])
        (control / 'ready.json').write_text(json.dumps(ready, indent=2)+'\n')
    print(f'TP{rank}: real16 fixture resident; control {control}', flush=True)
    try:
        while True:
            request = [None]
            if rank == 0:
                pending = sorted(control.glob('*.request.json'))
                if pending:
                    path = pending[0]
                    request[0] = dict(json.loads(path.read_text()), request_path=str(path))
                else:
                    time.sleep(1)
            dist.broadcast_object_list(request, src=0)
            job = request[0]
            if job is None:
                continue
            if job.get('command') == 'stop':
                if rank == 0:
                    Path(job['request_path']).rename(Path(job['request_path']).with_suffix('.consumed'))
                break
            steps = int(job.get('steps', 200))
            if steps < 200:
                raise ValueError('Resident A/B needs at least 200 steps per period')
            if steps > args.resident_max_steps:
                raise ValueError('Resident job exceeds the allocated continuation step reservation')
            from tools.check_deepseek_v41_tp4_continuation import continuation_page_count
            import torch

            needed_pages = continuation_page_count(args.context_tokens, steps)
            pages = stage.shared.block_table[:needed_pages].cpu()
            if pages.numel() != needed_pages or not bool((pages > 0).all()) \
                    or torch.unique(pages).numel() != needed_pages:
                raise ValueError('Visible continuation pages must be nonzero and distinct')
            for cache in stage.shared.sources.values():
                if int(pages.max()) >= cache.index.shape[0] // (128 // cache.ratio):
                    raise ValueError('Continuation page mapping exceeds its physical cache')
            name = job['candidate']
            directory = control / job['id']
            directory.mkdir(exist_ok=True)
            begun = time.monotonic()
            if rank == 0:
                def expire(path=directory, candidate_name=name, start=begun):
                    partial = dict(status='candidate_timeout', candidate=candidate_name,
                                   elapsed_s=time.monotonic()-start, formal_gain_credit=False,
                                   partial_periods='periods.json', worker_exit_code=124)
                    (path / 'result.json').write_text(json.dumps(partial, indent=2)+'\n')
                    # Elastic retires this task's other ranks; foreign card owners are untouched.
                    os._exit(124)
                deadline = threading.Timer(45 * 60, expire)
                deadline.daemon = True
                deadline.start()
            wait_for_loading(directory, rank, dist)
            if job.get('command') == 'diagnose_index_mirror':
                import torch

                saved_path = Path(job['reference_result'])
                saved = json.loads(saved_path.read_text())
                saved_periods = json.loads((saved_path.parent / saved['periods_file']).read_text())
                forced = next(p for p in saved_periods if p['arm'] == 'A')['ranks'][rank]['tokens'][:32]
                outputs = []
                for name in ('long_index_paged_mme', 'long_index_mme'):
                    replay = arm(name)
                    records = []

                    def observe(program, hidden, position, records=records):
                        try:
                            records.append(observe_mirror_step(program, hidden, position))
                        except (AttributeError, KeyError, TypeError, ValueError) as error:
                            records.append(dict(position=position, observer_error=repr(error)))

                    tokens, _, _ = chain(True, len(forced), measure=False, engine=replay,
                                         forced_tokens=forced, observer=observe)
                    records[-1]['terminal_sample_token'] = tokens[-1]
                    torch.save(records, directory / f'{name}-rank{rank}.pt')
                    outputs.append(records)
                checks = []
                for old, new in zip(*outputs, strict=True):
                    if 'observer_error' in old or 'observer_error' in new:
                        checks.append(dict(position=old['position'],
                                           observer_error=[old.get('observer_error'), new.get('observer_error')]))
                        continue
                    a, b = old['hidden'].float().flatten(), new['hidden'].float().flatten()
                    keys_exact = all(torch.equal(k['mirror'][k['visible']], k['canonical'][k['visible']])
                                     for record in (old, new) for k in record['keys'].values())
                    checks.append(dict(position=old['position'], keys_exact=keys_exact,
                                       indices_exact=all(torch.equal(v, new['indices'][k])
                                                         for k, v in old['indices'].items()),
                                       hidden_exact=torch.equal(a, b), max_abs=float((a-b).abs().max()),
                                       relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-30)),
                                       cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                                       terminal_sample_token=[old.get('terminal_sample_token'),
                                                              new.get('terminal_sample_token')]))
                (directory / f'checks-rank{rank}.json').write_text(json.dumps(checks, indent=2)+'\n')
                dist.barrier()
                if rank == 0:
                    (directory / 'result.json').write_text(json.dumps(dict(status='diagnostics_observer_failed'
                        if any('observer_error' in row for row in checks) else 'diagnostics_completed',
                        timed=False, formal_gain_credit=False, checks_file='checks-rank*.json'), indent=2)+'\n')
                    Path(job['request_path']).rename(Path(job['request_path']).with_suffix('.consumed'))
                    deadline.cancel()
                    deadline = None
                continue
            if job.get('reference_result'):
                saved_path = Path(job['reference_result'])
                saved = json.loads(saved_path.read_text())
                saved_periods = json.loads((saved_path.parent / saved['periods_file']).read_text())
                label = job.get('reference_arm', 'B')
                saved_periods = [p for p in saved_periods if p['arm'] == label]
                if saved['status'] != 'completed' or saved['context_tokens'] != args.context_tokens:
                    raise ValueError('Saved reference must match the complete continuation context')
                if saved['steps'] != steps or not saved_periods:
                    raise ValueError('Saved reference must match the measured step count')
                expected = saved_periods[0]['ranks'][rank]['tokens']
                assert all(p['ranks'][rank]['tokens'] == expected for p in saved_periods)
                candidate = arm(name)
                counts = preparation_counts()
                periods = []
                for _ in range(3):
                    wait_for_loading(directory, rank, dist)
                    dist.barrier()
                    tokens, host_ms, device_ms = chain(True, steps, engine=candidate, warm_steps=32)
                    assert counts == preparation_counts(), 'Hot recompilation invalidates candidate timing'
                    local = dict(rank=rank, tokens=tokens, delivery_ns=report['token_delivery_ns'],
                                 host_ms=host_ms, device_ms=device_ms,
                                 checked_first_device_position=report.get('checked_first_device_position'))
                    ranks = [None] * dist.get_world_size()
                    dist.all_gather_object(ranks, local)
                    assert all(row['tokens'] == tokens for row in ranks), 'Ranks disagree on tokens'
                    if tokens != expected:
                        if rank == 0:
                            (directory / 'feedback-mismatch.json').write_text(json.dumps(
                                dict(expected=expected, observed=ranks, formal_gain_credit=False), indent=2) + '\n')
                        raise AssertionError('Candidate differs from saved feedback tokens')
                    if rank == 0:
                        delivery = [max(row['delivery_ns'][i] for row in ranks) for i in range(steps + 1)]
                        intervals = [(b - a) / 1e6 for a, b in zip(delivery, delivery[1:])]
                        periods.append(dict(arm='B', ranks=ranks, token_intervals_ms=intervals,
                                            summary=summarize(intervals)))
                        (directory / 'periods.json').write_text(json.dumps(periods, indent=2) + '\n')
                if rank == 0:
                    old = summarize(v for p in saved_periods for v in p['token_intervals_ms'])
                    new = summarize(v for p in periods for v in p['token_intervals_ms'])
                    delta = old['median_ms'] - new['median_ms']
                    result = dict(status='completed', candidate=name, steps=steps, context_tokens=args.context_tokens,
                                  reference_result=str(saved_path), reference_arm=label, baseline_measured=False,
                                  comparison=dict(baseline=old, candidate=new, saving_ms=delta,
                                                  validity_threshold_ms=2 * old['iqr_ms'],
                                                  effective=delta > 2 * old['iqr_ms'], formal_gain_credit=False),
                                  no_profiler=True, no_hot_compilation=True, feedback_tokens_exact=True,
                                  checked_first_device_positions=[row['checked_first_device_position']
                                                                  for row in ranks],
                                  graphs=[graph_info(candidate)], periods_file='periods.json',
                                  formal_quality_pending=True)
                    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                    Path(job['request_path']).rename(Path(job['request_path']).with_suffix('.consumed'))
                    print(json.dumps(result), flush=True)
                    deadline.cancel()
                    deadline = None
                continue
            reference_name = job.get('baseline', 'baseline')
            reference = arm(reference_name)
            candidate = arm(name)
            # Resolve both warmed contracts before the no-hot-compilation gate.
            chain(True, 2, measure=False, engine=reference, warm_steps=6)
            counts = preparation_counts()
            periods = []
            for label in ('A', 'B', 'A', 'B', 'A', 'B'):
                if time.monotonic() - begun > 45 * 60:
                    raise TimeoutError('Candidate reached the 45-minute limit')
                wait_for_loading(directory, rank, dist)
                dist.barrier()
                tokens, host_ms, device_ms = chain(True, steps, engine=reference if label == 'A' else candidate,
                                                   warm_steps=32)
                assert counts == preparation_counts(), 'Hot recompilation invalidates the A/B measurement'
                local = dict(rank=rank, tokens=tokens, delivery_ns=report['token_delivery_ns'],
                             host_ms=host_ms, device_ms=device_ms,
                             checked_first_device_position=report.get('checked_first_device_position'))
                ranks = [None] * dist.get_world_size()
                dist.all_gather_object(ranks, local)
                assert all(row['tokens'] == tokens for row in ranks), 'Ranks disagree on generated tokens'
                if rank == 0:
                    delivery = [max(row['delivery_ns'][i] for row in ranks) for i in range(steps+1)]
                    intervals = [(b-a)/1e6 for a, b in zip(delivery, delivery[1:])]
                    period = dict(arm=label, ranks=ranks, token_intervals_ms=intervals, summary=summarize(intervals))
                    periods.append(period)
                    (directory / 'periods.json').write_text(json.dumps(periods, indent=2)+'\n')
                    print(f"{name} {label}: {period['summary']['median_ms']:.6f} ms, "
                          f"IQR {period['summary']['iqr_ms']:.6f}", flush=True)
            infos = [graph_info(reference), graph_info(candidate)]
            if rank == 0:
                result = dict(status='completed', baseline=reference_name, candidate=name, steps=steps, order='ABABAB',
                              elapsed_s=time.monotonic()-begun, graphs=infos, comparison=compare_periods(periods),
                              context_tokens=args.context_tokens,
                              checked_first_device_positions=[row['checked_first_device_position'] for row in ranks],
                              no_profiler=True, no_hot_compilation=True, four_rank_tokens_equal=True,
                              statistic_unit='Four-rank latest token delivery interval, milliseconds',
                              baseline_drift_ms=max(p['summary']['median_ms'] for p in periods if p['arm']=='A')
                              - min(p['summary']['median_ms'] for p in periods if p['arm']=='A'),
                              formal_quality_pending=True, periods_file='periods.json')
                result['cold_compiler_settings'] = COMPILER_CANDIDATES.get(name, {})
                result['within_arm_feedback_stable'] = all(
                    all(p['ranks'][0]['tokens'] == next(q for q in periods if q['arm'] == label)['ranks'][0]['tokens']
                        for p in periods if p['arm'] == label) for label in ('A', 'B'))
                if not result['within_arm_feedback_stable']:
                    result['comparison']['effective'] = False
                    result['status'] = 'unstable_feedback'
                if name in ('dense_fp8_swa_norm_quant', 'dense_fp8_swa_norm_handoff'):
                    result['cross_arm_feedback_exact'] = all(
                        p['ranks'][0]['tokens'] == periods[0]['ranks'][0]['tokens'] for p in periods)
                    result['numerical_reference_pending'] = True
                    result['gain_ledger_eligible'] = False
                if reference_name == 'long_index_paged_mme':
                    result['cross_arm_feedback_exact'] = all(
                        p['ranks'][0]['tokens'] == periods[0]['ranks'][0]['tokens'] for p in periods)
                    if not result['cross_arm_feedback_exact']:
                        result['comparison']['effective'] = False
                        result['status'] = 'numerical_contract_failed'
                if name in COMPILER_CANDIDATES:
                    # Runtime compiler settings are absent from Bridge's recipe
                    # cache key. A/B timing alone cannot establish distinct arms.
                    result['status'] = 'compiler_cache_isolation_unverified'
                    result['comparison']['effective'] = False
                if name in (*COMPILER_CANDIDATES, 'dense_fp8_static_int32', 'dense_fp8_swa_packed') \
                        and reference_name in ('dense_fp8', 'dense_fp8_swa_packed'):
                    result['compiler_token_exact'] = all(p['ranks'][0]['tokens'] == periods[0]['ranks'][0]['tokens']
                                                         for p in periods)
                    if not result['compiler_token_exact']:
                        result['comparison']['effective'] = False
                        result['status'] = 'numerical_contract_failed'
                (directory / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
                Path(job['request_path']).rename(Path(job['request_path']).with_suffix('.consumed'))
                print(json.dumps(result), flush=True)
                deadline.cancel()
                deadline = None
    finally:
        if deadline is not None:
            deadline.cancel()
        for name, replay in arms.items():
            if name != 'baseline':
                replay.close()


def observe_mirror_step(program, hidden, position):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4

    keys = {}
    for source, cache in program.shared.sources.items():
        row = position // cache.ratio
        visible_rows = (position + 1) // cache.ratio
        probes = torch.linspace(0, max(0, visible_rows - 1), 33).long().tolist()
        ids = torch.tensor([*probes, row], dtype=torch.int64, device=hidden.device)
        physical = program.shared.physical_rows(ids, cache.ratio).long()
        packed = cache.index.index_select(0, physical).cpu()
        mirror = cache.index_mirror.index_select(0, ids).cpu()
        valid = (ids.cpu() < visible_rows)
        keys[source] = dict(rows=ids.cpu(), visible=valid, mirror=mirror,
                            canonical=unpack_fp4(packed, 128, 32))
    indices = {str(b.layer): b.attention.selection.indices[:1].cpu().clone()
               for b in program.layers if b.attention.owns_index}
    return dict(position=position, hidden=hidden.cpu().clone(), keys=keys, indices=indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--candidate', choices=('dense_fp8', 'tail', 'dense_fp8_tail',
                                              'dense_fp8_static_int32', 'dense_fp8_swa_packed',
                                              'dense_fp8_swa_norm_quant', 'dense_fp8_swa_norm_handoff',
                                              'long_index_packed', 'long_index_mme', 'long_index_paged_mme',
                                              *COMPILER_CANDIDATES))
    parser.add_argument('--baseline', choices=('baseline', 'dense_fp8', 'dense_fp8_swa_packed',
                                               'dense_fp8_swa_norm_quant', 'long_index_packed',
                                               'long_index_paged_mme'), default='baseline')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--reference-result', type=Path,
                        help='Reuse saved matching-context timing and feedback tokens; time only the candidate')
    parser.add_argument('--reference-arm', choices=('A', 'B'), default='B')
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--diagnose-index-mirror', action='store_true')
    args = parser.parse_args()
    if not args.stop and not args.candidate:
        parser.error('Select a candidate or --stop')
    if args.diagnose_index_mirror and not args.reference_result:
        parser.error('--diagnose-index-mirror needs the saved same-state A/B result')
    if args.steps < 200:
        parser.error('--steps must be at least 200')
    args.control_dir.mkdir(parents=True, exist_ok=True)
    job_id = f'{time.time_ns()}-{uuid.uuid4().hex[:8]}'
    job = dict(id=job_id, command='stop' if args.stop else 'measure', candidate=args.candidate,
               baseline=args.baseline, steps=args.steps)
    if args.diagnose_index_mirror:
        job['command'] = 'diagnose_index_mirror'
    if args.reference_result:
        job.update(reference_result=str(args.reference_result.resolve()), reference_arm=args.reference_arm)
    target = args.control_dir / f'{job_id}.request.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(job, indent=2)+'\n')
    temporary.rename(target)
    print(json.dumps(dict(job=job, request=str(target), result=str(args.control_dir/job_id/'result.json'))), flush=True)


if __name__ == '__main__':
    main()
