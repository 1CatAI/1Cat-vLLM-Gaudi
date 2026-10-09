# SPDX-License-Identifier: Apache-2.0
"""Full Target probability-overlap gate on saved, identical request prefixes.

The proposal distribution is immutable: these candidates change only Target
computation, not MTP/Markov or sampling. This conditional gate does
not replace full serving quality or establish future-round acceptance.
"""
import hashlib
import ast
import json
import statistics
import time
from pathlib import Path


def router_source_identity(source):
    """Hash the complete Router/shared implementation, excluding other trials."""
    source = Path(source)
    tree = ast.parse((source / 'vllm_gaudi/models/deepseek_v41_program.py').read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PreparedMoE')
    quantizer = (source / 'vllm_gaudi/ops/deepseek_v41_woa_fp8.py').read_bytes()
    return hashlib.sha256(ast.dump(owner, include_attributes=False).encode() + quantizer).hexdigest()


def reuse_router_acceptance(proof, workspace, native):
    """Reuse unchanged full-prefix quality; timing still uses actual real16 A/B."""
    if (proof.get('kind') != 'conditional_router_acceptance'
            or proof.get('candidate') not in ('router_shared', 'router_ready_fp8')):
        raise ValueError('Conditional proof belongs to a different candidate')
    summary = Path(proof['summary'])
    if hashlib.sha256(summary.read_bytes()).hexdigest() != proof['summary_sha256']:
        raise ValueError('Conditional acceptance evidence changed')
    result = json.loads(summary.read_text())
    if (not result['passed'] or len(result['cases']) != 3 or result['candidate'] != proof['candidate']
            or result['mean_alpha_delta'] < -result['numerical_allowance']):
        raise ValueError('Whole-model conditional acceptance did not pass')
    if router_source_identity(workspace) != proof['router_source_sha256']:
        raise ValueError('Router/shared code differs from its full-prefix probability proof')
    for name, expected in proof['native_binaries'].items():
        if hashlib.sha256((Path(native) / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Router/shared numerical kernels differ from their full-prefix proof')
    return dict(qualified=True, scope='Unchanged full40 real-prefix alpha, real16 finite outputs required',
                mean_alpha_delta=result['mean_alpha_delta'], path=str(summary),
                future_serving_quality_qualified=False)


def qualify(programs, plans, raw, engram_cases, reset, root, rank, report, save, *, timing=True):
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_tp_group
    from vllm_gaudi.models.deepseek_v41_program import output_head_projection

    report['acceptance_helper_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['timing_performed'] = timing
    report['acceptance_scope'] = (
        '40-layer native Target from identical saved KV/history and teacher-forced C6 IDs; '
        'unchanged saved official proposal probabilities. Future-round MTP state changes remain a serving gate.'
    )
    for case, data in enumerate(raw):
        identity = dict(request_id=data['request_id'], context_prefix_tokens=data['context_prefix_tokens'],
                        ids=data['ids'].tolist(), positions=data['positions'].tolist(),
                        history=data['cursor_history'].tolist())
        prefix_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        ids, positions = data['ids'].to('hpu'), data['positions'].to('hpu')
        source = data['groups'][0]
        values = source['residual'].to('hpu'), source['pre'].to('hpu'), positions, ids
        logits = []
        for arm, (program, plan) in enumerate(zip(programs, plans, strict=True)):
            reset(arm, case)
            hidden, _, _ = plan(*values, engram_cases[case], fused_text_io=True)
            score = output_head_projection(hidden, program.weights.head.weight, bf16=program.bf16_head)
            torch.hpu.synchronize()
            logits.append(score.cpu().clone())
        # Keep rank-local FP32 proposal values unchanged; full vocabulary is
        # reconstructed offline after all ranks finish. No second top_p on q.
        path = root / f'fixed-prefix-case{case}-rank{rank}.pt'
        torch.save(dict(identity=identity, prefix_sha256=prefix_hash, teacher_forced=True,
                        reference_target_logits=logits[0][:5], candidate_target_logits=logits[1][:5],
                        draft_probabilities=data['proposal'].clone(),
                        reference_precision='Captured reference arm; deployment baseline fingerprinted by launcher',
                        draft_changed=False), path)
        report['checks'].append(dict(case=case, fixed_prefix_exported=True, file=str(path),
                                     logits_finite=all(bool(torch.isfinite(x).all()) for x in logits)))
        save()
    if not timing:
        report['status'] = 'diagnostic_completed'
        save()
        return
    # The same resident full-Target plans also supply the missing complete
    # device-chain timing; do not restart a service for this component gate.
    for iteration in range(3):
        medians = []
        for arm, plan in enumerate(plans):
            device_ms, wall_ms = [], []
            for sample in range(6):
                case = sample % len(raw)
                data = raw[case]
                source = data['groups'][0]
                values = (source['residual'].to('hpu'), source['pre'].to('hpu'),
                          data['positions'].to('hpu'), data['ids'].to('hpu'))
                reset(arm, case)
                # Match the regular real16 harness: reset drains each local
                # device, then align the host ranks outside the timer. Without
                # this boundary the first peer wait includes another rank's
                # untimed cache restoration, rather than Target device cost.
                dist.barrier(group=get_tp_group().cpu_group)
                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                clock = time.perf_counter()
                begin.record()
                plan(*values, engram_cases[case], fused_text_io=True)
                end.record()
                end.synchronize()
                device_ms.append(begin.elapsed_time(end))
                wall_ms.append((time.perf_counter()-clock)*1000)
            medians.append(statistics.median(device_ms))
            report['rounds'].append(dict(iteration=iteration, arm=arm, device_ms=device_ms,
                                        wall_ms=wall_ms, median_device_ms=medians[-1]))
        report.setdefault('paired_saved_ms', []).append(medians[0]-medians[1])
    report['status'] = 'diagnostic_completed' 
    save()


def summarize(run):
    import torch
    from check_deepseek_v41_teacher_forced_acceptance import distribution

    run = Path(run)
    reports = [json.loads((run / f'request-c6-rank{rank}.json').read_text()) for rank in range(4)]
    if any(row['status'] != 'diagnostic_completed' for row in reports):
        raise ValueError('Four native Target ranks must finish')
    count = len(reports[0]['checks'])
    if not 3 <= count <= 5 or any(len(row['checks']) != count for row in reports):
        raise ValueError('Three to five aligned real prefixes required')
    cases = []
    for index in range(count):
        ranks = [torch.load(run / f'fixed-prefix-case{index}-rank{rank}.pt', weights_only=True) for rank in range(4)]
        if len({row['prefix_sha256'] for row in ranks}) != 1 or any(row['draft_changed'] for row in ranks):
            raise ValueError('Prefix mismatch or changed draft cannot use immutable proposal probabilities')
        target = [torch.cat([row[name] for row in ranks], -1)
                  for name in ('reference_target_logits', 'candidate_target_logits')]
        q = torch.cat([row['draft_probabilities'] for row in ranks], -1).double()
        if q.shape != (5, 129280) or any(t.shape != q.shape for t in target):
            raise ValueError('Full official vocabulary and five positions required')
        mass = q.sum(-1)
        if not (torch.isfinite(q).all() and (q >= 0).all() and torch.allclose(
                mass, torch.ones_like(mass), atol=2e-6, rtol=0)):
            raise ValueError('Saved proposal must be a normalized official distribution')
        p0, p1 = [distribution(value, 1., .95) for value in target]
        a0, a1 = torch.minimum(p0, q).sum(-1), torch.minimum(p1, q).sum(-1)
        delta = a1-a0
        cases.append(dict(prefix_sha256=ranks[0]['prefix_sha256'], identity=ranks[0]['identity'],
                          alpha_reference=a0.tolist(), alpha_candidate=a1.tolist(), alpha_delta=delta.tolist(),
                          mean_delta=float(delta.mean()), minimum_delta=float(delta.min()),
                          proposal_mass=mass.tolist(), target_max_abs=float((target[1]-target[0]).abs().max()),
                          reference_top1=p0.argmax(-1).tolist(), candidate_top1=p1.argmax(-1).tolist()))
    # This is a cohort mean, not a demand that every floating-point position
    # move in the same direction. Preserve every position's change for audit.
    mean_delta = sum(row['mean_delta'] for row in cases)/len(cases)
    savings = []
    has_full_timing = all(len(rank['rounds']) == 6 for rank in reports)
    if has_full_timing:
        for iteration in range(3):
            slowest = [max(next(item['median_device_ms'] for item in rank['rounds']
                                if item['iteration'] == iteration and item['arm'] == arm)
                           for rank in reports) for arm in (0, 1)]
            savings.append(slowest[0]-slowest[1])
    # The scored cost is the slowest TP rank, in each paired invocation.
    # Keep every local delta for diagnosis; a faster noncritical rank is not
    # an independent round and must not add a fourth performance gate.
    timing_passed = has_full_timing and all(value > 0 for value in savings)
    candidates = {row['candidate'] for row in reports}
    if len(candidates) != 1:
        raise ValueError('Ranks must qualify the same Target candidate')
    result = dict(candidate=candidates.pop(), status='conditional_acceptance_checked',
                  reference='current source parent on the same recorded real prefix',
                  temperature=1., top_p=.95, alpha_definition='sum_v min(p_k(v),q_k(v))',
                  cases=cases, mean_alpha_delta=mean_delta, numerical_allowance=1e-7,
                  passed=mean_delta >= -1e-7, end_to_end_quality_passed=False,
                  full_target_native_pairs_saved_ms=savings,
                  rank_local_paired_saved_ms=[rank.get('paired_saved_ms', []) for rank in reports],
                  timing_rule='Three paired slowest-rank round deltas positive; forecast gate is separate',
                  projected_saved_ms_per_round=statistics.median(savings) if savings else None,
                  full_target_timing_passed=timing_passed,
                  micro_passed=timing_passed and mean_delta >= -1e-7,
                  limitation=('Common saved prefix/proposal state; complete candidate-prefill and '
                              'future draft state require serving quality'))
    (run/'fixed-prefix-acceptance-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
