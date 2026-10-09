# SPDX-License-Identifier: Apache-2.0
"""Join native draft shards to unchanged, same-prefix production Target logits."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(run, target, candidate, native):
    import torch
    from check_deepseek_v41_teacher_forced_acceptance import evaluate

    reports = [json.loads((run / f'mtp-teacher-rank{rank}.json').read_text()) for rank in range(4)]
    if any(row['status'] != 'completed_conditional_teacher' or row['candidate'] != candidate for row in reports):
        raise ValueError('All four native draft workers must finish the same numerical candidate')
    count = len(reports[0]['cases'])
    if not 3 <= count <= 5 or any(len(row['cases']) != count for row in reports):
        raise ValueError('Three to five identical real cases required on every rank')
    result = []
    for case in range(count):
        draft = [torch.load(run / f'mtp-prefix-case{case}-rank{rank}.pt', weights_only=True, map_location='cpu')
                 for rank in range(4)]
        targets = (draft if candidate == 'vocab_head_fp8' else
                   [torch.load(target / f'fixed-prefix-case{case}-rank{rank}.pt',
                               weights_only=True, map_location='cpu') for rank in range(4)])
        prefixes = [row['prefix_sha256'] for row in draft + targets]
        if len(set(prefixes)) != 1 or any(row['identity'] != draft[0]['identity'] for row in draft + targets):
            raise ValueError('Draft and unchanged Target identities differ')
        p = torch.cat([row['reference_target_logits'] for row in targets], -1)
        changed_p = (torch.cat([row['candidate_target_logits'] for row in targets], -1)
                     if candidate == 'vocab_head_fp8' else p)
        before = torch.cat([row['reference_draft_logits'] for row in draft], -1)
        after = torch.cat([row['candidate_draft_logits'] for row in draft], -1)
        joined = dict(teacher_forced=True, prefix_sha256=[prefixes[0]] * 4,
                      reference_target_logits=p, candidate_target_logits=changed_p,
                      reference_draft_logits=before, candidate_draft_logits=after)
        row = evaluate(joined)
        row.update(case=case, delta=row['alpha_delta'])
        result.append(row)
    mean_delta = sum(row["mean_delta"] for row in result)/count
    result = dict(candidate=candidate, teacher_forced=True, temperature=1., top_p=.95,
                reference=('Same actual normalized Target hidden and native teacher C5/prescribed Markov inputs; '
                           'BF16 parent versus FP8 Target/draft head p/q' if candidate == 'vocab_head_fp8' else
                           'Same frozen current production Target p; BF16 parent versus candidate native C5/Markov q'),
                cases=result, passed=mean_delta >= -1e-7,
                mean_alpha_delta=mean_delta,
                all_retained_positions_non_decreasing=all(row['passed'] for row in result),
                gate='Retain established MTP439 aggregate teacher acceptance: mean alpha over all15fixed positions; '
                     'individual negative deltas are reported, not claimed unchanged',
                policy_reference='MTP_TEACHER_439_DECISION.json',
                complete_request_acceptance_unqualified=True,
                mtp_anchor_alignment='C5 positions begin at current Target anchor position',
                native_teacher_root=str(run), target_logits_root=str(run if candidate == 'vocab_head_fp8' else target),
                native_binaries={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in native.glob('*.so')},
                target_record_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (run.glob('mtp-prefix-case*-rank*.pt') if candidate == 'vocab_head_fp8'
                                                else target.glob('fixed-prefix-case*-rank*.pt'))},
                performance_measured=False, end_to_end_quality_passed=False)
    if candidate in ('draft_packed_mla', 'draft_kv_decode', 'draft_tail'):
        sources = ('vllm_gaudi/ops/deepseek_v41_paged_attention.py', 'vllm_gaudi/ops/deepseek_v41_math.py')
        result['draft_attention_source_sha256'] = {
            name: hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest() for name in sources}
        result['target_producer_scope'] = 'Unchanged full40 same-prefix Target logits'
    if candidate == 'draft_tail':
        result['combined_switches'] = ['VLLM_HPU_DSV41_DSPARK_DRAFT_KV_DECODE',
                                      'VLLM_HPU_DSV41_DSPARK_MTP_K128']
        result['captured_arms'] = [row['captured_draft_tail_arms'] for row in reports]
    if candidate == 'draft_mhc':
        sources = ('vllm_gaudi/models/deepseek_v41_program.py', 'vllm_gaudi/ops/deepseek_v41_math.py')
        result['draft_mhc_source_sha256'] = {
            name: hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest() for name in sources}
    if candidate == 'draft_shared_fp8':
        sources = ('vllm_gaudi/ops/deepseek_v41_draft_shared_fp8.py',
                   'vllm_gaudi/ops/deepseek_v41_woa_fp8.py', 'vllm_gaudi/models/deepseek_v41_program.py')
        result['draft_shared_source_sha256'] = {
            name: hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest() for name in sources}
    if candidate == 'draft_query_fp8':
        sources = ('vllm_gaudi/ops/deepseek_v41_draft_query_fp8.py',
                   'vllm_gaudi/ops/deepseek_v41_paged_attention.py',
                   'vllm_gaudi/ops/deepseek_v41_woa_fp8.py', 'vllm_gaudi/models/deepseek_v41_program.py')
        result['draft_query_source_sha256'] = {
            name: hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest() for name in sources}
    if candidate == 'draft_dense_fp8':
        sources = ('vllm_gaudi/ops/deepseek_v41_draft_dense_fp8.py',
                   'vllm_gaudi/ops/deepseek_v41_paged_attention.py',
                   'vllm_gaudi/ops/deepseek_v41_qkv.py', 'vllm_gaudi/ops/deepseek_v41_woa_fp8.py',
                   'vllm_gaudi/models/deepseek_v41_program.py')
        result['draft_dense_source_sha256'] = {
            name: hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest() for name in sources}
    if candidate == 'vocab_head_fp8':
        sources = ('vllm_gaudi/ops/deepseek_v41_vocab_head_fp8.py',
                   'vllm_gaudi/ops/deepseek_v41_woa_fp8.py', 'vllm_gaudi/models/deepseek_v41_program.py')
        result['head_source_sha256'] = {name: hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest()
                                       for name in sources}
        result['target_producer_scope'] = 'Frozen actual-request normalized Target hidden; head-only p/q change'
        result['full_target_producer_reexecuted'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--candidate', choices=('mtp_fp8', 'mtp_sat', 'mtp_k128', 'draft_mhc', 'vocab_head_fp8',
                                               'draft_packed_mla', 'draft_kv_decode', 'draft_shared_fp8',
                                               'draft_query_fp8', 'draft_dense_fp8', 'draft_tail'),
                        required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run.resolve(), args.target.resolve(), args.candidate, args.native.resolve())
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], mean_alpha_delta=report['mean_alpha_delta'],
                         per_case=[row['alpha_delta'] for row in report['cases']]), indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
