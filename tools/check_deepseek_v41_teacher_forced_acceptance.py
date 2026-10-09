# SPDX-License-Identifier: Apache-2.0
"""Audit fixed-prefix p/q changes independently of a sampled output's RNG.

Each saved case must identify one teacher-forced prompt/continuation prefix and
contain full-vocabulary reference/candidate Target and draft logits on that
same prefix. It is invalid to substitute different sampled trajectories or
rank-local fragments. This is an acceptance gate, not an end-to-end speed test.
"""
import argparse
import hashlib
import json
from pathlib import Path


def distribution(logits, temperature, top_p):
    import torch

    scores = logits.double() / temperature
    sorted_scores, order = scores.sort(dim=-1, descending=True, stable=True)
    probabilities = sorted_scores.softmax(-1)
    retained = torch.where(probabilities.cumsum(-1) - probabilities < top_p, probabilities, 0.)
    retained /= retained.sum(-1, keepdim=True)
    return torch.zeros_like(retained).scatter(-1, order, retained)


def evaluate(case, temperature=1., top_p=.95, allowance=1e-7):
    import torch

    if not case.get('teacher_forced') or not case.get('prefix_sha256'):
        raise ValueError('Teacher-forced same-prefix identity is required')
    prefixes = case['prefix_sha256']
    if len(prefixes) != 4 or len(set(prefixes)) != 1:
        raise ValueError('Reference/candidate Target and draft must share the exact prefix')
    names = ('reference_target_logits', 'reference_draft_logits',
             'candidate_target_logits', 'candidate_draft_logits')
    tensors = [case[name].cpu() for name in names]
    if any(value.ndim != 2 or value.shape[0] != 5 or value.shape[1] != 129280 or
           not torch.isfinite(value).all() for value in tensors):
        raise ValueError('Five positions and full official vocabulary logits required')
    p, q, candidate_p, candidate_q = [distribution(value, temperature, top_p) for value in tensors]
    baseline = torch.minimum(p, q).sum(-1)
    candidate = torch.minimum(candidate_p, candidate_q).sum(-1)
    delta = candidate - baseline
    return dict(alpha_reference=baseline.tolist(), alpha_candidate=candidate.tolist(),
                alpha_delta=delta.tolist(), mean_delta=float(delta.mean()), minimum_delta=float(delta.min()),
                passed=bool((delta >= -allowance).all()), allowance=allowance,
                target_max_abs=float((tensors[2].float() - tensors[0].float()).abs().max()),
                draft_max_abs=float((tensors[3].float() - tensors[1].float()).abs().max()),
                prefix_sha256=prefixes[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixtures', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= len(args.fixtures) <= 5:
        parser.error('Use three to five actual fixed-prefix cases')
    import torch

    results = []
    for path in args.fixtures:
        case = torch.load(path, weights_only=True, map_location='cpu')
        row = evaluate(case)
        row.update(source=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        results.append(row)
    report = dict(reference='Official/C1 production same-prefix Target and proposal logits',
                  temperature=1., top_p=.95, alpha_definition='sum_v min(p_k(v), q_k(v))',
                  tests=results, passed=all(row['passed'] for row in results),
                  end_to_end_quality_passed=False)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
