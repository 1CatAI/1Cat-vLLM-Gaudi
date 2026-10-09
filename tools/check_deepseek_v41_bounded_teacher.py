# SPDX-License-Identifier: Apache-2.0
"""Reuse native teacher logits to audit certified nucleus versus exact repair.

This checks conditional probability semantics, not a new device measurement.
Device FP32 agreement and seeded returned transactions belong to the native
protocol gate. Uncovered rows must retain the exact distribution through repair.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from check_deepseek_v41_teacher_forced_acceptance import distribution

    reports = []
    sources = []
    for case in range(3):
        drafts, targets = [], []
        for rank in range(4):
            draft_path = args.draft / f'mtp-prefix-case{case}-rank{rank}.pt'
            target_path = args.target / f'fixed-prefix-case{case}-rank{rank}.pt'
            draft = torch.load(draft_path, weights_only=True, map_location='cpu')
            target = torch.load(target_path, weights_only=True, map_location='cpu')
            if draft['prefix_sha256'] != target['prefix_sha256'] or not target['teacher_forced']:
                raise ValueError('Teacher-forced Target and draft must share the actual prefix')
            drafts.append(draft)
            targets.append(target)
            for path in (draft_path, target_path):
                sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if len({row['prefix_sha256'] for row in drafts}) != 1:
            raise ValueError('Rank prefix mismatch')
        logits = torch.cat([row['reference_draft_logits'] for row in drafts], -1).double()
        p = distribution(torch.cat([row['reference_target_logits'] for row in targets], -1), 1., .95)
        q = distribution(logits, 1., .95)
        scores, ids = logits.sort(dim=-1, descending=True, stable=True)
        mass = (scores - scores[:, :1]).exp()
        mass /= mass.sum(-1, keepdim=True)
        small = mass[:, :64]
        certified = small.sum(-1) >= .95
        kept = torch.where(small.cumsum(-1) - small < .95, small, 0.)
        kept /= kept.sum(-1, keepdim=True)
        bounded = torch.zeros_like(q).scatter(-1, ids[:, :64], kept)
        repaired = torch.where(certified[:, None], bounded, q)
        a, b = torch.minimum(p, q).sum(-1), torch.minimum(p, repaired).sum(-1)
        delta = b - a
        reports.append(dict(case=case, prefix_sha256=drafts[0]['prefix_sha256'],
                            nucleus_size=(q > 0).sum(-1).tolist(), covered=certified.tolist(),
                            alpha_reference=a.tolist(), alpha_candidate=b.tolist(), delta=delta.tolist(),
                            probability_max_abs=float((repaired - q).abs().max()),
                            passed=bool((delta >= -1e-7).all())))
    result = dict(candidate='certified K64 draft nucleus', teacher_forced=True, temperature=1., top_p=.95,
                  sources=sources, cases=reports, passed=all(row['passed'] for row in reports),
                  scope='Conditional exact-probability audit on saved native teacher prefixes; unchanged Target/MTP',
                  limits='FP64 semantic calculation. Native FP32 sampler agreement is separately checked by gate442; '
                         'future full-request coverage frequency and readable output still require serving.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
