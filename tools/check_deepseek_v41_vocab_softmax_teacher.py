# SPDX-License-Identifier: Apache-2.0
"""Numerical gate for full-q normalization on three saved teacher prefixes.

This gate measures no performance: the native producer/consumer protocol must
be timed separately. Only reference draft logits are reused, never an older
numerical candidate's logits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = os.environ.get('PT_HPU_RECIPE_CACHE_CONFIG', '').replace('{rank}', '0')
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators, prepare_native_libraries

    prepare_native_libraries()
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from check_deepseek_v41_teacher_forced_acceptance import distribution

    torch.hpu.set_device(0)
    load_native_operators(required=('custom_deepseek_v41_vocab_softmax_f32_gaudi2',))
    operation = torch.ops.custom_op.custom_deepseek_v41_vocab_softmax_f32_gaudi2
    functions = [torch.compile(fn, backend='hpu_backend', fullgraph=True, dynamic=False)
                 for fn in (lambda x: x.softmax(-1), lambda x: operation(x))]
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(status='running', cases=[], performance_measured=False, formal_gain_ms=0,
                  scope='M1 full vocabulary q normalization; native production protocol timing pending')

    def save():
        (root / 'softmax-teacher.json').write_text(json.dumps(report, indent=2) + '\n')

    def nucleus(probabilities, order):
        probabilities = probabilities.double().gather(-1, order)
        retained = torch.where(probabilities.cumsum(-1) - probabilities < .95, probabilities, 0.)
        retained /= retained.sum(-1, keepdim=True)
        return torch.zeros_like(retained).scatter(-1, order, retained)

    save()
    try:
        with torch.inference_mode():
            for case in range(3):
                paths = [args.draft / f'mtp-prefix-case{case}-rank{rank}.pt' for rank in range(4)]
                target_paths = [args.target / f'fixed-prefix-case{case}-rank{rank}.pt' for rank in range(4)]
                draft = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
                target = [torch.load(p, map_location='cpu', weights_only=True) for p in target_paths]
                joined = draft + target
                if len({x['prefix_sha256'] for x in joined}) != 1 or any(
                        x['identity'] != draft[0]['identity'] for x in joined):
                    raise ValueError('Target and draft teacher prefix identity mismatch')
                q_logits = torch.cat([x['reference_draft_logits'] for x in draft], -1).float()
                p_logits = torch.cat([x['reference_target_logits'] for x in target], -1)[:5]
                if q_logits.shape != (5, 129280) or p_logits.shape != q_logits.shape:
                    raise ValueError('Five real full vocabulary rows required')
                outputs = [[], []]
                for row in q_logits:
                    operand = row.unsqueeze(0).contiguous().to('hpu')
                    for arm, fn in enumerate(functions):
                        report['active_case'] = case
                        report['active_arm'] = arm
                        save()
                        outputs[arm].append(fn(operand).cpu())
                before, after = [torch.cat(x) for x in outputs]
                finite = bool(torch.isfinite(after).all())
                probability_ok = torch.allclose(before, after, atol=2e-7, rtol=2e-5)
                order = q_logits.argsort(dim=-1, descending=True, stable=True)
                p = distribution(p_logits, 1., .95)
                qb, qa = [nucleus(x, order) for x in (before, after)]
                alpha_before, alpha_after = [torch.minimum(p, x).sum(-1) for x in (qb, qa)]
                delta = alpha_after - alpha_before
                report['cases'].append(dict(case=case, prefix_sha256=draft[0]['prefix_sha256'],
                    fixture_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + target_paths},
                    finite=finite, probability_tolerance_passed=bool(probability_ok),
                    maximum_abs=float((after - before).abs().max()),
                    row_mass=after.sum(-1).tolist(), support_changes=(qb.ne(0) != qa.ne(0)).sum(-1).tolist(),
                    alpha_reference=alpha_before.tolist(), alpha_candidate=alpha_after.tolist(),
                    alpha_delta=delta.tolist(), mean_delta=float(delta.mean())))
                save()
                if not finite or not probability_ok:
                    raise AssertionError('Probability tolerance failed')
            report['mean_alpha_delta'] = sum(x['mean_delta'] for x in report['cases']) / 3
            report['passed'] = report['mean_alpha_delta'] >= -1e-7
            report['status'] = 'completed'
            if not report['passed']:
                raise AssertionError('Aggregate teacher acceptance decreased')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
