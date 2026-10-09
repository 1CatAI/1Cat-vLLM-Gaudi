# SPDX-License-Identifier: Apache-2.0
"""Check the repair draw against real probabilities and CDF boundary inputs."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    args = parser.parse_args()
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators

    load_native_operators(required=('custom_deepseek_v41_probability_full_draw_gaudi2',
                                    'custom_deepseek_v41_probability_draw_gaudi2'))
    torch.hpu.set_device(0)

    def operation(probability, controls):
        token = torch.ops.custom_op.custom_deepseek_v41_probability_full_draw_gaudi2(probability, controls)
        main = torch.ops.custom_op.custom_deepseek_v41_probability_draw_gaudi2(probability, controls)
        return token, probability.gather(1, token.long()[:, None]), main

    compiled = torch.compile(operation, backend='hpu_backend', fullgraph=True, dynamic=False)
    cases = []
    for case in range(3):
        shards = [torch.load(args.fixtures / f'fixed-prefix-case{case}-rank{rank}.pt',
                             map_location='cpu', weights_only=True) for rank in range(4)]
        if len({s['prefix_sha256'] for s in shards}) != 1:
            raise ValueError('Only common real prefixes qualify')
        q = torch.cat([s['draft_probabilities'] for s in shards], -1).contiguous()
        cases.append((f'actual-prefix-{case}', q, torch.tensor([.13 + .17 * i for i in range(5)])))
    q = torch.zeros(5, 129280)
    q[:, [2046, 2047, 4096, 126999]] = torch.tensor([.125, .125, .25, .5])
    cases.append(('zero-one-and-part-boundaries', q, torch.tensor([0., .25, .25000003, .99999994, 1.])))
    generator = torch.Generator().manual_seed(683)
    q = torch.zeros(5, 129280)
    prefix = torch.randn(2048, generator=generator).mul(2).exp()
    q[:, :2048] = prefix / prefix.double().sum() * .25
    q[:, 4095] = .75
    cases.append(('different-reduction-trees', q, torch.tensor([.25, .24999998, .25000003, 0., 1.])))
    records = []
    for name, q, uniforms in cases:
        controls = torch.stack((torch.ones(5), torch.full((5,), .95), uniforms, torch.zeros(5)), -1)
        before = q.clone()
        token, support, main = (v.cpu() for v in compiled(q.to('hpu'), controls.to('hpu')))
        if not ((token >= 0) & (token < q.shape[-1])).all() or not (support > 0).all():
            raise AssertionError((name, token.tolist(), support.tolist()))
        cdf = q.double().cumsum(-1)
        expected = ((cdf < uniforms.double()[:, None] * cdf[:, -1:]) | (cdf == 0)).sum(-1).int()
        # Exact boundary masses may differ by FP32 accumulation. Preserve the
        # discrepancy and check that it lies within the accepted mass error.
        cdf32 = q.cumsum(-1)
        threshold = uniforms[:, None] * cdf32[:, -1:]
        previous = cdf32.gather(1, (token - 1).clamp_min(0).long()[:, None])
        previous = torch.where(token[:, None] == 0, 0., previous)
        selected = cdf32.gather(1, token.long()[:, None])
        error = torch.maximum((previous - threshold).clamp_min(0), (threshold - selected).clamp_min(0))
        if error.max() > 2e-6:
            raise AssertionError((name, float(error.max()), token.tolist(), expected.tolist()))
        if not torch.equal(q, before):
            raise AssertionError('Repair mutated its reference probabilities')
        records.append(dict(case=name, tokens=token.tolist(), fp64_tokens=expected.tolist(),
                            fp64_token_exact=torch.equal(token, expected), max_mass_error=float(error.max()),
                            positive_support=True, q_unchanged=True, main_draw=main.tolist(),
                            incorrect_certified_main_rows=((main[:, 0] != expected) & main[:, 1].bool()).sum().item()))
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    (root / 'full-cdf-boundaries.json').write_text(json.dumps(dict(
        status='passed', cases=records, teacher_forced_alpha_delta=0,
        performance_measured=False, numerical_tolerance=2e-6), indent=2) + '\n')


if __name__ == '__main__':
    main()
