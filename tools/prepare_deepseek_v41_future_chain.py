# SPDX-License-Identifier: Apache-2.0
"""Freeze real per-layer producer inputs and gates for native chain A/B."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--peer-fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre, hc_post
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    torch.set_num_threads(8)
    report = dict(status='running', layers=[20, 21, 22, 23], cases=3, ranks=4,
                  gates='Frozen official CPU FP32 hc_pre, propagated using saved attention/MoE reductions',
                  producer='Actual per-layer WO_B projection input', performance_qualified=False)
    manifest = args.output / 'fixtures.json'
    try:
        for rank in range(4):
            directory = args.output / f'rank{rank}'
            directory.mkdir()
            shard = PreparedV41Shard(args.prepared, 0, rank)
            controls = {layer: {part: [shard.tensor(f'layers.{layer}.hc_{part}_{key}', 'cpu').float()
                                      for key in ('fn', 'scale', 'base')]
                                for part in ('attn', 'ffn')} for layer in range(20, 24)}
            for case in range(3):
                state = torch.load(args.fixtures / f'rank{rank}/c6-{case}.pt', map_location='cpu',
                                   weights_only=True, mmap=True)
                captures = torch.load(args.peer_fixtures / f'peer-inputs-rank{rank}-case{case}.pt',
                                      map_location='cpu', weights_only=True, mmap=True)[0]
                if not state['request_context_qualified'] or not state['groups_qualified']:
                    raise ValueError('Fixture source is not a qualified production request')
                residual, previous = state['groups'][4]['residual'], state['groups'][4]['pre']
                for capture in captures:
                    layer = capture['layer']
                    if layer not in controls:
                        raise ValueError('Captured layers differ')
                    _, pre, post, comb = hc_pre(residual, previous, *controls[layer]['attn'])
                    value = dict(layer=layer, case=case, rank=rank,
                                 projection_input=capture['attention']['projection_input'].bfloat16().contiguous(),
                                 residual=residual.contiguous(), pre=pre.contiguous(), post=post.contiguous(),
                                 comb=comb.contiguous(), actual_request_context=True)
                    torch.save(value, directory / f'layer{layer}-case{case}.pt')
                    residual = hc_post(capture['attention']['reduced'].bfloat16(), residual, post, comb)
                    _, previous, post, comb = hc_pre(residual, pre, *controls[layer]['ffn'])
                    residual = hc_post(capture['moe']['reduced'].bfloat16(), residual, post, comb)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        manifest.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
