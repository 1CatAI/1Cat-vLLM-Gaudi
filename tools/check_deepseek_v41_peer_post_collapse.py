# SPDX-License-Identifier: Apache-2.0
"""Check the common C1 post boundary with recorded real four-rank packets."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--peer-fixtures', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    load_native_operators(required=('custom_deepseek_v41_peer_mhc_post_collapse_gaudi2',))
    shard = PreparedV41Shard(args.prepared, 0, 0)
    fn, scale, base = [shard.tensor('layers.20.hc_attn_' + name, 'hpu')
                       for name in ('fn', 'scale', 'base')]
    weight = fn.bfloat16()

    def producer(residual, pre):
        return hc_pre(residual, pre, fn, scale, base, 1e-20, 1e-6, 20,
                      packed_fn=fn, decode=True, control_mme_weight=weight)

    def reference(peers, residual, post, comb, pre):
        value = peers[0].float()
        for rank in range(1, peers.shape[0]):
            value = value + peers[rank].float()
        return torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
            value.bfloat16(), residual, post, comb, pre)

    def candidate(peers, residual, post, comb, pre):
        return torch.ops.custom_op.custom_deepseek_v41_peer_mhc_post_collapse_gaudi2(
            peers, residual, post, comb, pre)

    functions = [torch.compile(fn, backend='hpu_backend', fullgraph=True, dynamic=False)
                 for fn in (producer, reference, candidate)]
    checks = []
    for case in range(3):
        record = torch.load(args.fixtures / f'c6-{case}.pt', map_location='cpu', weights_only=True)
        if not record.get('request_context_qualified') or not record.get('groups_qualified'):
            raise ValueError('Only qualified actual request fixtures are accepted')
        residual, previous = [record['groups'][4][key].to('hpu') for key in ('residual', 'pre')]
        packets = [torch.load(args.peer_fixtures / f'peer-inputs-rank{rank}-case{case}.pt',
                              map_location='cpu', weights_only=True)[0][0]['attention']['partial']
                   for rank in range(4)]
        peers = torch.stack(packets).to('hpu')
        _, pre, post, comb = functions[0](residual, previous)
        operands = peers, residual, post.contiguous(), comb.contiguous(), pre.contiguous()
        outputs = [[tensor.cpu() for tensor in fn(*operands)] for fn in functions[1:]]
        errors = [dict(exact=torch.equal(a, b), max_abs=float((a.float()-b.float()).abs().max()),
                       rms=float((a.float()-b.float()).square().mean().sqrt()))
                  for a, b in zip(*outputs, strict=True)]
        checks.append(dict(case=case, errors=errors))
    report = dict(checks=checks, passed=all(e['exact'] for c in checks for e in c['errors']),
                  reference='Production C1 fused post/collapse, ordered FP32 peer sum, BF16 boundary',
                  performance_measured=False, scope='Actual layer20 weights, request inputs and four-rank partials')
    (Path(os.environ['DSV41_RUN_EVIDENCE'])/'peer-post-collapse.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise AssertionError('Peer fusion changed the qualified C1 post/collapse boundary')


if __name__ == '__main__':
    main()
