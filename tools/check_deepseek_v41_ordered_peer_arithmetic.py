# SPDX-License-Identifier: Apache-2.0
"""Check C1 peer arithmetic using three archived real activation distributions.

Four mHC streams are reshaped as peer packets. These are real activations,
not captured communication outputs: this isolates arithmetic and lane layout,
and supplies neither a communication performance vote nor a model tolerance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators
    from vllm_gaudi.ops.deepseek_v41_ordered_peer_sum import ordered_peer_sum, ordered_peer_sum_reference

    load_native_operators(required=('custom_deepseek_v41_ordered_peer_sum_gaudi2',))
    custom = torch.compile(ordered_peer_sum, backend='hpu_backend', fullgraph=True, dynamic=False)
    generic = torch.compile(ordered_peer_sum_reference, backend='hpu_backend', fullgraph=True, dynamic=False)
    rows = []
    for case in range(3):
        path = args.fixtures / f'c6-{case}.pt'
        data = torch.load(path, map_location='cpu', weights_only=True)
        if not data.get('request_context_qualified') or not data.get('groups_qualified'):
            raise ValueError('Archived request and group identities must be qualified')
        residual = data['groups'][5]['residual']
        if residual.shape != (6, 4, 5120) or residual.dtype != torch.bfloat16:
            raise ValueError('Expected the actual C6 four-stream residual')
        shards = residual.permute(1, 0, 2).contiguous().reshape(4, 30720)
        reference = ordered_peer_sum_reference(shards)
        actual, compiled = [fn(shards.to('hpu')).cpu() for fn in (custom, generic)]
        rows.append(dict(case=case, source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         custom_exact_to_c1_rank_order=torch.equal(actual.view(torch.int16),
                                                                  reference.view(torch.int16)),
                         generic_exact_to_c1_rank_order=torch.equal(compiled.view(torch.int16),
                                                                   reference.view(torch.int16)),
                         custom_max_abs=float((actual.float()-reference.float()).abs().max()),
                         generic_max_abs=float((compiled.float()-reference.float()).abs().max()),
                         custom_different_words=int((actual.view(torch.int16)!=reference.view(torch.int16)).sum()),
                         generic_different_words=int((compiled.view(torch.int16)!=reference.view(torch.int16)).sum())))
    report = dict(cases=rows, passed=all(row['custom_exact_to_c1_rank_order'] for row in rows),
                  scope='Arithmetic/layout only. Four actual mHC streams, not recorded peer outputs.',
                  performance_measured=False, model_precision_qualified=False)
    (Path(os.environ['DSV41_RUN_EVIDENCE'])/'ordered-peer-arithmetic.json').write_text(
        json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise AssertionError('C1 peer primitive changed ordered FP32 arithmetic or BF16 layout')


if __name__ == '__main__':
    main()
