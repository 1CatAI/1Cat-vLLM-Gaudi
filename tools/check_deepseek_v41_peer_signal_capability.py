# SPDX-License-Identifier: Apache-2.0
"""Compile a real C6 peer/control producer with an external output signal.

This is an API/numerical capability check, not a performance microbenchmark.
The native consumer/epoch gate must follow before any speed measurement.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--peer-fixtures', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = os.environ.get('PT_HPU_RECIPE_CACHE_CONFIG', '').replace('{rank}', '0')
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_native_libraries

    prepare_native_libraries()
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    torch.ops.load_library(args.library)
    operation = torch.ops.custom_op.custom_deepseek_v41_peer_signal_probe_gaudi2
    # Different static flag/code paths own independent cached recipes.
    def reference(x, w, control, cw):
        return operation(x, w, control, cw, False)

    def candidate(x, w, control, cw):
        return operation(x, w, control, cw, True)

    functions = [torch.compile(fn, backend='hpu_backend', fullgraph=True, dynamic=False)
                 for fn in (reference, candidate)]
    shard = PreparedV41Shard(args.prepared, 0, 0)
    weights = shard.dense('layers.20.attn.wo_b.weight', 'cpu').bfloat16().contiguous().to('hpu')
    control_weights = shard.tensor('layers.20.hc_ffn_fn', 'cpu').bfloat16().contiguous().to('hpu')
    checks = []
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    for case in range(3):
        record = torch.load(args.fixtures / f'c6-{case}.pt', map_location='cpu', weights_only=True)
        if not record.get('request_context_qualified') or not record.get('groups_qualified'):
            raise ValueError('Expected actual qualified request fixtures')
        operands = torch.load(args.peer_fixtures / f'peer-inputs-rank0-case{case}.pt',
                              map_location='cpu', weights_only=True)[0]
        x = operands['projection_input'].bfloat16().contiguous().to('hpu')
        residual = record['groups'][4]['residual'].flatten(1).float()
        high = residual.bfloat16()
        low = (residual-high.float()).bfloat16()
        control = torch.cat((high, low)).contiguous().to('hpu')
        outputs = [[v.cpu() for v in fn(x, weights, control, control_weights)] for fn in functions]
        errors = [dict(exact=torch.equal(a, b), maximum_abs=float((a.float()-b.float()).abs().max()))
                  for a, b in zip(*outputs, strict=True)]
        checks.append(dict(case=case, input_shape=list(x.shape), control_shape=list(control.shape), errors=errors))
        (root / 'peer-signal-capability.json').write_text(json.dumps(
            dict(checks=checks, performance_qualified=False), indent=2)+'\n')
        if not all(e['exact'] for e in errors):
            raise AssertionError('Marking readiness changed the producer arithmetic')
    report = dict(checks=checks, passed=True, performance_qualified=False,
                  native_consumer_qualified=False, external_recipe_binding_verified=False,
                  scope='Checkpoint layer20 matrices and actual request inputs; control/peer numerical capability only')
    (root / 'peer-signal-capability.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
