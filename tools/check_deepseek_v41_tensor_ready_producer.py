# SPDX-License-Identifier: Apache-2.0
"""Untimed ordinary producer handoff: isolate compiler from native bindings."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--peer-fixtures', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = os.environ.get('PT_HPU_RECIPE_CACHE_CONFIG', '').format(rank=0)
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment, load_native_operators
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    torch.hpu.set_device(0)
    load_native_operators()
    torch.ops.load_library(args.library)
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(status='running', performance_measured=False, cases=[])
    try:
        with torch.inference_mode():
            shard = PreparedV41Shard(args.prepared, pp_rank=0, tp_rank=0)
            weight = shard.dense('layers.20.attn.wo_b.weight', 'cpu').bfloat16().contiguous().to('hpu')
            cw = shard.tensor('layers.20.hc_ffn_fn', 'cpu').bfloat16().contiguous().to('hpu')
            compiled = []
            for external in (False, True):
                def body(x, control, external=external):
                    partial, projected = torch.ops.custom_op.custom_deepseek_v41_peer_signal_probe_gaudi2(
                        x, weight, control, cw, False)
                    partial = torch.ops.custom_op.custom_deepseek_v41_peer_ready_identity_gaudi2(
                        partial.reshape(1, -1), external).reshape(6, 5120)
                    return partial, projected
                compiled.append(torch.compile(body, fullgraph=True, dynamic=False, backend='hpu_backend'))
            for case in (0, 1, 2, 2, 1, 0):
                packet = torch.load(args.peer_fixtures/f'peer-inputs-rank0-case{case}.pt', weights_only=True,
                                    map_location='cpu')[0]
                fixture = torch.load(args.fixtures/f'rank0/c6-{case}.pt', weights_only=True, map_location='cpu')
                x = packet['projection_input'].bfloat16().contiguous().to('hpu')
                flat = fixture['groups'][4]['residual'].float().flatten(1).to('hpu')
                hi = flat.bfloat16()
                control = torch.cat((hi, (flat-hi.float()).bfloat16())).contiguous()
                reference = torch.ops.custom_op.custom_deepseek_v41_peer_signal_probe_gaudi2(
                    x, weight, control, cw, False)
                results = [fn(x, control) for fn in compiled]
                torch.hpu.synchronize()
                ref, arms = [v.cpu() for v in reference], [[v.cpu() for v in arm] for arm in results]
                errors = [dict(exact=torch.equal(a, b), max_abs=float((a.float()-b.float()).abs().max()))
                          for a, b in zip(*arms, strict=True)]
                references = [[float((a.float()-b.float()).abs().max()) for a, b in zip(arm, ref, strict=True)]
                              for arm in arms]
                report['cases'].append(dict(case=case, arm_errors=errors, ordinary_reference_max_abs=references))
                if not all(e['exact'] for e in errors) or any(v > 0.015625 for arm in references for v in arm):
                    torch.save(dict(case=case, reference=ref, arms=arms), root/'producer-mismatch.pt')
                    raise AssertionError('Ordinary producer handoff exceeds the existing MME envelope')
            report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (root/'producer.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
