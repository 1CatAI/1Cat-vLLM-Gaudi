# SPDX-License-Identifier: Apache-2.0
"""Bounded one-card FFN correctness admission, no performance vote.

Native four-layer replay owns performance qualification. This small admission
checks numeric producer/consumer contracts before communication/model capture.
"""
import argparse
import copy
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', choices=('cooperative_silu','router_shared_bf16','router_batched_f32'),
                        default='cooperative_silu')
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators
    from vllm_gaudi.models.deepseek_v41_program import PreparedMoE, _weight_tree, load_weight_tree
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut

    torch.set_num_threads(1)
    flag='VLLM_HPU_DSV41_DSPARK_'+args.candidate.upper()
    os.environ[flag] = '1'
    load_native_operators()
    torch.hpu.set_device(0)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    specs = {k: v for k, v in shard.specs.items()
             if k.startswith(('layers.20.ffn.', 'layers.20.ffn_norm.'))}
    weights = _weight_tree(specs)
    report = dict(status='running', performance_vote=False, checks=[])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    def weight_views(module):
        result=copy.copy(module)
        result._buffers=module._buffers.copy()
        result._parameters=module._parameters.copy()
        result._modules={k:weight_views(v)if v is not None else None for k,v in module._modules.items()}
        return result

    with torch.inference_mode():
        load_weight_tree(shard, weights, 'hpu', specs)
        layer = weights.layers.get_submodule('20')
        lookup = mxfp4_bf16_lut('hpu')
        cfg=json.loads((args.prepared/'config.json').read_text())['text_config']
        epsilon=cfg['rms_norm_eps']
        if args.candidate in ('router_shared_bf16','router_batched_f32'):
            from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
            sidecar=DenseFP8Sidecar(Path(os.environ['VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR']),shard)
            selected={k:v for k,v in specs.items()if '.shared_experts.'in k and k.endswith('weight')}
            load_weight_tree(shard,weights,'hpu',selected,dense_sidecar=sidecar,
                             dense_config={k:[20]for k in ('shared_w1','shared_w3','shared_w2')})
        models = []
        for arm in (0, 1):
            os.environ[flag] = str(arm)
            moe = PreparedMoE(weight_views(layer.ffn), 6, True, lookup, lambda x, **kw: x, tensor_parallel_size=4)
            moe.layer=20
            moe.prepare_split_scale_planes()
            if args.candidate in ('router_shared_bf16','router_batched_f32'):
                moe.prepare_shared_gate_up_weight()
                moe.prepare_router_shared_bf16_weight(shard)
                moe.prepare_router_batched_weight()
            if not moe.c6_token_wide_sat:
                raise ValueError('Admission must use the actual SAT weight contract')
            models.append(moe)

        def parent(x, ids, routing, q, sx):
            if args.candidate in ('router_shared_bf16','router_batched_f32'):
                return models[0](x,torch.zeros(x.shape[0],device=x.device,dtype=torch.bool),
                                 decode=True,prequant=(q,sx))
            return models[0]._forward_n256_fp8(x, ids, routing, prequant=(q, sx))

        def candidate(x, ids, routing, q, sx):
            if args.candidate in ('router_shared_bf16','router_batched_f32'):
                return models[1](x,torch.zeros(x.shape[0],device=x.device,dtype=torch.bool),
                                 decode=True,prequant=(q,sx))
            return models[1]._forward_n256_fp8(x, ids, routing, prequant=(q, sx))

        def producer(residual, pre):
            x = (residual.float()*pre.unsqueeze(-1)).sum(1).bfloat16()
            x, q, sx = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                x.contiguous(), layer.ffn_norm.weight, epsilon)
            logits = torch.nn.functional.linear(x.float(), layer.ffn.gate.weight.float())
            ids, routing = torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(
                logits.contiguous(), layer.ffn.gate.bias, layer.ffn.gate.bias_vl,
                torch.zeros(x.shape[0], dtype=torch.bool, device=x.device))
            return x, ids, routing, q, sx

        produce = torch.compile(producer, backend='hpu_backend', fullgraph=True, dynamic=False)
        functions = [torch.compile(f, backend='hpu_backend', fullgraph=True, dynamic=False)
                     for f in (parent, candidate)]
        for case, path in enumerate(sorted((args.fixtures/'rank0').glob('c6-*.pt'))[:3]):
            saved = torch.load(path, map_location='cpu', weights_only=True)
            source = saved['groups'][4]
            inputs = produce(source['residual'].to('hpu'), source['pre'].to('hpu'))
            if args.candidate=='router_shared_bf16':
                normalized=inputs[0]
                paired=torch.ops.custom_op.custom_deepseek_v41_dense_bf16_pair_gaudi2(normalized.contiguous())
                sq,ss=torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(normalized.contiguous())
                restored=(sq.float()*ss).bfloat16()
                original_exact=torch.equal(paired[:normalized.shape[0]].cpu(),normalized.cpu())
                rounded_exact=torch.equal(paired[normalized.shape[0]:].cpu(),restored.cpu())
                if not original_exact or not rounded_exact:
                    raise AssertionError("Dual producer changed C1 original/FP8 rounded activation")
                report.setdefault('producer_checks',[]).append(dict(case=case,original_exact=original_exact,
                                                                    shared_effective_exact=rounded_exact))
            expected = functions[0](*inputs).cpu()
            # Repeat different cases and reuse the same graph: stale scratch
            # must not turn an incomplete row into a certified scale.
            actual = functions[1](*inputs).cpu()
            finite = bool(torch.isfinite(actual).all())
            delta=expected.float()-actual.float()
            relative=float(delta.norm()/expected.float().norm().clamp_min(1e-30))
            row = dict(case=case, exact=torch.equal(expected, actual), finite=finite,
                       max_abs=float(delta.abs().max()),relative_l2=relative,epsilon=epsilon)
            row['passed']=finite and (row['exact'] if args.candidate=='cooperative_silu'else relative<=.005)
            report['checks'].append(row)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(row), flush=True)
            if not row['passed']:
                raise AssertionError('FFN numeric boundary failed before native timing')
        torch.hpu.synchronize()
    report['status'] = 'capability_pass_native_chain_pending'
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
