# SPDX-License-Identifier: Apache-2.0
"""Locate numerical divergence in the rejected W13 BF16 handoff, not time it."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-ffn', action='store_true',
                        help='Compare current K2, historical full-K and scaled full-K against identical C1 rows')
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators
    from vllm_gaudi.models.deepseek_v41_program import PreparedMoE, _weight_tree, load_weight_tree
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut

    torch.set_num_threads(1)
    load_native_operators()
    torch.hpu.set_device(0)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    specs = {name: item for name, item in shard.specs.items()
             if name.startswith(('layers.20.ffn.', 'layers.20.ffn_norm.'))}
    tree = _weight_tree(specs)
    report = dict(status='running', performance_vote=False,
                  boundary_materialization=True, checks=[])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    with torch.inference_mode():
        load_weight_tree(shard, tree, 'hpu', specs)
        layer = tree.layers.get_submodule('20')
        weights = layer.ffn.experts
        epsilon = json.loads((args.prepared / 'config.json').read_text())['text_config']['rms_norm_eps']
        lookup = mxfp4_bf16_lut('hpu')

        def produce(residual, pre):
            x = (residual.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
            x, q, sx = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                x.contiguous(), layer.ffn_norm.weight, epsilon)
            logits = torch.nn.functional.linear(x.float(), layer.ffn.gate.weight.float())
            ids, routing = torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(
                logits.contiguous(), layer.ffn.gate.bias, layer.ffn.gate.bias_vl,
                torch.zeros(x.shape[0], dtype=torch.bool, device=x.device))
            return x, ids, routing, q, sx

        def boundaries(q, ids, routing, sx):
            return torch.ops.custom_op.custom_deepseek_v41_w13_boundary_probe_gaudi2(
                q, ids, routing, weights.w13_q16, weights.w13_s16, lookup, weights.w13_fp8_channel, sx)

        produce = torch.compile(produce, backend='hpu_backend', fullgraph=True, dynamic=False)
        boundaries = torch.compile(boundaries, backend='hpu_backend', fullgraph=True, dynamic=False)
        channels = weights.w13_fp8_channel.cpu().float().reshape(384, 1280)
        if args.full_ffn:
            moe = PreparedMoE(layer.ffn, 6, True, lookup, lambda x, **kw: x, tensor_parallel_size=4)
            moe.layer = 20
            moe.prepare_split_scale_planes()

            def current(x, ids, routing, q, sx):
                return moe._forward_n256_fp8(x, ids, routing, prequant=(q, sx))

            def c1(x, ids, routing, q, sx):
                return moe._forward_n256_fp8(x, ids, routing, ordinary_decode=True, prequant=(q, sx))

            def full(x, ids, routing, q, sx):
                return torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_sat_fp8_gaudi2(
                    x, ids, routing, weights.w13_q16, weights.w2_q16, weights.w13_s16, weights.w2_s16,
                    lookup, weights.w13_fp8_channel, weights.w2_fp8_channel, q, sx, True)

            def scaled_full(x, ids, routing, q, sx):
                return torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_scaled_w13_fp8_gaudi2(
                    x, ids, routing, weights.w13_q16, weights.w2_q16, weights.w13_s16, weights.w2_s16,
                    lookup, weights.w13_fp8_channel, weights.w2_fp8_channel, q, sx, True)

            ffn = {name: torch.compile(fn, backend='hpu_backend', fullgraph=True, dynamic=False)
                   for name, fn in (('current_k2', current), ('c1', c1), ('legacy_full_k', full),
                                    ('scaled_full_k', scaled_full))}
            report['current_source_path'] = dict(w13_k2=moe.dspark_w13_k_pipeline,
                                                 w2_reduce_n256=moe.dspark_w2_reduce_n256,
                                                 scale_planes=moe.dspark_split_scale_planes,
                                                 w13_stages=moe.dspark_w13_k_pipeline_stages)
        for case, path in enumerate(sorted((args.fixtures / 'rank0').glob('c6-*.pt'))[:3]):
            saved = torch.load(path, map_location='cpu', weights_only=True)['groups'][4]
            x, ids, routing, q, sx = produce(saved['residual'].to('hpu'), saved['pre'].to('hpu'))
            if args.full_ffn:
                operands = (x, ids, routing, q, sx)
                reference = torch.cat([ffn['c1'](*(t[i:i + 1].contiguous() for t in operands)).cpu()
                                       for i in range(x.shape[0])])
                values = {name: ffn[name](*operands).cpu()
                          for name in ('current_k2', 'legacy_full_k', 'scaled_full_k')}
                row = dict(case=case, source=str(path), reference='Same producer/routes/FP8 inputs, C1 row execution',
                           ffn={})
                for name, actual in values.items():
                    diff = actual.float() - reference.float()
                    row['ffn'][name] = dict(finite=bool(torch.isfinite(actual).all()),
                                           exact=torch.equal(actual, reference), max_abs=float(diff.abs().max()),
                                           relative_l2=float(diff.norm() / reference.float().norm().clamp_min(1e-30)))
                old, selected = values['legacy_full_k'], values['scaled_full_k']
                delta = old.float() - selected.float()
                row['scaled_vs_legacy_full_k'] = dict(exact=torch.equal(old, selected),
                                                      max_abs=float(delta.abs().max()),
                                                      relative_l2=float(delta.norm() /
                                                                        old.float().norm().clamp_min(1e-30)))
                report['checks'].append(row)
                torch.save(dict(reference_c1=reference, **values),
                           args.output.parent / f'ffn-references-case{case}.pt')
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(row), flush=True)
                continue
            outputs = [tensor.cpu() for tensor in boundaries(q, ids, routing, sx)]
            product, scaled, old_q, new_q, old_sx, new_sx = outputs
            routes = ids.cpu().long().reshape(-1)
            scales = sx.cpu().float().repeat_interleave(6, 0).reshape(-1, 1, 1)
            expected = ((product.reshape(36, 1, 1280).float() * channels[routes].unsqueeze(1)) * scales).bfloat16()
            difference = scaled.float() - expected.float()
            old_bytes, new_bytes = old_q.view(torch.uint8), new_q.view(torch.uint8)
            row = dict(case=case, source=str(path), epsilon=epsilon,
                       scale_exact=torch.equal(scaled.view(torch.int16), expected.view(torch.int16)),
                       scale_max_abs=float(difference.abs().max()),
                       scale_relative_l2=float(difference.norm() / expected.float().norm().clamp_min(1e-30)),
                       middle_fp8_exact=torch.equal(old_bytes, new_bytes),
                       differing_fp8_values=int((old_bytes != new_bytes).sum()),
                       middle_scale_exact=torch.equal(old_sx, new_sx),
                       middle_scale_max_abs=float((old_sx - new_sx).abs().max()))
            report['checks'].append(row)
            # Keep the actual contrasting boundaries for CPU diagnosis. This
            # is a numeric fixture, not an SDK graph/trace export.
            torch.save(dict(product=product, scaled=scaled, expected_scaled=expected,
                            old_q_bytes=old_bytes, new_q_bytes=new_bytes,
                            old_scale=old_sx, new_scale=new_sx, ids=routes,
                            source_scale=scales, channels=channels[routes]),
                       args.output.parent / f'w13-boundaries-case{case}.pt')
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
        torch.hpu.synchronize()
    report['status'] = 'numeric_boundaries_recorded_no_performance_qualification'
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
