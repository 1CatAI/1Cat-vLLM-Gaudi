# SPDX-License-Identifier: Apache-2.0
"""Real Q producer -> canonical cache -> MLA -> wo_a/wo_b, native AB3.

One-card component gate; four-card real16 and teacher qualification remain
mandatory before any ledger credit. No profiler or compiler graph export.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--matrix-oracle', action='store_true', help='Untimed isolate transposed 4D MME support')
    parser.add_argument('--publish-oracle', action='store_true', help='Untimed isolate the compound cache/MLA boundary')
    parser.add_argument('--diagnostic-stages', action='store_true', help='Untimed retain producer/consumer boundaries')
    parser.add_argument('--woa-oracle', type=Path, help='Untimed reuse retained actual wo_a operands')
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators, prepare_environment
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree
    from vllm_gaudi.ops.deepseek_v41_attention_layout import (
        interleave_output_weight, interleave_query_weight, pack_heads, unpack_heads,
    )
    from vllm_gaudi.ops.deepseek_v41_math import quantize_activation, rms_norm, rotary_table
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from deepseek_v41_micro_replay import RecipeRecorder
    from deepseek_v41_resident_ab import device_load

    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(status='loading', performance_vote=False, ledger_credit_ms=0,
                  default_enabled=False, checks=[], primitives=[], periods=[], scope='single_card_four_real_layers')

    def save():
        (root / 'ATTENTION_LAYOUT_COMPONENT.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    load_native_operators()
    torch.hpu.set_device(0)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    indices = range(20, 24)
    specs = {name: item for name, item in shard.specs.items()
             if any(name.startswith((f'layers.{i}.attn.', f'layers.{i}.attn_norm.')) for i in indices)
             and '.indexer.' not in name and '.compressor.' not in name}
    tree = _weight_tree(specs)
    load_weight_tree(shard, tree, 'hpu', specs)
    config = json.loads((args.prepared / 'config.json').read_text())['text_config']
    eps = config['rms_norm_eps']
    scaling = config['rope_scaling']
    table = rotary_table(64, 32768, config['compress_rope_theta'], scaling['original_max_position_embeddings'],
                         scaling['factor'], scaling['beta_fast'], scaling['beta_slow'])
    phase = torch.cat((table[..., 0], table[..., 1]), -1).contiguous().to('hpu')
    weights = []
    for i in indices:
        layer = tree.layers.get_submodule(str(i))
        attn = layer.attn
        wa = attn.wo_a.weight.reshape(2, 1024, 4096)
        weights.append((layer.attn_norm.weight, attn.wq_a.weight, attn.q_norm.weight,
                        attn.wq_b.weight, interleave_query_weight(attn.wq_b.weight.cpu(), 16).to('hpu'),
                        wa, interleave_output_weight(wa.cpu()).to('hpu'), attn.wo_b.weight,
                        attn.attn_sink.float()))
    scale = torch.tensor([512**-.5], device='hpu')
    if args.woa_oracle:
        rows = []
        for path in sorted(args.woa_oracle.glob('producer-consumer-*.pt'))[:3]:
            saved = torch.load(path,weights_only=True,map_location='cpu')
            for layer,item in enumerate(weights):
                old_x = saved['parent'][7+5*layer+2].reshape(6,2,4096)
                new_x = saved['candidate'][7+5*layer+2].reshape(6,2,4096)
                wa,pwa = item[5],item[6]
                wc,pc = wa.cpu(),pwa.cpu()
                recovered = pc.reshape(2,1024,512,8).transpose(2,3).reshape_as(wc)
                cpu = torch.einsum('tgd,gnd->tgn',old_x.float(),wc.float())
                def mm(x,w):
                    return torch.einsum('tgd,gnd->tgn',x,w)
                fn = torch.compile(mm,backend='hpu_backend',fullgraph=True,dynamic=False)
                eager = mm(new_x.to('hpu'),pwa).cpu()
                compiled_wo = fn(new_x.to('hpu'),pwa).cpu()
                row = dict(source=str(path),layer=20+layer,weights_inverse_exact=torch.equal(recovered,wc),
                           weights_finite=bool(torch.isfinite(pc).all()),results=[])
                for name,actual in [('eager',eager),('compiled',compiled_wo)]:
                    delta = actual.float()-cpu
                    row['results'].append(dict(name=name,finite=bool(torch.isfinite(actual).all()),
                        max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/cpu.norm().clamp_min(1e-30))))
                rows.append(row)
                report['woa_oracle']=rows
                save()
        report['status']='woa_oracle_no_performance_vote'
        save()
        return

    def chain(arm):
        def run(residual, pre, positions, swas, main, selected, pages, lengths):
            x = (residual.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
            outputs = []
            debug = []
            bank = mask = values = None
            for local, item in enumerate(weights):
                norm, wqa, qnorm, wqb, packed_q, wa, packed_wa, wb, sink = item
                value = rms_norm(x, norm, eps, native_decode=True)
                qr = rms_norm(torch.nn.functional.linear(quantize_activation(value), wqa), qnorm, eps)
                query = torch.nn.functional.linear(quantize_activation(qr), packed_q if arm else wqb)
                query = query.reshape(6, 16, 512)
                query_raw = query
                if not arm:
                    query = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(query, positions, phase)
                if not local:
                    operands = (query, swas[local], main, selected, positions, pages, sink, scale, lengths, 1)
                    if arm:
                        output, bank, mask, values = torch.ops.custom_op.custom_deepseek_v41_interleaved_publish_mla_gaudi2(
                            *operands, phase)
                    else:
                        output, bank, mask, values = torch.ops.custom_op.custom_deepseek_v41_main_split_publish_mla_gaudi2(
                            *operands)
                elif arm:
                    output = torch.ops.custom_op.custom_deepseek_v41_interleaved_reuse_mla_gaudi2(
                        query, swas[local], bank, mask, positions, sink, scale, lengths, values, phase)
                else:
                    output = torch.ops.custom_op.custom_deepseek_v41_main_split_reuse_mla_gaudi2(
                        query, swas[local], bank, mask, positions, sink, scale, lengths, values)
                if not arm:
                    output = torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(output, positions, phase)
                # Preserve the actual BF16 wo_a path: no new activation codec.
                if arm:
                    projected = torch.bmm(output,packed_wa.transpose(1,2)).transpose(0,1).flatten(1)
                else:
                    projected = torch.einsum('tgd,gnd->tgn',output.reshape(6,2,4096),wa).flatten(1)
                result = torch.nn.functional.linear(quantize_activation(projected), wb)
                outputs.append(result)
                if args.diagnostic_stages:
                    debug.extend((qr,query_raw,output,projected,quantize_activation(projected)))
            # Bank consumers in following layers are part of the timed graph.
            return (*outputs, bank, mask, values, *debug)
        return run

    recorder = RecipeRecorder(root)
    compiled = [torch.compile(chain(arm), backend='hpu_backend', fullgraph=True, dynamic=False) for arm in range(2)]
    cases = []
    for path in sorted((args.fixtures / 'rank0').glob('c6-*.pt'))[:3]:
        raw = torch.load(path, weights_only=True, map_location='cpu')

        def state(name):
            return torch.load(Path(raw['decoder_state_root']) / raw['decoder_state_files'][name],
                              weights_only=True, map_location='cpu')

        source = raw['groups'][4]
        rows = raw['positions'].numel()
        pos = raw['positions'].to('hpu')
        case = (source['residual'].to('hpu'), source['pre'].to('hpu'), pos,
                tuple(state(f'layers.{i}.attention.swa').to('hpu') for i in indices),
                state('shared.sources.20.main').to('hpu'), state('shared.topk.2.indices')[:rows].contiguous().to('hpu'),
                state('shared.block_table').to('hpu'), (pos + 1).int())
        cases.append(case)
        # Primitive proof isolates changed shuffle/rounding from GEMM order.
        query = torch.nn.functional.linear(
            quantize_activation(source['residual'][:, 0, :1280].contiguous().to('hpu')), weights[0][3]).reshape(6, 16, 512)
        packed = pack_heads(query).reshape(6, 2, 4096)
        expected = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(query, pos, phase).cpu()
        actual = torch.ops.custom_op.custom_deepseek_v41_interleaved_query_rope_gaudi2(packed, pos, phase).cpu()
        actual = unpack_heads(actual.reshape(6, 2, 512, 8))
        inverse = torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(query, pos, phase).cpu()
        packed_f32 = pack_heads(query).reshape(6, 2, 4096).float()
        roundtrip = torch.ops.custom_op.custom_deepseek_v41_interleaved_pv_bf16_gaudi2(packed_f32, pos, phase).cpu()
        roundtrip = unpack_heads(roundtrip.transpose(0,1).contiguous().reshape(6, 2, 512, 8))
        report['primitives'].append(dict(source=str(path), query_exact=torch.equal(actual, expected),
                                         pv_boundary_exact=torch.equal(roundtrip, inverse),
                                         query_max_abs=float((actual.float()-expected.float()).abs().max()),
                                         pv_max_abs=float((roundtrip.float()-inverse.float()).abs().max())))
        save()
        if not all(report['primitives'][-1][k] for k in ('query_exact', 'pv_boundary_exact')):
            raise AssertionError('Interleaved primitive changed RoPE or BF16 rounding')
        if args.publish_oracle:
            roped_query = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(query,pos,phase)
            operands = (case[3][0],case[4],case[5],pos,case[6],weights[0][8],scale,case[7])
            original = torch.ops.custom_op.custom_deepseek_v41_main_split_publish_mla_gaudi2(
                roped_query,*operands,1)
            expected_out = pack_heads(torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(
                original[0],pos,phase)).reshape(6,2,4096).transpose(0,1).contiguous().cpu()

            def publish(q,*a):
                return torch.ops.custom_op.custom_deepseek_v41_interleaved_publish_mla_gaudi2(q,*a,1,phase)

            packed_view = packed.reshape(6,16,512)
            eager = publish(packed_view,*operands)[0].cpu()
            replayed = torch.compile(publish,backend='hpu_backend',fullgraph=True,dynamic=False)(
                packed_view,*operands)[0].cpu()
            diagnostic = dict(source=str(path),outputs=[])
            for name,actual in [('eager',eager),('compiled',replayed)]:
                delta = actual.float()-expected_out.float()
                diagnostic['outputs'].append(dict(name=name,finite=bool(torch.isfinite(actual).all()),
                    max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/expected_out.float().norm().clamp_min(1e-30))))
            packed_projection = torch.nn.functional.linear(
                quantize_activation(source['residual'][:,0,:1280].contiguous().to('hpu')),weights[0][4])
            unpacked_projection = unpack_heads(packed_projection.reshape(6,2,512,8)).cpu()
            diagnostic['query_projection_exact'] = torch.equal(unpacked_projection,query.cpu())
            report.setdefault('publish_oracle',[]).append(diagnostic)
            torch.save(dict(reference=expected_out,eager=eager,compiled=replayed),
                       root/f'publish-oracle-{len(cases)-1}.pt')
            save()
            continue
        if args.matrix_oracle:
            reference = torch.ops.custom_op.custom_deepseek_v41_main_split_publish_mla_gaudi2(
                torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(query,pos,phase), case[3][0],case[4],
                case[5],pos,case[6],weights[0][8],scale,case[7],1)
            keys, values = reference[1],reference[3]
            q = pack_heads(expected.to('hpu')).contiguous()
            k = keys.reshape(6,1,640,512)
            fn = torch.compile(lambda a,b: torch.ops.custom_op.custom_deepseek_v41_interleaved_matrix_gaudi2(
                a,b,True,True),backend='hpu_backend',fullgraph=True,dynamic=False)
            native = fn(q,k).cpu()
            cpu = torch.matmul(q.cpu().float().transpose(-1,-2),k.cpu().float().transpose(-1,-2))
            probability = torch.softmax(cpu,dim=-1).contiguous()
            pv_native = fn(values.reshape(6,1,640,512), probability.to('hpu')).cpu()
            pv_cpu = torch.matmul(values.cpu().reshape(6,1,640,512).transpose(-1,-2),
                                  probability.transpose(-1,-2))
            def query_pipeline(a,b,p,c):
                rotated = torch.ops.custom_op.custom_deepseek_v41_interleaved_query_rope_gaudi2(
                    a.reshape(6,2,4096),p,c).reshape(6,2,512,8)
                return torch.ops.custom_op.custom_deepseek_v41_interleaved_matrix_gaudi2(rotated,b,True,True)

            def value_pipeline(a,b,p,c):
                product = torch.ops.custom_op.custom_deepseek_v41_interleaved_matrix_gaudi2(a,b,True,True)
                return torch.ops.custom_op.custom_deepseek_v41_interleaved_pv_bf16_gaudi2(
                    product.reshape(6,2,4096),p,c)

            qp = torch.compile(query_pipeline,backend='hpu_backend',fullgraph=True,dynamic=False)(
                packed,k,pos,phase).cpu()
            vp = torch.compile(value_pipeline,backend='hpu_backend',fullgraph=True,dynamic=False)(
                values.reshape(6,1,640,512),probability.to('hpu'),pos,phase).cpu().transpose(0,1).contiguous()
            expected_v = torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(
                unpack_heads(pv_cpu.bfloat16()).to('hpu'),pos,phase).cpu()
            expected_v = pack_heads(expected_v).reshape(6,2,4096)
            diagnostic = dict(source=str(path),matrices=[])
            for name,a,b in [('qk',cpu,native),('pv',pv_cpu,pv_native),('query_rope_qk_pipeline',cpu,qp),
                             ('pv_rope_pipeline',expected_v.float(),vp.float())]:
                diagnostic['matrices'].append(dict(name=name,finite=bool(torch.isfinite(b).all()),
                    max_abs=float((a-b).abs().max()),relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-30))))
            report.setdefault('matrix_oracle',[]).append(diagnostic)
            torch.save(dict(qk_reference=cpu,qk_native=native,pv_reference=pv_cpu,pv_native=pv_native),
                       root/f'matrix-oracle-{len(cases)-1}.pt')
            save()
            continue
        results = [[t.cpu() for t in fn(*case)] for fn in compiled]
        if args.diagnostic_stages:
            rows = []
            for index,(a,b) in enumerate(zip(results[0][7:],results[1][7:],strict=True)):
                kind = index % 5
                if kind==2:
                    b=b.transpose(0,1).contiguous()
                if kind in (1,2):
                    b = unpack_heads(b.reshape(6,2,512,8))
                delta = b.float()-a.float()
                rows.append(dict(layer=20+index//5,stage=('q_norm','q_projection','mla_inverse_rope','wo_a','wo_a_codec')[kind],
                    finite=bool(torch.isfinite(b).all()),exact=torch.equal(a,b),max_abs=float(delta.abs().max()),
                    relative_l2=float(delta.norm()/a.float().norm().clamp_min(1e-30))))
            report.setdefault('diagnostic_stages',[]).append(dict(source=str(path),stages=rows))
            torch.save(dict(parent=results[0],candidate=results[1]),root/f'producer-consumer-{len(cases)-1}.pt')
            save()
            continue
        checks = []
        for a, b in zip(*results, strict=True):
            delta = b.float() - a.float()
            relative = float(delta.norm()/a.float().norm().clamp_min(1e-30))
            checks.append(dict(shape=list(a.shape), exact=torch.equal(a, b), relative_l2=relative,
                               max_abs=float(delta.abs().max()), finite=bool(torch.isfinite(b).all())))
        report['checks'].append(dict(source=str(path), outputs=checks))
        save()
        if any(not c['finite'] or c['relative_l2']>.01 for c in checks[:4]):
            raise AssertionError('Changed matrix layout exceeds component tolerance')
        if not all(c['exact'] for c in checks[4:]):
            raise AssertionError('Canonical cache planes or mask changed')
    if len(cases) != 3:
        raise ValueError('Expected three real request cases')
    report['status'] = ('stages_materialized_no_performance_vote' if args.diagnostic_stages
                        else 'publish_oracle_recorded_no_performance_vote' if args.publish_oracle
                        else 'matrix_oracle_recorded_no_performance_vote' if args.matrix_oracle
                        else 'correctness_passed_teacher_pending')
    save()
    if args.checks_only or args.matrix_oracle or args.publish_oracle or args.diagnostic_stages:
        torch.distributed.destroy_process_group()
        return
    plans = []
    # Four real layer weights (well above SRAM capacity), three changing
    # prefixes, identical native compute execution in both arms.
    for fn in compiled:
        plans.append(recorder.prepare(fn, [case[0] for case in cases], [case[1:] for case in cases]))
    report['recipes'] = [p.recipes for p in plans]
    for label in 'ABABAB':
        plan = plans[label == 'B']
        for _ in range(20):
            plan()
        torch.hpu.synchronize()
        load = device_load()
        events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)) for _ in range(200)]
        start_wall = time.perf_counter()
        for begin, end in events:
            begin.record()
            plan()
            end.record()
            end.synchronize()
        wall = time.perf_counter()-start_wall
        samples = [begin.elapsed_time(end)/len(cases) for begin, end in events]
        row = dict(arm=label, median_device_ms=statistics.median(samples), samples_ms=samples,
                   drained_host_ms=wall*1000/len(samples)/len(cases), competing_load=load)
        report['periods'].append(row)
        save()
        print(label, row['median_device_ms'], flush=True)
    savings = [report['periods'][2*i]['median_device_ms']-report['periods'][2*i+1]['median_device_ms']
               for i in range(3)]
    report.update(status='native_component_complete_no_ledger_credit', paired_saved_ms=savings,
                  direction_passed=all(value>0 for value in savings),
                  projected_full38_ms=statistics.median(savings)*38/4,
                  excluded_unchanged=('KV writes, selector and compressor; TP communication/next-layer residual '
                                      'remain for the mandatory real16 gate'))
    save()
    for plan in plans:
        plan.close()
    torch.distributed.destroy_process_group()
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','periods','primitives')}), flush=True)


if __name__ == '__main__':
    main()
