# SPDX-License-Identifier: Apache-2.0
"""Actual-query FP16-PV capability through production inverse RoPE and BF16 wo_a.

This bounded native chain precedes the four-card real16 gate. Its local timings
are not whole-round gain, and it does not qualify changed numeric acceptance.
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
    parser.add_argument('--iterations', type=int, default=32)
    parser.add_argument('--direct', action='store_true')
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment, prepare_native_libraries

    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    prepare_native_libraries()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_math import quantize_activation, rotary_table
    from deepseek_v41_micro_replay import RecipeRecorder

    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    operation = ('custom_deepseek_v41_main_fp16_direct_publish_mla_gaudi2' if args.direct else
                 'custom_deepseek_v41_main_fp16_publish_mla_gaudi2')
    load_native_operators(required=(operation,))
    recorder = RecipeRecorder(root)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    config = json.loads((args.prepared / 'config.json').read_text())['text_config']
    groups = config['o_groups'] // 4
    report = dict(status='running', checks=[], timings=[], serving_selected=False,
                  performance_credit_ms=0, communication_included=False,
                  boundary='Packed state producer -> QK/softmax/PV -> inverse RoPE -> production wo_a')
    replays = []
    try:
        with torch.inference_mode():
            fixtures = [torch.load(args.fixtures / f'attention-operands-rank0-case{c}.pt',
                                  map_location='cpu', weights_only=True)[0] for c in range(3)]
            inputs, weights = [], []
            for layer in range(4):
                item = fixtures[layer % 3]
                count, heads, _ = item['query'].shape
                ratio = item['ratio']
                # Reconstruct a request-local packed page bank from preserved
                # exact selected rows. No union or metadata preparation is timed.
                selected = item['selected']
                valid = selected >= 0
                capacity = max(32768 // ratio, int(selected.max()) + 1)
                main = torch.zeros(capacity, 288, dtype=torch.uint8)
                main.index_copy_(0, selected[valid].long(), item['packed_main'][valid])
                if not torch.equal(main[selected.clamp_min(0).long()][valid], item['packed_main'][valid]):
                    raise AssertionError('Saved duplicate KV rows disagree')
                pages = torch.arange((capacity + 127 // ratio) // (128 // ratio), dtype=torch.int32)
                # The producer writes exactly the preserved SWA rows in stream
                # order before the gather, retaining a real state dependency.
                slots = (item['positions'] & 255).long()
                packed_rows = item['swa'].index_select(0, slots).to('hpu')
                swa = item['swa'].to('hpu')
                positions = item['positions'].to('hpu', dtype=torch.int32)
                phase_config = config['rope_scaling']
                theta = config['compress_rope_theta'] if ratio == 2 else config['rope_theta']
                table = rotary_table(64, 32768, theta, phase_config['original_max_position_embeddings'],
                                     phase_config['factor'], phase_config['beta_fast'], phase_config['beta_slow'])
                phase = torch.cat((table[..., 0], table[..., 1]), -1).contiguous().to('hpu')
                dense = shard.dense(f'layers.{layer}.attn.wo_a.weight', 'hpu')
                woa = dense.reshape(groups, 1024, 4096).transpose(1, 2).contiguous()
                inputs.append(item['query'].to('hpu'))
                weights.append((swa, main.to('hpu'), selected.to('hpu'), positions, pages.to('hpu'),
                                item['sink'].to('hpu'), item['scale'].to('hpu'),
                                torch.full((count,), 640, dtype=torch.int32, device='hpu'), ratio,
                                packed_rows, phase, woa))

            def body(candidate):
                def execute(q, swa, main, selection, positions, pages, sink, scale, lengths,
                            ratio, packed_rows, phase, woa):
                    swa.index_copy_(0, (positions & 255).long(), packed_rows)
                    op = ((torch.ops.custom_op.custom_deepseek_v41_main_fp16_direct_publish_mla_gaudi2
                           if args.direct else torch.ops.custom_op.custom_deepseek_v41_main_fp16_publish_mla_gaudi2)
                          if candidate else
                          torch.ops.custom_op.custom_deepseek_v41_main_split_publish_mla_gaudi2)
                    attention, bank, mask, widened = op(q, swa, main, selection, positions,
                                                       pages, sink, scale, lengths, ratio)
                    rotated = torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(
                        attention, positions, phase)
                    operand = quantize_activation(rotated.reshape(q.shape[0], groups, 4096))
                    projected = torch.einsum('tgk,gkn->tgn', operand, woa).bfloat16()
                    return projected, attention, bank, mask, widened
                return execute

            functions = [torch.compile(body(c), backend='hpu_backend', fullgraph=True, dynamic=False)
                         for c in (False, True)]
            # Three changing prefixes, four immutable real weight matrices.
            for index, (x, w) in enumerate(zip(inputs, weights, strict=True)):
                values = [tuple(y.cpu() for y in fn(x, *w)) for fn in functions]
                a, b = values
                delta = b[1].float() - a[1].float()
                row = dict(layer=index, max_abs=float(delta.abs().max()),
                           relative_l2=float(delta.norm() / a[1].float().norm().clamp_min(1e-30)),
                           consumer_max_abs=float((b[0].float() - a[0].float()).abs().max()),
                           states_exact=all(torch.equal(left.to(right.dtype).view(torch.uint8), right.view(torch.uint8))
                                            for left, right in zip(a[2:], b[2:], strict=True)),
                           finite=all(bool(torch.isfinite(y).all()) for y in b))
                report['checks'].append(row)
                if not row['finite'] or not row['states_exact'] or row['relative_l2'] > .002:
                    raise AssertionError(('FP16 consumer exceeds numeric gate', row))
            for fn in functions:
                replay = recorder.prepare(fn, inputs, weights)
                replays.append(replay)
                expected = [tuple(y.cpu() for y in fn(x, *w)) for x, w in zip(inputs, weights, strict=True)]
                replay()
                torch.hpu.synchronize()
                for i, row in enumerate(expected):
                    for j, saved in enumerate(row):
                        current = replay.outputs[i * 5 + j].cpu()
                        if not torch.equal(saved.view(torch.uint8), current.view(torch.uint8)):
                            raise AssertionError('Native and compiled outputs differ')
                for _ in range(8):
                    replay()
                torch.hpu.synchronize()
            for iteration in range(3):
                pair = {}
                for arm in (0, 1):
                    start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    wall = time.perf_counter()
                    start.record()
                    for _ in range(args.iterations):
                        replays[arm]()
                    stop.record()
                    stop.synchronize()
                    pair[str(arm)] = dict(device_ms=start.elapsed_time(stop) / args.iterations,
                                         host_wall_ms=(time.perf_counter() - wall) * 1000 / args.iterations)
                report['timings'].append(dict(iteration=iteration, arms=pair,
                                             saved_ms=pair['0']['device_ms'] - pair['1']['device_ms']))
            saved = [r['saved_ms'] for r in report['timings']]
            report.update(status='capability_passed', local_native_gain=all(v > 0 for v in saved),
                          median_four_layer_saved_ms=statistics.median(saved),
                          teacher_acceptance_qualified=False, real16_qualified=False)
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        for replay in replays:
            replay.close()
        (root / 'mla-fp16-pv.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
