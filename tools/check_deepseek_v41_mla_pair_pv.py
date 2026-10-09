# SPDX-License-Identifier: Apache-2.0
"""Complete C6 MLA writer/gather/QK/softmax/PV through real BF16 wo_a.

The candidate omits widened FP32 KV and splits FP32 probabilities into BF16
high/low operands for two FP32-accumulating MME products. Internal numerical
errors are recorded; serving quality and real16 category gains remain gates.
"""
import argparse
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from deepseek_v41_micro_replay import RecipeRecorder
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, rotary_table
    from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    torch.ops.load_library(os.environ['DSV41_UNIQUE_OPERATOR_LIBRARY'])
    recorder = RecipeRecorder(root)
    configure_post_graph_directory(root / 'graphs' / 'rank0')
    shard = PreparedV41Shard(args.prepared, 0, 0)
    config = json.loads((args.prepared / 'config.json').read_text())['text_config']
    heads = config['num_attention_heads'] // 4
    groups = config['o_groups'] // 4

    def body(candidate):
        def execute(value, swa, main, selected, positions, pages, sink, scale, lengths, ratio, wo_a, phase):
            q = value[:, :heads].contiguous()
            new_main, new_swa = value[:, heads].contiguous(), value[:, heads + 1].contiguous()
            logical = torch.div(positions, ratio, rounding_mode='floor')
            page_rows = 128 // ratio
            physical = pages.index_select(0, (logical // page_rows).long()) * page_rows + logical % page_rows
            main.index_copy_(0, physical.long(), pack_fp4(new_main, 16))
            swa.index_copy_(0, (positions & 255).long(), pack_swa(new_swa))
            operation = (torch.ops.custom_op.custom_deepseek_v41_logical_mla_pair_pv_gaudi2 if candidate else
                         torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2)
            output = operation(q, swa, main, selected, positions, pages, sink, scale, lengths, ratio, True)
            rotated = torch.ops.custom_op.custom_deepseek_v41_prefill_rope_inverse_bf16_gaudi2(
                output, positions, phase)
            projected = torch.einsum('tgd,gdr->tgr', rotated.reshape(-1, groups, heads // groups * 512), wo_a)
            return projected.flatten(1), output
        return execute

    functions = [torch.compile(body(candidate), backend='hpu_backend', fullgraph=True, dynamic=False)
                 for candidate in (False, True)]
    report = dict(status='running', cases=[], mla_heads=heads, consumer='real prepared BF16 wo_a',
                  serving_qualified=False, credited_e2e_ms=0)
    try:
        with torch.inference_mode():
            torch.manual_seed(4166)
            for count in (2, 6):
                inputs, weights, errors = [], [], []
                for layer in range(8):
                    ratio = (1, 2)[layer % 2]
                    capacity = 32768 // ratio
                    main = pack_fp4(torch.randn(capacity, 512).bfloat16(), 16).to('hpu')
                    swa = pack_swa(torch.randn(256, 512).bfloat16()).to('hpu')
                    pages = torch.arange(256, dtype=torch.int32).roll(17).to('hpu')
                    positions = torch.arange(16384, 16384 + count, dtype=torch.int32).to('hpu')
                    selected = torch.randint(0, 16384 // ratio, (count, 512), dtype=torch.int32).to('hpu')
                    selected[:, :count].copy_(torch.div(positions, ratio, rounding_mode='floor')[None, :])
                    selected[:, -2:] = -1
                    sink = torch.randn(heads).to('hpu')
                    scale = torch.tensor([512**-.5], device='hpu')
                    lengths = torch.full((count,), 640, dtype=torch.int32, device='hpu')
                    value = torch.randn(count, heads + 2, 512).bfloat16().to('hpu')
                    if ratio == 2:
                        value[:, heads].copy_(value[::2, heads].repeat_interleave(2, 0))
                    dense = shard.dense(f'layers.{layer}.attn.wo_a.weight', 'hpu')
                    wo_a = dense.reshape(groups, 1024, -1).transpose(1, 2).contiguous()
                    scaling = config['rope_scaling']
                    theta = config['compress_rope_theta'] if ratio == 2 else config['rope_theta']
                    table = rotary_table(64, 32768, theta,
                                         scaling['original_max_position_embeddings'], scaling['factor'],
                                         scaling['beta_fast'], scaling['beta_slow'])
                    phase = torch.cat((table[..., 0], table[..., 1]), -1).contiguous().to('hpu')
                    current = (swa, main, selected, positions, pages, sink, scale, lengths, ratio, wo_a, phase)
                    inputs.append(value)
                    weights.append(current)
                    a, b = [tuple(t.cpu() for t in fn(value, *current)) for fn in functions]
                    errors.append(dict(layer=layer, ratio=ratio,
                                       attention_max_abs=float((a[1].float() - b[1].float()).abs().max()),
                                       consumer_max_abs=float((a[0].float() - b[0].float()).abs().max()),
                                       attention_changed=int((a[1] != b[1]).sum()),
                                       consumer_changed=int((a[0] != b[0]).sum())))
                    if not all(bool(torch.isfinite(t).all()) for t in b):
                        raise AssertionError('Pair PV is non-finite')
                    if errors[-1]['attention_max_abs'] > .004 or errors[-1]['consumer_max_abs'] > .02:
                        raise AssertionError(('Pair arithmetic exceeds the component error gate', errors[-1]))
                replays = [recorder.prepare(fn, inputs, weights) for fn in functions]
                for fn, replay in zip(functions, replays, strict=True):
                    expected = [tuple(t.cpu() for t in fn(x, *w)) for x, w in zip(inputs, weights, strict=True)]
                    replay()
                    torch.hpu.synchronize()
                    for index, values in enumerate(expected):
                        for output_index, reference in enumerate(values):
                            actual = replay.outputs[index * 2 + output_index].cpu()
                            if not torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)):
                                raise AssertionError('Native replay differs from its compiled arm')
                    for _ in range(8):
                        replay()
                    torch.hpu.synchronize()
                periods = []
                for arm in (0, 1, 0, 1, 0, 1):
                    start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.samples):
                        replays[arm]()
                    stop.record()
                    stop.synchronize()
                    periods.append(dict(arm=arm, ms_per_layer=start.elapsed_time(stop) / args.samples / 8))
                means = [statistics.fmean(p['ms_per_layer'] for p in periods if p['arm'] == a) for a in (0, 1)]
                report['cases'].append(dict(rows=count, errors=errors, periods=periods,
                                            means_ms=means, real_weight_layers=list(range(8))))
                for replay in replays:
                    replay.close()
            report['status'] = 'passed_component_gate'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        (root / 'mla-pair-pv.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
