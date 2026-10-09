# SPDX-License-Identifier: Apache-2.0
"""Gate common C6 FP8 projections through cache writes, MLA and the next norm.

No new projection implementation is used: both arms call the model's linear
helper and use the native cache/attention consumers. This is a component gate;
its timings do not establish a real16 or serving gain.
"""
import argparse
import json
import os
from pathlib import Path
import statistics


class Projection:
    def __init__(self, weight, scale, candidate, direct):
        self.weight, self.channel_scale = weight, scale
        self.scale = True
        self.dense_fp8 = candidate
        self.dense_fp8_direct_input = candidate and direct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('sidecar', type=Path)
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from deepseek_v41_micro_replay import RecipeRecorder
    from vllm_gaudi.models.deepseek_v41_program import linear
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, rms_norm, rotary_table
    from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    recorder = RecipeRecorder(root)
    configure_post_graph_directory(root / 'graphs/rank0')
    shard = PreparedV41Shard(args.prepared, 0, 0)
    sidecar = DenseFP8Sidecar(args.sidecar, shard)
    config = json.loads((args.prepared / 'config.json').read_text())['text_config']
    heads, groups, eps = config['num_attention_heads'] // 4, config['o_groups'] // 4, config['rms_norm_eps']

    def body(candidate):
        def execute(x, wqkv, cqkv, wqb, cqb, wob, cob, qnorm, kvnorm, woa,
                    swa, main, selected, positions, pages, sink, attnscale, lengths, phase, nextnorm):
            qkv = linear(x, Projection(wqkv, cqkv, candidate, True))
            qa, kv = qkv[:, :1280].contiguous(), qkv[:, 1280:].contiguous()
            q = linear(rms_norm(qa, qnorm, eps), Projection(wqb, cqb, candidate, False))
            q = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(
                q.reshape(-1, heads, 512).contiguous(), positions, phase)
            kv = torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_bf16_gaudi2(
                kv.contiguous(), kvnorm, positions, phase, eps)
            physical = pages.index_select(0, (positions // 128).long()) * 128 + positions % 128
            main.index_copy_(0, physical.long(), pack_fp4(kv, 16))
            swa.index_copy_(0, (positions & 255).long(), pack_swa(kv))
            attended = torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(
                q, swa, main, selected, positions, pages, sink, attnscale, lengths, 1, True)
            rotated = torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(
                attended, positions, phase)
            intermediate = torch.einsum('tgd,gdr->tgr',
                                        rotated.reshape(-1, groups, heads // groups * 512), woa).flatten(1)
            partial = linear(intermediate, Projection(wob, cob, candidate, False))
            # First actual consumer after projection, preserving BF16 addition.
            return rms_norm((partial + x).bfloat16(), nextnorm, eps), partial
        return execute

    functions = [torch.compile(body(candidate), backend='hpu_backend', fullgraph=True, dynamic=False)
                 for candidate in (False, True)]
    report = dict(status='running', heads=heads, rows=[], real16_qualified=False, credited_e2e_ms=0,
                  limitations='Local partial output; TP reduction is covered only by subsequent real16 gate.')
    try:
        with torch.inference_mode():
            torch.manual_seed(4660)
            # Immutable matrices exceed the chip SRAM working set in each arm.
            matrices = []
            for layer in range(20, 28):
                prefix = f'layers.{layer}.attn.'
                original, proposed = [], []
                for name in ('wq_a', 'wkv', 'wq_b', 'wo_b'):
                    original.extend((shard.dense(prefix + name + '.weight', 'hpu'),
                                     torch.ones(1, device='hpu')))
                    proposed.extend((sidecar.tensor(prefix + name + '.weight', 'hpu'),
                                     sidecar.tensor(prefix + name + '.channel_scale', 'hpu')))
                original = [torch.cat((original[0].cpu(), original[2].cpu())).to('hpu'), original[1], *original[4:]]
                proposed = [torch.cat((proposed[0].cpu(), proposed[2].cpu())).to('hpu'),
                            torch.cat((proposed[1].cpu(), proposed[3].cpu()), dim=1).to('hpu'), *proposed[4:]]
                norm = [shard.tensor(prefix + name + '.weight', 'hpu') for name in ('q_norm', 'kv_norm')]
                woa = shard.dense(prefix + 'wo_a.weight', 'hpu').reshape(groups, 1024, -1).transpose(1, 2).contiguous()
                sink = shard.tensor(prefix + 'attn_sink', 'hpu')
                nextnorm = shard.tensor(f'layers.{layer}.ffn_norm.weight', 'hpu')
                matrices.append((original, proposed, norm, woa, sink, nextnorm))
            scaling = config['rope_scaling']
            table = rotary_table(64, 32768, config['rope_theta'],
                                 scaling['original_max_position_embeddings'], scaling['factor'],
                                 scaling['beta_fast'], scaling['beta_slow'])
            phase = torch.cat((table[..., 0], table[..., 1]), -1).contiguous().to('hpu')
            for count in (2, 6):
                inputs, operands, errors = [], [[], []], []
                for layer, (original, proposed, norm, woa, sink, nextnorm) in enumerate(matrices):
                    x = torch.randn(count, 5120).bfloat16().to('hpu')
                    inputs.append(x)
                    positions = torch.arange(16384, 16384 + count, dtype=torch.int32).to('hpu')
                    pages = torch.arange(256, dtype=torch.int32).roll(17).to('hpu')
                    selected = torch.randint(0, 16384, (count, 512), dtype=torch.int32).to('hpu')
                    selected[:, :count].copy_(positions[None, :])
                    selected[:, -2:] = -1
                    main = pack_fp4(torch.randn(32768, 512).bfloat16(), 16).to('hpu')
                    swa = pack_swa(torch.randn(256, 512).bfloat16()).to('hpu')
                    common = (*norm, woa, swa, main, selected, positions, pages, sink,
                              torch.tensor([512**-.5], device='hpu'),
                              torch.full((count,), 640, dtype=torch.int32, device='hpu'), phase, nextnorm)
                    for arm, weights in enumerate((original, proposed)):
                        operands[arm].append((*weights, *common))
                    a, b = [tuple(t.cpu() for t in fn(x, *operands[arm][-1]))
                            for arm, fn in enumerate(functions)]
                    if not all(bool(torch.isfinite(t).all()) for t in b):
                        raise AssertionError('Non-finite dense FP8 attention consumer')
                    errors.append(dict(layer=layer + 20,
                                       max_abs=[float((aa.float() - bb.float()).abs().max())
                                                for aa, bb in zip(a, b, strict=True)],
                                       rms=[float((aa.float() - bb.float()).square().mean().sqrt())
                                            for aa, bb in zip(a, b, strict=True)]))
                replays = [recorder.prepare(fn, inputs, operands[arm]) for arm, fn in enumerate(functions)]
                try:
                    for arm, replay in enumerate(replays):
                        for x in inputs:
                            x.add_(.03125)
                        expected = [tuple(t.cpu() for t in functions[arm](x, *w))
                                    for x, w in zip(inputs, operands[arm], strict=True)]
                        replay()
                        torch.hpu.synchronize()
                        for index, values in enumerate(expected):
                            for output_index, reference in enumerate(values):
                                if not torch.equal(reference, replay.outputs[index * 2 + output_index].cpu()):
                                    raise AssertionError('Native dense replay differs from its compiled arm')
                        for _ in range(8):
                            replay()
                        torch.hpu.synchronize()
                    timings = []
                    for arm in (0, 1, 0, 1, 0, 1):
                        start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.samples):
                            replays[arm]()
                        end.record()
                        end.synchronize()
                        timings.append(dict(arm=arm, ms_per_layer=start.elapsed_time(end) / args.samples / 8))
                    report['rows'].append(dict(count=count, errors=errors, timings=timings,
                                               means_ms=[statistics.fmean(t['ms_per_layer'] for t in timings
                                                                         if t['arm'] == arm) for arm in (0, 1)]))
                finally:
                    torch.hpu.synchronize()
                    for replay in replays:
                        replay.close()
            report['status'] = 'passed_component_contract'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        (root / 'c6-dense-restore.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
