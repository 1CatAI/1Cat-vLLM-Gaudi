# SPDX-License-Identifier: Apache-2.0
"""Native C6 Q norm/projection/RoPE through the real attention consumer."""
import argparse
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--tiled', action='store_true')
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from deepseek_v41_micro_replay import RecipeRecorder
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, quantize_activation, rms_norm, rotary_table
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory

    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    if args.tiled:
        torch.ops.load_library(os.environ['DSV41_UNIQUE_OPERATOR_LIBRARY'])
    recorder = RecipeRecorder(root)
    configure_post_graph_directory(root / 'graphs/rank0')
    shard = PreparedV41Shard(args.prepared, 0, 0)
    sidecar = DenseFP8Sidecar(args.prepared / 'sidecars/attention_dense_fp8', shard)
    report = dict(status='running', rows=6, heads=16, tiled=args.tiled, real16_qualified=False, ledger_credit_ms=0)
    with torch.inference_mode():
        torch.manual_seed(6420)
        inputs, weights = [], []
        positions = torch.arange(16384, 16390, device='hpu', dtype=torch.int32)
        rotary = rotary_table(64, 32768, 160000)
        table = torch.cat((rotary[..., 0], rotary[..., 1]), -1).contiguous().to('hpu')
        pages = torch.arange(256, device='hpu', dtype=torch.int32)
        ids = torch.arange(512, device='hpu', dtype=torch.int32).repeat(6, 1)
        lengths = torch.full((6,), 640, device='hpu', dtype=torch.int32)
        scale = torch.tensor([512**-.5], device='hpu')
        for layer in range(20, 24):
            prefix = f'layers.{layer}.attn.'
            x = torch.randn(6, 1280).bfloat16().to('hpu')
            inputs.append(x)
            norm = shard.tensor(prefix + 'q_norm.weight', 'hpu')
            w = sidecar.tensor(prefix + 'wq_b.weight', 'hpu')
            channel = sidecar.tensor(prefix + 'wq_b.channel_scale', 'hpu')
            sink = shard.tensor(prefix + 'attn_sink', 'hpu')
            swa = pack_swa(torch.randn(256, 512).bfloat16()).to('hpu')
            main = pack_fp4(torch.randn(32768, 512).bfloat16(), 16).to('hpu')
            weights.append((norm, w, channel, positions, table, swa, main, ids, pages, sink, scale, lengths))

        def consumer(q, swa, main, ids, positions, pages, sink, scale, lengths):
            return torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(
                q.reshape(6, 16, 512), swa, main, ids, positions, pages, sink, scale, lengths, 1, True)

        def baseline(x, norm, w, channel, positions, table, swa, main, ids, pages, sink, scale, lengths):
            if args.tiled:
                q = torch.ops.custom_op.custom_deepseek_v41_q_norm_projection_rope_gaudi2(
                    x, norm, w, channel, positions, table, 1e-6)
                return consumer(q, swa, main, ids, positions, pages, sink, scale, lengths)
            value = quantize_activation(rms_norm(x, norm, 1e-6))
            q = torch.ops.custom_op.custom_deepseek_v41_q_projection_rope_gaudi2(
                value.contiguous(), w, channel, positions, table)
            return consumer(q, swa, main, ids, positions, pages, sink, scale, lengths)

        def candidate(x, norm, w, channel, positions, table, swa, main, ids, pages, sink, scale, lengths):
            op = (torch.ops.custom_op.custom_deepseek_v41_q_norm_projection_rope_tiled_gaudi2
                  if args.tiled else torch.ops.custom_op.custom_deepseek_v41_q_norm_projection_rope_gaudi2)
            q = op(x, norm, w, channel, positions, table, 1e-6)
            return consumer(q, swa, main, ids, positions, pages, sink, scale, lengths)

        functions = [torch.compile(f, backend='hpu_backend', fullgraph=True, dynamic=False)
                     for f in (baseline, candidate)]
        errors = []
        for _ in range(2):
            for x, operands in zip(inputs, weights, strict=True):
                x.add_(.0625)
                a, b = [f(x, *operands).cpu() for f in functions]
                if not bool(torch.isfinite(b).all()):
                    raise AssertionError('Non-finite fused query consumer')
                errors.append(dict(max_abs=float((a.float() - b.float()).abs().max()), changed=int((a != b).sum())))
        replays = [recorder.prepare(f, inputs, weights) for f in functions]
        try:
            for f, replay in zip(functions, replays, strict=True):
                expected = [f(x, *w).cpu() for x, w in zip(inputs, weights, strict=True)]
                replay()
                torch.hpu.synchronize()
                if not all(torch.equal(a, b.cpu()) for a, b in zip(expected, replay.outputs, strict=True)):
                    raise AssertionError('Native query replay changed its compiled consumer')
                for _ in range(32):
                    replay()
                torch.hpu.synchronize()
            timings = []
            for arm in (0, 1, 0, 1, 0, 1):
                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                begin.record()
                for _ in range(args.steps):
                    replays[arm]()
                end.record()
                end.synchronize()
                timings.append(dict(arm=arm, device_ms=begin.elapsed_time(end) / args.steps))
            report.update(status='passed_component', errors=errors, timings=timings,
                          means_ms=[statistics.fmean(x['device_ms'] for x in timings if x['arm'] == a)
                                    for a in (0, 1)])
        finally:
            torch.hpu.synchronize()
            for replay in replays:
                replay.close()
    (root / 'c6-query-fusion.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
