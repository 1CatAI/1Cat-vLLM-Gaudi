# SPDX-License-Identifier: Apache-2.0
"""Real index projections -> candidate addresses -> stock gather/MME scoring.

Checkpoint embedding-derived fixtures, not captured layer-24 hidden states.
Native compute replay measures the complete local chain; communication is unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fused-coordinates', action='store_true')
    parser.add_argument('--whole-coordinates', action='store_true', help='Compare whole metadata materialization against sliced fused parent')
    parser.add_argument('--physical-only', action='store_true', help='Compile candidate graph only; no repeated timings')
    parser.add_argument('--compression-ratio', type=int, choices=(1, 2), default=2)
    parser.add_argument('--key-rows', type=int, default=16384)
    parser.add_argument('--candidate-blocks', type=int, default=256)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.candidate_blocks <= 2048 and args.key_rows >= 128):
        parser.error('Require 1..2048 candidate blocks and >=128 mirror rows')
    args.prepared, args.output = args.prepared.resolve(), args.output.resolve()
    if any('DUMP' in name for name in os.environ):
        raise RuntimeError('DUMP settings are forbidden')
    os.environ['GRAPH_VISUALIZATION'] = '1'
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from safetensors import safe_open
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_default_fastpaths
    prepare_default_fastpaths(args.prepared)
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm, fp4_roundtrip, quantize_activation, rotary_table, apply_rope
    from vllm_gaudi.ops.deepseek_v41_index_mirror import mirror_index_tile
    from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
    from tools.deepseek_v41_micro_replay import RecipeRecorder
    from tools.audit_deepseek_v41_physical_nodes import audit
    from tools.deepseek_v41_resident_ab import graph_compilation_count, compare_periods, device_load

    torch.set_num_threads(1)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    recorder = RecipeRecorder(args.output)
    graph_compilation_count()
    shards = [PreparedV41Shard(args.prepared, 0, rank) for rank in range(4)]
    wq = torch.cat([s.dense('layers.24.attn.indexer.wq_b.weight', 'cpu') for s in shards]).to('hpu')
    wp = torch.cat([s.tensor('layers.24.attn.indexer.weights_proj.weight', 'cpu') for s in shards]).to('hpu')
    qnorm = shards[0].tensor('layers.24.attn.q_norm.weight', 'hpu')
    wk = shards[0].tensor('layers.20.attn.indexer.wk.weight', 'cpu')
    knorm = shards[0].tensor('layers.20.attn.indexer.k_norm.weight', 'cpu')
    with safe_open(args.prepared / 'pp0-tp0.safetensors', framework='pt', device='cpu') as checkpoint:
        fixtures = [checkpoint.get_slice('embed.weight')[token:token+1].clone() for token in (17, 41, 128, 512, 1024)]
    key_rows = rms_norm(torch.nn.functional.linear(torch.cat(fixtures)[:, :512].contiguous(), wk), knorm, 1e-20)
    key_rows = fp4_roundtrip(key_rows, 32)
    keys = key_rows.repeat(((args.key_rows + 4) // 5, 1))[:args.key_rows].contiguous().to('hpu')
    config = json.loads((args.prepared / 'config.json').read_text())['text_config']
    scaling = config['rope_scaling']
    phase = rotary_table(64, 32768, config['compress_rope_theta'], scaling['original_max_position_embeddings'],
                         scaling['factor'], scaling['beta_fast'], scaling['beta_slow']).to('hpu')
    x = fixtures[0].to('hpu')
    position = torch.tensor([16384], dtype=torch.int32, device='hpu')
    pool = torch.arange(args.candidate_blocks, dtype=torch.int32, device='hpu')
    offsets = torch.arange(8, dtype=torch.int32, device='hpu')

    def chain(value, pos, blocks, offsets, wq, wp, norm, phase, keys, *, fused=False, whole=False):
        qr = rms_norm(value[:, :1280].contiguous(), norm, 1e-20)
        query = torch.nn.functional.linear(quantize_activation(qr), wq).reshape(1, 32, 128)
        query = fp4_roundtrip(apply_rope(query, pos, phase), 32)
        weights = torch.nn.functional.linear(value, wp) * (128**-.5 * 32**-.5)
        safe = None
        if fused:
            logical, safe = torch.ops.custom_op.custom_deepseek_v41_candidate_coordinates_gaudi2(
                blocks.reshape(1, -1), keys.shape[0] - 1, whole)
        else:
            logical = torch.where(blocks[:, None] >= 0, blocks[:, None] * 8 + offsets[None, :], -1).reshape(-1)
        logical = logical.reshape(1, -1)
        pieces = [mirror_index_tile(query, weights, keys, pos, logical[..., start:start+2048], args.compression_ratio, 8,
                    safe_rows=None if safe is None else safe[..., start:start+2048])
                  for start in range(0, logical.shape[-1], 2048)]
        return (torch.cat(pieces, -1).contiguous(),)

    parameters = (position, pool, offsets, wq, wp, qnorm, phase, keys)
    plans, outputs, counts = [], [], []
    repeats = 32
    previous_graphs = set()
    arms = (('B', True),) if args.physical_only else (('A', False), ('B', True))
    for label, clamps in arms:
        directory = args.output / label
        directory.mkdir()
        (directory / ".graph_dumps").mkdir()
        os.chdir(directory)
        def entry(*inputs, fused=args.whole_coordinates or (clamps and args.fused_coordinates),
                  whole=clamps and args.whole_coordinates):
            return chain(*inputs, fused=fused, whole=whole)
        compiled = torch.compile(entry, backend=make_backend(static_int32=True, static_factories=True,
                                                             static_clamps=clamps and not (args.fused_coordinates or args.whole_coordinates)),
                                 fullgraph=True, dynamic=False)
        compiled(x, *parameters)[0].cpu()
        if not args.physical_only:
            plan = recorder.prepare(compiled, [x] * repeats, [parameters] * repeats)
            plans.append(plan)
        current_graphs = set(args.output.rglob('*PostGraph-symbol.pbtxt'))
        graphs = [audit(p) for p in sorted(current_graphs - previous_graphs)]
        previous_graphs = current_graphs
        counts.append(dict(arm=label, graphs=graphs))
    os.chdir(args.output)
    if args.physical_only:
        (args.output / 'physical_nodes.json').write_text(json.dumps(counts, indent=2)+'\n')
        return
    checks = []
    for index, fixture in enumerate(fixtures):
        x.copy_(fixture.to('hpu'))
        # Include sentinel rows and both valid/overflow addresses.
        rows = (torch.arange(args.candidate_blocks, dtype=torch.int32) * (index+1)) % max(1, args.key_rows // 8)
        rows[:4] = torch.tensor([-1, -1000, args.key_rows // 8, args.key_rows // 4], dtype=torch.int32)
        pool.copy_(rows.to('hpu'))
        position.fill_((127, 255, 8191, 16384, 32767)[index])
        torch.hpu.synchronize()
        values = []
        for plan in plans:
            plan()
            torch.hpu.synchronize()
            values.append(plan.outputs[0].cpu())
        exact = torch.equal(values[0].view(torch.uint8), values[1].view(torch.uint8))
        checks.append(dict(fixture=index, exact=exact))
        if not exact:
            raise RuntimeError(f'Address candidate changed index scores at fixture {index}')
    (args.output / 'physical_nodes.json').write_text(json.dumps(counts, indent=2)+'\n')
    before = graph_compilation_count()
    periods = []
    for label in 'ABABAB':
        plan = plans[label == 'B']
        for _ in range(8):
            plan()
        torch.hpu.synchronize()
        events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)) for _ in range(200)]
        load = device_load()
        for begin, end in events:
            begin.record()
            plan()
            end.record()
        torch.hpu.synchronize()
        values = [begin.elapsed_time(end) / repeats for begin, end in events]
        periods.append(dict(arm=label, token_intervals_ms=values, competing_load=load))
        print(label, statistics.median(values), flush=True)
    if graph_compilation_count() != before:
        raise RuntimeError('Compilation occurred in timed replay')
    result = dict(checks=checks, periods=periods, comparison=compare_periods(periods, device_events=True),
                  physical_graphs_present=all(row['graphs'] for row in counts),
                  fixture_scope=__doc__, geometry=dict(compression_ratio=args.compression_ratio, key_rows=args.key_rows, candidate_blocks=args.candidate_blocks, score_tiles=(args.candidate_blocks*8+2047)//2048), formal_gain_credit=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    for plan in plans:
        plan.close()


if __name__ == '__main__':
    import torch
    with torch.inference_mode():
        main()
