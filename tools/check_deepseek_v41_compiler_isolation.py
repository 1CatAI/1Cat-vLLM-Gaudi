# SPDX-License-Identifier: Apache-2.0
"""Prove compiler-setting arms own different physical expert recipes before A/B."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import habana_frameworks.torch.core  # noqa: F401
import vllm_gaudi.distributed.tp2_fused_ar_norm  # noqa: F401

from deepseek_v41_resident_ab import compiler_settings, compare_periods, device_load, summarize
from deepseek_v41_micro_replay import RecipeRecorder
from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
from vllm_gaudi.ops.deepseek_v41_expert_n256 import _fused_decode, prepare_expert


def weights(n, k, generator):
    q = generator.integers(-32768, 32768, (n // 128, k * 32), dtype=np.int16)
    scales = np.full((n // 128, k * 4), 127 << 7, dtype=np.uint16)
    q, scales, channel, _ = prepare_expert(q, scales, compact_scales=True)
    def move(array, bf16=False):
        tensor = torch.from_numpy(array.view(np.int16)).repeat(6, 1, 1)
        return (tensor.view(torch.bfloat16) if bf16 else tensor).to('hpu')
    return move(q), move(scales), move(channel, True)


def physical_graphs(directory):
    graphs = {}
    for path in directory.rglob('*.post.json'):
        graph = json.loads(path.read_text())['graphs'][0]
        nodes = [n for n in graph['nodes'] if not n.get('is_logical', False)
                 and n.get('engine') in ('TPC', 'MME')]
        if not any('expert_n256' in n['guid'] for n in nodes):
            continue
        graphs[graph['name']] = dict(name=graph['name'], recipe_debug_id=graph['recipe_debug_id'],
                                     path=str(path), kernels=[n['guid'] for n in nodes],
                                     bundle_nodes=sum('/bundle_' in n['name'] for n in nodes))
    return list(graphs.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-bytes', type=int, default=0)
    parser.add_argument('--native-replay', action='store_true')
    options = parser.parse_args()
    if os.environ.get('PT_HPU_ENABLE_JIT_GRAPH_NAME_HASH') != '1':
        raise RuntimeError('Graph-name cache isolation must be enabled before Bridge initialization')
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    # Recipe observation is installed at compilation, before either arm exists.
    recorder = RecipeRecorder(root) if options.native_replay else None
    rng = np.random.default_rng(30)
    q13, s13, c13 = weights(1280, 5120, rng)
    q2, s2, c2 = weights(5120, 640, rng)
    lookup = mxfp4_bf16_lut('hpu')
    cpu_x = torch.from_numpy(rng.standard_normal((1, 5120)).astype(np.float32)).to(torch.bfloat16)
    x = cpu_x.to('hpu')
    ids = torch.arange(6, dtype=torch.int32).reshape(1, 6).to('hpu')
    routing = torch.full((1, 6), 1 / 6, dtype=torch.float32, device='hpu')
    arguments = (x, ids, routing, q13, q2, s13, s2, lookup, c13, c2, True, True)
    arms, outputs = {}, {}
    for name, settings in (('baseline', {}), ('unsliced', {
        'SRAM_SLICER_MAX_CAPACITY_BYTES': str(options.capacity_bytes)})):
        folder = root / 'compiler' / name
        folder.mkdir(parents=True, exist_ok=True)
        fn = torch.compile(_fused_decode, backend=make_backend(), fullgraph=True, dynamic=False)
        with compiler_settings(dict(settings, DUMP_POST_GRAPHS=str(folder))):
            outputs[name] = fn(*arguments).cpu()
        arms[name] = fn
    error = (outputs['baseline'].float() - outputs['unsliced'].float()).abs()
    exact = torch.equal(outputs['baseline'], outputs['unsliced'])
    changed = []
    for step in range(3):
        # Include a different input and expert ordering after capture; this is
        # a real downstream MoE consumer, not a decoder-only byte comparison.
        x.copy_((cpu_x * (1 + step / 16)).to('hpu'))
        ids.copy_(torch.arange(6, dtype=torch.int32).reshape(1, 6).roll(step, 1).to('hpu'))
        a, b = [arms[name](*arguments).cpu() for name in ('baseline', 'unsliced')]
        changed.append(dict(step=step, exact=torch.equal(a, b),
                            max_absolute_error=float((a.float()-b.float()).abs().max())))
    graphs = {name: physical_graphs(root / 'compiler' / name) for name in arms}
    identities = [{g['name'] for g in graphs[name]} for name in arms]
    isolated = all(identities) and not identities[0].intersection(identities[1])
    report = dict(status='passed' if isolated and exact and all(r['exact'] for r in changed) else 'failed',
                  physical_recipe_isolated=isolated, exact=exact, max_absolute_error=float(error.max()),
                  changed_input_consumers=changed, graphs=graphs, performance_gain_credit=False,
                  capacity_bytes=options.capacity_bytes,
                  note='Small compile/dataflow contract only; no real16 or formal timing')
    (root / 'COMPILER_ISOLATION_CONTRACT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if report['status'] != 'passed':
        raise RuntimeError('Compiler isolation or consumer numerical contract did not pass')
    plans = {}
    if options.native_replay:
        def tuple_output(function):
            def forward(*args):
                return (function(*args),)
            return forward
        plans = {name: recorder.prepare(tuple_output(fn), [x] * 32, [arguments[1:]] * 32)
                 for name, fn in arms.items()}
    periods = []
    for label in ('A', 'B', 'A', 'B', 'A', 'B'):
        fn = arms['baseline' if label == 'A' else 'unsliced']
        for _ in range(32):
            plans['baseline' if label == 'A' else 'unsliced']() if plans else fn(*arguments)
        torch.hpu.synchronize()
        events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)) for _ in range(200)]
        load = device_load()
        for start, end in events:
            start.record()
            plans['baseline' if label == 'A' else 'unsliced']() if plans else fn(*arguments)
            end.record()
            if plans:
                end.synchronize()
        torch.hpu.synchronize()
        values = [start.elapsed_time(end) / (32 if plans else 1) for start, end in events]
        periods.append(dict(arm=label, token_intervals_ms=values, competing_load=load, summary=summarize(values)))
        print(label, periods[-1]['summary'], flush=True)
    timing = dict(periods=periods, comparison=compare_periods(periods, device_events=True),
                  scope='Single expert producer/consumer chain through native replay' if plans else
                        'Single expert producer/consumer chain, device events include exposed submission gaps',
                  capacity_bytes=options.capacity_bytes, replay_repeats=32 if plans else 1,
                  no_real16_gain_credit=True)
    (root / 'SMALL_CHAIN_AB.json').write_text(json.dumps(timing, indent=2)+'\n')
    for plan in plans.values():
        plan.close()
    print(json.dumps(timing['comparison']), flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
