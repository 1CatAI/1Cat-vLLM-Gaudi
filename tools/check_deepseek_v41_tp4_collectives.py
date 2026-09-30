# SPDX-License-Identifier: Apache-2.0
"""Measure TP4 collective producer/consumer chains to isolate fixed host delays."""
import argparse
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--defer-profile-drain", action="store_true")
args = parser.parse_args()
rank = int(os.environ['LOCAL_RANK'])
os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
import torch
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives

torch.hpu.set_device(rank)
bind_worker_cpu(rank)
init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://', local_rank=rank, backend='hccl')
config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
root = Path(os.environ['DSV41_RUN_EVIDENCE'])
report = dict(rank=rank, cases=[])
with set_current_vllm_config(config), torch.inference_mode():
    initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
    reduce, gather = stage_collectives(rank, False)
    programs = []
    for kind, shape in [('reduce', (1, 5120)), ('gather', (1, 6, 256)),
                        ('gather', (1, 8, 128)), ('gather', (1, 8))]:
        def body(value, kind=kind):
            produced = value + .25
            collected = reduce(produced) if kind == 'reduce' else gather(produced, 1)
            return collected * 2
        program = torch.compile(body, backend='hpu_backend', fullgraph=True, dynamic=False)
        value = torch.full(shape, float(rank), dtype=torch.bfloat16, device='hpu')
        expected = (sum(range(4)) + 1) * 2 if kind == 'reduce' else torch.cat(
            [torch.full(shape, (peer + .25) * 2, dtype=torch.bfloat16) for peer in range(4)], 1)
        result = program(value).cpu()
        assert torch.equal(result, torch.full_like(result, expected) if isinstance(expected, int) else expected)
        timings = []
        for step in range(12):
            value.add_(.0078125)
            torch.hpu.synchronize()
            started = time.perf_counter()
            result = program(value).cpu()
            timings.append((time.perf_counter() - started) * 1000)
        report['cases'].append(dict(kind=kind, shape=shape, host_sync_ms=timings))
        programs.append((kind, shape, program, value))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.HPU], record_shapes=True) as profiler:
        for kind, shape, program, value in programs:
            for step in range(3):
                with torch.profiler.record_function(f'TP4::{kind}::{shape}::iteration{step}'):
                    value.add_(.0078125)
                    result = program(value)
                    if not args.defer_profile_drain:
                        result = result.cpu()
    torch.hpu.synchronize()
    profiler.export_chrome_trace(str(root / f'collective-rank{rank}.json'))
report['status'] = 'passed'
(root / f'collective-rank{rank}-results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
