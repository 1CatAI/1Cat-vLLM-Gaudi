# SPDX-License-Identifier: Apache-2.0
"""Measure C1/C6 BF16 peer payloads through the same compiled collective."""
import json
import os
from pathlib import Path
import time


def main():
    rank, tp = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = os.environ.get('PT_HPU_RECIPE_CACHE_CONFIG',
                                                              '').replace('{rank}', str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(tensor_parallel_size=tp, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel, init_distributed_environment,
                                  initialize_model_parallel)
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans

    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    report = dict(status='running',
                  rank=rank,
                  tp_size=tp,
                  rows=[],
                  scope='compiled collective operator; not whole-stage native joint replay or model TPOT',
                  formal_qualification=False)
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp))), \
            torch.inference_mode():
        try:
            init_distributed_environment(world_size=tp,
                                         rank=rank,
                                         distributed_init_method='env://',
                                         local_rank=rank,
                                         backend='hccl')
            initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            reduce, gather = stage_collectives(rank, True, tp)
            for count in (1, 6):
                value = torch.full((count, 5120), rank + .125, dtype=torch.bfloat16, device='hpu')
                for kind, operation in [('gather', lambda x: gather(x, 1)), ('reduce', reduce)]:
                    execute = torch.compile(operation, backend='hpu_backend', fullgraph=True, dynamic=False)
                    actual = execute(value)
                    expected = (torch.cat(
                        [torch.full((count, 5120), r + .125, dtype=torch.bfloat16)
                         for r in range(tp)], 1) if kind == 'gather' else torch.full(
                             (count, 5120), sum(r + .125 for r in range(tp)), dtype=torch.bfloat16))
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                    for _ in range(32):
                        execute(value)
                    torch.hpu.synchronize()
                    samples = []
                    for _ in range(3):
                        start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                        start.record()
                        host = time.perf_counter_ns()
                        for _ in range(128):
                            execute(value)
                        stop.record()
                        stop.synchronize()
                        samples.append(
                            dict(device_us=start.elapsed_time(stop) * 1000 / 128,
                                 host_us=(time.perf_counter_ns() - host) / 1000 / 128))
                    report['rows'].append(
                        dict(tokens=count,
                             payload_bytes=value.numel() * value.element_size(),
                             operation=kind,
                             exact=True,
                             samples=samples))
            report['status'] = 'passed'
        except Exception as exc:
            report.update(status='failed', error=repr(exc))
            raise
        finally:
            (root / f'peer-rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
            if torch.distributed.is_initialized():
                shutdown_prepared_group_plans()
                destroy_model_parallel()
                destroy_distributed_environment()


if __name__ == '__main__':
    main()
