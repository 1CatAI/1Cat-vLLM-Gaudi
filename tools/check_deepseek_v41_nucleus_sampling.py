# SPDX-License-Identifier: Apache-2.0
"""Bounded-candidate sampling through four-rank exchange and token consumption."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--candidates', type=int, default=128)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import torch.distributed as dist
    from functools import partial
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_sampling import (
        local_nucleus_candidates, sample_nucleus_candidates, sample_probabilities,
    )
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives, _Snapshot
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.tp2_prepared_plan import (
        collect_prepared_group_replays, record_native_decoder_outputs,
        replay_native_decoder, shutdown_prepared_group_plans,
    )
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(rank=rank, status='running', candidates=args.candidates, segments=[], correctness=[])
    path = root / f'sampling-rank{rank}.json'
    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')
    torch.hpu.set_device(rank)
    torch.set_num_threads(1)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://',
                                 local_rank=rank, backend='hccl')
    config = EngineArgs(model=str(args.prepared), dtype='bfloat16', tensor_parallel_size=4,
                        pipeline_parallel_size=1, load_format='dsv41_prepared', max_model_len=524288,
                        max_num_seqs=32, max_num_batched_tokens=16384, block_size=128,
                        enable_prefix_caching=True, async_scheduling=True).create_engine_config()
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            _, full_gather = stage_collectives(rank, False, tp_size=4)
            _, peer_gather = stage_collectives(rank, True, tp_size=4, native_fp32_gather=True)
            gen = torch.Generator().manual_seed(914)
            full = torch.randn(1, 129280, generator=gen) * 8
            logits = full[:, rank * 32320:(rank + 1) * 32320].contiguous().to('hpu')
            controls = torch.tensor([[1., .95, .5, -1.]], device='hpu')
            reference = torch.compile(partial(sample_probabilities, filtered=True),
                                      backend='hpu_backend', fullgraph=True, dynamic=False)
            owner = torch.nn.Module()
            metadata = SimpleNamespace(native_completion=None)
            adapter = DecoderTopology('deepseek_v41_sampling_probe', (1,), 0, False, extra_collectives=1)
            fixed_logits, fixed_controls = logits.clone(), controls.clone()
            roots = dict(hidden_states=fixed_logits, pre_mix=fixed_controls, positions=None, input_ids=None,
                         attention_inputs=(), metadata=metadata, state_generation=1, state_tensors=())
            def body(value, settings):
                packet = local_nucleus_candidates(value, settings, rank, candidates=args.candidates)
                token, covered = sample_nucleus_candidates(peer_gather(packet, 1), settings,
                                                           candidates=args.candidates)
                # The next device token/input consumer is retained in both arms.
                return torch.cat((token + 1, covered.to(torch.int32)), -1)
            candidate = torch.compile(body, backend='hpu_backend', fullgraph=True, dynamic=False)
            def call_candidate():
                dynamic = dict(roots, hidden_states=logits, pre_mix=controls)
                result = replay_native_decoder(owner, **dynamic)
                if result is not None:
                    return result[0]
                fixed_logits.copy_(logits)
                fixed_controls.copy_(controls)
                with collect_prepared_group_replays(owner=owner, adapter=adapter,
                                                   snapshot=lambda: _Snapshot(()), **roots) as context:
                    context['group_index'] = 0
                    result = candidate(fixed_logits, fixed_controls)
                    record_native_decoder_outputs(result)
                return result
            def call_reference():
                token = reference(full_gather(logits, -1), controls)
                return torch.cat((token + 1, torch.ones_like(token)), -1)
            for ordinal in range(32):
                controls[:, 2] = (ordinal + .5) / 32
                expected = call_reference().cpu()
                actual = call_candidate().cpu()
                covered = bool(actual[0, 1])
                exact = torch.equal(actual[:, :1], expected[:, :1])
                report['correctness'].append(dict(ordinal=ordinal, covered=covered, exact=exact))
                save()
                assert not covered or exact
            for step, arm in enumerate(('A', 'B', 'A', 'B', 'A', 'B')):
                call = call_reference if arm == 'A' else call_candidate
                dist.barrier()
                torch.hpu.synchronize()
                samples = []
                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                begin.record()
                for ordinal in range(200):
                    controls[:, 2] = ((ordinal + step * 17) % 200 + .5) / 200
                    started = time.perf_counter_ns()
                    call().cpu()
                    samples.append((time.perf_counter_ns() - started) / 1e6)
                end.record()
                end.synchronize()
                quantiles = statistics.quantiles(samples, n=4)
                report['segments'].append(dict(arm=arm, median_ms=statistics.median(samples),
                    iqr_ms=quantiles[2] - quantiles[0], device_ms=begin.elapsed_time(end) / 200,
                    host_ms=samples))
                save()
            report['status'] = 'passed'
            shutdown_prepared_group_plans()
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
