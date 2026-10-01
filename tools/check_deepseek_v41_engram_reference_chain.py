# SPDX-License-Identifier: Apache-2.0
"""Qualify scheduler-boundary Engram staging through its real TP4 consumers."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    cache = os.environ.get('PT_HPU_RECIPE_CACHE_CONFIG')
    if cache:
        os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = cache.replace('{rank}', str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree, linear
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_engram_fp8 import EngramFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_host import stage_device_engram_rows
    from vllm_gaudi.ops.deepseek_v41_math import engram_update, hc_pre
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://', local_rank=rank,
                                 backend='hccl')
    config = EngineArgs(model=str(args.prepared), dtype='bfloat16', tensor_parallel_size=4,
                        pipeline_parallel_size=1, load_format='dsv41_prepared', max_model_len=1048576,
                        max_num_seqs=32, max_num_batched_tokens=8192, block_size=128,
                        enable_prefix_caching=False, async_scheduling=True).create_engine_config()
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    output = root / f'chain-rank{rank}.json'
    report = dict(rank=rank, status='running', cases=[], samples={},
                  scope='C1 fallback unpack/copy -> four-rank all_gather -> real FP8 Wkv -> Engram update -> mHC/RMS',
                  reference_reason='No archived component timing includes the eager scheduler-boundary staging path.',
                  limitations=['Not a complete-model or EOS measurement.', 'C2/C6 only protect staging geometry.'])
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            reduce, gather = stage_collectives(rank, False)
            reduce(torch.ones(1, device='hpu', dtype=torch.bfloat16)).cpu()
            bind_worker_helpers(rank)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            specs = {name: spec for name, spec in shard.specs.items()
                     if any(name.startswith(f'layers.{layer}.{part}') for layer in (1, 14)
                            for part in ('engram.', 'attn_norm.', 'hc_attn_'))}
            tree = _weight_tree(specs)
            load_weight_tree(shard, tree, 'hpu', specs,
                             engram_sidecar=EngramFP8Sidecar(args.prepared / 'sidecars/engram_fp8', shard))
            text = json.loads((args.prepared / 'config.json').read_text())['text_config']
            eps, hc_eps, iterations = (text[k] for k in ('rms_norm_eps', 'hc_eps', 'hc_sinkhorn_iters'))
            candidate = torch.compile(stage_device_engram_rows, backend=make_backend(), fullgraph=True, dynamic=False)

            def consumer(rows, residual, previous, active, w):
                projected = linear(gather(rows, 1).flatten(1), w.engram.wkv)
                updated = engram_update(residual, projected, w.engram.q_weight, w.engram.k_weight, active, eps)
                values = hc_pre(updated, previous, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base,
                                eps, hc_eps, iterations, packed_fn=w.hc_attn_fn)
                return (updated, *values)

            consume = torch.compile(consumer, backend=make_backend(), fullgraph=True, dynamic=False)
            torch.manual_seed(9217 + rank)
            residual = torch.randn(1, 4, 5120).bfloat16().to('hpu')
            previous = torch.full((1, 4), .25, device='hpu')
            active = torch.tensor([True], device='hpu')
            # Same two-layer packet backing/stride and persistent BF16 output
            # as serving; the second half is a canary against alias writes.
            packet = torch.empty(3168, dtype=torch.uint8, device='hpu')
            packed = packet[:1584].view(1, 6, 264)
            destination = torch.empty((1, 6, 256), dtype=torch.bfloat16, device='hpu')
            fixtures = []
            for step in range(16):
                host = torch.full((3168,), 197, dtype=torch.uint8)
                rows = host[:1584].view(1, 6, 264)
                codes = (torch.arange(1536).reshape(1, 6, 256) + step * 17 + rank * 11) % 126
                rows[..., :256] = codes.to(torch.uint8)
                rows[..., 256:] = 119 + step % 5
                fixtures.append(host.to('hpu'))
            saved = []
            for step, source in enumerate(fixtures):
                packet.copy_(source)
                w = tree.layers.get_submodule(str(1 if step % 2 == 0 else 14))
                stage_device_engram_rows(packed, destination)
                expected = tuple(x.cpu() for x in consume(destination, residual, previous, active, w))
                expected_rows = destination.cpu()
                pointer = destination.data_ptr()
                candidate(packed, destination)
                actual = tuple(x.cpu() for x in consume(destination, residual, previous, active, w))
                assert destination.data_ptr() == pointer
                assert torch.equal(destination.cpu().view(torch.int16), expected_rows.view(torch.int16))
                assert torch.equal(packet.cpu()[1584:], source.cpu()[1584:])
                for a, b in zip(actual, expected, strict=True):
                    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                    assert torch.isfinite(a).all()
                saved.append(expected)
                report['cases'].append(dict(step=step, layer=1 if step % 2 == 0 else 14, exact=True))
            for count in (2, 6, 1):
                source = torch.cat([fixtures[count % 16][:1584].view(1, 6, 264)] * count, 0)
                reference = torch.empty(count, 6, 256, dtype=torch.bfloat16, device='hpu')
                result = torch.empty_like(reference)
                stage_device_engram_rows(source, reference)
                candidate(source, result)
                assert torch.equal(result.cpu().view(torch.int16), reference.cpu().view(torch.int16))
                report['cases'].append(dict(tokens=count, exact=True, scope='staging geometry only'))
            torch.save(saved, root / f'consumer-reference-rank{rank}.pt')
            bind_worker_helpers(rank)
            for arm, stage in [('reference', stage_device_engram_rows), ('candidate', candidate)]:
                samples = []
                for step, source in enumerate(fixtures):
                    packet.copy_(source)
                    w = tree.layers.get_submodule(str(1 if step % 2 == 0 else 14))
                    torch.hpu.synchronize()
                    start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    before = time.perf_counter_ns()
                    start.record()
                    stage(packed, destination)
                    result = consume(destination, residual, previous, active, w)
                    end.record()
                    end.synchronize()
                    samples.append(dict(host_ms=(time.perf_counter_ns()-before)/1e6,
                                        device_elapsed_ms=start.elapsed_time(end)))
                    for a, b in zip(result, saved[step], strict=True):
                        assert torch.equal(a.cpu().contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                report['samples'][arm] = samples
                report[arm + '_host_median_ms'] = statistics.median(x['host_ms'] for x in samples)
                report[arm + '_device_median_ms'] = statistics.median(x['device_elapsed_ms'] for x in samples)
                output.write_text(json.dumps(report, indent=2) + '\n')
            report['status'] = 'exact_complete_consumer_chain'
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status='failed', error=repr(error))
    finally:
        output.write_text(json.dumps(report, indent=2) + '\n')
    if report['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
