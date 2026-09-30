# SPDX-License-Identifier: Apache-2.0
"""Gate packed TP4 MLA -> output projections -> real hc_post at long positions.

Both implementations consume identical packed pages, sparse/repeated selections,
four real TP shards from four KV-source layers, and changing queries/residuals.
Their output projections pass through the production AllReduce and hc_post.
The parent component reference is missing for this exact boundary and measured once.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--iterations', type=int, default=64)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--direct-gather', action='store_true')
    parser.add_argument('--reuse-reference', type=Path)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    if '{rank}' in os.environ.get('PT_HPU_RECIPE_CACHE_CONFIG', ''):
        os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = os.environ['PT_HPU_RECIPE_CACHE_CONFIG'].replace('{rank}', str(rank))
    if os.getenv('GRAPH_VISUALIZATION') == '1':
        graph_dir = Path(os.environ['GRAPH_VISUALIZATION_DIR']) / f'rank{rank}'
        graph_dir.mkdir(parents=True, exist_ok=True)
        os.environ['GRAPH_VISUALIZATION_DIR'] = str(graph_dir)
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel,
                                  init_distributed_environment, initialize_model_parallel)
    from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, rotary_table, hc_post
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://', local_rank=rank,
                                 backend='hccl')
    config = EngineArgs(model=str(args.prepared), dtype='bfloat16', tensor_parallel_size=4,
                        pipeline_parallel_size=1, load_format='dsv41_prepared', max_model_len=1048576,
                        max_num_seqs=32, max_num_batched_tokens=8192, block_size=128,
                        enable_prefix_caching=False, async_scheduling=True).create_engine_config()
    op = torch.ops.custom_op
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(rank=rank, status='running', scope=__doc__, checks=[], timings={})

    def save():
        (root / f'projection-rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')

    def reference(q, residual, post, comb, swa, main, rows, ids, sink, scale, lengths,
                  wa, wa_scale, positions, phase, wb, wb_scale):
        output = op.custom_deepseek_v41_paged_mla_mme_gaudi2(q, swa, main, rows, ids, sink, scale, lengths)
        output = op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(output, positions, phase)
        output = op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2(
            output.reshape(q.shape[0], 2, 4096).contiguous(), wa, wa_scale)
        output = op.custom_deepseek_v41_dense_fp8_gaudi2(output.contiguous(), wb, wb_scale)
        return hc_post(reduce(output, ready_outputs=(post, comb)), residual, post, comb)

    def candidate(q, residual, post, comb, swa, main, rows, ids, sink, scale, lengths,
                  wa, wa_scale, positions, phase, wb, wb_scale):
        if args.direct_gather:
            output = op.custom_deepseek_v41_paged_mla_direct_mme_gaudi2(
                q, swa, main, rows, ids, sink, scale, lengths)
            output = op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(output, positions, phase)
            output = op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2(
                output.reshape(q.shape[0], 2, 4096).contiguous(), wa, wa_scale)
            output = op.custom_deepseek_v41_dense_fp8_gaudi2(output.contiguous(), wb, wb_scale)
        else:
            output = op.custom_deepseek_v41_paged_mla_woa_wob_fp8_roundtrip_gaudi2(
                q, swa, main, rows, ids, sink, scale, lengths, wa, wa_scale, positions, phase, wb, wb_scale)
        return hc_post(reduce(output, ready_outputs=(post, comb)), residual, post, comb)

    try:
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            reduce, gather = stage_collectives(rank, False)
            reduce(torch.ones(1, device='hpu', dtype=torch.bfloat16)).cpu()
            bind_worker_helpers(rank)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            wa = WoaFP8Sidecar(args.prepared / 'sidecars/wo_a_fp8', shard)
            wb = DenseFP8Sidecar(args.prepared / 'sidecars/attention_dense_fp8', shard)
            config = json.loads((args.prepared / 'config.json').read_text())['text_config']
            scaling = config['rope_scaling']
            table = rotary_table(64, 32768, config['compress_rope_theta'],
                                 scaling['original_max_position_embeddings'], scaling['factor'],
                                 scaling['beta_fast'], scaling['beta_slow'])
            phase = torch.cat((table[..., 0], table[..., 1]), -1).contiguous().to('hpu')
            del table
            constants = []
            for layer in (2, 8, 14, 20):
                prefix = f'layers.{layer}.attn.'
                ratio = config['compress_ratios'][layer]
                main_rows = 1880704 // ratio  # Archived component pool; full serving geometry checked separately.
                logical_rows = 32768 // ratio
                generator = torch.Generator().manual_seed(260929 + layer)
                swa = pack_swa(torch.randn(256, 512, dtype=torch.bfloat16, generator=generator)).to('hpu')
                packed = pack_fp4(torch.randn(logical_rows, 512, dtype=torch.bfloat16, generator=generator), 16)
                page_width = 128 // ratio
                logical = torch.arange(logical_rows, dtype=torch.int64)
                # Non-identity, sparse physical page mapping near the end of
                # the unchanged full-capacity pool. Logical rows stay ordered.
                page = main_rows // page_width - 1 - 3 * (logical // page_width)
                physical = page * page_width + logical.remainder(page_width)
                assert int(physical.min()) >= 0
                main = torch.zeros(main_rows, 288, dtype=torch.uint8, device='hpu')
                main.index_copy_(0, physical.to('hpu'), packed.to('hpu'))
                constants.append((ratio, physical, swa, main, shard.tensor(prefix + 'attn_sink', 'hpu').float(),
                                  wa.tensor(prefix + 'wo_a.weight', 'hpu'),
                                  wa.tensor(prefix + 'wo_a.channel_scale', 'hpu'),
                                  wb.tensor(prefix + 'wo_b.weight', 'hpu'),
                                  wb.tensor(prefix + 'wo_b.channel_scale', 'hpu')))
            fixtures = []
            for seed, position in enumerate((16384, 16532, 19372, 32767)):
                generator = torch.Generator().manual_seed(6300 + seed)
                query_generator = torch.Generator().manual_seed(7300 + seed + 100 * rank)
                group = []
                for ratio, physical, swa, main, sink, weight_a, scale_a, weight_b, scale_b in constants:
                    selected = (torch.arange(512) * (37 + seed * 2) + 13) % ((position + 1) // ratio)
                    selected[3] = selected[2]  # Duplicate selections preserve their multiplicity.
                    selected[::17] = -1
                    rows = torch.cat((torch.arange(256), physical[selected.clamp_min(0)] + 256)).int().reshape(1, 768)
                    window = torch.arange(position - 127, position + 1).remainder(256)
                    ids = torch.cat((window, torch.where(selected >= 0, torch.arange(512) + 256, -1))).int()
                    q = torch.randn(1, 16, 512, dtype=torch.bfloat16,
                                    generator=query_generator) * (0.25 + seed * 0.25)
                    residual = torch.randn(1, 4, 5120, dtype=torch.bfloat16, generator=generator)
                    post = torch.rand(1, 4, generator=generator) * 2
                    comb = torch.softmax(torch.randn(1, 4, 4, generator=generator), -1)
                    group.append((q.to('hpu'), residual.to('hpu'), post.to('hpu'), comb.to('hpu'), swa, main,
                                  rows.to('hpu'), ids.reshape(1, 640).to('hpu'), sink,
                                  torch.tensor([512**-.5], device='hpu'),
                                  torch.tensor([640], dtype=torch.int32, device='hpu'), weight_a, scale_a,
                                  torch.tensor([position], dtype=torch.int32, device='hpu'), phase, weight_b, scale_b))
                fixtures.append(tuple(group))
            parent = torch.compile(reference, backend=make_backend(), fullgraph=True, dynamic=False)
            changed = torch.compile(candidate, backend=make_backend(), fullgraph=True, dynamic=False)
            for index, group in enumerate(fixtures):
                expected = tuple(parent(*row).cpu() for row in group)
                device_actual = tuple(changed(*row) for row in group)
                actual = tuple(value.cpu() for value in device_actual)
                copies = gather(device_actual[0].flatten(), dim=0).cpu().reshape(4, -1)
                assert torch.equal(copies, copies[:1].expand_as(copies)), 'AllReduce consumers differ across ranks'
                checks = [dict(layer=layer, exact_bits=torch.equal(a.view(torch.int16), b.view(torch.int16)),
                               max_abs=float((a.float() - b.float()).abs().max()), finite=bool(torch.isfinite(a).all()))
                          for layer, a, b in zip((2, 8, 14, 20), actual, expected, strict=True)]
                report['checks'].append(dict(fixture=index, layers=checks))
                torch.save(dict(parent=expected, candidate=actual), root / f'outputs-rank{rank}-{index}.pt')
                save()
                assert all(c['exact_bits'] and c['finite'] for c in checks), checks
            # Each call has the same graph boundary as one production
            # Attention output. Retain the hc_post dependency between calls
            # while changing queries and the four real projection weights.
            # No token readback or whole-device drain occurs inside a block.
            def block(function, iteration):
                output = fixtures[iteration % len(fixtures)][0][1]
                for step in range(args.steps):
                    group = fixtures[(iteration + step) % len(fixtures)]
                    for row in group:
                        output = function(row[0], output, *row[2:])
                return output

            expected = block(parent, 0).cpu()
            actual = block(changed, 0).cpu()
            assert torch.equal(expected.view(torch.int16), actual.view(torch.int16)), 'Feedback consumer mismatch'
            report['feedback_exact'] = True
            report['steps_per_sample'] = args.steps
            report['projections_per_step'] = len(constants)
            arms = [('parent', parent), ('candidate', changed)]
            if args.reuse_reference:
                saved = json.loads((args.reuse_reference / f'projection-rank{rank}.json').read_text())
                assert saved['status'] == 'exact_complete_consumer_chain_measured'
                assert saved['steps_per_sample'] == args.steps and saved['projections_per_step'] == len(constants)
                report['timings']['parent'] = saved['timings']['parent']
                report['reference_reused'] = str(args.reuse_reference)
                report['baseline_measured'] = False
                arms = [('candidate', changed)]
            for name, function in arms:
                device, wall = [], []
                for iteration in range(args.iterations + 8):
                    torch.hpu.synchronize()
                    torch.distributed.barrier()
                    begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    started = time.perf_counter_ns()
                    begin.record()
                    output = block(function, iteration)
                    end.record()
                    end.synchronize()
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    if iteration >= 8:
                        device.append(begin.elapsed_time(end) / args.steps)
                        wall.append(elapsed / args.steps)
                assert bool(torch.isfinite(output.cpu()).all()), 'Nonfinite feedback consumer'
                report['timings'][name] = dict(device_ms=device, wall_ms=wall,
                                               device_median_ms=statistics.median(device),
                                               wall_median_ms=statistics.median(wall))
                save()
            if args.direct_gather:
                report['edge_checks'] = []
                # Correctness only, after timing. Match the current full
                # serving pool and address its final non-identity pages.
                for source_index in (0, 3):
                    row = fixtures[0][source_index]
                    ratio, physical = constants[source_index][:2]
                    full_rows = 6179072 // ratio
                    delta = full_rows - row[5].shape[0]
                    full_main = torch.zeros(full_rows, 288, dtype=torch.uint8, device='hpu')
                    packed_rows = row[5].index_select(0, physical.to('hpu'))
                    full_main.index_copy_(0, (physical + delta).to('hpu'), packed_rows)
                    for length in (0, 1, 127, 639, 640):
                        edge = list(row)
                        edge[5] = full_main
                        source_rows = row[6].cpu()
                        source_rows = torch.where(source_rows >= 256, source_rows + delta, source_rows)
                        source_rows[0, 259] = -1
                        source_rows[0, 260] = full_rows + 256
                        edge[6] = source_rows.to('hpu')
                        indices = row[7].cpu()
                        indices[0, :6] = torch.tensor([-1, 768, 259, 260, 258, 258], dtype=torch.int32)
                        edge[7] = indices.to('hpu')
                        edge[10] = torch.tensor([length], dtype=torch.int32, device='hpu')
                        expected = parent(*edge).cpu()
                        actual = changed(*edge).cpu()
                        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16)), (source_index, length)
                        report['edge_checks'].append(dict(layer=(2, 8, 14, 20)[source_index], length=length,
                                                          pool_rows=full_rows, invalid_physical=True, exact_bits=True))
                        save()
            report.update(status='exact_complete_consumer_chain_measured',
                          memory_allocated=torch.hpu.memory_allocated(),
                          limitation='Component only; shared B2/C6 integration and fullmodel remain separate gates')
            destroy_model_parallel()
            destroy_distributed_environment()
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save()
        print(json.dumps({k: v for k, v in report.items() if k != 'timings'}), flush=True)


if __name__ == '__main__':
    main()
