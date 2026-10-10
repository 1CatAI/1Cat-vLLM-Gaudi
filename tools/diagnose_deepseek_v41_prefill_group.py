# SPDX-License-Identifier: Apache-2.0
"""Bounded TP4 prefill stage diagnosis with real weights and serving operators.

This samples one source-owner group at production shapes. It is not a full
request trace, and group timing must not be called the measured full-model split.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--start', type=int, default=2)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=16384)
    parser.add_argument('--probe-frozen-cache', action='store_true')
    parser.add_argument('--synchronized-phases', action='store_true')
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = os.environ.get('PT_HPU_RECIPE_CACHE_CONFIG',
                                                              '').replace('{rank}', str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment, load_native_operators
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.models.deepseek_v41_program import PreparedDecoderLayer, _weight_tree, load_weight_tree
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_prefill_regions import freeze_prefill_regions
    load_native_operators()
    torch.hpu.set_device(rank)
    torch.set_num_threads(1)
    bind_worker_cpu(rank)
    bind_worker_helpers(rank)
    init_distributed_environment(world_size=4,
                                 rank=rank,
                                 distributed_init_method='env://',
                                 local_rank=rank,
                                 backend='hccl')
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    path = root / f'prefill-group-rank{rank}.json'
    report = dict(status='loading',
                  rank=rank,
                  tokens=args.tokens,
                  layers=list(range(args.start, args.start + args.layers)),
                  scope=__doc__,
                  full_model_requests=0,
                  formal_gain_credit=False,
                  synchronization_added=args.synchronized_phases,
                  rounds=[])

    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')

    save()
    with set_current_vllm_config(config), torch.inference_mode():
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
        reduce, gather = stage_collectives(rank, False, 4)
        shard = PreparedV41Shard(args.prepared, 0, rank)
        text = json.loads((args.prepared / 'config.json').read_text())['text_config']
        layers = report['layers']
        if any(i in (1, 14) for i in layers):
            raise ValueError('This bounded group does not include the host Engram producer')
        prefixes = tuple(f'layers.{i}.' for i in layers)
        specs = {name: spec for name, spec in shard.specs.items() if name.startswith(prefixes)}
        weights = _weight_tree(specs)
        load_weight_tree(shard,
                         weights,
                         'hpu',
                         specs,
                         expert_n256_layers=layers,
                         woa_sidecar=WoaFP8Sidecar(os.environ['VLLM_HPU_DSV41_WO_A_FP8_SIDECAR'], shard),
                         woa_layers=layers,
                         dense_sidecar=DenseFP8Sidecar(os.environ['VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR'], shard),
                         dense_config={
                             'wq_b': layers,
                             'wo_b': layers,
                             'shared_w1': layers,
                             'shared_w3': layers,
                             'shared_w2': layers
                         })
        shared = PagedCSA2SharedState(text, 0, 40, 'hpu', 32768, prefill_tokens=args.tokens, tensor_parallel_size=4)
        shared.block_table[:256].copy_(torch.arange(1, 257, dtype=torch.int32, device='hpu'))
        for cache in shared.sources.values():
            cache.main = torch.zeros(257 * 128 // cache.ratio, 288, dtype=torch.uint8, device='hpu')
            cache.index = torch.zeros(257 * 128 // cache.ratio, 68, dtype=torch.uint8, device='hpu')
        lookup = mxfp4_bf16_lut(torch.device('hpu'))
        blocks = []
        records = None

        @contextmanager
        def span(name, layer):
            if records is None or torch.compiler.is_compiling():
                yield
                return
            if args.synchronized_phases:
                # This intentionally serializes diagnostic boundaries. Report
                # elapsed wall time, never hardware-kernel exclusive duration.
                torch.hpu.synchronize()
                began = time.perf_counter_ns()
                try:
                    yield
                finally:
                    torch.hpu.synchronize()
                    records.append(dict(name=name, layer=layer, synchronized_ms=(time.perf_counter_ns() - began) / 1e6))
                return
            first, last = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
            began = time.perf_counter_ns()
            first.record()
            try:
                yield
            finally:
                last.record()
                records.append(
                    dict(name=name, layer=layer, first=first, last=last,
                         host_ms=(time.perf_counter_ns() - began) / 1e6))

        def instrument(module, name, layer):
            original = module.forward

            def invoke(*inputs, **kwargs):
                if torch.compiler.is_compiling():
                    return original(*inputs, **kwargs)
                with span(name, layer):
                    return original(*inputs, **kwargs)

            module.forward = invoke

        for i in layers:
            normal = shard.manifest['normal_scales'][f'layers.{i}.ffn.experts'][rank]
            block = PreparedDecoderLayer(weights.layers.get_submodule(str(i)),
                                         text,
                                         i,
                                         shared,
                                         normal,
                                         lookup,
                                         reduce,
                                         gather,
                                         'hpu',
                                         tensor_parallel_size=4)
            block.prepare_mhc_control_weights()
            block.moe.prepare_shared_gate_up_weight()
            block.moe.prepare_split_scale_planes()
            attention = block.attention
            attention.prefill_tp_rank = rank
            attention.prepare_qkv_input_weight()
            attention.prepare_compressor_input_weight()
            attention.prepare_index_gain_weight()
            attention.woa_fp8 = attention.woa_output_roundtrip = True
            attention.set_search_length(args.tokens)
            attention.prefill_token_end = args.tokens
            instrument(attention, 'attention', i)
            instrument(block.moe, 'moe_including_router_shared_and_exchange', i)
            blocks.append(block)
        generator = torch.Generator().manual_seed(58215)
        # Shape-accurate activations, not saved full-request hidden states.
        residual = torch.randn(args.tokens // 4, 4, 5120, generator=generator).bfloat16().to('hpu')
        previous = torch.full((args.tokens // 4, 4), .25, device='hpu')
        positions = torch.arange(args.tokens, dtype=torch.int32, device='hpu')
        image_mask = torch.zeros(args.tokens, dtype=torch.bool, device='hpu')

        def run():
            shared.prefill_kv_generation += 1
            shared.prefill_main_workspace.begin(shared.prefill_kv_generation)
            value, pre = residual, previous
            for block in blocks:
                with span('layer', block.layer):
                    value, pre, _ = block(value, pre, positions, image_mask, prefill_sequence=True)
            return value, pre

        report['status'] = 'warming'
        save()
        for _ in range(2):
            output = run()
            torch.hpu.synchronize()
        freeze_prefill_regions()
        report['status'] = 'measuring'
        save()
        from vllm_gaudi.ops import deepseek_v41_prefill_regions as regions
        retained_regions = regions._function_regions
        arms = ('retained', 'frozen_empty') if args.probe_frozen_cache else ('retained', )
        for repeat in range(3 * len(arms)):
            arm = arms[repeat % len(arms)]
            regions._function_regions = retained_regions if arm == 'retained' else {}
            records = []
            begin = time.perf_counter_ns()
            output = run()
            torch.hpu.synchronize()
            wall = (time.perf_counter_ns() - begin) / 1e6
            rows = []
            anchor = None if args.synchronized_phases else next(item['first'] for item in records
                                                                if item['name'] == 'layer')
            for item in records:
                if args.synchronized_phases:
                    rows.append(item)
                    continue
                first, last = item.pop('first'), item.pop('last')
                item['device_start_ms'] = anchor.elapsed_time(first)
                item['device_end_ms'] = anchor.elapsed_time(last)
                item['device_ms'] = first.elapsed_time(last)
                rows.append(item)
            records = None
            report['rounds'].append(
                dict(repeat=repeat,
                     arm=arm,
                     host_drained_ms=wall,
                     spans=rows,
                     finite=bool(output[0].isfinite().all().cpu())))
            save()
        regions._function_regions = retained_regions
        report['status'] = 'passed'
        save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
