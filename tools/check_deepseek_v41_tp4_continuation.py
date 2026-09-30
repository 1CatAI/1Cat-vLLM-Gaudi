# SPDX-License-Identifier: Apache-2.0
"""Real head/token/Engram/16-layer continuation, with production TP4 shards.

The first layer-14 consumer is included. This is a component gate, not a
measurement of EngineCore IPC or a substitute for full serving qualification.
"""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--speed-probe', action='store_true',
                        help='Candidate timing screen with exact saved tokens; broader state/lifecycle gate deferred')
    parser.add_argument('--qualify-positive', action='store_true',
                        help='Continue state/lifecycle qualification in this process only if all ranks gain speed')
    parser.add_argument('--reuse-reference', type=Path,
                        help='Reuse archived continuation timing and exact tokens; never time the old arm again')
    parser.add_argument('--diagnose-state', action='store_true',
                        help='Untimed native-vs-original position copy comparison with saved full states')
    parser.add_argument('--state-reference-original-copy', action='store_true',
                        help='Reconstruct only a missing comparable byte-state oracle, without timing the old copy')
    parser.add_argument('--state-reference-paged-projection', action='store_true',
                        help='Compare new packed projections with the original chain on the same initial buffers')
    parser.add_argument('--state-reference-paged-gather', action='store_true',
                        help='Compare direct packed gather with the original chain on the same initial buffers')
    parser.add_argument('--state-reference-visible-prefix', action='store_true',
                        help='Exercise the runner visible-prefix contract against unpruned same-buffer math')
    parser.add_argument('--state-reference-logical-mla', action='store_true',
                        help='Compare logical MLA with current directpacked gather using identical initial state')
    parser.add_argument('--state-reference-index-mirror', action='store_true',
                        help='Compare derived-key scoring with canonical packed scoring from identical buffers')
    parser.add_argument('--state-reference-shared-main', action='store_true')
    parser.add_argument('--state-reference-native-groups', action='store_true')
    parser.add_argument('--shared-stage-replay', action='store_true',
                        help='Use the production shared StageReplay and native input/Engram dependency graph')
    parser.add_argument('--tp2-ordered-selection', action='store_true',
                        help='Screen historical TP2 cutoff semantics with TP4 scoring; model quality remains pending')
    parser.add_argument('--selection-trace-only', action='store_true',
                        help='Capture eight real-chain steps after continuous warmup; do not collect a speed result')
    parser.add_argument('--state-reference-decode-metadata', action='store_true')
    parser.add_argument('--state-reference-post-collapse', action='store_true')
    parser.add_argument('--state-reference-interlayer-collapse', action='store_true')
    parser.add_argument('--state-reference-feature-silu', action='store_true')
    parser.add_argument('--state-reference-mirror-partition', action='store_true',
                        help='Compare contiguous rank partition with original full mirror scoring, untimed')
    parser.add_argument('--production-visible-prefix', action='store_true',
                        help='Use the current production visible-prefix bounds without an unpruned comparison arm')
    parser.add_argument('--candidate-local-index-queries', action='store_true',
                        help='Call the maintained cold weight preparation used by the normal loader')
    parser.add_argument('--diagnose-local-query-state', action='store_true',
                        help='Compare full continuation state from identical initial buffers; never collect timing')
    parser.add_argument('--continuous-warm-steps', type=int, default=0,
                        help='Warm inside the same state-continuous chain before starting its timers')
    parser.add_argument('--measure-continuous-reference', action='store_true',
                        help='Measure a missing reference only for the new continuous timing contract')
    args = parser.parse_args()
    if args.diagnose_local_query_state and not args.candidate_local_index_queries:
        parser.error('Local-query state diagnostic requires that candidate')
    if args.continuous_warm_steps < 0:
        parser.error('continuous warm steps must be non-negative')
    if args.measure_continuous_reference and not args.continuous_warm_steps:
        parser.error('continuous reference measurement requires continuous warm steps')
    if args.continuous_warm_steps and not (args.measure_continuous_reference or args.reuse_reference):
        parser.error('continuous measurement requires an archived or explicitly missing component reference')
    if args.measure_continuous_reference and not (args.state_reference_paged_projection
                                                  or args.state_reference_paged_gather
                                                  or args.state_reference_visible_prefix):
        parser.error('continuous comparison requires a separate original-attention reference')
    if args.speed_probe and (not args.reuse_reference or not args.continuous_warm_steps):
        parser.error('Speed screening requires the saved continuous comparison')
    if args.tp2_ordered_selection and (not args.speed_probe or args.qualify_positive
                                     or args.state_reference_native_groups):
        parser.error('Ordered selection currently requires its isolated speed screen and explicit pending quality gate')
    if args.selection_trace_only and (not args.tp2_ordered_selection or not args.speed_probe):
        parser.error('Selection trace requires the ordered selection speed-screen fixture')
    if args.qualify_positive and not (args.speed_probe and args.state_reference_native_groups):
        parser.error('Conditional qualification requires the native-groups speed probe')
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    for name in ('PT_HPU_RECIPE_CACHE_CONFIG',):
        if '{rank}' in os.environ.get(name, ''):
            os.environ[name] = os.environ[name].replace('{rank}', str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import torch.distributed as dist
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.models.deepseek_v41_program import (
        PreparedStage, PreparedDecoderLayer, PreparedInput, CompiledStage, _weight_tree, load_weight_tree)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives, stage_state_tensors, _Snapshot
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_engram_fp8 import EngramFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_host import EngramHost
    from vllm_gaudi.ops.deepseek_v41_inputs import PositionBank
    from vllm_gaudi.ops.deepseek_v41_math import unpack_swa
    from vllm_gaudi.ops.deepseek_v41_completion import resolve_device_runtime
    from vllm_gaudi.ops.deepseek_v41_config import decode_source_prefix_bound
    from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://', local_rank=rank,
                                 backend='hccl')
    config = EngineArgs(model=str(args.prepared), dtype='bfloat16', tensor_parallel_size=4,
                         pipeline_parallel_size=1, load_format='dsv41_prepared', max_model_len=1048576,
                         max_num_seqs=32, max_num_batched_tokens=8192, block_size=128,
                         enable_prefix_caching=False, async_scheduling=True).create_engine_config()
    output = Path(os.environ['DSV41_RUN_EVIDENCE']) / f'continuation-rank{rank}.json'
    report = dict(rank=rank, status='running', scope='real head -> token -> Engram -> layers0-15 -> head', cases=[])
    host = None
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            if args.shared_stage_replay:
                from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
                initialize_tp2_fused_ar_norm_runtime()
            reduce, gather = stage_collectives(rank, args.shared_stage_replay, tp_size=4)
            # Initialize the actual four-rank communicator before constructing
            # the compute-only device producer's lifetime holder.
            reduce(torch.ones(1, device='hpu', dtype=torch.bfloat16)).cpu()
            # The communicator initializes eager pools after the main thread
            # was pinned. Assign their normal helper CPUs before weight loading
            # so cold preparation does not serialize every pool on one core.
            bind_worker_helpers(rank)
            if args.state_reference_mirror_partition:
                from check_deepseek_v41_tp4_mirror_partition import check_mirror_partition
                report['mirror_partition_contract'] = check_mirror_partition(gather)
                output.write_text(json.dumps(report, indent=2) + '\n')
            if args.state_reference_shared_main:
                from check_deepseek_v41_shared_main import check_shared_main_contract
                report['shared_main_native_contract'] = check_shared_main_contract()
                output.write_text(json.dumps(report, indent=2) + '\n')
            if args.state_reference_post_collapse:
                from check_deepseek_v41_mhc_post_collapse import check_mhc_post_collapse_contract
                report['mhc_post_collapse_contract'] = check_mhc_post_collapse_contract(
                    output.with_name(f'mhc-first-difference-rank{rank}.pt'))
                output.write_text(json.dumps(report, indent=2) + '\n')
            if args.state_reference_interlayer_collapse:
                from check_deepseek_v41_mhc_post_collapse import check_mhc_attention_handoff
                report['mhc_attention_handoff_contract'] = check_mhc_attention_handoff(
                    output.with_name(f'mhc-handoff-difference-rank{rank}.pt'))
                output.write_text(json.dumps(report, indent=2) + '\n')
            if args.state_reference_feature_silu:
                from check_deepseek_v41_feature_silu import check_feature_silu_contract
                report['feature_silu_contract'] = check_feature_silu_contract(
                    output.with_name(f'feature-silu-difference-rank{rank}.pt'))
                output.write_text(json.dumps(report, indent=2) + '\n')
            bridge, _ = resolve_device_runtime(4)
            host = EngramHost(args.prepared, rank, 'hpu', max_tokens=8192,
                               resident_tables=json.loads(args.bindings.read_text()))
            host.activate('chain', reset=True)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            text = json.loads((args.prepared / 'config.json').read_text())['text_config']
            layers = range(16)
            specs = {name: spec for name, spec in shard.specs.items()
                     if name in ('norm.weight', 'embed.weight', 'head.weight')
                     or name.startswith(tuple(f'layers.{i}.' for i in layers))}
            tree = _weight_tree(specs)
            load_weight_tree(shard, tree, 'hpu', specs,
                             woa_sidecar=WoaFP8Sidecar(args.prepared / 'sidecars/wo_a_fp8', shard),
                             woa_layers=layers,
                             dense_sidecar=DenseFP8Sidecar(args.prepared / 'sidecars/attention_dense_fp8', shard),
                             dense_config={'wq_b': layers, 'wo_b': layers},
                             engram_sidecar=EngramFP8Sidecar(args.prepared / 'sidecars/engram_fp8', shard))
            stage = torch.nn.Module()
            stage.weights, stage.config = tree, {'text_config': text}
            stage.length, stage.search_length = 1048576, 32768
            stage.fp8_decode, stage.expert_n256, stage.bf16_head = True, True, True
            stage.tensor_parallel_size, stage.pp_rank, stage.tp_rank = 4, 0, rank
            stage.dspark, stage.is_last_stage, stage.generation = False, True, 1
            stage.precision_fingerprint = ('fp8_decode', 'n256', 'woa_fp8', 'bf16_head')
            stage.reduce, stage.all_gather = reduce, gather
            stage.shared = PagedCSA2SharedState(text, 0, 16, 'hpu', stage.length, tensor_parallel_size=4)
            stage.runtime_indexer = stage.shared.runtime_indexer
            stage.shared.block_table[:256].copy_(torch.arange(1, 257, dtype=torch.int32, device='hpu'))
            for cache in stage.shared.sources.values():
                cache.main = torch.zeros(257 * 128 // cache.ratio, 288, dtype=torch.uint8, device='hpu')
                cache.index = torch.zeros(257 * 128 // cache.ratio, 68, dtype=torch.uint8, device='hpu')
            lookup = mxfp4_bf16_lut(torch.device('hpu'))
            stage.layers = torch.nn.ModuleList()
            for layer in layers:
                normal = shard.manifest['normal_scales'][f'layers.{layer}.ffn.experts'][rank]
                block = PreparedDecoderLayer(tree.layers.get_submodule(str(layer)), text, layer, stage.shared,
                                             normal, lookup, reduce, gather, 'hpu', tensor_parallel_size=4)
                block.moe.prepare_shared_gate_up_weight()
                block.prepare_mhc_control_weights()
                block.attention.prefill_tp_rank = rank
                block.attention.prepare_qkv_input_weight()
                block.attention.prepare_compressor_input_weight()
                if args.candidate_local_index_queries:
                    block.attention.prepare_tp4_index_query_weights()
                block.attention.woa_fp8 = block.attention.woa_output_roundtrip = True
                if block.attention.prepared_output:
                    block.attention.prepare_output_weight()
                stage.layers.append(block)
            if args.tp2_ordered_selection:
                report['ordered_selection_layers'] = [block.layer for block in stage.layers
                                                       if block.attention.owns_index]
                report['selection_contract'] = 'historical TP2 ordered cutoff ties; unchanged TP4 scoring'
                report['model_quality_pending'] = True
            report['local_index_query_layers'] = [block.layer for block in stage.layers
                                                  if block.attention.tp4_local_index_queries]
            for block in stage.layers:
                attention = block.attention
                if attention.tp4_local_index_queries:
                    assert attention.weights.indexer.wq_b.weight is getattr(attention, f'_tp4_index_query_shard_{rank}')
                    assert attention.weights.indexer.weights_proj.weight is getattr(
                        attention, f'_tp4_index_score_shard_{rank}')
            for name in ('_forward_impl', 'forward', '_head_projection', 'sample_greedy', 'sample_greedy_token'):
                setattr(stage, name, MethodType(getattr(PreparedStage, name), stage))
            if args.shared_stage_replay:
                from vllm_gaudi.ops.deepseek_v41_replay import StageReplay
                decoder = StageReplay(stage)
                stage.replay_owner = decoder
            else:
                decoder = CompiledStage(stage, prepared_tp4=True, native_tp4=args.state_reference_native_groups)
            embedding = torch.compile(PreparedInput(tree.embed, rank, reduce), backend=make_backend(),
                                       fullgraph=True, dynamic=False)
            sampler = torch.compile(stage.sample_greedy_token, backend='hpu_backend', fullgraph=True, dynamic=False)
            bank = PositionBank(stage.length, 128, 'hpu')
            ids = torch.empty(1, dtype=torch.int64, device='hpu')
            positions = torch.empty(1, dtype=torch.int32, device='hpu')
            controls = bridge.PreparedControlInputs(ids, positions)
            bank.prepare_compiled_copies({1: positions,
                                          2: torch.empty(2, dtype=torch.int32, device='hpu'),
                                          6: torch.empty(6, dtype=torch.int32, device='hpu')})
            position_preparations = bank.copy_preparations
            report['position_copy_mode'] = 'compiled_framework_copy'
            # Include the descending C8192 -> C1024 startup boundary that
            # the first complete model exposed. The final two chunks retain
            # the same exact 16K state used for the continuation comparison.
            prefill_cases = ((0, 8192), (8192, 8192)) if args.speed_probe else (
                (0, 8192), (0, 1024), (0, 8192), (8192, 8192))
            for start, count in prefill_cases:
                if start == 0:
                    mutable = {id(value) for value in stage_state_tensors(stage)}
                    for name, value in stage.named_buffers():
                        leaf = name.rsplit('.', 1)[-1]
                        if id(value) in mutable and leaf != 'block_table':
                            value.fill_(-1 if leaf in ('indices', 'candidate_pool') else 0)
                    host.reset('chain')
                tokens = list(range(7 + start, 7 + start + count))
                ticket = host.prepare('chain', tokens, defer_wait=True)
                residual, pre = embedding(torch.tensor(tokens, dtype=torch.int64, device='hpu'))
                rows = host.wait(ticket)
                pos = torch.arange(start, start + count, dtype=torch.int32, device='hpu')
                for block in stage.layers:
                    block.attention.set_search_length(start + count)
                    block.attention.prefill_token_end = start + count
                hidden = stage(residual, pre, pos, torch.tensor(tokens, device='hpu'), rows)[0]
                host.complete(ticket, len(tokens))
                assert torch.isfinite(hidden).all().item()
                print(f'TP{rank}: real 16-layer prefill {start}+{count} complete', flush=True)
            seed_hidden = hidden[-1:].clone()
            del hidden, residual, pre, rows, ticket, pos
            if stage.shared.index_mirror_tokens:
                stage.shared.prepare_index_mirror(16384)
                report['index_mirror_capacity'] = stage.shared.index_mirror_tokens
                report['index_mirror_rebuilds'] = stage.shared.index_mirror_rebuilds
            if args.state_reference_visible_prefix or args.production_visible_prefix:
                stage.decode_token_bound = decode_source_prefix_bound(16385, stage.search_length, 4)
                report['decode_token_bound'] = stage.decode_token_bound
            for block in stage.layers:
                block.attention.set_search_length(32768)
                block.attention.prefill_token_end = None
                if args.state_reference_visible_prefix or args.production_visible_prefix:
                    block.attention.set_decode_visible_tokens(stage.decode_token_bound)
            initial_history = host.history.history.copy()
            state = _Snapshot(stage_state_tensors(stage))

            def reset():
                torch.hpu.synchronize()
                state.restore()
                host.reset('chain')
                host.history.history = initial_history.copy()
                host.history.position = 16384
                torch.hpu.synchronize()

            # Warm both stable argument contracts and prove device hash/decode
            # against the exact host-produced packed bytes before any timing.
            for mode in (('device',) if args.shared_stage_replay else ('host', 'fallback', 'device')):
                device = mode == 'device'
                reset()
                selected = sampler(seed_hidden)
                value = int(selected.cpu()[0, 0])
                controls.upload([value], 16384)
                if device:
                    host.prepare_device_c1('chain', selected.view(1))
                ticket = host.prepare('chain', [value], defer_wait=True)
                packed = host.wait(ticket)
                if mode == 'fallback':
                    host.stage_device_c1_reference('chain', packed[0])
                if mode != 'host':
                    decoded = host.consume_device_c1('chain')
                    torch.testing.assert_close(decoded.cpu(), unpack_swa(packed[0], 256).cpu(), rtol=0, atol=0)
                    rows = decoded, packed[1]
                    token_input = selected.view(1) if device else ids
                else:
                    rows, token_input = packed, ids
                if args.shared_stage_replay:
                    for _ in range(4):
                        warmed = decoder.from_input_ids(positions, token_input, rows)[0].cpu()
                        if decoder.input_variant_ready(32768):
                            break
                else:
                    residual, pre = embedding(token_input)
                    warmed = decoder(residual, pre, positions, token_input, rows)[0].cpu()
                if args.shared_stage_replay:
                    assert torch.isfinite(warmed).all()
                elif mode == 'host':
                    warm_reference = warmed
                else:
                    torch.testing.assert_close(warmed, warm_reference, rtol=0, atol=0)
                host.complete(ticket, 1)
            if args.shared_stage_replay:
                assert decoder.input_variant_ready(32768)
            else:
                assert decoder.prefix_groups == 3 and decoder.prefix_ready(32768)
            bind_worker_helpers(rank)

            original_projection = None
            native_state_oracle = args.state_reference_native_groups and (not args.speed_probe or args.qualify_positive)
            if (native_state_oracle or args.diagnose_local_query_state or args.state_reference_logical_mla
                    or args.state_reference_index_mirror
                    or args.state_reference_shared_main or args.state_reference_decode_metadata
                    or args.state_reference_post_collapse or args.state_reference_interlayer_collapse
                    or args.state_reference_feature_silu or args.state_reference_mirror_partition
                    or args.state_reference_paged_projection or args.state_reference_paged_gather
                    or args.state_reference_visible_prefix):
                # The numerical reference shares immutable weights and the
                # initial mutable allocations, but owns separate compiled
                # groups with the original packed MLA output chain.
                reference_stage = copy.copy(stage)
                reference_stage._modules = dict(stage._modules)
                if args.state_reference_visible_prefix:
                    reference_stage.decode_token_bound = None
                blocks = []
                for block in stage.layers:
                    cloned = copy.copy(block)
                    cloned._modules = dict(block._modules)
                    cloned.attention = copy.copy(block.attention)
                    cloned.attention._buffers = dict(block.attention._buffers)
                    if args.diagnose_local_query_state:
                        cloned.attention.tp4_local_index_queries = False
                    if args.state_reference_mirror_partition:
                        cloned.attention.tp4_mirror_selection = False
                    if args.state_reference_logical_mla:
                        assert cloned.attention.paged_mla_logical
                        cloned.attention.paged_mla_logical = False
                    if args.state_reference_post_collapse:
                        assert cloned.mhc_post_collapse
                        cloned.mhc_post_collapse = False
                    if args.state_reference_interlayer_collapse:
                        assert cloned.mhc_interlayer_collapse
                        cloned.mhc_interlayer_collapse = False
                    if args.state_reference_feature_silu:
                        cloned.moe = copy.copy(block.moe)
                        cloned.moe._modules = dict(block.moe._modules)
                        assert cloned.moe.feature_silu
                        cloned.moe.feature_silu = False
                    if args.state_reference_decode_metadata:
                        assert cloned.attention.shared_decode_metadata
                        cloned.attention.shared_decode_metadata = False
                    if args.state_reference_shared_main:
                        assert cloned.attention.shared_main_mla
                        cloned.attention.shared_main_mla = False
                    if args.state_reference_index_mirror:
                        assert cloned.attention.index_mirror_scores and stage.shared.index_mirror_valid
                        cloned.attention.index_mirror_scores = False
                    if args.state_reference_paged_gather or args.state_reference_paged_projection:
                        attribute = 'paged_mla_direct' if args.state_reference_paged_gather else 'paged_mla_projection'
                        assert getattr(cloned.attention, attribute)
                        setattr(cloned.attention, attribute, False)
                    if args.state_reference_visible_prefix:
                        cloned.attention.set_decode_visible_tokens(None)
                    blocks.append(cloned)
                reference_stage.layers = torch.nn.ModuleList(blocks)
                original_projection = CompiledStage(reference_stage, prepared_tp4=True, native_tp4=False)
                reset()
                selected = sampler(seed_hidden)
                value = int(selected.cpu()[0, 0])
                controls.upload([value], 16384)
                host.prepare_device_c1('chain', selected.view(1))
                ticket = host.prepare('chain', [value], defer_wait=True)
                packed = host.wait(ticket)
                rows = host.consume_device_c1('chain'), packed[1]
                residual, pre = embedding(selected.view(1))
                wanted = original_projection(residual, pre, positions, selected.view(1), rows)[0].cpu()
                torch.testing.assert_close(wanted, warm_reference, rtol=0, atol=0)
                host.complete(ticket, 1)
                assert original_projection.prefix_ready(32768)

            def chain(device, steps, *, measure=True, native_positions=True, engine=None, warm_steps=0, trace=None):
                engine = decoder if engine is None else engine
                reset()
                selected = sampler(seed_hidden)
                readback = bridge.copy_sampled_tokens_to_host(selected)
                torch.hpu.synchronize()
                output_tokens = []
                device_start = torch.hpu.Event(enable_timing=True) if measure else None
                device_end = torch.hpu.Event(enable_timing=True) if measure else None
                started = None
                iteration_marks = []
                for index in range(warm_steps + steps):
                    if trace is not None and index == warm_steps:
                        torch.hpu.synchronize()
                        dist.barrier()
                        trace.start()
                    if measure and index == warm_steps:
                        started = time.perf_counter_ns()
                        device_start.record()
                        iteration_marks.append(time.perf_counter_ns())
                    if trace is not None and index >= warm_steps:
                        from vllm_gaudi.ops.deepseek_v41_native_trace import scope
                        context = scope(f'v41::real_selection_chain::step{index-warm_steps}::rank{rank}')
                    else:
                        context = nullcontext()
                    with context:
                        position = 16384 + index
                        if device:
                            if native_positions:
                                bank.copy_into(positions, position)
                            else:
                                positions.copy_(bank.view(position, 1))
                            host.prepare_device_c1('chain', selected.view(1))
                            token_input = selected.view(1)
                            if args.shared_stage_replay:
                                prefix_started = engine.input_variant_ready(stage.search_length)
                                if prefix_started:
                                    engine.begin_segmented_from_input_ids(positions, token_input)
                            else:
                                residual, pre = embedding(token_input)
                                residual, pre = engine.prefix(residual, pre, positions, token_input,
                                                               (host.device_rows, host.prefix_late_placeholder))
                        cpu, done = readback
                        done.synchronize()
                        value = int(cpu[0, 0])
                        output_tokens.append(value)
                        ticket = host.prepare('chain', [value], defer_wait=True, device_layer1=device)
                        packed = host.wait(ticket)
                        if device:
                            rows = host.consume_device_c1('chain'), packed[1]
                            if args.shared_stage_replay:
                                hidden = (engine.finish_segmented(positions, token_input, rows)
                                          if prefix_started else engine.from_input_ids(positions, token_input, rows))[0]
                            else:
                                hidden = engine.suffix(residual, pre, positions, token_input, rows)[0]
                        else:
                            controls.upload([value], position)
                            residual, pre = embedding(ids)
                            hidden = engine(residual, pre, positions, ids, packed)[0]
                        selected = sampler(hidden)
                        readback = bridge.copy_sampled_tokens_to_host(selected)
                        host.complete(ticket, 1)
                    if measure and index >= warm_steps:
                        iteration_marks.append(time.perf_counter_ns())
                if measure:
                    device_end.record()
                cpu, done = readback
                done.synchronize()
                output_tokens.append(int(cpu[0, 0]))
                if measure:
                    device_end.synchronize()
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    report['candidate_submission_intervals_ms'] = [
                        (stop - start) / 1e6
                        for start, stop in zip(iteration_marks[:-1], iteration_marks[1:], strict=True)]
                    report['candidate_final_drain_ms'] = (time.perf_counter_ns() - iteration_marks[-1]) / 1e6
                    return output_tokens, elapsed / steps, device_start.elapsed_time(device_end) / steps
                torch.hpu.synchronize()
                return output_tokens, None, None

            def state_fingerprints():
                fingerprints = []
                for tensor in stage_state_tensors(stage):
                    value = tensor.cpu().contiguous()
                    fingerprints.append(dict(shape=list(value.shape), dtype=str(value.dtype),
                                             sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()))
                return fingerprints

            if args.diagnose_local_query_state:
                names = {id(value): name for name, value in stage.named_buffers()}
                state_names = [names[id(value)] for value in stage_state_tensors(stage)]
                reference_tokens, _, _ = chain(True, args.steps, measure=False, engine=original_projection,
                                                warm_steps=args.continuous_warm_steps)
                reference_state = [value.cpu().clone() for value in stage_state_tensors(stage)]
                actual_tokens, _, _ = chain(True, args.steps, measure=False, warm_steps=args.continuous_warm_steps)
                actual_state = [value.cpu() for value in stage_state_tensors(stage)]
                differences = []
                for name, wanted, actual in zip(state_names, reference_state, actual_state, strict=True):
                    if not torch.equal(wanted, actual):
                        changed = wanted != actual
                        rows = changed.reshape(changed.shape[0], -1).any(-1).nonzero().flatten().tolist()
                        differences.append(dict(name=name, elements=int(changed.sum()), rows=rows[:256],
                                                changed_rows=len(rows)))
                        torch.save(dict(reference=wanted, actual=actual),
                                   output.with_name(f'local-query-difference-rank{rank}-{name}.pt'))
                # Preserve the full pools, including the null-page scratch.
                # These diagnose cross-process hashes without hiding any byte
                # from the identical-buffer comparison above.
                torch.save(dict(names=state_names, tensors=actual_state),
                           output.with_name(f'local-query-state-rank{rank}.pt'))
                report.update(status='local_query_state_diagnostic_complete_not_performance',
                              tokens_exact=actual_tokens == reference_tokens, tokens=actual_tokens,
                              state_differences=differences, state_names=state_names,
                              no_timing_collected=True)
                output.write_text(json.dumps(report, indent=2) + '\n')
                assert actual_tokens == reference_tokens and not differences
                return

            if args.diagnose_state:
                names = {id(value): name for name, value in stage.named_buffers()}
                report['state_names'] = [names[id(value)] for value in stage_state_tensors(stage)]
                archived = json.loads((args.reuse_reference / f'continuation-rank{rank}.json').read_text())
                diagnostic = {}
                for label, use_native in (('native', True), ('original', False)):
                    tokens, _, _ = chain(True, args.steps, measure=False, native_positions=use_native)
                    fingerprints = state_fingerprints()
                    values = tuple(value.cpu().contiguous() for value in stage_state_tensors(stage))
                    torch.save(values, output.with_name(f'state-{label}-rank{rank}.pt'))
                    diagnostic[label] = dict(tokens=tokens, fingerprints=fingerprints,
                                             archived_state_equal=fingerprints == archived['final_state_fingerprints'])
                    report['state_diagnostic'] = diagnostic
                    output.write_text(json.dumps(report, indent=2) + '\n')
                report['position_state_equal'] = (diagnostic['native']['fingerprints'] ==
                                                   diagnostic['original']['fingerprints'])
                report['position_tokens_equal'] = diagnostic['native']['tokens'] == diagnostic['original']['tokens']
                report['status'] = 'state_diagnostic_complete_not_performance'
                return

            def preparation_counts():
                if args.shared_stage_replay:
                    from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats
                    stats = prepared_group_stats()
                    return [stats['prepares'], stats['native_captures']]
                return [chunk.preparations for chunk in decoder.chunks]

            before = preparation_counts()
            if args.state_reference_native_groups:
                from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats
                report['native_groups_before_timing'] = prepared_group_stats()
            if args.measure_continuous_reference:
                assert args.measure_continuous_reference and original_projection is not None
                expected, baseline_ms, baseline_device_ms = chain(
                    True, args.steps, engine=original_projection, warm_steps=args.continuous_warm_steps)
                expected_state = state_fingerprints()
                if args.reuse_reference:
                    archived = json.loads((args.reuse_reference / f'continuation-rank{rank}.json').read_text())
                    assert expected[:len(archived['tokens'])] == archived['tokens']
                report.update(
                    baseline_measured=True, reference_timing_reused=False,
                    historical_correctness_prefix_reused=str(args.reuse_reference),
                    reference_reason='Saved 32-step periods restart after bulk state restore and include a large '
                                     'nonstationary prefix. They cannot resolve steady complete-chain gains. '
                                     'One missing component reference uses this continuous boundary; no full B.',
                    continuous_warm_steps=args.continuous_warm_steps,
                    reference_submission_intervals_ms=report.pop('candidate_submission_intervals_ms'),
                    reference_final_drain_ms=report.pop('candidate_final_drain_ms'))
            elif args.reuse_reference:
                archived = json.loads((args.reuse_reference / f'continuation-rank{rank}.json').read_text())
                if archived['status'] != 'component_exact':
                    completed = json.loads((args.reuse_reference / 'CONTRACT_COMPLETION.json').read_text())
                    assert completed['status'] == 'component_qualified_with_contract21'
                    assert archived['exact_final_state']
                    assert completed['full_model_validation_pending']
                assert archived['steps'] == args.steps
                assert archived.get('continuous_warm_steps', 0) == args.continuous_warm_steps
                report['continuous_warm_steps'] = args.continuous_warm_steps
                baseline_ms = archived['candidate_ms_per_token']
                baseline_device_ms = archived['candidate_device_ms_per_token']
                expected = archived['tokens']
                expected_state = archived.get('final_state_fingerprints')
                if (native_state_oracle or args.state_reference_logical_mla or args.state_reference_index_mirror
                        or args.state_reference_shared_main or args.state_reference_decode_metadata
                    or args.state_reference_post_collapse or args.state_reference_interlayer_collapse
                    or args.state_reference_feature_silu or args.state_reference_mirror_partition
                        or args.state_reference_paged_projection
                        or args.state_reference_paged_gather
                        or args.state_reference_visible_prefix):
                    oracle, _, _ = chain(True, args.steps, measure=False, engine=original_projection,
                                          warm_steps=args.continuous_warm_steps)
                    assert oracle == expected
                    report['archived_original_state_fingerprints'] = expected_state
                    expected_state = state_fingerprints()
                    report['state_oracle'] = 'same initial buffers; selected attention optimization disabled; untimed'
                    if args.state_reference_native_groups:
                        report['state_oracle'] = 'same initial buffers and math; ordinary compiled groups; untimed'
                    if args.state_reference_mirror_partition:
                        report['state_oracle'] = 'same initial buffers; original full mirror scoring/selection; untimed'
                elif args.state_reference_original_copy:
                    oracle, _, _ = chain(True, args.steps, measure=False, native_positions=False)
                    assert oracle == expected
                    current_state = state_fingerprints()
                    report['archived_original_state_equal'] = current_state == expected_state
                    report['archived_original_state_fingerprints'] = expected_state
                    expected_state = current_state
                    report['state_oracle'] = 'same initial buffers; original position copy; untimed'
                elif expected_state is None:
                    # Older archives preserved tokens and a comparison flag but
                    # not the mutable state. Reconstruct only the missing state
                    # oracle, without collecting a new baseline measurement.
                    oracle, _, _ = chain(False, args.steps, measure=False)
                    assert oracle == expected
                    expected_state = state_fingerprints()
                report.update(reference_reused=str(args.reuse_reference), baseline_measured=False,
                              state_oracle_reconstructed=(native_state_oracle or
                                                          args.state_reference_mirror_partition or
                                                          args.state_reference_feature_silu or
                                                          args.state_reference_post_collapse or
                                                          args.state_reference_interlayer_collapse or
                                                          args.state_reference_decode_metadata or
                                                          args.state_reference_shared_main or
                                                          args.state_reference_logical_mla or
                                                          args.state_reference_index_mirror or
                                                          args.state_reference_visible_prefix or
                                                          args.state_reference_paged_gather or
                                                          args.state_reference_paged_projection or
                                                          args.state_reference_original_copy or
                                                          'final_state_fingerprints' not in archived))
            else:
                expected, baseline_ms, baseline_device_ms = chain(False, args.steps)
                expected_state = state_fingerprints()
            # A separate numerical oracle runs different recipes. Restore the
            # candidate's steady execution before timing, matching archives
            # whose same-graph oracle already supplied these warm steps.
            if not args.continuous_warm_steps:
                warmed, _, _ = chain(True, args.steps, measure=False)
                assert warmed == expected
                assert state_fingerprints() == expected_state
                report['candidate_warm_steps_after_oracle'] = args.steps
            if args.selection_trace_only:
                from vllm_gaudi.ops.deepseek_v41_native_trace import NativeTrace
                trace = NativeTrace(cpu_trace_dir=output.parent / f'traces/rank{rank}', scope_only=True)
                try:
                    traced, _, _ = chain(True, 8, measure=False, warm_steps=args.continuous_warm_steps, trace=trace)
                finally:
                    if trace.running:
                        trace.stop()
                assert traced == expected[:len(traced)]
                assert before == preparation_counts()
                report.update(status='real_chain_selection_trace_complete', trace_steps=8,
                              trace_metadata=trace.metadata, no_timing_collected=True,
                              tokens=traced, no_hot_compilation=True, final_state_gate_pending=True,
                              model_quality_pending=True, gain_ledger_credit=False)
                output.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(dict(rank=rank, status=report['status'], trace_steps=8)), flush=True)
                return
            actual, candidate_ms, candidate_device_ms = chain(
                True, args.steps, warm_steps=args.continuous_warm_steps)
            if stage.runtime_indexer:
                # The algorithm reference is TP2 C1. Generic TP4 TopK ties
                # and its shard-add order are not a numerical acceptance gate.
                all_tokens = [None] * dist.get_world_size()
                dist.all_gather_object(all_tokens, actual)
                assert all(tokens == actual for tokens in all_tokens), "TP ranks disagree on generated tokens"
                report.update(selection_reference='TP2 C1 ordered threshold/emit; actual local head geometry',
                              generic_tp4_tokens_equal=actual == expected, four_rank_tokens_equal=True,
                              model_quality_pending=True)
            else:
                assert actual == expected, (actual, expected)
            speed_positive = False
            if args.qualify_positive:
                # Decide after timing, identically on every rank, before the
                # expensive wider-bucket and allocation-lifetime checks.
                decision = torch.tensor([candidate_ms < baseline_ms], device='hpu', dtype=torch.int32)
                dist.all_reduce(decision, op=dist.ReduceOp.MIN)
                speed_positive = bool(decision.cpu().item())
                report['speed_gate_positive_all_ranks'] = speed_positive
            if args.speed_probe and not speed_positive:
                from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats
                native_stats = prepared_group_stats()
                if args.state_reference_native_groups or args.shared_stage_replay:
                    assert native_stats['native_replays'] > 0
                after = preparation_counts()
                assert before == after, ('hot compilation', before, after)
                report.update(status='speed_probe_exact_tokens_state_gate_pending',
                              candidate_ms_per_token=candidate_ms, candidate_device_ms_per_token=candidate_device_ms,
                              baseline_ms_per_token=baseline_ms, baseline_device_ms_per_token=baseline_device_ms,
                              steps=args.steps, tokens=actual, no_hot_compilation=True,
                              native_groups=native_stats, final_state_gate_pending=True, lifecycle_gate_pending=True)
                print(json.dumps(report), flush=True)
                return
            observed_state = state_fingerprints()
            report.update(candidate_ms_per_token=candidate_ms, candidate_device_ms_per_token=candidate_device_ms,
                          tokens=actual, observed_state_fingerprints=observed_state,
                          archived_state_fingerprints=expected_state)
            output.write_text(json.dumps(report, indent=2) + '\n')
            assert observed_state == expected_state, 'exact final-state fingerprints differ'
            after = [chunk.preparations for chunk in decoder.chunks]
            if args.state_reference_native_groups:
                report['native_groups_after_timing'] = prepared_group_stats()
                assert report['native_groups_after_timing']['native_replays'] > 0
            assert before == after, ('unexpected hot compilation', before, after)
            if args.state_reference_mirror_partition:
                report['mirror_partition_layers'] = [block.layer for block in stage.layers
                                                       if block.attention.tp4_mirror_selection]
                assert report['mirror_partition_layers']
            report.update(status='c1_exact_buckets_pending', steps=args.steps, tokens=actual,
                          baseline_ms_per_token=baseline_ms, candidate_ms_per_token=candidate_ms,
                          baseline_device_ms_per_token=baseline_device_ms,
                          candidate_device_ms_per_token=candidate_device_ms,
                          device_timer='stream event span including exposed host submission gaps',
                          final_state_fingerprints=observed_state,
                          no_hot_compilation=True, exact_final_state=True)
            output.write_text(json.dumps(report, indent=2) + '\n')
            # The ordinary B>1 contract remains shared with the same decoder.
            # Rebind state once to exercise invalidation, then reuse the slot.
            reset()
            old_states = {id(value): value for value in stage_state_tensors(stage)}
            fresh_states = {key: value.clone() for key, value in old_states.items()}
            retired = [(value, value.cpu()) for value in old_states.values()]
            for name, value in list(stage.named_buffers(remove_duplicate=False)):
                if id(value) in fresh_states:
                    parent, _, leaf = name.rpartition('.')
                    setattr(stage.get_submodule(parent), leaf, fresh_states[id(value)])
            stage.generation += 1
            assert not decoder.prefix_ready(32768)
            reference = CompiledStage(stage, prepared_tp4=False)
            cases = [(2, None), (6, None)]
            if args.state_reference_visible_prefix or args.production_visible_prefix:
                cases.extend((1, bound) for bound in (20480, 24576, 20480))
            elif args.state_reference_paged_projection or args.state_reference_paged_gather:
                cases.append((1, None))
            for count, visible_bound in cases:
                if visible_bound is not None:
                    stage.decode_token_bound = visible_bound
                    for block in stage.layers:
                        block.attention.set_decode_visible_tokens(visible_bound)
                local_ids = torch.arange(17, 17 + count, dtype=torch.int64, device='hpu')
                local_positions = torch.arange(16384, 16384 + count, dtype=torch.int32, device='hpu')
                ticket = host.prepare('chain', list(range(17, 17 + count)), defer_wait=True)
                rows = host.wait(ticket)
                residual, pre = embedding(local_ids)
                candidate_residual, candidate_pre = residual.clone(), pre.clone()
                saved = _Snapshot(stage_state_tensors(stage))
                wanted = reference(residual, pre, local_positions, local_ids, rows)[0].cpu()
                wanted_states = [value.cpu() for value in stage_state_tensors(stage)]
                saved.restore()
                bank.copy_into(local_positions, 16384)
                observed = decoder(candidate_residual, candidate_pre, local_positions, local_ids, rows)[0].cpu()
                torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
                for observed_state, wanted_state in zip(stage_state_tensors(stage), wanted_states, strict=True):
                    torch.testing.assert_close(observed_state.cpu(), wanted_state, rtol=0, atol=0)
                host.complete(ticket, count)
                report['cases'].append(dict(tokens=count, exact_output=True, exact_state=True,
                                            decode_token_bound=getattr(stage, 'decode_token_bound', None),
                                            contract='physical_state_rebind_and_slot_reuse'))
                host.reset('chain')
                host.history.history = initial_history.copy()
                host.history.position = 16384
                saved.restore()
                del saved, wanted_states
            for value, wanted in retired:
                torch.testing.assert_close(value.cpu(), wanted, rtol=0, atol=0)
            if args.state_reference_index_mirror:
                from vllm_gaudi.ops.deepseek_v41_state import PagedStageState
                # These untimed ownership cases run after the scored chain.
                # Grow only this fixture's physical pool so both independent
                # requests and the first token above 32K have real live pages.
                old_pools = [(cache, cache.main, cache.index) for cache in stage.shared.sources.values()]
                stage.replay_owner = None
                pages = PagedStageState(stage)
                pages.allocate(514, 'hpu')
                for cache, main, index in old_pools:
                    cache.main[:main.shape[0]].copy_(main)
                    cache.index[:index.shape[0]].copy_(index)
                del old_pools
                oracle_stage = copy.copy(stage)
                oracle_stage._modules = dict(stage._modules)
                blocks = []
                for block in stage.layers:
                    cloned = copy.copy(block)
                    cloned._modules = dict(block._modules)
                    cloned.attention = copy.copy(block.attention)
                    cloned.attention._buffers = dict(block.attention._buffers)
                    cloned.attention.index_mirror_scores = False
                    blocks.append(cloned)
                oracle_stage.layers = torch.nn.ModuleList(blocks)
                oracle = CompiledStage(oracle_stage, prepared_tp4=True)
                report['mirror_lifecycle_cases'] = []
                transitions = [
                    ('first_request', 'a', list(range(1, 257)), 16384, 32768),
                    ('different_request', 'b', list(range(257, 513)), 16384, 32768),
                    ('request_reuse', 'a', list(range(1, 257)), 16384, 32768),
                    ('page_remap', 'a', list(range(256, 0, -1)), 16384, 32768),
                    ('last_bounded_token', 'a', list(range(1, 257)), 32767, 32768),
                    ('first_unbounded_token', 'a', list(range(1, 258)), 32768, 65536),
                    ('return_to_bounded', 'a', list(range(1, 257)), 16384, 32768),
                ]
                for label, request_id, block_ids, position, search in transitions:
                    pages.activate(request_id, block_ids)
                    bound = decode_source_prefix_bound(position + 1, search, 4)
                    stage.search_length = oracle_stage.search_length = search
                    stage.decode_token_bound = oracle_stage.decode_token_bound = bound
                    for block in (*stage.layers, *oracle_stage.layers):
                        block.attention.set_search_length(search)
                        block.attention.set_decode_visible_tokens(bound)
                    if search <= stage.shared.index_mirror_tokens:
                        stage.shared.prepare_index_mirror(position)
                    else:
                        stage.shared.invalidate_index_mirror()
                    local_ids = torch.tensor([17], dtype=torch.int64, device='hpu')
                    local_positions = torch.tensor([position], dtype=torch.int32, device='hpu')
                    ticket = host.prepare('chain', [17], defer_wait=True)
                    rows = host.wait(ticket)
                    residual, pre = embedding(local_ids)
                    candidate_residual, candidate_pre = residual.clone(), pre.clone()
                    saved = _Snapshot(stage_state_tensors(stage))
                    wanted = oracle(residual, pre, local_positions, local_ids, rows)[0].cpu()
                    wanted_states = [value.cpu() for value in stage_state_tensors(stage)]
                    saved.restore()
                    if search <= stage.shared.index_mirror_tokens:
                        stage.shared.invalidate_index_mirror()
                        for cache in stage.shared.sources.values():
                            cache.index_mirror.fill_(17)
                        stage.shared.prepare_index_mirror(position)
                    # No host wait between the restore producer and decoder.
                    bank.copy_into(local_positions, position)
                    observed = decoder(candidate_residual, candidate_pre, local_positions, local_ids, rows)[0].cpu()
                    torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
                    for value, expected_value in zip(stage_state_tensors(stage), wanted_states, strict=True):
                        torch.testing.assert_close(value.cpu(), expected_value, rtol=0, atol=0)
                    report['mirror_lifecycle_cases'].append(dict(case=label, position=position, search=search,
                                                                  exact_output=True, exact_state=True))
                    host.complete(ticket, 1)
                    host.reset('chain')
                    host.history.history = initial_history.copy()
                    host.history.position = 16384
                    saved.restore()
                    del saved, wanted_states
            assert bank.copy_preparations == position_preparations
            report.update(status='component_exact', steps=args.steps, tokens=actual,
                          baseline_ms_per_token=baseline_ms, candidate_ms_per_token=candidate_ms,
                          no_hot_compilation=True, exact_final_state=True, engram=dict(host.audit),
                          position_copy_preparations=bank.copy_preparations,
                          allocated_bytes=torch.hpu.memory_allocated(), peak_bytes=torch.hpu.max_memory_allocated())
            print(json.dumps(report), flush=True)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2)+'\n')
        if args.state_reference_native_groups or args.speed_probe:
            from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans
            shutdown_prepared_group_plans()
        if host is not None:
            host.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
