# SPDX-License-Identifier: Apache-2.0
"""Four-rank canonical writes -> Full/Reindex -> real publish/reuse MLA.

The launcher must hold all module leases. Component timing includes query
exchange, derived-key maintenance and both downstream attention consumers.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import FunctionType, SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capacity', type=int, default=524288)
    parser.add_argument('--end', type=int, default=82944)
    parser.add_argument('--search', type=int, default=131072)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    os.environ['VLLM_HPU_DSV41_GRAPH_REPLAY'] = '1'
    os.environ['VLLM_HPU_TP2_NATIVE_JOINT_PLAN'] = '1'
    import torch
    import torch.distributed as dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_config import decode_source_prefix_bound
    from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select
    from vllm_gaudi.ops.deepseek_v41_index_mirror import initialize_index_mirror, prepare_index_mirror
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, fp4_roundtrip
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
    from vllm_gaudi.ops.tp2_prepared_plan import (
        register_tp2_prepared_group_pass, collect_prepared_group_replays, replay_native_decoder,
    )
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology

    if not 32768 < args.end <= args.search <= args.capacity or args.search % 128 or args.steps < 200:
        raise ValueError('Expected an aligned long-context geometry and at least 200 timed steps')
    torch.set_num_threads(1)
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    bind_worker_helpers(rank)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://', local_rank=rank,
                                 backend='hccl')
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(status='preparing', rank=rank, scope=__doc__, end=args.end, search=args.search,
                  capacity=args.capacity, cases=[], full_model_qualified=False)

    def save():
        (args.output / f'rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    with set_current_vllm_config(config), torch.inference_mode():
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
        initialize_tp2_fused_ar_norm_runtime()
        register_tp2_prepared_group_pass()
        _, gather = stage_collectives(rank, True, 4)
        ops = torch.ops.custom_op
        for ratio in (1, 2):
            generator = torch.Generator().manual_seed(8791 + ratio)
            page_rows = 128 // ratio
            shared = torch.nn.Module()
            shared.length = args.capacity
            pages_cpu = (torch.randperm(args.capacity // 128, generator=generator) + 1).int()
            shared.register_buffer('block_table', pages_cpu.to('hpu'))
            cache = torch.nn.Module()
            cache.ratio = ratio
            packed_cpu = pack_fp4(torch.randn(args.capacity // ratio + page_rows, 128,
                                              generator=generator).bfloat16(), 32)
            main_cpu = torch.randint(0, 256, (args.capacity // ratio + page_rows, 288),
                                     generator=generator, dtype=torch.uint8)
            main_cpu[:, 256:] = 120
            cache.register_buffer('index', packed_cpu.to('hpu'))
            cache.register_buffer('main', main_cpu.to('hpu'))
            shared.sources = torch.nn.ModuleDict({'source': cache})
            initialize_index_mirror(shared, 4, 'hpu')
            prepare_index_mirror(shared, args.end - 1)
            rings = []
            for _ in range(2):
                ring = torch.randint(0, 127, (256, 528), generator=generator, dtype=torch.uint8)
                ring[:, 512:] = 119
                rings.append(ring.to('hpu'))
            banks = []
            for _ in range(3):
                full_query = fp4_roundtrip(torch.randn(1, 32, 128, generator=generator).bfloat16() / 32, 32)
                gains = (torch.randn(1, 32, generator=generator) * .02).bfloat16()
                mla = (torch.randn(1, 64, 512, generator=generator) / 32).bfloat16()
                index = torch.randn(1, 128, generator=generator).bfloat16()
                latent = torch.randn(1, 512, generator=generator).bfloat16()
                banks.append(tuple(t.to('hpu') for t in (
                    full_query[:, rank * 8:(rank + 1) * 8].contiguous(),
                    gains[:, rank * 8:(rank + 1) * 8].contiguous(),
                    mla[:, rank * 16:(rank + 1) * 16].contiguous(), index, latent)))
            position = torch.tensor([args.end - 1], dtype=torch.int32).to('hpu')
            logical = torch.tensor([(args.end - 1) // ratio], dtype=torch.int64).to('hpu')
            physical = (pages_cpu[logical.cpu() // page_rows].long() * page_rows + logical.cpu() % page_rows).to('hpu')
            pool = torch.full((1, 2048), -1, dtype=torch.int32, device='hpu')
            sink = torch.zeros(16, dtype=torch.float32, device='hpu')
            scale = torch.tensor([512 ** -.5], device='hpu')
            lengths = torch.tensor([640], dtype=torch.int32, device='hpu')
            visible = decode_source_prefix_bound(args.end, args.search, 4) // ratio
            fixed = (cache.index, cache.main, shared.block_table, cache.index_mirror,
                     *rings, position, logical, physical, pool, sink, scale, lengths)

            def chain(q, gains, mla, index, latent, packed, main, pages, mirror, swa0, swa1,
                      pos, logical, physical, pool, sink, scale, lengths, use_mirror,
                      ratio=ratio, visible=visible, search=args.search):
                q, gains = gather(q, 1), gather(gains, 1)
                packed.index_copy_(0, physical, pack_fp4(index, 32))
                main.index_copy_(0, physical, pack_fp4(latent, 16))
                if use_mirror:
                    mirror.index_copy_(0, logical, fp4_roundtrip(index, 32))
                columns = visible if use_mirror else search // ratio
                ids, blocks = runtime_index_select(
                    q, gains, packed, pages, pos, pool, ratio=ratio, capacity=search // ratio,
                    local_heads=8, search_rows=columns, decoded_keys=mirror if use_mirror else None,
                    publish_candidates=True, ordered_candidates=True)
                reused, _ = runtime_index_select(
                    q, gains, packed, pages, pos, blocks, ratio=ratio, capacity=search // ratio,
                    local_heads=8, search_rows=columns, decoded_keys=mirror if use_mirror else None,
                    reindex=True, ordered_candidates=True)
                first, decoded, mask = ops.custom_deepseek_v41_main_publish_mla_gaudi2(
                    mla, swa0, main, reused, pos, pages, sink, scale, lengths, ratio)
                output = ops.custom_deepseek_v41_main_reuse_mla_gaudi2(
                    first, swa1, decoded, mask, pos, sink, scale, lengths)
                return ids, blocks, reused, output

            arms = []
            for mode in (False, True):
                entry = FunctionType(chain.__code__.replace(co_name=f'long_mirror_{ratio}_{mode}'),
                                     chain.__globals__, argdefs=chain.__defaults__, closure=chain.__closure__)
                arms.append(torch.compile(entry, backend=make_backend(), fullgraph=True, dynamic=False))

            class Snapshot:
                def __init__(self, cache=cache, physical=physical, logical=logical):
                    self.rows = [(cache.index, physical), (cache.main, physical), (cache.index_mirror, logical)]
                    self.saved = [value.index_select(0, index).clone() for value, index in self.rows]

                def restore(self):
                    for (value, index), saved in zip(self.rows, self.saved, strict=True):
                        value.index_copy_(0, index, saved)

            adapter = DecoderTopology('deepseek_v41_index_micro', (1,), 0, False, 2)
            owners = [torch.nn.Module(), torch.nn.Module()]
            for owner in owners:
                owner.generation = 0
            staged = [[value.clone() for value in banks[0]] for _ in owners]

            def invoke(arm, bank, arms=arms, fixed=fixed, owners=owners, staged=staged,
                       snapshot=Snapshot, adapter=adapter):
                q, w, mla, index, latent = bank
                owner = owners[arm]
                metadata = SimpleNamespace(native_completion=None,
                                           inputs=dict(q=q, w=w, mla=mla, index=index, latent=latent))
                roots = dict(adapter=adapter, owner=owner, snapshot=snapshot, metadata=metadata,
                             state_generation=0,
                             state_tensors=(fixed[0], fixed[1], fixed[3]) if arm else (fixed[0], fixed[1]))
                ready = replay_native_decoder(owner, **{k: v for k, v in roots.items() if k != 'owner'})
                if ready is not None:
                    return ready
                # Discovery binds dedicated input buffers, never immutable
                # fixture banks that later replay staging would overwrite.
                for destination, source in zip(staged[arm], bank, strict=True):
                    destination.copy_(source)
                q, w, mla, index, latent = staged[arm]
                metadata.inputs = dict(q=q, w=w, mla=mla, index=index, latent=latent)
                with collect_prepared_group_replays(**roots) as context:
                    result = arms[arm](q, w, mla, index, latent, *fixed, bool(arm))
                    context['outputs'] = result
                return result

            checks = []
            case = dict(ratio=ratio, visible_rows=visible, checks=checks, periods=[])
            report['cases'].append(case)
            for bank in banks:
                old, new = invoke(0, bank), invoke(1, bank)
                torch.hpu.synchronize()
                exact = [torch.equal(a.cpu(), b.cpu()) for a, b in zip(old, new, strict=True)]
                checks.append(exact)
                save()
                assert all(exact), (ratio, exact)
            bind_worker_helpers(rank)
            save()
            for _ in range(1800):
                if (args.output / 'timing-ready.json').exists():
                    break
                time.sleep(1)
            else:
                raise TimeoutError('Timing lease did not become ready')
            for arm in (0, 1, 0, 1, 0, 1):
                dist.barrier()
                torch.hpu.synchronize()
                host, device = [], []
                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                for step in range(args.steps):
                    started = time.perf_counter_ns()
                    begin.record()
                    invoke(arm, banks[step % 3])
                    end.record()
                    end.synchronize()
                    host.append((time.perf_counter_ns() - started) / 1e6)
                    device.append(begin.elapsed_time(end))
                q1, _, q3 = statistics.quantiles(host, n=4, method='inclusive')
                case['periods'].append(dict(arm=arm, wall_ms=host, device_ms=device,
                                            median_ms=statistics.median(host), iqr_ms=q3 - q1))
                save()
            combined = [[v for p in case['periods'] if p['arm'] == arm for v in p['wall_ms']] for arm in (0, 1)]
            a, b = [statistics.median(v) for v in combined]
            q1, _, q3 = statistics.quantiles(combined[0], n=4, method='inclusive')
            case.update(baseline_ms=a, candidate_ms=b, delta_ms=a - b, gate_ms=2 * (q3 - q1),
                        effective=a - b > 2 * (q3 - q1))
            save()
        report['status'] = 'completed'
        save()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
