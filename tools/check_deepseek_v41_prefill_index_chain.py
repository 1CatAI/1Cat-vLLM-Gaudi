# SPDX-License-Identifier: Apache-2.0
"""Four-rank prefill Q exchange -> Full -> Reindex -> actual MLA consumer.

Run under a launcher that holds the four module leases. This is a synthetic
component, not full-model performance or output-quality qualification.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--source-rows", type=int, default=65536)
    parser.add_argument("--capacity", type=int, default=524288)
    parser.add_argument("--visible", type=int, default=62464)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--mla-only", action="store_true")
    parser.add_argument("--paged-mla", action="store_true",
                        help="Include bounded selected-KV decode inside the query-owner MLA recipe")
    parser.add_argument("--mla-ratio", type=int, choices=(1, 2), default=1)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    import torch
    import torch.distributed as dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel, get_tp_group
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention, compiled_flash_prefill_mla
    from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import (
        full_prefill_index_selection, tp_full_prefill_index_selection, tp_prefill_reindex_selection,
    )
    from vllm_gaudi.ops.deepseek_v41_prefill_index_exchange import exchange_prefill_index_queries
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_prefill_sequence import sequence_prefill_mla
    from vllm_gaudi.ops.deepseek_v41_prefill_mla import sparse_prefill_mla

    if (args.tokens < 1024 or args.tokens % 4
            or not args.tokens <= args.visible <= args.source_rows <= args.capacity):
        raise ValueError("Expected finite large-M queries and a bounded source prefix")
    if args.paged_mla and not args.mla_only:
        raise ValueError("Paged MLA qualification uses the MLA-only complete producer/consumer chain")
    if args.paged_mla and args.source_rows > args.capacity // args.mla_ratio:
        raise ValueError("Paged MLA source exceeds its compressed context capacity")
    torch.set_num_threads(1)
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    bind_worker_helpers(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://",
                                 local_rank=rank, backend="hccl")
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
    report = dict(status="preparing", rank=rank, scope=__doc__, tokens=args.tokens,
                  source_rows=args.source_rows, capacity=args.capacity, visible=args.visible,
                  mla_only=args.mla_only, paged_mla=args.paged_mla, mla_ratio=args.mla_ratio,
                  checks_only=args.checks_only, checks=[], periods=[],
                  baseline_reason="No archived four-rank large-M long-prefix producer/consumer measurement",
                  full_model_requests=0, ledger_credit=False)
    args.output.mkdir(parents=True, exist_ok=True)

    def save():
        (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    with set_current_vllm_config(config), torch.inference_mode():
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
        group = get_tp_group()
        _, gather = stage_collectives(rank, False, 4)
        generator = torch.Generator().manual_seed(57131)
        # Identical canonical cache across ranks; distinct head shards below.
        packed = torch.cat((torch.zeros(128, 68, dtype=torch.uint8),
                            pack_fp4(torch.randn(args.capacity, 128, generator=generator).bfloat16(), 32)), 0)
        main = pack_fp4(torch.randn(args.source_rows, 512, generator=generator).bfloat16(), 16)
        table = (torch.randperm(args.capacity // 128, generator=generator) + 1).int().to("hpu")
        packed = packed.to("hpu")
        swa_rows = args.tokens + 127
        swa = torch.randn(swa_rows, 512, generator=generator).bfloat16()
        cache = torch.cat((swa, unpack_fp4(main, 512, 16)), 0).to("hpu")
        paged_main = None
        if args.paged_mla:
            page_rows = 128 // args.mla_ratio
            physical_main = torch.zeros(args.capacity // args.mla_ratio + page_rows, 288, dtype=torch.uint8)
            logical = torch.arange(args.source_rows)
            mapping = table.cpu()[logical // page_rows].long() * page_rows + logical % page_rows
            physical_main.index_copy_(0, mapping, main)
            paged_main = physical_main.to("hpu")
        del main
        positions = torch.arange(args.visible - args.tokens, args.visible, dtype=torch.int32).to("hpu")
        banks = []
        for _ in range(3):
            q = unpack_fp4(pack_fp4(torch.randn(args.tokens, 32, 128, generator=generator).bfloat16(), 32), 128, 32)
            weights = (torch.randn(args.tokens, 32, generator=generator) * .02).bfloat16()
            mla_q = torch.randn(args.tokens, 64, 512, generator=generator).bfloat16()
            banks.append((q[:, rank * 8:(rank + 1) * 8].contiguous().to("hpu"),
                          weights[:, rank * 8:(rank + 1) * 8].contiguous().to("hpu"),
                          mla_q[:, rank * 16:(rank + 1) * 16].contiguous().to("hpu")))
        full_sink = torch.linspace(-1, 1, 64, dtype=torch.float32).to("hpu")
        sink = full_sink[rank * 16:(rank + 1) * 16].contiguous()
        mla_choices = torch.randint(0, args.source_rows, (args.tokens, 512), generator=generator)
        mla_choices = mla_choices.remainder(positions.cpu()[:, None] + 1).sort(-1).values.int().to("hpu")

        def physical(rows, ratio=1):
            width = 128 // ratio
            return table.index_select(0, (rows.flatten() // width).long()).reshape(rows.shape) * width + rows % width

        owner = SimpleNamespace(tensor_parallel_size=4, index_heads=8, ratio=1,
                                cache=SimpleNamespace(index=packed), shared=SimpleNamespace(physical_rows=physical))
        owner._scores = lambda p, rows, q, w: PagedCSA2Attention._scores(owner, p, rows, q, w)
        owner._prefill_scores = owner._scores
        owner._merge_topk = PagedCSA2Attention._merge_topk

        def old_reindex(q, weights, blocks):
            outputs = []
            for start in range(0, args.tokens, 128):
                pool = blocks[start:start + 128]
                rows = pool[..., None] * 8 + torch.arange(8, device="hpu", dtype=torch.int32)
                rows = torch.where(pool[..., None] >= 0, rows, -1).flatten(1)
                p = positions[start:start + 128]
                ids, _, _ = PagedCSA2Attention._stream_topk(owner, p, rows, q[start:start + 128],
                                                          weights[start:start + 128], native_scores=True)
                count = (p + 1).unsqueeze(-1)
                ids = torch.where((ids >= 0) & (ids < count), ids, args.source_rows).sort(-1).values
                outputs.append(torch.where(ids < args.source_rows, ids, -1).int())
            return torch.cat(outputs)

        def attention_ids(choices):
            window = (torch.arange(args.tokens, device="hpu", dtype=torch.int32)[:, None]
                      + torch.arange(128, device="hpu", dtype=torch.int32)[None, :])
            return torch.cat((window, torch.where(choices >= 0, choices + swa_rows, -1)), -1).contiguous()

        def consume(query, choices):
            ids = attention_ids(choices)
            outputs = []
            for start in range(0, args.tokens, 2048):
                stop = min(start + 2048, args.tokens)
                signature = (stop-start, (16, 512), tuple(cache.shape), 640)
                outputs.append(compiled_flash_prefill_mla(signature)(
                    query[start:stop].contiguous(), cache, ids[start:stop].contiguous(), sink))
            return torch.cat(outputs)

        def old_paged_tile(query, packed, pages, swa, indices, sink, ratio):
            # Match the actual replicated selected-row producer used above
            # the flat-cache workspace budget, including generic FP4 decode.
            selected = indices[:, 128:]
            width = 128 // ratio
            logical = selected.clamp_min(0).long()
            physical = pages[(logical // width).flatten()].reshape(logical.shape).long() * width + logical % width
            main = unpack_fp4(packed.index_select(0, physical.flatten()), 512, 16)
            local = torch.arange(selected.numel(), dtype=torch.int32, device=query.device).reshape_as(selected)
            local = torch.where(selected >= 0, local + swa.shape[0], -1)
            cache = torch.cat((swa, main), 0)
            ids = torch.cat((indices[:, :128], local), -1)
            lengths = torch.full((query.shape[0],), 640, dtype=torch.int32, device=query.device)
            return sparse_prefill_mla(query, cache, ids, sink, lengths, query_tile=256)

        old_paged = torch.compile(old_paged_tile, backend="hpu_backend", fullgraph=True, dynamic=False)

        def paged_ids():
            ids = attention_ids(mla_choices)
            ids[:, 128:] = mla_choices
            return ids

        def chain(which, inputs):
            query, weights, mla_q = inputs
            if args.mla_only:
                if which == 0:
                    if args.paged_mla:
                        ids = paged_ids()
                        output = torch.cat([
                            old_paged(mla_q[start:start + 128].contiguous(), paged_main, table,
                                      cache[:swa_rows].contiguous(), ids[start:start + 128].contiguous(),
                                      sink, args.mla_ratio) for start in range(0, args.tokens, 128)
                        ], 0)
                    else:
                        output = consume(mla_q, mla_choices)
                elif args.paged_mla:
                    ids = paged_ids()
                    output = sequence_prefill_mla(
                        mla_q, cache[:swa_rows].contiguous(), ids, full_sink, rank,
                        group=group.device_group, retire_chunks=False,
                        paged_main=(paged_main, table, args.mla_ratio))
                else:
                    output = sequence_prefill_mla(
                        mla_q, cache, attention_ids(mla_choices), full_sink, rank,
                        group=group.device_group, retire_chunks=False)
                return (output,)
            if which == 0:
                q, w = gather(query, 1), gather(weights, 1)
                selected, blocks = full_prefill_index_selection(q, w, packed, table, positions, 1,
                                                                args.source_rows, True, args.visible, 4)
                downstream = old_reindex(q, w, blocks)
            else:
                local = exchange_prefill_index_queries(query, weights, rank, group=group.device_group)
                q, w = local.query, local.weights
                selected, blocks = tp_full_prefill_index_selection(
                    q, w, packed, table, positions, 1, args.source_rows, True, rank, gather,
                    args.visible, 4, query_partitioned=True)
                downstream = tp_prefill_reindex_selection(
                    q, w, packed, table, positions, blocks, 1, args.source_rows, rank, gather,
                    tensor_parallel_size=4, query_partitioned=True)
            return selected, blocks, downstream, consume(mla_q, downstream)

        for change, inputs in enumerate(banks):
            old, new = chain(0, inputs), chain(1, inputs)
            torch.hpu.synchronize()
            check = dict(change=change, exact=[torch.equal(a.cpu(), b.cpu()) for a, b in zip(old, new)])
            report["checks"].append(check)
            save()
            if not all(check["exact"]):
                report["status"] = "correctness_failed"
                save()
                raise RuntimeError(f"Selection/MLA contract differs: {check}")
        if args.checks_only:
            report["status"] = "correctness_passed_no_timing"
            save()
            dist.destroy_process_group()
            return
        bind_worker_helpers(rank)
        # The lease launcher watches competing memory growth and CPU pressure.
        # Preparation may overlap loading; timed periods must wait for its gate.
        for _ in range(1800):
            if (args.output / "timing-ready.json").exists():
                break
            time.sleep(1)
        else:
            raise TimeoutError("Competing workload did not reach the timing gate")
        dist.barrier(group=group.device_group)
        bind_worker_helpers(rank)
        report["thread_affinity"] = {
            task.name: sorted(os.sched_getaffinity(int(task.name)))
            for task in Path(f"/proc/{os.getpid()}/task").iterdir()
        }
        save()
        for arm in (0, 1, 0, 1, 0, 1):
            dist.barrier(group=group.device_group)
            torch.hpu.synchronize()
            host, device = [], []
            begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
            for step in range(args.steps):
                started = time.perf_counter_ns()
                begin.record()
                chain(arm, banks[step % len(banks)])
                end.record()
                end.synchronize()
                host.append((time.perf_counter_ns() - started) / 1e6)
                device.append(begin.elapsed_time(end))
                if step % 20 == 19:
                    report["progress"] = dict(arm=arm, steps=step+1, median_ms=statistics.median(host))
                    save()
            q1, _, q3 = statistics.quantiles(host, n=4, method="inclusive")
            report["periods"].append(dict(arm=arm, samples_ms=host, device_ms=device,
                                           median_ms=statistics.median(host), iqr_ms=q3-q1))
            save()
        combined = [[v for p in report["periods"] if p["arm"] == arm for v in p["samples_ms"]] for arm in (0, 1)]
        a, b = [statistics.median(values) for values in combined]
        q1, _, q3 = statistics.quantiles(combined[0], n=4, method="inclusive")
        report.update(status="completed", baseline_median_ms=a, candidate_median_ms=b,
                      delta_ms=a-b, gate_ms=2*(q3-q1), effective=a-b > 2*(q3-q1))
        save()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
