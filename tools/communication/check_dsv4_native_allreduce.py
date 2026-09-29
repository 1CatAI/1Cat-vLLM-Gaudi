#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Changing-input execution check for V4 ordinary native AllReduce nodes."""
import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import habana_frameworks.torch.distributed.hccl  # noqa: F401

from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge, _RUNTIME_ATTR
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
from vllm_gaudi.ops.tp2_prepared_plan import (collect_prepared_group_replays, record_native_decoder_outputs,
                                              replay_native_decoder, register_tp2_prepared_group_pass,
                                              prepared_group_stats, shutdown_prepared_group_plans)


def exercise_allocator_compaction():
    """Fragment a small diagnostic pool while captured commands remain live."""
    torch.hpu.synchronize()
    before = torch.hpu.memory_stats()
    limit, used = before["Limit"], before["InUse"]
    if limit > 4 * 1024**3:
        raise RuntimeError("Memory-pressure contract requires a <=4 GiB diagnostic pool")
    chunk = 32 * 1024**2
    count = (limit - used) // chunk - 1
    if count < 8:
        raise RuntimeError("Diagnostic pool has insufficient headroom for compaction")
    buffers = [torch.full((chunk, ), i + 1, dtype=torch.uint8, device="hpu") for i in range(count)]
    torch.hpu.synchronize()
    for i in range(0, count, 2):
        buffers[i] = None
    gc.collect()
    torch.hpu.synchronize()
    # Each freed hole is one chunk and the remaining tail is <2 chunks.
    # Four chunks require moving live, non-captured allocations.
    combined = torch.full((4 * chunk, ), 217, dtype=torch.uint8, device="hpu")
    torch.hpu.synchronize()
    for i in range(1, count, 2):
        assert bool((buffers[i].cpu() == i + 1).all()), "Compaction corrupted live storage"
    assert bool((combined.cpu() == 217).all())
    result = {
        "before": before,
        "fragment_count": count,
        "chunk_bytes": chunk,
        "requested_bytes": 4 * chunk,
        "after": torch.hpu.memory_stats()
    }
    del combined, buffers
    gc.collect()
    torch.hpu.synchronize()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-intermediates", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reshape-output", action="store_true")
    parser.add_argument("--parallel-reductions", action="store_true")
    parser.add_argument("--compiled-embedding", action="store_true")
    parser.add_argument("--profile-before-capture", action="store_true")
    parser.add_argument("--profile-recapture", action="store_true")
    parser.add_argument("--memory-pressure", action="store_true")
    parser.add_argument("--request-batch",
                        type=int,
                        choices=(1, 2, 4, 8, 16, 32, 64),
                        help="Qualify native V4.1 peer exchange through its dependent consumer")
    args = parser.parse_args()
    if args.request_batch:
        modules = os.environ["HABANA_VISIBLE_MODULES"].split(",")
        os.environ["HLS_MODULE_ID"] = modules[int(os.environ["LOCAL_RANK"])]
    torch.hpu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    warm = torch.ones(128, dtype=torch.bfloat16, device="hpu")
    dist.all_reduce(warm)
    torch.hpu.synchronize()
    bridge = _load_bridge(args.bridge.resolve())
    bridge.set_use_tensor_ids(True)
    bridge.set_prepared_communication(True)
    backend = dist.group.WORLD._get_backend(torch.device("hpu"))
    setattr(torch, _RUNTIME_ATTR, (bridge, backend, bridge.communicator_id(backend)))
    register_tp2_prepared_group_pass()

    def reduce(partial):
        if args.request_batch:
            return partial + torch.ops.vllm_gaudi.tp2_exchange_peer(partial)
        return torch.ops.vllm_gaudi.tp2_allreduce_plain(partial)

    def program(x):
        partial1 = x + 2
        first = reduce(partial1)
        partial2 = x + 4 if args.parallel_reductions else first + 2
        second = reduce(partial2)
        result = (first + second) * 0.25 if args.parallel_reductions else second * 0.25
        if args.reshape_output:
            result = result.view(1, 4, 1024)
        return (result, partial1, first, partial2, second) if args.inspect_intermediates else result

    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    owner = torch.nn.Identity()
    topology = DecoderTopology("deepseek_v41_batch_micro" if args.request_batch else "deepseek_v4", (1, ), 2, False)
    if args.compiled_embedding:
        from vllm_gaudi.ops.deepseek_v4_native import embedding_partial
        embedding = torch.compile(embedding_partial, backend="hpu_backend", fullgraph=True, dynamic=False)
        vocabulary = (torch.arange(rank * 16, (rank + 1) * 16) * 2 + 1).bfloat16()
        vocabulary = vocabulary[:, None].expand(16, 4096).contiguous().to("hpu")
    profiler = None
    compactions = []
    profile_enabled = args.profile or args.profile_before_capture or args.profile_recapture
    for step in range(34):
        if args.memory_pressure and step in (4, 12, 24):
            compactions.append(exercise_allocator_compaction())
        if profile_enabled and step == (0 if args.profile_before_capture else 2):
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(args.output), use_gzip=True))
            profiler.start()
            if args.profile_recapture:
                from vllm_gaudi.ops.tp2_prepared_plan import recapture_native_decoder_programs
                recapture_native_decoder_programs()
        if args.compiled_embedding:
            token = (step * 7) % 32
            tokens = torch.tensor([token], dtype=torch.int32, device="hpu")
            x = embedding(tokens, vocabulary, rank * 16, (rank + 1) * 16, 0, 32, 32)
        else:
            width = args.request_batch * 5120 if args.request_batch else 4096
            x = torch.full((1, width), float(step + rank), dtype=torch.bfloat16).to("hpu")
        roots = dict(hidden_states=x, positions=None, residual=None, metadata=SimpleNamespace(is_prompt=False))
        result = replay_native_decoder(owner, **roots)
        if result is None:
            with collect_prepared_group_replays(owner=owner,
                                                adapter=topology,
                                                snapshot=lambda: SimpleNamespace(restore=lambda: None),
                                                **roots):
                output = compiled(x)
                record_native_decoder_outputs(output)
            result = output, None
        ticket = getattr(roots["metadata"], "native_completion", None)
        if ticket is None and args.request_batch and step == 0:
            # Cold discovery executes once before the prepared graph exists.
            # Subsequent calls must use the replay's actual consumer event.
            ticket = bridge.record_native_completion()
        if ticket is None:
            raise RuntimeError("Steady replay did not publish its consumer completion")
        if not ticket.query():
            ticket.synchronize()
        assert ticket.query(), "The decoder consumer did not complete"
        if args.inspect_intermediates:
            intermediate = [value.cpu().flatten()[:8].float().tolist() for value in result[0]]
            print(json.dumps(dict(rank=rank, step=step, intermediate=intermediate)), flush=True)
        actual = (result[0][0] if args.inspect_intermediates else result[0]).cpu()
        expected = torch.full_like(actual, (token if args.compiled_embedding else step) + 3.5)
        if not torch.equal(actual, expected):
            raise AssertionError(f"rank={rank} step={step}: actual={actual[0,:8]}, expected={expected[0,:8]}")
        if profiler is not None and step == 9:
            profiler.stop()
    stats = prepared_group_stats()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(
        json.dumps(dict(status="exact", steps=34, compactions=compactions, **stats), indent=2))
    print(json.dumps(dict(rank=rank, status="exact", **stats)), flush=True)
    shutdown_prepared_group_plans()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
