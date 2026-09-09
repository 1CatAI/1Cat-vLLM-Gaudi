# SPDX-License-Identifier: Apache-2.0
"""Stress the 43-layer replay boundary with an asynchronous token recurrence."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import habana_frameworks.torch.distributed.hccl  # noqa: F401

from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge, _RUNTIME_ATTR
from vllm_gaudi.ops.deepseek_v4_native import embedding_partial
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
from vllm_gaudi.ops.tp2_prepared_plan import (
    collect_prepared_group_replays, prepared_group_stats, record_native_decoder_outputs,
    register_tp2_prepared_group_pass, replay_native_decoder, shutdown_prepared_group_plans)


class Group(torch.nn.Module):
    def __init__(self, layers, ordinal, packed_metadata):
        super().__init__()
        self.layers, self.ordinal = layers, ordinal
        self.packed_metadata = packed_metadata

    def forward(self, value, positions, metadata):
        if self.packed_metadata and self.ordinal == 0:
            value = value + positions[:1].bfloat16() + metadata[:1].bfloat16() * 2
        for _ in range(self.layers):
            value = torch.ops.vllm_gaudi.tp2_allreduce_plain(value * 0.5)
            value = torch.ops.vllm_gaudi.tp2_allreduce_plain((value + 1) * 0.5)
        return (value + self.ordinal) - self.ordinal


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--packed-metadata", action="store_true")
    parser.add_argument("--interleave-prefill", action="store_true")
    parser.add_argument("--retain-host-positions", action="store_true")
    parser.add_argument("--position-dtype", choices=("int32", "int64"), default="int64")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    os.sched_setaffinity(0, {38 if rank == 0 else 50})
    torch.hpu.set_device(rank)
    dist.init_process_group("hccl")
    warm = torch.ones(128, dtype=torch.bfloat16, device="hpu")
    dist.all_reduce(warm)
    torch.hpu.synchronize()
    bridge = _load_bridge(args.bridge.resolve())
    bridge.set_use_tensor_ids(True)
    bridge.set_prepared_communication(True)
    backend = dist.group.WORLD._get_backend(torch.device("hpu"))
    setattr(torch, _RUNTIME_ATTR, (bridge, backend, bridge.communicator_id(backend)))
    register_tp2_prepared_group_pass()
    topology = DecoderTopology("deepseek_v4", (8, 8, 8, 8, 8, 3), 2, False)
    owner = torch.nn.Identity()
    groups = [torch.compile(Group(n, i, args.packed_metadata), backend="hpu_backend", fullgraph=True, dynamic=False)
              for i, n in enumerate(topology.group_layers)]
    embed = torch.compile(embedding_partial, backend="hpu_backend", fullgraph=True, dynamic=False)
    vocabulary = (torch.arange(rank * 16, (rank + 1) * 16) * 2 + 1).bfloat16()
    vocabulary = vocabulary[:, None].expand(16, 4096).contiguous().to("hpu")
    indices = torch.arange(129280, dtype=torch.float32, device="hpu").view(1, -1)

    def sample(hidden, ids):
        target = (((hidden[:, :1].float() - 44) * 0.5 + 1) % 32)
        logits = -(ids - target).abs()
        return logits.argmax(dim=-1).to(torch.int32).reshape(1, 1)

    sampler = torch.compile(sample, backend="hpu_backend", fullgraph=True, dynamic=False)
    input_buffer = torch.zeros(1, dtype=torch.int32, device="hpu")
    previous = torch.zeros((1, 1), dtype=torch.int32, device="hpu")
    packed_cpu = [torch.zeros(611, dtype=torch.int32, pin_memory=True) for _ in range(2)]
    packed_device = [torch.empty(611, dtype=torch.int32, device="hpu") for _ in range(2)]
    fixed_pack = torch.empty_like(packed_device[0])
    expected_tokens, expected_token = [], 0
    retained_host_positions = []
    completions, pending = [], []
    for step in range(args.steps):
        # Match the model's two-slot metadata retirement without draining the
        # current token or its sampler output on every iteration.
        if len(completions) >= 2 and not completions[-2].query():
            completions[-2].synchronize()
        if args.interleave_prefill and step > 0 and step % 32 == 0:
            prefill = torch.full((64, 4096), rank + 1., dtype=torch.bfloat16, device="hpu")
            dist.all_reduce(prefill)
        slot = step % 2
        packed_cpu[slot].fill_((step * 7) % 17)
        packed_device[slot].copy_(packed_cpu[slot], non_blocking=True)
        positions_cpu = torch.tensor([2 * (step % 13)], dtype=getattr(torch, args.position_dtype), pin_memory=True)
        if args.retain_host_positions:
            retained_host_positions.append(positions_cpu)
        positions = torch.empty_like(positions_cpu, device="hpu").copy_(positions_cpu, non_blocking=True)
        input_buffer.copy_(previous.flatten())
        tokens = torch.zeros(1, dtype=torch.int32).to("hpu", non_blocking=True)
        tokens[:1] = input_buffer[:1]
        value = embed(tokens, vocabulary, rank * 16, (rank + 1) * 16, 0, 32, 32)
        dist.all_reduce(value)
        roots = dict(hidden_states=value, positions=positions, residual=None,
                     metadata=SimpleNamespace(is_prompt=False), metadata_destination=fixed_pack,
                     metadata_pack=packed_device[slot])
        result = replay_native_decoder(owner, **roots)
        if result is None:
            fixed_pack.copy_(packed_device[slot])
            with collect_prepared_group_replays(owner=owner, adapter=topology,
                                               snapshot=lambda: SimpleNamespace(restore=lambda: None),
                                               **roots) as context:
                for i, group in enumerate(groups):
                    context["group_index"] = i
                    value = group(value, positions, fixed_pack)
                record_native_decoder_outputs(value)
            result = value, None
        completions.append(roots["metadata"].native_completion)
        previous = torch.cat([sampler(result[0], indices).flatten()]).view(1, 1)
        host, ticket = bridge.copy_sampled_tokens_to_host(previous)
        pending.append((host, ticket))
        delta = step % 13 + (step * 7) % 17 if args.packed_metadata else 0
        expected_token = (expected_token + 1 + delta) % 32
        expected_tokens.append(expected_token)
    rows = []
    for step, (host, ticket) in enumerate(pending):
        ticket.synchronize()
        actual, expected = int(host[0, 0]), expected_tokens[step]
        rows.append(dict(step=step, actual=actual, expected=expected, exact=actual == expected))
    torch.hpu.synchronize()
    stats = prepared_group_stats()
    result = dict(passed=all(row["exact"] for row in rows), rows=rows, **stats)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(rank=rank, passed=result["passed"], first_mismatch=next(
        (row for row in rows if not row["exact"]), None), **stats)), flush=True)
    shutdown_prepared_group_plans()
    dist.destroy_process_group()
    assert result["passed"], "Asynchronous token recurrence diverged"


if __name__ == "__main__":
    main()
