# SPDX-License-Identifier: Apache-2.0
"""Bounded four-rank TP4 correctness checks before full-model admission.

Run with torch.distributed.run under the evidence/device lease launcher.
This checks real expert shards, optimized N256 MoE and TP collectives. It
does not establish full-model memory capacity, quality or performance.
"""

import argparse
import json
import math
import os
from pathlib import Path
from itertools import permutations


def validate_bf16_reduction(actual, operands):
    """Allow BF16 collective summation order, never arbitrary error tolerance."""
    import torch
    # A ring can use a different rank order per output partition. Check each
    # lane against all legal four-input BF16 chains and balanced trees.
    matched = torch.zeros_like(actual, dtype=torch.bool)
    for order in permutations(range(4)):
        chain = operands[order[0]]
        for rank in order[1:]:
            chain = (chain.float() + operands[rank].float()).bfloat16()
        matched |= actual == chain
        left = (operands[order[0]].float() + operands[order[1]].float()).bfloat16()
        right = (operands[order[2]].float() + operands[order[3]].float()).bfloat16()
        matched |= actual == (left.float() + right.float()).bfloat16()
    fp32 = operands.float().sum(0).bfloat16()
    matched |= actual == fp32
    if not matched.all():
        raise AssertionError(f"TP4 reduction differs from all legal BF16 trees at {int((~matched).sum())} lanes")
    return float((actual.float() - fp32.float()).abs().max())


def read_experts(shard, name, count=2):
    import torch
    source = shard.catalog[name]
    shape = (count, *source.shape[1:])
    with shard.path.open("rb") as stream:
        stream.seek(source.offset)
        raw = bytearray(stream.read(math.prod(shape) * 2))
    if len(raw) != math.prod(shape) * 2:
        raise ValueError("Truncated prepared expert tensor")
    dtype = torch.int16 if name.endswith("q16") else torch.bfloat16
    return torch.frombuffer(raw, dtype=dtype).clone().reshape(shape)


def decode_reference(q, scales):
    import numpy as np
    import torch
    from vllm_gaudi.ops.deepseek_v41_weights import restore_q16, restore_s16
    n, k = q.shape[1] * 128, q.shape[2] // 32
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    result = []
    for expert in range(q.shape[0]):
        packed = torch.from_numpy(restore_q16(q[expert].numpy(), (n, k))).long()
        bits = scales[expert].view(torch.int16).numpy().view(np.uint16)
        codes = torch.from_numpy(restore_s16(bits, (n, k))).float()
        ids = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
        result.append((lut[ids] * torch.exp2(codes - 127).repeat_interleave(32, -1)).bfloat16())
    return torch.stack(result)


def moe_reference(value, ids, routing, w13, w2):
    import torch
    import torch.nn.functional as F
    output = torch.zeros_like(value, dtype=torch.float32)
    clipped = 0
    for route in range(ids.shape[1]):
        for expert in range(w13.shape[0]):
            rows = ids[:, route] == expert
            if not rows.any():
                continue
            gate, up = F.linear(value[rows].float(), w13[expert].float()).bfloat16().float().chunk(2, -1)
            clipped += int(((gate > 10) | (up.abs() > 10)).sum())
            middle = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10) * routing[rows, route, None]).bfloat16()
            output[rows] += F.linear(middle.float(), w2[expert].float()).bfloat16().float()
    return output.bfloat16(), clipped


def device_moe_reference(value, ids, routing, w13, w2):
    """Unfused BF16 HPU oracle using independently decoded checkpoint weights."""
    import torch
    import torch.nn.functional as F
    output = torch.zeros_like(value, dtype=torch.float32)
    for route in range(ids.shape[1]):
        contribution = torch.zeros_like(value, dtype=torch.float32)
        for expert in range(w13.shape[0]):
            gate, up = F.linear(value, w13[expert]).float().chunk(2, -1)
            middle = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10) * routing[:, route, None]).bfloat16()
            down = F.linear(middle, w2[expert]).float()
            contribution += torch.where((ids[:, route] == expert)[:, None], down, 0)
        output += contribution
    return output.bfloat16()


def validate_grouped_boundaries(inputs, weights, lookup, normal, cpu_weights, actual):
    """Check each route before cancellation, then require exact route reduction.

    MME tiling can perturb FP32 sums at a BF16 rounding midpoint. Opposite
    routed contributions amplify that difference relative to their near-zero
    sum. Keep the original tolerance per route, and check the final ordering
    and sum exactly instead of relaxing the output tolerance.
    """
    import torch
    import torch.nn.functional as F
    from vllm_gaudi.ops.deepseek_v41_grouped_prefill import compiled_project, ordered_reduce
    from vllm_gaudi.ops.deepseek_v41_route_blocks import device_route_blocks
    value, ids, routing = inputs
    experts, slots, _, counts = device_route_blocks(ids.cpu(), weights[0].shape[0], 128)
    active = int(((counts + 127) // 128).sum())
    routes = torch.empty(ids.numel(), value.shape[-1], dtype=torch.bfloat16)
    visited = torch.zeros(ids.numel(), dtype=torch.int32)
    max_error = 0.
    for start in range(0, active, 8):
        group_slots = slots[start:min(start + 8, active)].clone()
        group_ids = experts[:, start:start + group_slots.shape[0]].flatten().clone().reshape(1, -1)
        selected = value.cpu()[group_slots.clamp(min=0) // 6].to("hpu")
        route = routing.cpu().flatten()[group_slots.clamp(min=0)].masked_fill(group_slots < 0, 0).to("hpu")
        decoded = [weight[group_ids.flatten().long()].transpose(-1, -2).contiguous().to("hpu")
                   for weight in cpu_weights]
        gate, up = torch.bmm(selected, decoded[0]).float().chunk(2, -1)
        middle = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10) * route[..., None]).bfloat16()
        expected = torch.bmm(middle, decoded[1]).cpu().flatten(0, 1)
        result = compiled_project((group_slots.shape[0], 128, weights[0].shape[0], bool(normal)))(
            selected, route, group_ids.to("hpu"), *weights, lookup, normal).cpu()
        torch.testing.assert_close(result.float(), expected.float(), atol=.03125, rtol=.03)
        max_error = max(max_error, float((result.float() - expected.float()).abs().max()))
        valid = group_slots.flatten() >= 0
        destinations = group_slots.flatten()[valid]
        routes[destinations] = result[valid]
        visited[destinations] += 1
    assert (visited == 1).all()
    torch.testing.assert_close(actual, ordered_reduce(routes.reshape(value.shape[0], 6, -1)), atol=0, rtol=0)
    return {"per_route_tolerance": "unchanged: atol=0.03125 rtol=0.03", "per_route_max_abs": max_error,
            "route_visits": "exactly_once", "final_ordered_reduction": "bit_exact"}


def diagnostic_generic_moe(value, ids, routing, w13_q16, w2_q16, w13_s16, w2_s16, lookup, normal_scales):
    """Bound generic TP4 dequant workspace without changing token arithmetic."""
    import torch
    op = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2
    # Large generic batches can spill their decoded [T*6,K,N] weight stream
    # to HBM. Bound the compound node independently of the scheduler's prompt
    # budget; attention and request state retain the complete transaction.
    tile = 32
    if value.shape[0] <= tile:
        return op(value, ids, routing, w13_q16, w2_q16, w13_s16, w2_s16, lookup, normal_scales)
    outputs = []
    for start in range(0, value.shape[0], tile):
        stop = min(value.shape[0], start + tile)
        outputs.append(op(value[start:stop].contiguous(), ids[start:stop].contiguous(),
                          routing[start:stop].contiguous(), w13_q16, w2_q16, w13_s16, w2_s16, lookup, normal_scales))
    return torch.cat(outputs, dim=0)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--implementation", choices=("n256", "generic"), default="n256")
    parser.add_argument("--resident-experts", type=int, choices=(2, 384), default=2)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel,
                                  get_pp_group, init_distributed_environment, initialize_model_parallel)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.v1.worker.deepseek_v41_runner import PPBuffers
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://", local_rank=rank,
                                 backend="hccl")
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
    with set_current_vllm_config(config):
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
        output = Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"components-rank{rank}.json"
        report = {"rank": rank, "tp": 4, "pp": 1, "status": "running",
                  "implementation": args.implementation, "cases": []}
        try:
            with torch.inference_mode():
                assert get_pp_group().is_first_rank and get_pp_group().is_last_rank
                pp = PPBuffers("hpu", capacity=128, dspark=False, device_commit=False)
                reduce, gather = stage_collectives(rank, False)
                shard = PreparedV41Shard(args.prepared, 0, rank)
                names = ("w13_q16", "w2_q16", "w13_s16", "w2_s16")
                cpu = [read_experts(shard, "layers.0.ffn.experts." + name) for name in names]
                w13, w2 = decode_reference(cpu[0], cpu[2]), decode_reference(cpu[1], cpu[3])
                channels = None
                if args.implementation == "n256":
                    import numpy as np
                    from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert, restore_expert
                    converted = []
                    for projection in (0, 1):
                        rows = []
                        for expert in range(cpu[projection].shape[0]):
                            q = cpu[projection][expert].numpy()
                            scales = cpu[projection + 2][expert].view(torch.int16).numpy().view(np.uint16)
                            prepared = prepare_expert(q, scales)
                            restored = restore_expert(prepared[0], prepared[1])
                            assert np.array_equal(restored[0], q) and np.array_equal(restored[1], scales)
                            rows.append(prepared)
                        values = [torch.from_numpy(np.stack([row[i] for row in rows]).view(np.int16))
                                  for i in range(3)]
                        values[2] = values[2].view(torch.bfloat16)
                        converted.append([value.to("hpu") for value in values])
                    weights = [converted[0][0], converted[1][0], converted[0][1], converted[1][1]]
                    channels = [converted[0][2], converted[1][2]]
                    if args.resident_experts == 384:
                        from vllm_gaudi.ops.deepseek_v41_expert_n256 import load_projection
                        del weights, channels, converted
                        first = load_projection(shard, "layers.0.ffn.experts.w13", "hpu")
                        second = load_projection(shard, "layers.0.ffn.experts.w2", "hpu")
                        weights = [first[0], second[0], first[1], second[1]]
                        channels = [first[2], second[2]]
                    report["resident_experts"] = args.resident_experts
                else:
                    weights = [value.to("hpu") for value in cpu]
                lookup = mxfp4_bf16_lut(torch.device("hpu"))
                normal = shard.manifest["normal_scales"]["layers.0.ffn.experts"][rank]
                for count in (1, 6, 7, 128, 1):
                    step = len(report["cases"])
                    torch.manual_seed(4104 + step)
                    value = (torch.randn(count, 5120) * (8 if step % 2 else 1)).bfloat16()
                    ids = (torch.arange(count * 6).reshape(count, 6) + step).remainder(2).int()
                    routing = torch.rand(count, 6)
                    routing = routing / routing.sum(-1, keepdim=True) * 1.5
                    torch.hpu.synchronize()
                    resident_bytes = torch.hpu.memory_allocated()
                    torch.hpu.reset_peak_memory_stats()
                    print(f"TP{rank} C{count}: submitting prepared MoE", flush=True)
                    inputs = (value.to("hpu"), ids.to("hpu"), routing.to("hpu"))
                    if args.implementation == "n256":
                        if count > 6:
                            from vllm_gaudi.ops.deepseek_v41_grouped_prefill import run_device_grouped_prefill
                            def execute():
                                return run_device_grouped_prefill(*inputs, *weights, lookup, normal, *channels)
                            cpu_expected, clipped = moe_reference(value, ids, routing, w13, w2)
                            reference_weights = (w13.to("hpu"), w2.to("hpu"))
                            expected = device_moe_reference(*inputs, *reference_weights).cpu()
                            del reference_weights
                        else:
                            namespace = torch.ops.custom_op
                            from vllm_gaudi.ops.deepseek_v41_expert_n256 import run_fused_decode
                            def execute():
                                return run_fused_decode(*inputs, *weights, lookup, *channels, normal)
                            expected = namespace.custom_deepseek_v41_expert_n256_moe_fp8_gaudi2(
                                *inputs, *weights, lookup, *channels, normal).cpu()
                            clipped = None
                        actual = execute()
                    else:
                        def execute():
                            return diagnostic_generic_moe(*inputs, *weights, lookup, normal)
                        actual = execute()
                        expected, clipped = moe_reference(value, ids, routing, w13, w2)
                    host = actual.cpu()
                    torch.save({"actual": host, "expected": expected,
                                "cpu_expected": cpu_expected if count > 6 and args.implementation == "n256" else None},
                               output.with_name(f"moe-rank{rank}-step{step}.pt"))
                    boundary_check = None
                    try:
                        torch.testing.assert_close(host.float(), expected.float(), atol=.03125, rtol=.03)
                    except AssertionError:
                        if args.implementation != "n256" or count <= 6:
                            raise
                        boundary_check = validate_grouped_boundaries(inputs, weights, lookup, normal, (w13, w2), host)
                    assert torch.isfinite(host).all()
                    relative_rms = float((host.float() - expected.float()).square().mean().sqrt()
                                         / expected.float().square().mean().sqrt().clamp_min(1e-30))
                    if args.implementation == "n256" and count <= 6:
                        assert relative_rms < .001, relative_rms
                    torch.hpu.synchronize()
                    import time
                    started = time.perf_counter()
                    for _ in range(5):
                        execute()
                    torch.hpu.synchronize()
                    chain_ms = (time.perf_counter() - started) * 1000 / 5
                    # Check the complete producer -> gather/reduce -> consumer
                    # ordering with changing shape and values across requests.
                    copies = gather(actual, dim=0).cpu().reshape(4, count, 5120)
                    combined = reduce(actual.clone()).cpu()
                    torch.save({"actual": combined, "rank_operands": copies},
                               output.with_name(f"reduce-rank{rank}-step{step}.pt"))
                    reduction_error = validate_bf16_reduction(combined, copies)
                    constant = torch.full_like(actual, rank + 1)
                    assert (reduce(constant).cpu() == 10).all()
                    heads = torch.full((count, 8, 128), rank + step, dtype=torch.bfloat16, device="hpu")
                    gathered = gather(heads, dim=1).cpu()
                    assert gathered.shape == (count, 32, 128)
                    for peer in range(4):
                        assert (gathered[:, peer * 8:(peer + 1) * 8] == peer + step).all()
                    assert pp.finish_single(count, 42) == (count, [42])
                    report["cases"].append({"tokens": count, "clipped_values": clipped,
                                             "boundary_check": boundary_check,
                                             "relative_rms_error": relative_rms, "warm_host_sync_ms": chain_ms,
                                             "max_abs_error": float((host.float() - expected.float()).abs().max()),
                                         "finite": True, "tp_reduce_gather": "passed",
                                         "bf16_collective_vs_fp32_max_abs": reduction_error,
                                             "resident_bytes": resident_bytes,
                                             "peak_allocated_bytes": torch.hpu.max_memory_allocated()})
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    print(f"TP{rank} C{count}: passed", flush=True)
                assert pp.sends == pp.receives == 0
                pp.drain()
                report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            output.write_text(json.dumps(report, indent=2) + "\n")
            if report["status"] == "passed":
                destroy_model_parallel()
                destroy_distributed_environment()
            else:
                # A peer may already be waiting in the next collective.
                # Let torchrun terminate this owned group on a failed rank.
                import traceback
                traceback.print_exc()
                os._exit(1)


if __name__ == "__main__":
    main()
