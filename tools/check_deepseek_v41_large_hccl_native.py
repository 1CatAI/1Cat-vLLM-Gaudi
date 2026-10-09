# SPDX-License-Identifier: Apache-2.0
"""Real C6 head/stock AllGather/consumer on the DSpark-private native bridge.

This is a capacity prerequisite for the complete sampled protocol. It never
changes the shared communication source or substitutes a PCIe peer transport.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    rank, size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace(
        "{rank}", str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment, load_native_operators

    prepare_environment(args.prepared, tensor_parallel_size=size, pipeline_parallel_size=1)
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    import torch.distributed as dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel, get_tp_group
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_draft_body_replay import NativeDraftBody
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats, shutdown_prepared_group_plans

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(rank=rank, status="loading", capability_passed=False, micro_qualified=False,
                  checks=[], rounds=[], qualified_gain_ms=0,
                  scope="Capacity prerequisite: real C6 FP32 head -> stock HCCL -> exact logits/checksum",
                  credited_end_to_end_ms=0)

    def save():
        (root / f"sampled-control-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    plan = None
    try:
        config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=size))
        with set_current_vllm_config(config), torch.inference_mode():
            bind_worker_cpu(rank)
            torch.hpu.set_device(rank)
            load_native_operators()
            init_distributed_environment(world_size=size, rank=rank, local_rank=rank,
                                         distributed_init_method="env://", backend="hccl")
            initialize_model_parallel(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            _, gather = stage_collectives(rank, True, size)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            # Frozen actual normalized Target inputs, not random messages.
            files = sorted((args.fixtures / f"rank{rank}").glob("c6-*.pt"))
            if not 3 <= len(files) <= 5 or args.samples < 3:
                raise ValueError("Need3-5 actual production cases and three AB pairs")
            inputs = []
            for path in files:
                data = torch.load(path, weights_only=True)
                if not data.get("request_context_qualified") or data["hidden"].shape != (6, 5120):
                    raise ValueError("C6 head input is not an actual request fixture")
                inputs.append((data["hidden"].to("hpu"), data["positions"].to("hpu")))
                report.setdefault("fixtures", []).append(dict(path=str(path),
                                                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            # Use the checkpoint's BF16 values as FP32, matching serving205.
            weight = shard.tensor("head.weight", "hpu").float()
            wire_view = torch.ops.custom_op.custom_deepseek_v41_sampling_wire_view_gaudi2

            class HeadWire(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("weight", weight, False)
                    self.tensor_parallel_size = size
                    # Protocol ownership is reused; this prerequisite has no
                    # MTP state and performs only its one actual collective.
                    self.layers = tuple(SimpleNamespace(attention=SimpleNamespace(swa=None)) for _ in range(3))

                def forward_local(self, value, positions):
                    logits = torch.nn.functional.linear(value.float(), self.weight)
                    wire = wire_view(logits.contiguous(), False)
                    result = torch.ops.vllm_gaudi.tp_peer_allgather(wire.reshape(1, -1).contiguous(), size)
                    result = result.reshape(size, *wire.shape).permute(1, 0, 2).contiguous()
                    full = wire_view(result.reshape(wire.shape[0], -1), True)
                    return full, full.sum(-1)

            class WirePlan(NativeDraftBody):
                def __init__(self, stage):
                    super().__init__(stage, generation=1)
                    self.adapter = DecoderTopology("deepseek_v41_dspark_large_vocab", (1,), 0, False, 1)
                    self.states = ()

                def mutable_states(self):
                    return ()

                @staticmethod
                def _validate(hidden, proposed, control, auxiliary, positions):
                    if proposed.shape != (6, 5120) or positions.shape != (6,):
                        raise ValueError("Large collective capability requires actual C6 geometry")

            stage = HeadWire()

            def reference(value, positions):
                logits = torch.nn.functional.linear(value.float(), weight)
                wire = wire_view(logits.contiguous(), False)
                full = wire_view(gather(wire, dim=-1).contiguous(), True)
                return full, full.sum(-1)

            reference = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
            plan = WirePlan(stage)
            plan.prepare(*inputs[0])
            plan.require_ready()
            for case, values in enumerate(inputs):
                expected = tuple(x.cpu() for x in reference(*values))
                actual = tuple(x.cpu() for x in plan(*values))
                exact = all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
                report["checks"].append(dict(case=case, head_wire_and_consumer_exact=exact))
                save()
                agreement = [None] * size
                dist.all_gather_object(agreement, exact, group=get_tp_group().cpu_group)
                if not all(agreement):
                    raise AssertionError("Large stock native HCCL changed actual head logits or rank order")
            report["capability_passed"] = True
            for function in (reference, plan):
                for _ in range(2):
                    function(*inputs[0])
                    torch.hpu.synchronize()
            for iteration in range(3):
                for arm, function in (("parent", reference), ("native", plan)):
                    timings = []
                    for _ in range(args.samples):
                        torch.hpu.synchronize()
                        first, last = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                        first.record()
                        function(*inputs[0])
                        last.record()
                        last.synchronize()
                        timings.append(first.elapsed_time(last))
                    report["rounds"].append(dict(iteration=iteration, arm=arm, device_ms=timings,
                                                 median_device_ms=statistics.median(timings)))
                    save()
            report.update(status="passed", native_stats=prepared_group_stats(),
                          message_bf16_words=6 * 32320 * 2, full_protocol_qualified=False)
            save()
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save()
        raise
    finally:
        if plan is not None:
            plan.close()
        shutdown_prepared_group_plans()


if __name__ == "__main__":
    main()
