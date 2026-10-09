# SPDX-License-Identifier: Apache-2.0
"""Check the shared target projection through local or TP greedy selection."""

import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0")) if args.distributed else 0
    if args.distributed:
        os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.models.deepseek_v41_program import output_head_projection
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4 if args.distributed else 1))
    with set_current_vllm_config(config):
        torch.hpu.set_device(rank)
        torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
        select = lambda logits: logits.argmax(-1)
        if args.distributed:
            from vllm.distributed import init_distributed_environment, initialize_model_parallel
            from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
            from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
            from vllm_gaudi.ops.deepseek_v41_verify import vocab_parallel_argmax
            from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers

            bind_worker_cpu(rank)
            init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://",
                                         local_rank=rank, backend="hccl")
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            _, gather = stage_collectives(rank, True, 4)
            select = lambda logits: vocab_parallel_argmax(logits, rank, gather)
        shard = PreparedV41Shard(args.prepared, 0, rank)
        root = Path(os.environ["DSV41_RUN_EVIDENCE"])
        result = dict(
            status="running",
            scope="projection through four-rank greedy selection" if args.distributed else
                  "local projection through actual greedy consumer; no TP communication",
            rank=rank,
            formal_qualified=False,
            rows=[],
        )

        def save():
            filename = f"projection-rank{rank}.json" if args.distributed else "projection-result.json"
            (root / filename).write_text(json.dumps(result, indent=2) + "\n")

        save()
        with torch.inference_mode():
            for name, rows in [("head.weight", 6)]:
                weight = shard.tensor(name, "hpu").contiguous()
                if weight.dtype != torch.bfloat16:
                    raise ValueError(f"Original checkpoint projection is not BF16: {name} {weight.dtype}")
                fp32 = weight.float()
                checkpoint = weight.cpu().float()

                def legacy(value, matrix):
                    logits = torch.nn.functional.linear(value.float(), matrix)
                    return (logits, select(logits))

                def candidate(value, matrix):
                    logits = output_head_projection(value, matrix, bf16=True)
                    return (logits, select(logits))

                baseline = torch.compile(legacy, backend="hpu_backend", fullgraph=True, dynamic=False)
                proposed = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
                one = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
                value = torch.empty(rows, 5120, dtype=torch.bfloat16, device="hpu")
                generator = torch.Generator().manual_seed(6416)
                fixtures = [torch.randn(rows, 5120, generator=generator).bfloat16() for _ in range(args.steps)]
                for fixture in fixtures[:2]:
                    value.copy_(fixture)
                    baseline(value, fp32)
                    proposed(value, weight)
                    one(value[:1].clone(), weight)
                torch.hpu.synchronize()
                blocks = []
                errors = []
                tokens = []
                c1_tokens = []
                for use_b in (False, True):
                    for index, fixture in enumerate(fixtures):
                        value.copy_(fixture)
                        logits, ids = proposed(value, weight) if use_b else baseline(value, fp32)
                        out = logits.cpu()
                        selected = ids.cpu()
                        reference = torch.nn.functional.linear(fixture.float(), checkpoint)
                        errors.append(
                            dict(
                                candidate=use_b,
                                step=index,
                                max_abs=float((out - reference).abs().max()),
                                rms=float((out - reference).square().mean().sqrt()),
                            )
                        )
                        tokens.append(dict(candidate=use_b, step=index, ids=selected.tolist()))
                        if use_b:
                            single = torch.cat(
                                [one(value[row : row + 1].clone(), weight)[1].cpu() for row in range(rows)]
                            )
                            c1_tokens.append(
                                dict(
                                    step=index,
                                    exact=torch.equal(selected, single),
                                    c6=selected.tolist(),
                                    c1=single.tolist(),
                                )
                            )
                torch.hpu.synchronize()
                if args.distributed:
                    torch.distributed.barrier()
                for use_b in (False, True, False, True, False, True):
                    if args.distributed:
                        torch.distributed.barrier()
                    for fixture in fixtures:
                        value.copy_(fixture)
                        proposed(value, weight) if use_b else baseline(value, fp32)
                    torch.hpu.synchronize()
                    wall = []
                    device = []
                    for index, fixture in enumerate(fixtures):
                        value.copy_(fixture)
                        torch.hpu.synchronize()
                        begin, end = (torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True))
                        started = time.perf_counter_ns()
                        begin.record()
                        logits, ids = proposed(value, weight) if use_b else baseline(value, fp32)
                        end.record()
                        end.synchronize()
                        wall.append((time.perf_counter_ns() - started) / 1000000.0)
                        device.append(begin.elapsed_time(end))
                    blocks.append(
                        dict(candidate=use_b, device_ms=device, wall_ms=wall, mean_device_ms=statistics.mean(device))
                    )
                record = dict(
                    weight=name,
                    shape=list(weight.shape),
                    rows=rows,
                    blocks=blocks,
                    errors=errors,
                    tokens=tokens,
                    c6_vs_c1=c1_tokens,
                    paired_saved_ms=[blocks[i]["mean_device_ms"] - blocks[i + 1]["mean_device_ms"] for i in (0, 2, 4)],
                )
                result["rows"].append(record)
                save()
                del baseline, proposed, one, value, weight, fp32, checkpoint
                torch.hpu.synchronize()
            result["status"] = "complete"
            save()
            if any(not item["exact"] for row in result["rows"] for item in row["c6_vs_c1"]):
                result["status"] = "failed_c6_c1_head_tokens"
                save()
                raise AssertionError("C6 head differs from the same C1 native head")
        if args.distributed:
            from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
            from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans

            shutdown_prepared_group_plans()
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
