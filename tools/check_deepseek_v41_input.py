# SPDX-License-Identifier: Apache-2.0
"""Check the real TP2 embedding/input graph before full-model measurement."""

import argparse
import json
import os
from pathlib import Path

rank = int(os.environ["LOCAL_RANK"])
evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace(
    "{rank}", str(rank))
if os.environ.get("GRAPH_VISUALIZATION") == "1":
    os.environ["GRAPH_VISUALIZATION_DIR"] += f"/rank{rank}"
    Path(os.environ["GRAPH_VISUALIZATION_DIR"]).mkdir(parents=True, exist_ok=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment  # noqa: E402
prepare_environment()
import torch  # noqa: E402
from vllm.distributed import (  # noqa: E402
    init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment)
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime  # noqa: E402
from vllm_gaudi.models.deepseek_v41_program import PreparedInput  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--profile", action="store_true", help="Isolate post-warmup hardware profiler entry")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--int32-inputs", action="store_true", help="Read C1 IDs from an int32 commit-offset view")
    args = parser.parse_args()
    torch.hpu.set_device(rank)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=2, rank=rank, distributed_init_method="env://",
                                 local_rank=rank, backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    initialize_tp2_fused_ar_norm_runtime()
    shard = PreparedV41Shard(args.prepared, 0, rank)
    embedding = torch.nn.Module()
    embedding.register_buffer("weight", shard.tensor("embed.weight", "hpu"))
    reduce, _ = stage_collectives(rank, True)
    program = PreparedInput(embedding, rank, reduce)
    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    cases = []
    for step, ids in enumerate(([0], [64640], [129264], [129265],
                                [0, 1, 64639, 64640, 129264, 129265], [129279], [1])):
        if args.int32_inputs and len(ids) == 1:
            storage = torch.tensor([7, 1, 1, ids[0]], dtype=torch.int32, device="hpu")
            value = storage[3:4]
        else:
            value = torch.tensor(ids, dtype=torch.int64, device="hpu")
        ordinary = tuple(tensor.cpu() for tensor in program(value))
        # Only C1 is changed. Multi-token prefill retains its existing program;
        # then the subsequent C1 calls exercise the communication transition.
        actual = tuple(tensor.cpu() for tensor in (compiled(value) if len(ids) == 1 else program(value)))
        assert torch.equal(ordinary[0].view(torch.int16), actual[0].view(torch.int16))
        assert torch.equal(ordinary[1].view(torch.int32), actual[1].view(torch.int32))
        cases.append({"tokens": ids, "residual_shape": list(actual[0].shape),
                      "compiled_c1": len(ids) == 1, "bitwise_equal": True,
                      "dtype": str(value.dtype), "storage_offset": value.storage_offset()})
        print(f"Rank {rank} input case {step}: exact", flush=True)
        if step == 0:
            bind_worker_helpers(rank)
    if args.profile:
        torch.hpu.synchronize()
        print(f"Rank {rank}: start post-warmup profiler", flush=True)
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
                record_shapes=True, with_stack=args.with_stack,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(evidence / "traces"), use_gzip=True)) as prof:
            for step, ids in enumerate(([0], [64640], [129265], [129279])):
                print(f"Rank {rank}: profile ordinary input {step}", flush=True)
                value = torch.tensor(ids, dtype=torch.int64, device="hpu")
                ordinary = tuple(tensor.cpu() for tensor in program(value))
                print(f"Rank {rank}: profile compiled input {step}", flush=True)
                actual = tuple(tensor.cpu() for tensor in compiled(value))
                assert torch.equal(ordinary[0].view(torch.int16), actual[0].view(torch.int16))
                assert torch.equal(ordinary[1].view(torch.int32), actual[1].view(torch.int32))
                prof.step()
        print(f"Rank {rank}: profiler exported", flush=True)
    (evidence / f"rank{rank}-result.json").write_text(json.dumps({
        "passed": True, "purpose": "Real embedding TP2 compiled/eager input-chain check; no latency claim",
        "cases": cases, "peak_hpu_bytes": torch.hpu.max_memory_allocated()}, indent=2) + "\n")
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2))):
        main()
