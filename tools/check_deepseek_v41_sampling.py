# SPDX-License-Identifier: Apache-2.0
"""Device check of compact TP greedy selection and the C1 token copy contract."""

import faulthandler
import json
import os
from pathlib import Path
import signal

rank = int(os.environ["LOCAL_RANK"])
evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace(
    "{rank}", str(rank))

import torch  # noqa: E402
from vllm.distributed import (  # noqa: E402
    init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment,
    tensor_model_parallel_all_gather)
from vllm_gaudi.distributed.tp2_fused_ar_norm import (  # noqa: E402
    initialize_tp2_fused_ar_norm_runtime, _resolve_runtime)
from vllm_gaudi.ops.deepseek_v41_sampling import (  # noqa: E402
    local_greedy_candidate, select_greedy_candidate)


def sample(logits):
    local = local_greedy_candidate(logits, rank)
    return select_greedy_candidate(tensor_model_parallel_all_gather(local, dim=-1))


@torch.inference_mode()
def main():
    torch.hpu.set_device(rank)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=2, rank=rank, distributed_init_method="env://",
                                 local_rank=rank, backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    initialize_tp2_fused_ar_norm_runtime()
    bridge, _, _ = _resolve_runtime()
    stack_file = (evidence / f"rank{rank}-stacks.log").open("w")
    faulthandler.enable(file=stack_file, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=stack_file, all_threads=True, chain=False)
    compiled = torch.compile(sample, backend="hpu_backend", fullgraph=True, dynamic=False)
    logits = torch.empty((1, 64640), dtype=torch.float32, device="hpu")
    commit = torch.empty(4, dtype=torch.int32, device="hpu")
    next_input = torch.empty(1, dtype=torch.int64, device="hpu")
    source_view = commit[3:4]
    cases = []
    for step in range(10):
        generator = torch.Generator(device="cpu").manual_seed(41 + step)
        full = torch.randn((1, 129280), generator=generator, device="cpu")
        if step in (0, 1, 2):
            full[0, (step % 2) * 64640 + 3 + step] = 20
        elif step == 3:
            full[0, 3] = full[0, 64643] = 20
        elif step == 4:
            full.fill_(-float("inf"))
        elif step == 5:
            full[0, 8] = full[0, 64645] = float("nan")
        elif step == 6:
            full[0, 64648] = float("nan")
        elif step == 7:
            full[0, 7] = full[0, 64646] = float("inf")
        elif step == 8:
            full[:, :64640] = float("nan")
        else:
            full.fill_(float("nan"))
        cpu_expected = int(full.argmax(-1)[0])
        logits.copy_(full[:, rank * 64640:(rank + 1) * 64640])
        # The production HPU argmax has backend-specific NaN behavior. Compare
        # against its complete-logit path, retaining the CPU result separately.
        reference = tensor_model_parallel_all_gather(logits, dim=-1).argmax(-1)
        expected = int(reference.cpu()[0])
        selected = compiled(logits)
        host, done = bridge.copy_sampled_tokens_to_host(selected)
        done.synchronize()
        actual = int(host[0, 0])
        commit.copy_(torch.tensor([step + 1, 1, 1, actual], dtype=torch.int32, device="cpu"))
        next_input.copy_(source_view)
        copied = int(next_input.cpu()[0])
        cases.append({"step": step, "expected_hpu": expected, "expected_cpu": cpu_expected,
                      "actual": actual, "next_input": copied})
        (evidence / f"rank{rank}-cases.json").write_text(json.dumps(cases, indent=2) + "\n")
        print(f"RANK {rank} sample {step}: {actual}, expected {expected}, input {copied}", flush=True)
        assert actual == expected == copied
        if step == 0:
            bind_worker_helpers(rank)
    (evidence / f"rank{rank}-result.json").write_text(json.dumps({
        "purpose": "Actual TP2 compact greedy, native D2H and int32 commit slice to int64 input; no speed claim",
        "cases": cases, "passed": True}, indent=2) + "\n")
    destroy_model_parallel()
    destroy_distributed_environment()
    faulthandler.unregister(signal.SIGUSR1)
    faulthandler.disable()
    stack_file.close()


if __name__ == "__main__":
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2))):
        main()
