# SPDX-License-Identifier: Apache-2.0
"""Check the actual V2 device sampler/PP relay through the next embedding."""
import json
import os
from pathlib import Path
import time

# HPU imports must follow the per-rank device selection.
rank = int(os.environ["LOCAL_RANK"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]

import torch  # noqa: E402
import habana_frameworks.torch  # noqa: E402
import habana_frameworks.torch.distributed.hccl  # noqa: E402, F401 -- registers the HCCL backend
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_sampling import select_greedy_candidate  # noqa: E402
from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge, _verify_prepared_runtime  # noqa: E402


def sample(candidates):
    return select_greedy_candidate(candidates).to(torch.int32)


def consume(token, weight):
    return torch.nn.functional.embedding(token.reshape(-1).long(), weight)


def main():
    bind_worker_cpu(rank)
    torch.hpu.set_device(rank)
    torch.distributed.init_process_group("hccl")
    cpu_group = torch.distributed.new_group(backend="gloo")
    binary = Path(os.environ["VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE"])
    bridge = _load_bridge(binary)
    _verify_prepared_runtime(binary)
    producer = torch.compile(sample, backend="hpu_backend", fullgraph=True, dynamic=False)
    consumer = torch.compile(consume, backend="hpu_backend", fullgraph=True, dynamic=False)
    weight_cpu = torch.arange(512 * 5120).reshape(512, 5120).to(torch.bfloat16)
    weight = weight_cpu.to("hpu")
    inputs = [torch.tensor([[1., (i * 37 + 11) % 512, 0., 0.]], device="hpu") for i in range(40)]
    relay = torch.empty((1, 1), device="hpu", dtype=torch.int32)
    ids = torch.empty(1, device="hpu", dtype=torch.int32)
    host_record = torch.empty(4, dtype=torch.int32)
    consumer(producer(inputs[0]), weight)
    torch.hpu.synchronize()
    bind_worker_helpers(rank)
    result = {}
    for mode in ("cpu_gloo", "v2_device_relay"):
        times = []
        for generation, candidates in enumerate(inputs):
            started = time.perf_counter_ns()
            token = producer(candidates) if rank == 1 else relay
            if mode == "cpu_gloo":
                if rank == 1:
                    selected = int(token.cpu().item())
                    host_record.copy_(torch.tensor([generation, 1, 1, selected], dtype=torch.int32))
                torch.distributed.broadcast(host_record, src=1, group=cpu_group)
                ids.copy_(host_record[3:4])
                out = consumer(ids, weight)
            else:
                torch.distributed.broadcast(token, src=1)
                host, done = bridge.copy_sampled_tokens_to_host(token)
                # The device consumer is submitted before waiting for the
                # async scheduler record, as in the production V2 runner.
                out = consumer(token, weight)
                done.synchronize()
                selected = int(host[0, 0])
            torch.hpu.synchronize()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            expected = (generation * 37 + 11) % 512
            if not torch.equal(out.cpu(), weight_cpu[expected:expected + 1]):
                raise RuntimeError(f"{mode}: wrong embedding at generation {generation}")
            if mode == "v2_device_relay" and selected != expected:
                raise RuntimeError("Async host record differs from device consumer")
            if generation >= 8:
                times.append(elapsed)
        result[mode] = {"host_ms": times, "mean_ms": sum(times) / len(times), "correct": True}
    (Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
    print(rank, {k: v["mean_ms"] for k, v in result.items()}, flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
