# SPDX-License-Identifier: Apache-2.0
"""Compare completion records through the next device embedding consumer."""
import json
import os
from pathlib import Path
import time

rank = int(os.environ["LOCAL_RANK"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]

import torch  # noqa: E402
import habana_frameworks.torch  # noqa: E402, F401
import habana_frameworks.torch.distributed.hccl  # noqa: E402, F401
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers  # noqa: E402


def consumer(ids, weight):
    return torch.nn.functional.embedding(ids.long(), weight)


def main():
    bind_worker_cpu(rank)
    torch.hpu.set_device(rank)
    torch.distributed.init_process_group("hccl")
    cpu_group = torch.distributed.new_group(backend="gloo")
    weight_cpu = torch.arange(512 * 128).reshape(512, 128).to(torch.bfloat16)
    weight = weight_cpu.to("hpu")
    device_record = torch.empty(4, dtype=torch.int32, device="hpu")
    host_record = torch.empty(4, dtype=torch.int32)
    next_input = torch.empty(1, dtype=torch.int32, device="hpu")
    consume = torch.compile(consumer, backend="hpu_backend", fullgraph=True, dynamic=False)
    consume(torch.zeros_like(next_input), weight)
    torch.hpu.synchronize()
    bind_worker_helpers(rank)
    results = {}
    for mode in ("cpu_gloo", "original_hpu_record"):
        times = []
        for generation in range(56):
            token = (generation * 37 + 11) % 512
            started = time.perf_counter_ns()
            if rank == 1:
                record = torch.tensor([generation, 1, 1, token], dtype=torch.int32)
                if mode == "cpu_gloo":
                    host_record.copy_(record)
                else:
                    device_record.copy_(record)
            if mode == "cpu_gloo":
                torch.distributed.broadcast(host_record, src=1, group=cpu_group)
                committed = host_record.tolist()
                next_input.copy_(torch.tensor([committed[3]], dtype=torch.int32))
                ids = next_input
            else:
                torch.distributed.broadcast(device_record, src=1)
                committed = device_record.cpu().tolist()
                ids = device_record[3:4]
            output = consume(ids, weight)
            torch.hpu.synchronize()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if committed != [generation, 1, 1, token] or not torch.equal(output.cpu(), weight_cpu[token:token + 1]):
                raise RuntimeError(f"{mode}: stale completion/input generation {generation}")
            if generation >= 8:
                times.append(elapsed)
        results[mode] = {"host_ms": times, "mean_ms": sum(times) / len(times),
                         "consumer": "compiled BF16 embedding + HPU synchronize", "correct": True}
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    (root / f"rank{rank}.json").write_text(json.dumps(results, indent=2))
    print(rank, {k: v["mean_ms"] for k, v in results.items()}, flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
