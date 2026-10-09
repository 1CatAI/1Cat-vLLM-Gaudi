# SPDX-License-Identifier: Apache-2.0
"""Compare full-prefill and bounded decode selection-state backing stores."""
import json
import os
from pathlib import Path
import time


def update_selection(indices, candidates, selected, blocks):
    tokens = selected.shape[0]
    candidates[:tokens].copy_(blocks)
    indices[:tokens].copy_(selected)
    # Both persistent publications have consumers in the same compiled group.
    return indices[:tokens] + candidates[:tokens, :selected.shape[1]]


def main():
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG",
                                                              "").replace("{rank}", str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers

    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    bind_worker_helpers(rank)
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(rank=rank,
                  status="running",
                  measurements=[],
                  formal_qualification=False,
                  scope="selection-state update and consumer; no index scoring or model TPOT")
    try:
        with torch.inference_mode():
            execute = torch.compile(update_selection, backend="hpu_backend", fullgraph=True, dynamic=False)
            selected = torch.arange(6 * 512, dtype=torch.int32, device="hpu").reshape(6, 512)
            blocks = torch.arange(6 * 2048, dtype=torch.int32, device="hpu").reshape(6, 2048)
            for capacity in (8192, 128, 6):
                indices = torch.full((capacity, 512), -1, dtype=torch.int32, device="hpu")
                candidates = torch.full((capacity, 2048), -1, dtype=torch.int32, device="hpu")
                for _ in range(2):
                    actual = execute(indices, candidates, selected, blocks)
                    expected = selected.cpu() + blocks.cpu()[:, :512]
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                    torch.testing.assert_close(indices[:6].cpu(), selected.cpu(), rtol=0, atol=0)
                    torch.testing.assert_close(candidates[:6].cpu(), blocks.cpu(), rtol=0, atol=0)
                    if capacity > 6:
                        assert (indices[6:].cpu() == -1).all()
                        assert (candidates[6:].cpu() == -1).all()
                    selected.add_(1)
                    blocks.add_(3)
                for _ in range(64):
                    execute(indices, candidates, selected, blocks)
                torch.hpu.synchronize()
                samples = []
                for _ in range(3):
                    start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    start.record()
                    host = time.perf_counter_ns()
                    for _ in range(256):
                        execute(indices, candidates, selected, blocks)
                    stop.record()
                    stop.synchronize()
                    samples.append(
                        dict(device_us=start.elapsed_time(stop) * 1000 / 256,
                             host_us=(time.perf_counter_ns() - host) / 1000 / 256))
                report["measurements"].append(
                    dict(capacity=capacity,
                         rows=6,
                         backing_bytes=indices.numel() * 4 + candidates.numel() * 4,
                         exact=True,
                         samples=samples))
            report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (root / f"selection-state-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
