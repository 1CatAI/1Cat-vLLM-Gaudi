# SPDX-License-Identifier: Apache-2.0
"""Profile only the eight decode selectors and FullR1 block publication.

Inputs use the production BF16 score boundary and 16K-row workload shapes.
This small diagnostic excludes score generation, model weights and serving.
Its kernel budget is not an end-to-end or full-model trace qualification.
"""

import json
import os
from pathlib import Path


def main():
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    if "{rank}" in os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", ""):
        os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].replace("{rank}", str(rank))
    if os.getenv("GRAPH_VISUALIZATION") == "1":
        directory = Path(os.environ["GRAPH_VISUALIZATION_DIR"]) / f"rank{rank}"
        directory.mkdir(parents=True, exist_ok=True)
        os.environ["GRAPH_VISUALIZATION_DIR"] = str(directory)
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch.core  # noqa: F401
    from habana_frameworks.torch.hpu.metrics import metric_global
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_native_trace import NativeTrace, scope
    from vllm_gaudi.ops.deepseek_v41_ordered_selection import ordered_index_ids

    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    bind_worker_helpers(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    dist.init_process_group("gloo")
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(rank=rank, scope=__doc__, status="preparing", cases=[], gain_ledger_credit=False,
                  new_full_model_requests=0, profiled_steps=8,
                  owners=[2, 8, 14, 20, 24, 28, 32, 36],
                  expected_threshold_calls_per_step=9, expected_emit_calls_per_step=9,
                  limitations=["Precomputed scores isolate selector kernels; no claim of model latency or quality.",
                               "Four-card alignment is this component's CPU barrier, not model collectives."])

    def save():
        (root / f"selection-profile-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    def oracle(scores, width, candidates=None):
        ids = scores.argsort(dim=-1, descending=True, stable=True)[:, :width].sort(-1).values
        ids = torch.where(scores.gather(1, ids) > -torch.inf, ids, -1)
        if candidates is not None:
            mapped = candidates.gather(1, ids.clamp_min(0) // 8) * 8 + ids.clamp_min(0) % 8
            ids = torch.where(ids >= 0, mapped, -1)
        return ids.int()

    def full_r2(scores, pos, pool):
        return ordered_index_ids(scores, pos, pool, 2)

    def full_r1(scores, blocks, pos, pool):
        return (ordered_index_ids(scores, pos, pool, 1),
                ordered_index_ids(blocks, pos, pool, 1, blocks=True))

    def reindex(scores, pos, pool):
        return ordered_index_ids(scores, pos, pool, 1, reindex=True)

    functions = [torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
                 for fn in (full_r2, full_r1, reindex)]
    trace = None
    try:
        with torch.inference_mode():
            gen = torch.Generator().manual_seed(7701)
            dummy = torch.full((1, 2048), -1, dtype=torch.int32, device="hpu")
            banks = []
            for generation in range(2):
                position = 16514 + generation
                pos = torch.tensor([position], dtype=torch.int32)
                full = torch.randn(1, 20480, generator=gen).bfloat16().float()
                full[:, position + 1:] = -torch.inf
                blocks = full.reshape(1, -1, 8).amax(-1)
                blocks[:, position // 8] = torch.inf
                expected_pool = oracle(blocks, 2048)
                logical = (expected_pool[..., None] * 8 + torch.arange(8)).flatten(1)
                r2 = [torch.randn(1, 10240, generator=gen).bfloat16().float() for _ in range(3)]
                for scores in r2:
                    scores[:, (position + 1) // 2:] = -torch.inf
                selected = []
                for _ in range(4):
                    source = torch.randn(1, 20480, generator=gen).bfloat16().float()
                    source[:, position + 1:] = -torch.inf
                    selected.append(source.gather(1, logical.long()))
                expected = [oracle(x, 512) for x in r2] + [oracle(full, 512)] + [
                    oracle(x, 512, expected_pool) for x in selected]
                bank = dict(pos=pos.to("hpu"), r2=[x.to("hpu") for x in r2],
                            full=full.to("hpu"), blocks=blocks.to("hpu"),
                            selected=[x.to("hpu") for x in selected],
                            expected=expected, expected_pool=expected_pool)
                banks.append(bank)

            def step(bank):
                output = [functions[0](scores, bank["pos"], dummy) for scores in bank["r2"]]
                ids, pool = functions[1](bank["full"], bank["blocks"], bank["pos"], dummy)
                output.append(ids)
                output.extend(functions[2](scores, bank["pos"], pool) for scores in bank["selected"])
                return output, pool

            for generation, bank in enumerate(banks):
                output, pool = step(bank)
                assert torch.equal(pool.cpu(), bank["expected_pool"])
                assert all(torch.equal(got.cpu(), want) for got, want in zip(output, bank["expected"], strict=True))
                report["cases"].append(dict(generation=generation, eight_owner_ids_exact=True,
                                             ordered_candidate_publication_exact=True))
            for i in range(32):
                step(banks[i % 2])
            torch.hpu.synchronize()
            metric = metric_global("graph_compilation")
            before = dict(metric.stats())["TotalNumber"]
            dist.barrier()
            trace = NativeTrace(cpu_trace_dir=root / f"traces/rank{rank}", scope_only=True)
            trace.start()
            for i in range(report["profiled_steps"]):
                dist.barrier()
                with scope(f"v41::selection_token::step{i}::rank{rank}"):
                    output, pool = step(banks[i % 2])
                    torch.hpu.synchronize()
            trace.stop()
            assert dict(metric.stats())["TotalNumber"] == before
            assert torch.equal(pool.cpu(), banks[1]["expected_pool"])
            assert all(torch.equal(got.cpu(), want) for got, want in zip(output, banks[1]["expected"], strict=True))
            report.update(status="kernel_trace_complete", no_hot_compilation=True, metadata=trace.metadata)
            save()
            print(json.dumps(dict(rank=rank, status=report["status"],
                                  captures=trace.metadata["raw_files"])), flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save()
        raise
    finally:
        if trace is not None and trace.running:
            trace.stop()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
