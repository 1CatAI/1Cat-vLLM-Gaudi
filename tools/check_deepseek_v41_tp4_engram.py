# SPDX-License-Identifier: Apache-2.0
"""Qualify full-chunk TP4 Engram projection/update and its mHC consumer."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree, linear
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    from vllm_gaudi.ops.deepseek_v41_engram_fp8 import EngramFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_math import engram_update, prefill_engram_update, prefill_hc_input
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://", local_rank=rank,
                                 backend="hccl")
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    output = root / f"engram-rank{rank}.json"
    report = dict(rank=rank, status="running", cases=[],
                  boundary="four-way row gather -> real FP8 Engram projection -> gate/update -> mHC/RMS consumer")
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            _, gather = stage_collectives(rank, False)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            layers = (1, 14)
            specs = {name: spec for name, spec in shard.specs.items()
                     if any(name.startswith(f"layers.{layer}.{part}") for layer in layers
                            for part in ("engram.", "attn_norm.", "hc_attn_"))}
            tree = _weight_tree(specs)
            load_weight_tree(shard, tree, "hpu", specs,
                             engram_sidecar=EngramFP8Sidecar(args.prepared / "sidecars/engram_fp8", shard))
            text = json.loads((args.prepared / "config.json").read_text())["text_config"]
            eps, hc_eps, iterations = (text[key] for key in ("rms_norm_eps", "hc_eps", "hc_sinkhorn_iters"))
            for step, count in enumerate((8192, 128, 8192)):
                torch.manual_seed(9927 + step)
                residual = torch.randn(count, 4, 5120).bfloat16().to("hpu")
                previous = torch.full((count, 4), .25, device="hpu")
                active = torch.arange(count, device="hpu") % 7 != 2
                torch.manual_seed(5550 + 4 * step + rank)
                rows = torch.randn(count, 6, 256).bfloat16().to("hpu")
                for layer in layers:
                    w = tree.layers.get_submodule(str(layer))

                    def consume(value):
                        return prefill_hc_input(value, previous, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base,
                                                w.attn_norm.weight, eps, hc_eps, iterations, w.hc_attn_fn)[-1]

                    def execute(update):
                        projected = linear(gather(rows, 1).flatten(1), w.engram.wkv)
                        updated = update(residual, projected, w.engram.q_weight, w.engram.k_weight, active, eps)
                        return updated, consume(updated)

                    torch.hpu.synchronize()
                    torch.hpu.reset_peak_memory_stats()
                    resident = torch.hpu.memory_allocated()
                    reference = tuple(value.cpu() for value in execute(engram_update))
                    reference_peak = torch.hpu.max_memory_allocated()
                    candidate = tuple(value.cpu() for value in execute(prefill_engram_update))
                    for actual, expected in zip(candidate, reference, strict=True):
                        torch.testing.assert_close(actual.float(), expected.float(), atol=.03125, rtol=.03)
                        assert torch.isfinite(actual).all()
                    mask = active.cpu()
                    assert torch.equal(candidate[0][~mask], residual.cpu()[~mask])
                    torch.save(dict(actual=candidate, expected=reference), root / f"engram-rank{rank}-{step}-{layer}.pt")
                    torch.hpu.synchronize()
                    torch.hpu.reset_peak_memory_stats()
                    candidate_resident = torch.hpu.memory_allocated()
                    timings = []
                    for _ in range(3):
                        residual.add_(.001)
                        torch.hpu.synchronize()
                        start = time.perf_counter()
                        result = execute(prefill_engram_update)
                        torch.hpu.synchronize()
                        timings.append((time.perf_counter() - start) * 1000)
                        del result
                    report["cases"].append(dict(tokens=count, layer=layer,
                                                bitexact=[torch.equal(a, b) for a, b in zip(candidate, reference)],
                                                max_abs_error=[float((a.float() - b.float()).abs().max())
                                                               for a, b in zip(candidate, reference)],
                                                reference_resident_bytes=resident,
                                                reference_peak_bytes=reference_peak,
                                                candidate_resident_bytes=candidate_resident,
                                                candidate_peak_bytes=torch.hpu.max_memory_allocated(),
                                                host_sync_ms=timings))
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    print(f"TP{rank} Engram L{layer} C{count}: passed", flush=True)
            report["status"] = "passed"
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status="failed", error=repr(error))
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        if report["status"] != "passed":
            os._exit(1)


if __name__ == "__main__":
    main()
