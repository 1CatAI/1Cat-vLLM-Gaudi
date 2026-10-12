# SPDX-License-Identifier: Apache-2.0
"""Fresh-process frontend reuse through a real mHC native consumer boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import FunctionType, MethodType

import torch

from deepseek_v41_micro_replay import RecipeRecorder
from vllm_gaudi.compilation.deepseek_v41_frontend_cache import GuardedFrontendEntry
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard


class Boundary(torch.nn.Module):

    def __init__(self, shard, layer):
        super().__init__()
        fn = shard.tensor(f"layers.{layer}.hc_attn_fn", "hpu").float()
        for name, value in (("fn", fn), ("scale", shard.tensor(f"layers.{layer}.hc_attn_scale", "hpu").float()),
                            ("base", shard.tensor(f"layers.{layer}.hc_attn_base", "hpu").float())):
            self.register_buffer(name, value)
        high = fn.bfloat16()
        from vllm_gaudi import envs

        self.register_buffer(
            "mme", high if envs.VLLM_HPU_DSV41_DSPARK_MHC_HIGH_PLANE else torch.cat(
                (high, (fn - high.float()).bfloat16())).contiguous())

    def forward(self, residual, pre):
        from vllm_gaudi.ops.deepseek_v41_math import hc_post, hc_pre

        hidden, pre, post, comb = hc_pre(residual,
                                         pre,
                                         self.fn,
                                         self.scale,
                                         self.base,
                                         packed_fn=self.fn,
                                         control_mme_weight=self.mme,
                                         decode=True)
        after = hc_post(hidden, residual, post, comb)
        # Consume the gate slice at its real next boundary. The terminal
        # collapsed tensor owns its allocation; no diagnostic-only slice
        # output has to be reconstructed outside the native compute plan.
        collapsed = hc_pre(after, pre, self.fn, self.scale, self.base,
                           packed_fn=self.fn, control_mme_weight=self.mme, decode=True)[0]
        return (collapsed, )


def ordinary(owner, ordinal):
    method = Boundary.forward
    function = FunctionType(method.__code__.replace(co_name=f"frontend_reference_{ordinal}"), method.__globals__)
    return torch.compile(MethodType(function, owner), backend="hpu_backend", fullgraph=True, dynamic=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args()
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import habana_frameworks.torch.core  # noqa: F401
    from habana_frameworks.torch.dynamo.compile_backend.backends import hpu_backend
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers

    bind_worker_cpu(0)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    root.mkdir(parents=True, exist_ok=True)
    recorder = RecipeRecorder(root)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    runtime = json.loads(Path(os.environ["DSV41_RUNTIME_PROFILE"]).read_text())
    for name in ("TMPDIR", "HABANA_LOGS", "PT_HPU_RECIPE_CACHE_CONFIG"):
        runtime["environment"].pop(name, None)
    source = Path(__file__).resolve().parents[1]
    runtime["frontend_source_hashes"] = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((source / "vllm_gaudi").rglob("*.py"))
    }
    runtime["frontend_check_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    runtime["torch_version"] = torch.__version__
    identity = hashlib.sha256((args.prepared / "manifest.json").read_bytes() +
                              json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    records, pairs = [], []
    with torch.inference_mode():
        owner = Boundary(shard, args.layer)
        fixtures = [
            torch.load(args.fixtures / f"layer{args.layer}-case{case}.pt", weights_only=True, map_location="cpu")
            for case in range(3)
        ]
        for count in (1, 6):
            residual = fixtures[0]["residual"][:count].to("hpu").contiguous()
            pre = fixtures[0]["pre"][:count].to("hpu").contiguous()
            directory = args.artifacts / identity / f"C{count}"
            if args.restore_only and not any(directory.glob("*.json")):
                raise RuntimeError("Fresh-process restore requires a published frontend")
            seed = GuardedFrontendEntry(Boundary.forward, owner, hpu_backend, directory, identity=identity)
            seed(residual, pre)
            torch.hpu.synchronize()
            if args.restore_only and seed.stats["captures"]:
                raise RuntimeError("Restored frontend silently retraced its input")
            for case, fixture in enumerate(fixtures):
                residual.copy_(fixture["residual"][:count])
                pre.copy_(fixture["pre"][:count])
                torch.hpu.synchronize()
                reference = ordinary(owner, count * 10 + case)
                started = time.perf_counter()
                expected = reference(residual, pre)
                torch.hpu.synchronize()
                reference_s = time.perf_counter() - started
                started = time.perf_counter()
                restored = GuardedFrontendEntry(Boundary.forward, owner, hpu_backend, directory, identity=identity)
                actual = restored(residual, pre)
                torch.hpu.synchronize()
                restore_s = time.perf_counter() - started
                exact = all(
                    torch.equal(a.cpu().view(torch.uint8),
                                b.cpu().view(torch.uint8)) for a, b in zip(expected, actual, strict=True))
                if not exact:
                    raise RuntimeError("Cached frontend changed the real mHC consumer")
                pairs.append(
                    dict(count=count,
                         case=case,
                         ordinary_seconds=reference_s,
                         restore_seconds=restore_s,
                         exact=exact,
                         cache=dict(restored.stats)))
            compiled = seed

            def execute(x, previous, compiled=compiled):
                return compiled(x, previous)

            replay = recorder.prepare(execute, [residual], [(pre, )])
            bind_worker_helpers(0)
            for fixture in fixtures:
                residual.copy_(fixture["residual"][:count])
                pre.copy_(fixture["pre"][:count])
                expected = [v.cpu() for v in compiled(residual, pre)]
                replay()
                torch.hpu.synchronize()
                if not all(
                        torch.equal(a.view(torch.uint8),
                                    b.cpu().view(torch.uint8)) for a, b in zip(expected, replay.outputs, strict=True)):
                    raise RuntimeError("Restored frontend did not retain exact native replay")
            records.append(dict(count=count, native_recipes=replay.recipes, exact_inputs=3, cache=dict(seed.stats)))
            replay.close()
    report = dict(scope="real-weight mHC producer/consumer; not full model startup",
                  fresh_process=args.restore_only,
                  pairs=pairs,
                  native_checks=records,
                  model_startup_qualified=False,
                  median_prepare_seconds={
                      str(count):
                      dict(ordinary=statistics.median(p["ordinary_seconds"] for p in pairs if p["count"] == count),
                           restored=statistics.median(p["restore_seconds"] for p in pairs if p["count"] == count))
                      for count in (1, 6)
                  })
    (root / "frontend-cache-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["median_prepare_seconds"]), flush=True)


if __name__ == "__main__":
    main()
