# SPDX-License-Identifier: Apache-2.0
"""Isolate first-acquisition ViT/aligner recipe registration on one owned HPU.

This diagnostic uses the real prepared vision weights and image-grid geometry;
random patch values intentionally do not qualify visual output quality.
"""

import argparse
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment)
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config  # noqa: E402
from vllm.models.deepseek_v4.common.vision import DeepseekV4Aligner, DeepseekV4ViT  # noqa: E402
from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu  # noqa: E402


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.hpu.set_device(0)
    bind_worker_cpu(0)
    init_distributed_environment(world_size=1,
                                 rank=0,
                                 local_rank=0,
                                 backend="hccl",
                                 distributed_init_method="file://" + str(evidence / "pg-init"))
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel()
        config = DeepseekV41Config(**json.loads((args.prepared / "config.json").read_text()))
        shard = PreparedV41Shard(args.prepared, 0, 0)
        specs = {name: value for name, value in shard.specs.items() if name.startswith(("vision.", "aligner."))}
        weights = _weight_tree(specs)
        load_weight_tree(shard, weights, "hpu", specs)
        with torch.device("meta"):
            vision, aligner = DeepseekV4ViT(config), DeepseekV4Aligner(config)
        for name, module in (("vision", vision), ("aligner", aligner)):
            tree = weights.get_submodule(name)
            for target, parameter in list(module.named_parameters()):
                parent, _, attribute = target.rpartition(".")
                value = getattr(tree.get_submodule(parent), attribute)
                assert value.shape == parameter.shape
                setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))
        torch.manual_seed(41)
        patches = torch.randn(33 * 47, 3, 14, 14, dtype=torch.bfloat16, device="cpu").to("hpu")
        for _ in range(args.warmup):
            aligner(vision(patches, 33, 47), 33, 47)
        torch.hpu.synchronize()
        print(f"Vision loaded; first profiler start after {args.warmup} warmups", flush=True)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
            record_shapes=True,
            with_stack=False)
        profiler.start()
        with torch.profiler.record_function("v41::isolated_vision"):
            result = aligner(vision(patches, 33, 47), 33, 47)
            torch.hpu.synchronize()
        profiler.stop()
        profiler.export_chrome_trace(str(evidence / "vision.trace.json.gz"))
        from vllm_gaudi.ops.tp2_runtime_profile import verify_loaded_profile_libraries
        (evidence / "result.json").write_text(
            json.dumps(
                {
                    "status": "profile completed",
                    "scope": "isolated real ViT/aligner; random patches; no model quality/performance qualification",
                    "shape": list(result.shape),
                    "warmup": args.warmup,
                    "profile": verify_loaded_profile_libraries()
                },
                indent=2) + "\n")
    destroy_model_parallel()
    destroy_distributed_environment()
    print("Vision profiling diagnostic complete", flush=True)


if __name__ == "__main__":
    main()
