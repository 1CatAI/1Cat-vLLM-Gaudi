# SPDX-License-Identifier: Apache-2.0
"""Compare prepared compatibility prefill with stock MXFP4 on real weights."""

import argparse
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v4 import prepare_environment

prepare_environment()

import habana_frameworks.torch.core  # noqa: E402, F401
import torch  # noqa: E402

from check_deepseek_v4_mxfp4_prepared_checkpoint import CheckpointLayer  # noqa: E402
from vllm_gaudi.extension.ops import (  # noqa: E402
    VllmMixtureOfExpertsOpMXFP4,
    _mxfp4_fused_fwd,
)


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = actual.float() - expected.float()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "nonzero": int(torch.count_nonzero(delta)),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-tensors", action="store_true", help="Archive input and both exact comparison outputs")
    args = parser.parse_args()
    if not os.environ.get("HABANA_VISIBLE_MODULES") or not os.environ.get("HLS_MODULE_ID"):
        parser.error("select one free physical module explicitly")
    if args.tokens < 2:
        parser.error("prefill comparison requires at least two tokens")
    os.environ["VLLM_HPU_DSV4_MXFP4_PREPARED_MME"] = "1"
    os.environ["VLLM_HPU_DSV4_MXFP4_INDEXED_MME"] = "0"
    os.environ["VLLM_HPU_DSV4_TPC_MXFP4_INDEXED"] = "0"
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])

    reader = CheckpointLayer(args.model, args.layer, args.tp_rank)
    result = {
        "qualified": False,
        "model": str(args.model),
        "layer": args.layer,
        "tp_rank": args.tp_rank,
        "tokens": args.tokens,
        "source": "all 256 checkpoint experts reconstructed with the production loader contract",
        "baseline_run": False,
        "runtime_injection": False,
    }
    try:
        standard_cpu = [reader.loaded_expert(expert) for expert in range(256)]
        standard = tuple(
            torch.stack(tuple(expert[field] for expert in standard_cpu)).to("hpu")
            for field in range(4)
        )
        del standard_cpu
        standard_lists = tuple(tuple(value.unbind(0)) for value in standard)

        generator = torch.Generator().manual_seed(
            20260908 + args.layer * 1000 + args.tp_rank * 100
        )
        hidden = torch.randn(args.tokens, 4096, generator=generator).bfloat16().to("hpu")
        ids = torch.stack(
            [torch.randperm(256, generator=generator)[:6] for _ in range(args.tokens)]
        ).to(torch.int32).to("hpu")
        router = torch.rand(args.tokens, 6, generator=generator)
        router /= router.sum(dim=-1, keepdim=True)
        router = router.bfloat16().to("hpu")

        native = torch.compile(
            _mxfp4_fused_fwd,
            backend="hpu_backend",
            fullgraph=True,
            dynamic=False,
        )
        expected = native(
            hidden,
            ids,
            router,
            *standard_lists,
            32,
            "silu",
            0,
            255,
            0,
            0,
        ).cpu()
        torch.hpu.synchronize()
        if args.save_tensors:
            torch.save(dict(hidden_states=hidden.cpu(), expert_ids=ids.cpu(), router_weights=router.cpu(),
                            reference=expected), args.output.with_suffix(".reference.pt"))

        op = VllmMixtureOfExpertsOpMXFP4(
            256,
            256,
            0,
            255,
            tensor_parallel_size=2,
        )
        op.set_stacked_weights(*standard)
        if not op.prepare_stacked_weights():
            raise RuntimeError("full-shape prepared contract was not selected")
        torch.hpu.synchronize()
        op.release_standard_weight_references()
        del standard_lists
        del standard

        compatibility = torch.compile(
            op,
            backend="hpu_backend",
            fullgraph=True,
            dynamic=False,
        )
        actual = compatibility(hidden, ids, router).cpu()
        torch.hpu.synchronize()
        result["comparison"] = compare(actual, expected)
        if args.save_tensors:
            torch.save(dict(hidden_states=hidden.cpu(), expert_ids=ids.cpu(), router_weights=router.cpu(),
                            reference=expected, candidate=actual), args.output.with_suffix(".pt"))
        result["qualified"] = result["comparison"]["exact"]
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not result["qualified"]:
            raise SystemExit(2)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
