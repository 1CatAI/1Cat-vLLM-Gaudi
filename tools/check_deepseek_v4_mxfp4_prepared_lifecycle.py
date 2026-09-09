# SPDX-License-Identifier: Apache-2.0
"""Validate full-shape load preparation, release, and native fallback."""
import argparse
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v4 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from vllm_gaudi.extension.ops import (  # noqa: E402
    VllmMixtureOfExpertsOpMXFP4,
    _mxfp4_fused_fwd,
)


def allocated_bytes():
    try:
        return torch.accelerator.memory_allocated()
    except Exception:
        return torch.hpu.memory_allocated()


def reset_peak():
    try:
        torch.accelerator.reset_peak_memory_stats()
    except Exception:
        torch.hpu.reset_peak_memory_stats()


def peak_bytes():
    try:
        return torch.accelerator.max_memory_allocated()
    except Exception:
        return torch.hpu.max_memory_allocated()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("HABANA_VISIBLE_MODULES") or not os.environ.get("HLS_MODULE_ID"):
        parser.error("select a free physical module explicitly")
    os.environ["VLLM_HPU_DSV4_MXFP4_PREPARED_MME"] = "1"
    os.environ["VLLM_HPU_DSV4_MXFP4_INDEXED_MME"] = "0"
    os.environ["VLLM_HPU_DSV4_TPC_MXFP4_INDEXED"] = "0"
    os.environ["VLLM_HPU_MXFP4_DECODE_GATHER"] = "0"
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])

    sample = torch.load(args.sample, map_location="cpu", weights_only=True, mmap=True)
    standard = tuple(sample[name].to("hpu").repeat(64, 1, 1).contiguous()
                     for name in ("w13", "w2", "w13_scale", "w2_scale"))
    x = torch.randn(2, 4096, dtype=torch.bfloat16, device="hpu")
    ids = torch.tensor([[3, 0, 2, 1, 3, 0], [7, 4, 6, 5, 7, 4]], dtype=torch.int32, device="hpu")
    router = torch.randn(2, 6, dtype=torch.bfloat16, device="hpu")
    standard_lists = tuple(tuple(value[index] for index in range(256)) for value in standard)
    reference = torch.compile(_mxfp4_fused_fwd, backend="hpu_backend", fullgraph=True, dynamic=False)
    expected = reference(
        x,
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

    op = VllmMixtureOfExpertsOpMXFP4(256, 256, 0, 255, tensor_parallel_size=2)
    op.set_stacked_weights(*standard)
    before = allocated_bytes()
    reset_peak()
    if not op.prepare_stacked_weights():
        raise RuntimeError("full-shape prepared contract was not selected")
    torch.hpu.synchronize()
    after = allocated_bytes()
    peak = peak_bytes()
    descriptors = op.prepared_weight_descriptors()
    assert descriptors is not None
    expected_prepared_bytes = sum(value.numel() * value.element_size()
                                  for descriptor in descriptors for value in (descriptor.q16, descriptor.s16))
    op.release_standard_weight_references()
    del standard_lists
    del standard
    torch.hpu.synchronize()

    fallback = torch.compile(op, backend="hpu_backend", fullgraph=True, dynamic=False)
    before_fallback = allocated_bytes()
    reset_peak()
    actual = fallback(x, ids, router).cpu()
    torch.hpu.synchronize()
    after_fallback = allocated_bytes()
    fallback_peak = peak_bytes()
    if not torch.equal(actual, expected):
        delta = (actual.float() - expected.float()).abs()
        raise RuntimeError(f"prepared compatibility fallback differs: nonzero={int(torch.count_nonzero(delta))} "
                           f"max_abs={float(delta.max())}")
    restored_bytes = sum((
        descriptors[0].q16.numel() * descriptors[0].q16.element_size(),
        descriptors[1].q16.numel() * descriptors[1].q16.element_size(),
        descriptors[0].s16.numel(),
        descriptors[1].s16.numel(),
    ))
    limit_bytes = 2 * 1024**3
    prepare_peak_growth = peak - before
    fallback_peak_growth = fallback_peak - before_fallback
    prepare_under_limit = prepare_peak_growth <= limit_bytes
    fallback_under_limit = fallback_peak_growth <= limit_bytes
    if not prepare_under_limit or not fallback_under_limit:
        raise RuntimeError(
            "prepared lifecycle exceeded 2 GiB peak growth: "
            f"prepare={prepare_peak_growth}, compatibility={fallback_peak_growth}"
        )
    result = {
        "qualified": True,
        "full_shapes": [list(descriptor.original_shape) for descriptor in descriptors],
        "scale_shapes": [list(descriptor.scale_shape) for descriptor in descriptors],
        "layout_version": descriptors[0].layout_version,
        "generation": descriptors[0].generation,
        "normal_scales": descriptors[0].normal_scales,
        "prepared_bytes": expected_prepared_bytes,
        "prepare_live_growth_bytes": after - before,
        "prepare_peak_growth_bytes": prepare_peak_growth,
        "prepare_peak_under_2gib": prepare_under_limit,
        "compatibility_output_bytes": restored_bytes,
        "compatibility_output_under_2gib": restored_bytes <= limit_bytes,
        "compatibility_live_growth_bytes": after_fallback - before_fallback,
        "compatibility_peak_growth_bytes": fallback_peak_growth,
        "compatibility_peak_under_2gib": fallback_under_limit,
        "standard_references_released": op._standard_weights_released,
        "fallback_bit_exact": True,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
