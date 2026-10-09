# SPDX-License-Identifier: Apache-2.0
"""Validate C6 fused residual/collapse through its control and quant consumers."""

import json
import os
from pathlib import Path


def main():
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_math import hc_post

    torch.hpu.set_device(0)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    op = torch.ops.custom_op

    def consumer(updated, collapsed, control, norm):
        gates = op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(updated.flatten(1), control, 1e-20)
        normalized, quantized, scale = op.custom_deepseek_v41_ffn_norm_quant_gaudi2(collapsed, norm, 1e-6)
        return updated, collapsed, gates, normalized, quantized.view(torch.uint8), scale

    def candidate(value, residual, post, comb, pre, control, norm):
        updated, collapsed = op.custom_deepseek_v41_mhc_post_collapse_gaudi2(value, residual, post, comb, pre)
        return consumer(updated, collapsed, control, norm)

    def parent(value, residual, post, comb, pre, control, norm):
        updated = hc_post(value, residual, post, comb)
        collapsed = (updated.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
        return consumer(updated, collapsed, control, norm)

    fused = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
    legacy = torch.compile(parent, backend="hpu_backend", fullgraph=True, dynamic=False)
    generator = torch.Generator().manual_seed(641206)
    checks = []
    with torch.inference_mode():
        control = (torch.randn(24, 20480, generator=generator) * 0.01).to("hpu")
        norm = (torch.rand(5120, generator=generator) + 0.5).bfloat16().to("hpu")
        for magnitude in (0.0, 0.03125, 1.0, 256.0):
            for iteration in range(4):
                x = (torch.randn(6, 5120, generator=generator) * magnitude).bfloat16().to("hpu")
                residual = (torch.randn(6, 4, 5120, generator=generator) * magnitude).bfloat16().to("hpu")
                post = torch.rand(6, 4, generator=generator).to("hpu")
                comb = torch.rand(6, 4, 4, generator=generator).to("hpu")
                pre = torch.rand(6, 4, generator=generator).to("hpu")
                args = x, residual, post, comb, pre, control, norm
                observed = [v.cpu() for v in fused(*args)]
                independent = [fused(*(v[row : row + 1].clone() for v in args[:5]), control, norm) for row in range(6)]
                expected = [torch.cat([item[index].cpu() for item in independent]) for index in range(6)]
                original = [v.cpu() for v in legacy(*args)]
                exact = [
                    torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(observed, expected, strict=True)
                ]
                errors = [
                    dict(
                        exact=torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
                        max_abs=float((a.float() - b.float()).abs().max()),
                    )
                    for a, b in zip(observed, original, strict=True)
                ]
                checks.append(
                    dict(magnitude=magnitude, iteration=iteration, c6_vs_c1_exact=exact, legacy_errors=errors)
                )
                (root / "collapse-result.json").write_text(
                    json.dumps(dict(status="running", cases=checks, performance_measured=False), indent=2) + "\n"
                )
                if not all(exact):
                    raise AssertionError("C6 fused consumer differs from six C1 consumers")
        (root / "collapse-result.json").write_text(
            json.dumps(dict(status="passed", cases=checks, performance_measured=False), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
