# SPDX-License-Identifier: Apache-2.0
"""Compile the complete indexed BF16 MoE for placement inspection.

This is a candidate structural check, not a performance qualification. It
does not run a baseline or load a model. Set GRAPH_VISUALIZATION and its
output directory before launch to inspect Synapse's actual memory planning.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v4 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from vllm_gaudi.ops.deepseek_v4_mxfp4 import normal_e8m0_scales  # noqa: E402


def make_inputs(sample_path=None):
    """Saved synthetic contract, or four loaded experts reused in six slots."""
    torch.manual_seed(20260902)
    ids = torch.tensor([[251, 3, 127, 64, 191, 9]], dtype=torch.int32)
    router = torch.tensor([[.24, .21, .18, .15, .12, .10]], dtype=torch.bfloat16)
    x = (torch.randn(1, 4096, dtype=torch.bfloat16) * .01).to("hpu")
    w13 = torch.full((256, 2048, 2048), 0x11, dtype=torch.uint8, device="hpu")
    w2 = torch.full((256, 4096, 512), 0x11, dtype=torch.uint8, device="hpu")
    s13 = torch.full((256, 2048, 128), 120, dtype=torch.uint8, device="hpu")
    s2 = torch.full((256, 4096, 32), 120, dtype=torch.uint8, device="hpu")
    # Reuse the saved same-graph microbenchmark's selected-expert pattern.
    for slot, expert in enumerate(ids.flatten().tolist()):
        nibble = (1, 2, 3, 5, 9, 10)[slot]
        w13[expert].fill_(nibble | (nibble << 4))
        w2[expert].fill_(nibble | (nibble << 4))
        s13[expert].fill_(119 + slot)
        s2[expert].fill_(119 + slot)
    inputs = [x, ids.to("hpu"), router.to("hpu"), w13, w2, s13, s2]
    if sample_path is not None:
        sample = torch.load(sample_path, map_location="cpu", weights_only=True, mmap=True)
        for slot, expert in enumerate(ids.flatten().tolist()):
            for target, name in ((w13, "w13"), (w2, "w2"), (s13, "w13_scale"), (s2, "w2_scale")):
                target[expert].copy_(sample[name][slot % 4])
        inputs[0] = (torch.randn(1, 4096) * .75).bfloat16().to("hpu")
    torch.hpu.synchronize()
    return inputs, ids, router


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normal-scales", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("HABANA_VISIBLE_MODULES") or not os.environ.get("HLS_MODULE_ID"):
        parser.error("select free physical modules explicitly")
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    inputs, ids, router = make_inputs()
    normal = args.normal_scales and normal_e8m0_scales(*inputs[5:7])
    x = inputs[0]
    print("Compiling full TP2-shape indexed MoE", flush=True)
    op = torch.ops.custom_op.custom_deepseek_v4_mxfp4_indexed_mme_bf16_gaudi2
    fn = torch.compile(op, backend="hpu_backend", fullgraph=True, dynamic=False)
    output = fn(*inputs, normal).cpu()
    assert output.shape == (1, 4096) and torch.isfinite(output).all()
    # Independent scalar reference for this constant-row weight pattern.
    # It retains the BF16 boundaries between both matrix products, SiLU,
    # multiplication and router weighting; no candidate decoder is reused.
    magnitudes = torch.tensor([.5, 1., 1.5, 3., -.5, -1.])
    weights = magnitudes * torch.exp2(torch.arange(119, 125).float() - 127)
    x_cpu = x.cpu().float()

    def reference(order):
        routed = []
        for slot, index in enumerate(order):
            w = weights[index]
            gate = (x_cpu * w).sum(-1).bfloat16()
            activated = torch.nn.functional.silu(gate.float()).bfloat16() * gate
            down = (activated.float() * w * 1024).bfloat16()
            routed.append(down * router[0, slot])
        return torch.stack(routed).float().sum(0).bfloat16().expand(1, 4096)

    expected = reference(range(6))
    torch.testing.assert_close(output, expected, rtol=.008, atol=1e-7)
    print("Compiled and executed full TP2-shape MoE", flush=True)
    inputs[1].copy_(ids.flip(1))
    changed = fn(*inputs, normal).cpu()
    torch.testing.assert_close(changed, reference(reversed(range(6))), rtol=.008, atol=1e-7)
    inputs[1].copy_(ids)
    replay = fn(*inputs, normal).cpu()
    assert torch.equal(output, replay), "same IDs must reproduce the same output"
    assert not torch.equal(output, changed), "runtime expert IDs must affect route weighting"
    result = {
        "status": "executed; memory placement requires graph inspection",
        "qualified": False, "shape": {"batch": 1, "hidden": 4096, "intermediate": 1024,
                                       "experts": 256, "topk": 6},
        "torch": torch.__version__, "runtime_injection": False,
        "normal_scales": normal,
        "output_sha256": hashlib.sha256(output.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "replay_exact": True, "dynamic_ids_observed": True,
        "synthetic_reference_exact": torch.equal(output, expected),
        "output_first": output.float().flatten()[:8].tolist(),
        "timing": None, "baseline_run": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
