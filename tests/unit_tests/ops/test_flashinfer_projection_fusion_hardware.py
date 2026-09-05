# SPDX-License-Identifier: Apache-2.0
"""Run in a fresh process: adapter registration lives for the compiler lifetime."""

import json
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLASHINFER_GAUDI_RUN_HARDWARE_TESTS") != "1",
                                reason="explicit Gaudi2 hardware test opt-in required")


def test_graph_fusion_public_recipe_reentry_and_shape_fallback(tmp_path):
    import habana_frameworks.torch  # noqa: F401
    from habana_frameworks.torch.dynamo.compile_backend import config, passes
    from flashinfer_gaudi import load_native_extensions
    from tools.benchmark_flashinfer_native_projection import audit_trace, validate
    from vllm_gaudi.ops.flashinfer_projection_fusion import register_projection_fusion_pass, projection_fusion_stats

    assert load_native_extensions()
    config.use_eager_fallback = False
    torch.hpu.init()

    def projection(x, residual, gamma, weight, weight_scale):
        # Bridge may reuse the first private sum's storage for the second add.
        # The adapter must preserve this without ever mutating a graph input.
        summed = (x + residual) + x
        normed = torch.ops.hpu.rms_norm(summed, gamma, 1e-6, None, False)[0]
        normed = normed.view(-1, normed.shape[-1]).reshape(x.shape)
        scale = torch.ops.hpu.calculate_scale_for_cast(normed, 2, 0, -1, True, 240., 1.) + 1e-8 / 240.
        q = torch.ops.hpu.cast_to_fp8_v2(normed, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
        output = torch.ops.hpu.fp8_gemm_v2(q, False, weight, True, None, torch.bfloat16, scale.float(), weight_scale,
                                           None, False)
        return output, summed

    generator = torch.Generator().manual_seed(134)
    raw_weight = (torch.randn(34816, 5120, generator=generator, dtype=torch.bfloat16) * .02).to("hpu")
    weight_scale = raw_weight.float().abs().amax(-1) / 240.
    weight = torch.ops.hpu.cast_to_fp8_v2(raw_weight, weight_scale[:, None].reciprocal(), False, False,
                                          torch.float8_e4m3fn)[0]
    gamma = (torch.randn(5120, generator=generator) * .1 + 1.).bfloat16().to("hpu")
    samples = {
        batch: (torch.randn(batch, 5120, generator=generator).bfloat16().to("hpu"),
                torch.randn(batch, 5120, generator=generator).bfloat16().to("hpu"), gamma, weight, weight_scale)
        for batch in (1, 8, 32)
    }
    compiled = torch.compile(projection, backend="hpu_backend", fullgraph=True, dynamic=False)
    snapshots = {batch: tuple(value.cpu() for value in inputs[:2]) for batch, inputs in samples.items()}
    expected = {batch: tuple(value.cpu() for value in compiled(*inputs)) for batch, inputs in samples.items()}
    register_projection_fusion_pass(((8, 5120, 34816), ))
    adapted_graphs = []

    def audit_context(ctx):
        targets = [str(node.target) for node in ctx.graph_module.graph.nodes if node.op == "call_function"]
        if "custom_op.flashinfer_gaudi_add_rmsnorm_quant.default" in targets:
            assert targets.count("aten.add.Tensor") == 1
            assert targets.count("custom_op.flashinfer_gaudi_add_rmsnorm_quant.default") == 1
            assert not any(target.startswith("aten.add_") for target in targets)
            adapted_graphs.append(targets)
        return False

    passes.register_pass_at_optimization_pass(audit_context, passes.OptimizationPassPlacement.PRE_PLACEMENT)
    torch._dynamo.reset()
    with torch.inference_mode():
        for batch in (1, 8, 32, 8):
            validate(compiled(*samples[batch]), expected[batch])
    assert projection_fusion_stats()["compiled_matches"] == 1
    stream = torch.hpu.Stream()
    with torch.hpu.stream(stream):
        outputs = compiled(*samples[8])
    torch.hpu.synchronize()
    validate(outputs, expected[8])
    for batch, inputs in samples.items():
        for value, snapshot in zip(inputs[:2], snapshots[batch]):
            torch.testing.assert_close(value.cpu(), snapshot, rtol=0, atol=0)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]) as profiler:
        for _ in range(5):
            compiled(*samples[8])
        torch.hpu.synchronize()
    path = tmp_path / "trace.json"
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    # This context deliberately has one additional residual producer. Audit it
    # separately; do not weaken the complete projection benchmark's allowlist.
    # One logical TPC node emits multiple engine/slice activity records. The FX
    # audit checks node cardinality; runtime Launch events check recipe count.
    assert adapted_graphs
    assert any(event.get("cat") == "kernel" and event.get("name") == "add_fwd_bf16" for event in events)
    audit_trace(
        [event for event in events if not (event.get("cat") == "kernel" and event.get("name") == "add_fwd_bf16")])
