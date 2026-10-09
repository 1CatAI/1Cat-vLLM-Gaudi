# SPDX-License-Identifier: Apache-2.0
"""Static-scale compound ABI on checkpoint matrices; no performance qualification."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace("{rank}", "0")
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_native_libraries
    prepare_native_libraries()
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from vllm_gaudi import envs
    from vllm_gaudi.ops.deepseek_v41_hw_dense import ROUNDING_MARGIN, encode_weight, project, scale_covering
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm, quantize_activation
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import decode_e4m3fn, decode_gaudi2

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    library = Path(os.environ["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"])
    pattern = "hpu_dsv41_hw_dense_fused_pt2*.so" if envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE_FUSED_QUANT else (
        "hpu_dsv41_hw_dense_pt2*.so")
    torch.ops.load_library(next(library.glob(pattern)))
    shard = PreparedV41Shard(args.prepared, 0, 0)
    checks = []
    compiled = torch.compile(project, backend="hpu_backend", fullgraph=True, dynamic=False)
    for case, layer in enumerate((0, 20, 39)):
        data = torch.load(args.fixtures / f"rank0/c6-{case}.pt", weights_only=True)
        # Derived real hidden inputs check only the lowering/ABI. Full native
        # conditional acceptance uses true per-layer states in a separate gate.
        norm = shard.tensor(f"layers.{layer}.attn_norm.weight", "cpu")
        qnorm = shard.tensor(f"layers.{layer}.attn.q_norm.weight", "cpu")
        qweight = shard.dense(f"layers.{layer}.attn.wq_a.weight", "cpu")
        kvweight = shard.dense(f"layers.{layer}.attn.wkv.weight", "cpu")
        query_weight = shard.dense(f"layers.{layer}.attn.wq_b.weight", "cpu")
        value_raw = rms_norm(data["hidden"], norm)
        value = quantize_activation(value_raw)
        q = torch.nn.functional.linear(value, qweight)
        query_raw = rms_norm(q, qnorm)
        query = quantize_activation(query_raw)
        if envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE_FUSED_QUANT:
            value, query = value_raw, query_raw
        for name, x, weight, gamma in (("qkv", value, torch.cat((qweight, kvweight)), norm),
                                        ("query", query, query_weight, qnorm)):
            encoded, sw = encode_weight(weight)
            sx = scale_covering(x.shape[1] ** .5 * float(gamma.float().abs().max()) * ROUNDING_MARGIN)
            # The built-in converter retains E4M3 subnormals. Its native MME
            # honors their tensor exponent bias; the old custom C1 encoder's
            # explicit flush is a different arithmetic contract, not an ABI
            # reference. Model quality still needs whole-Target alpha.
            reference_input = (quantize_activation(x)
                               if envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE_FUSED_QUANT else x)
            codes = (reference_input.float() / sx).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
            qx = torch.from_numpy(decode_e4m3fn(codes) * sx)
            qw = torch.from_numpy(decode_gaudi2(encoded.view(torch.uint8).numpy()) * sw)
            reference = (qx @ qw.t()).bfloat16()
            x_device, w_device = x.to("hpu"), encoded.to("hpu")
            caster_input = (quantize_activation(x_device)
                            if envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE_FUSED_QUANT else x_device)
            q_device = torch.ops.hpu.cast_to_fp8_v2(caster_input, 1 / sx, False, False, torch.float8_e4m3fn)[0]
            ordinary = torch.ops.hpu.fp8_gemm_v2(q_device, False, w_device, True, None,
                                               torch.bfloat16, sx, sw, None, False)
            torch.hpu.synchronize()
            ordinary = ordinary.cpu()
            (root / "hw-dense-capability-stage.json").write_text(json.dumps(
                dict(stage="standard_scalar_diagnostic_not_reference", projection=name, case=case,
                     standard_max_abs=float((ordinary.float() - reference.float()).abs().max()),
                     performance_qualified=False), indent=2) + "\n")
            actual = compiled(x_device, w_device, sx, sw)
            torch.hpu.synchronize()
            actual = actual.cpu()
            error = actual.float() - reference.float()
            relative = float(error.norm() / reference.float().norm().clamp_min(1e-20))
            checks.append(dict(case=case, layer=layer, projection=name, rows=x.shape[0],
                               k=x.shape[1], n=weight.shape[0], input_scale=sx, weight_scale=sw,
                               maximum_abs=float(error.abs().max()), relative_l2=relative,
                               finite=bool(torch.isfinite(actual).all())))
            (root / "hw-dense-capability.json").write_text(json.dumps(
                dict(status="checking", checks=checks, performance_qualified=False), indent=2) + "\n")
            if not torch.isfinite(actual).all() or relative > .005:
                torch.save(dict(actual=actual, reference=reference, input=x, weight_codes=encoded.view(torch.uint8),
                                input_codes_cpu=torch.from_numpy(codes.copy()),
                                input_codes_public=q_device.cpu().view(torch.uint8),
                                input_scale=sx, weight_scale=sw), root / "hw-dense-reference-diagnostic.pt")
                raise ValueError("Hardware scalar lowering differs from its explicitly decoded FP8 operands")
    (root / "hw-dense-capability.json").write_text(json.dumps(
        dict(status="capability_passed", checks=checks, performance_qualified=False,
             acceptance_qualified=False, note="Derived real inputs; full production native gate remains required"),
        indent=2) + "\n")


if __name__ == "__main__":
    main()
