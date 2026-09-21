# SPDX-License-Identifier: Apache-2.0
"""Channel mapping and exact device encoding for both large projections."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_gaudi.ops.deepseek_v41_dense_fp8 import precision_config
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import (covering_scale, decode_gaudi2, encode_gaudi2,
                                               prepare_block32_rows, prepare_rows)


def test_block32_k_scales_and_woa_compatibility():
    rng = np.random.default_rng(371)
    for k in (1280, 4096):
        codes = rng.integers(0, 120, (64, k), dtype=np.uint8)
        scales = rng.integers(120, 133, (2, k // 32), dtype=np.uint8)
        q, s, _ = prepare_block32_rows(codes, scales)
        assert q.shape == codes.shape and s.shape == (64, 1)
        if k == 4096:
            old_q, old_s, _ = prepare_rows(codes, scales)
            assert np.array_equal(q, old_q) and np.array_equal(s, old_s)
        altered = scales.copy()
        altered[0, 0] += 1
        q2, s2, _ = prepare_block32_rows(codes, altered)
        assert not np.array_equal(decode_gaudi2(q) * s, decode_gaudi2(q2) * s2)
        assert np.array_equal(q[32:], q2[32:]) and np.array_equal(s[32:], s2[32:])
    with pytest.raises(ValueError):
        prepare_block32_rows(codes, np.full_like(scales, 255))
    with pytest.raises(ValueError):
        encode_gaudi2(np.array([np.nan], dtype=np.float32))


def test_precision_and_loading(monkeypatch):
    import torch
    from vllm_gaudi import envs
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_BF16_LM_HEAD", False)
    assert precision_config(None)["wq_b"] == list(range(40))
    specs = {f"layers.{i}.attn.wo_b.weight": {"dtype": "F8_E4M3", "shape": [5120, 4096]} for i in (0, 1)}
    calls = []

    def dense(name, device):
        calls.append(name)
        return torch.empty(5120, 4096, dtype=torch.bfloat16)

    def prepared(name, device):
        return (torch.ones(1, 5120) if name.endswith("channel_scale") else
                torch.empty(5120, 4096, dtype=torch.float8_e4m3fn))

    tree = _weight_tree(specs)
    shard = SimpleNamespace(specs=specs, dense=dense, check_identity=lambda: None)
    load_weight_tree(shard, tree, "cpu", dense_sidecar=SimpleNamespace(tensor=prepared),
                     dense_config={"version": 1, "wq_b": [], "wo_b": [0]})
    assert calls == ["layers.1.attn.wo_b.weight"]
    assert tree.layers.get_submodule("0").attn.wo_b.weight.dtype == torch.float8_e4m3fn
    assert tree.layers.get_submodule("0").attn.wo_b.dense_fp8


def test_engram_loading_bypasses_bf16_expansion_only_for_selected_weights(monkeypatch):
    import torch
    from vllm_gaudi import envs
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree

    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_BF16_LM_HEAD", False)
    specs = {
        f"layers.{layer}.engram.wkv.weight": {
            "dtype": "F8_E4M3",
            "shape": [8, 4]
        }
        for layer in (1, 14, 20)
    }
    calls = []

    def dense(name, device):
        calls.append(name)
        return torch.empty(8, 4, dtype=torch.bfloat16)

    def prepared(name, device):
        if name.endswith("channel_scale"):
            return torch.ones(1, 8)
        return torch.empty(8, 4, dtype=torch.float8_e4m3fn)

    tree = _weight_tree(specs)
    shard = SimpleNamespace(specs=specs, dense=dense, check_identity=lambda: None)
    load_weight_tree(shard, tree, "cpu", engram_sidecar=SimpleNamespace(tensor=prepared))
    assert calls == ["layers.20.engram.wkv.weight"]
    for layer in (1, 14):
        projection = tree.layers.get_submodule(str(layer)).engram.wkv
        assert projection.weight.dtype == torch.float8_e4m3fn
        assert projection.channel_scale.shape == (1, 8)
        assert projection.dense_fp8


@pytest.mark.skipif(os.environ.get("DSV41_TEST_HPU") != "1", reason="Requires explicit HPU lease")
@pytest.mark.parametrize("k,n", [(1280, 16384), (4096, 5120)])
def test_device_encoding_projection_and_changing_input(k, n):
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment()
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.manual_seed(k)
    values = torch.randn(3, k).bfloat16()
    values[0].zero_()
    values[1, :12] = torch.tensor([240, -240, 120, -120, .015625, -.015625, .0078125, -.0078125,
                                  1.0625, 1.1875, 0, -0.0])
    quant = torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2
    project = torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2
    compiled = torch.compile(project, backend="hpu_backend", fullgraph=True, dynamic=False)
    base = torch.randn(n, k).numpy().astype(np.float32)
    ws = covering_scale(np.max(np.abs(base), axis=1, keepdims=True))
    codes = encode_gaudi2(base / ws)
    weight = torch.from_numpy(codes).view(torch.float8_e4m3fn).to("hpu")
    scale = torch.from_numpy(ws.T.copy()).to("hpu")
    records = []
    for tokens in (1, 3):
        x = values[:tokens].to("hpu")
        for generation in range(2):
            host = values[:tokens].clone()
            if generation:
                host.add_(torch.linspace(-2, 2, k).bfloat16())
            x.copy_(host.to("hpu"))
            q, sx = quant(x)
            sx_ref = covering_scale(host.float().abs().amax(-1, keepdim=True).numpy())
            q_ref = encode_gaudi2(host.float().numpy() / sx_ref)
            assert np.array_equal(q.cpu().view(torch.uint8).numpy(), q_ref)
            assert np.array_equal(sx.cpu().numpy(), sx_ref)
            actual = compiled(x, weight, scale).cpu()
            ordinary = project(x, weight, scale).cpu()
            if not torch.equal(actual, ordinary):
                torch.save({"actual": actual, "ordinary": ordinary, "input": host, "codes": codes,
                            "channel_scale": ws, "q": q_ref, "sx": sx_ref},
                           Path(os.environ["DSV41_RUN_EVIDENCE"], f"dense-{k}-difference.pt"))
            equal = torch.equal(actual, ordinary)
            if not equal:
                # Check the unmodified runtime primitive at identical encoded
                # operands. Eager and graph compilers may select different MME
                # accumulation schedules; neither quantization nor scaling may
                # account for an unexplained discrepancy. Native replay versus
                # ordinary compiled execution is checked exactly by the micro.
                def product(a, b):
                    return torch.ops.hpu.fp8_gemm_v2(a, False, b, True, None, torch.float32,
                                                     None, None, None, False)

                raw_eager = product(q, weight).cpu()
                raw_graph = torch.compile(product, backend="hpu_backend", fullgraph=True,
                                            dynamic=False)(q, weight).cpu()
                sw_cpu, sx_cpu = scale.cpu(), sx.cpu()
                assert torch.equal((raw_eager * sw_cpu * sx_cpu).bfloat16(), ordinary)
                assert torch.equal((raw_graph * sw_cpu * sx_cpu).bfloat16(), actual)
            reference = (torch.from_numpy(decode_gaudi2(q_ref)) @ torch.from_numpy(decode_gaudi2(codes)).T)
            reference = (reference * torch.from_numpy(ws.T) * torch.from_numpy(sx_ref)).bfloat16()
            torch.testing.assert_close(actual, reference, rtol=.009, atol=.008)
            error = (actual.float() - reference.float()).abs()
            records.append({"tokens": tokens, "generation": generation, "max_abs": error.max().item(),
                            "rmse": error.square().mean().sqrt().item(), "eager_compiled_equal": equal,
                            "runtime_mme_difference_isolated": not equal})
    Path(os.environ["DSV41_RUN_EVIDENCE"], f"dense-{k}-{n}.json").write_text(json.dumps(records, indent=2))
