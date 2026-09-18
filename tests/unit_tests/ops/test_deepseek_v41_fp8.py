# SPDX-License-Identifier: Apache-2.0
"""Precision dispatch and exact prepared-channel contracts for ordinary C1."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_gaudi.models.deepseek_v41_program import PreparedMoE
from vllm_gaudi.ops.deepseek_v41_fp8 import channel_scales, precision_config
from vllm_gaudi.ops.deepseek_v41_weights import prepare_q16, prepare_s16


def test_n256_fused_finalize_is_c1_only(monkeypatch):
    from vllm_gaudi import envs

    flags = {
        "VLLM_HPU_DSV41_ROUTER_TOP6": True,
        "VLLM_HPU_DSV41_EXPERT_K128": False,
        "VLLM_HPU_DSV41_EXPERT_N256": False,
        "VLLM_HPU_DSV41_EXPERT_N256_FP8": True,
        "VLLM_HPU_DSV41_EXPERT_FUSED_QUANT": True,
        "VLLM_HPU_DSV41_EXPERT_FUSED_REDUCE": True,
        "VLLM_HPU_DSV41_BF16_ROUTER_GATE": False,
        "VLLM_HPU_DSV41_SHARED_GATE_UP": False,
        "VLLM_HPU_DSV41_EXPERT_COORD_PIPELINE": False,
        "VLLM_HPU_DSV41_FP8_DECODE": False,
    }
    for name, value in flags.items():
        monkeypatch.setattr(envs, name, value)

    calls = []

    def select(scores, *_args):
        tokens = scores.shape[0]
        return torch.zeros(tokens, 6, dtype=torch.int32), torch.ones(tokens, 6)

    def expert(kind):

        def run(value, *operands):
            assert operands[-3] is experts.w13_fp8_channel
            assert operands[-2] is experts.w2_fp8_channel
            assert operands[-1] is True
            calls.append((kind, value.shape[0]))
            return torch.zeros_like(value)

        return run

    monkeypatch.setattr(
        torch.ops, "custom_op",
        SimpleNamespace(custom_deepseek_v41_router_top6_gaudi2=select,
                        custom_deepseek_v41_expert_n256_moe_fused_reduce_fp8_gaudi2=expert("finalize"),
                        custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2=expert("body")))
    experts = SimpleNamespace(w13_q16=None,
                              w2_q16=None,
                              w13_s16=None,
                              w2_s16=None,
                              w13_channel=True,
                              w2_channel=True,
                              w13_fp8_channel=torch.ones(1, dtype=torch.bfloat16),
                              w2_fp8_channel=torch.ones(1, dtype=torch.bfloat16))
    weights = SimpleNamespace(gate=SimpleNamespace(weight=torch.zeros(384, 8),
                                                   bias=torch.zeros(384),
                                                   bias_vl=torch.zeros(384)),
                              experts=experts,
                              shared_experts=SimpleNamespace())
    moe = PreparedMoE(weights, 6, True, torch.zeros(128), lambda value: value)
    moe.shared_expert = lambda value: torch.zeros_like(value)
    for tokens in (1, 6):
        value = torch.zeros(tokens, 8, dtype=torch.bfloat16)
        assert torch.equal(moe(value, torch.zeros(tokens, dtype=torch.bool), fp8_decode=True), value)
    assert calls == [("finalize", 1), ("body", 6)]


def test_fp8_experts_only_run_for_enabled_native_c1(monkeypatch):
    calls = []

    def operator(kind):

        def run(*args):
            calls.append((kind, args[1].clone()))
            return torch.zeros_like(args[0])

        return run

    monkeypatch.setattr(
        torch.ops, "custom_op",
        SimpleNamespace(custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2=operator("bf16"),
                        custom_deepseek_v41_mxfp4_prepared_moe_fp8_gaudi2=operator("fp8")))
    torch.manual_seed(417)
    shared = SimpleNamespace(w1=SimpleNamespace(weight=torch.zeros(2, 4, dtype=torch.bfloat16)),
                             w3=SimpleNamespace(weight=torch.zeros(2, 4, dtype=torch.bfloat16)),
                             w2=SimpleNamespace(weight=torch.zeros(4, 2, dtype=torch.bfloat16)))
    weights = SimpleNamespace(gate=SimpleNamespace(weight=torch.randn(8, 4),
                                                   bias=torch.zeros(8),
                                                   bias_vl=torch.zeros(8)),
                              experts=SimpleNamespace(w13_q16=None, w2_q16=None, w13_s16=None, w2_s16=None),
                              shared_experts=shared)
    moe = PreparedMoE(weights, 6, True, torch.zeros(128), lambda value: value)
    moe.fp8 = True
    x, mask = torch.randn(1, 4).bfloat16(), torch.zeros(1, dtype=torch.bool)
    moe(x, mask)  # C1 prefill tail keeps the ordinary contract.
    moe(x, mask, fp8_decode=True)
    moe(-x, mask, fp8_decode=True)
    moe(x.expand(6, -1).contiguous(), mask.expand(6))
    with pytest.raises(ValueError, match="requires C1"):
        moe(x.expand(6, -1).contiguous(), mask.expand(6), fp8_decode=True)
    moe.fp8 = False
    moe(x, mask, fp8_decode=True)
    assert [kind for kind, _ in calls] == ["bf16", "fp8", "fp8", "bf16", "bf16"]
    assert torch.equal(calls[0][1], calls[1][1])
    assert not torch.equal(calls[1][1], calls[2][1])


def test_channel_scale_is_the_smallest_covering_power_and_rejects_unqualified_ranges():
    magnitudes = (0., .5, 1., 1.5, 2., 3., 4., 6.)
    for k in (128, 1152, 5120):
        for code, value in enumerate(magnitudes):
            raw = np.full((128, k // 2), code | (code << 4), dtype=np.uint8)
            q, shape = prepare_q16(raw)
            s, _ = prepare_s16(np.full((128, k // 32), 127, dtype=np.uint8), shape)
            bits, audit = channel_scales(q, s)
            exponent = int(bits[0, 0] >> 7) - 127
            assert (value == 0 and exponent == 0) or 240 * 2.**(exponent - 1) < value <= 240 * 2.**exponent
            assert audit["temporary_upper_bound_bytes"] < 2 * 2**30
    raw = np.full((128, 64), 0x77, dtype=np.uint8)
    q, shape = prepare_q16(raw)
    codes = np.full((128, 4), 127, dtype=np.uint8)
    codes[:, 0] = 100
    s, _ = prepare_s16(codes, shape)
    with pytest.raises(ValueError, match="subnormal"):
        channel_scales(q, s)
    for code in (0, 1, 255):
        s, _ = prepare_s16(np.full((128, 4), code, dtype=np.uint8), shape)
        with pytest.raises(ValueError, match="finite normal"):
            channel_scales(q, s)


def test_precision_configuration_rejects_unimplemented_or_ambiguous_selections(tmp_path):
    config = precision_config("")
    assert config["routed_experts"] == list(range(40))
    path = tmp_path / "precision.json"
    for field, value in (("attention", [0]), ("routed_experts", [0, 0]), ("routed_experts", [40])):
        path.write_text(json.dumps(dict(config, **{field: value})))
        with pytest.raises(ValueError):
            precision_config(path)
