# SPDX-License-Identifier: Apache-2.0
"""Projection selection must preserve routing scores and FP32 logits."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi import envs
from vllm_gaudi.models import deepseek_v41_program as program


def test_router_preserves_score_formula_and_real_mask(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_ROUTER_TOP6", True)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_EXPERT_K128", True)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_EXPERT_COORD_PIPELINE", False)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_FP8_DECODE", False)
    gate = SimpleNamespace(weight=torch.randn(384, 8), bias=torch.randn(384), bias_vl=torch.randn(384))
    weights = SimpleNamespace(gate=gate,
                              experts=SimpleNamespace(w13_q16=None, w2_q16=None, w13_s16=None, w2_s16=None),
                              shared_experts=SimpleNamespace(w1=None, w2=None, w3=None))
    observations = []

    def select(scores, text, image, mask):
        assert text is gate.bias and image is gate.bias_vl
        selected = torch.argsort(scores + torch.where(mask[:, None], image, text), descending=True, stable=True)[:, :6]
        raw = scores.gather(1, selected)
        routing = raw / (raw.sum(-1, keepdim=True) + 1e-20) * 1.5
        observations.append((scores.clone(), mask.clone(), selected, routing))
        return selected.int(), routing

    def experts(value, ids, routing, *_):
        assert torch.equal(ids, observations[-1][2].int())
        assert torch.equal(routing, observations[-1][3])
        return value

    monkeypatch.setattr(
        torch.ops, "custom_op",
        SimpleNamespace(custom_deepseek_v41_router_top6_gaudi2=select,
                        custom_deepseek_v41_mxfp4_prepared_moe_k128_bf16_gaudi2=experts))
    monkeypatch.setattr(program, "linear", lambda value, _: torch.zeros_like(value))
    moe = program.PreparedMoE(weights, 6, True, torch.empty(0), lambda x: x)
    for is_image in (False, True):
        x = torch.randn(1, 8).bfloat16()
        mask = torch.tensor([is_image])
        assert torch.equal(moe(x, mask), x)
        expected = torch.nn.functional.softplus(torch.nn.functional.linear(x.float(), gate.weight)).sqrt()
        assert torch.equal(observations[-1][0], expected)
        assert torch.equal(observations[-1][1], mask)


@pytest.mark.parametrize("bf16", [False, True])
def test_head_load_has_one_weight_in_selected_dtype(monkeypatch, bf16):
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_BF16_LM_HEAD", bf16)
    source = torch.randn(8, 16).bfloat16()
    specs = {"head.weight": {"shape": [8, 16], "dtype": "BF16"}}
    shard = SimpleNamespace(specs=specs, tensor=lambda *args: source, check_identity=lambda: None)
    tree = program._weight_tree(specs)
    program.load_weight_tree(shard, tree, "cpu")
    assert tree.head.weight.dtype == (torch.bfloat16 if bf16 else torch.float32)
    assert len(list(tree.buffers())) == 1
    if bf16:
        assert tree.head.weight is source


@pytest.mark.parametrize("bf16", [False, True])
def test_router_gate_load_has_one_weight_in_selected_dtype(monkeypatch, bf16):
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_BF16_ROUTER_GATE", bf16)
    source = torch.randn(384, 8).bfloat16()
    specs = {"layers.0.ffn.gate.weight": {"shape": [384, 8], "dtype": "BF16"}}
    shard = SimpleNamespace(specs=specs, tensor=lambda *args: source, check_identity=lambda: None)
    tree = program._weight_tree(specs)
    program.load_weight_tree(shard, tree, "cpu")
    assert tree.layers.get_submodule("0").ffn.gate.weight.dtype == (torch.bfloat16 if bf16 else torch.float32)


def test_router_bf16_gate_preserves_fp32_score_transform(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_BF16_ROUTER_GATE", True)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_ROUTER_TOP6", True)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_EXPERT_K128", True)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_EXPERT_COORD_PIPELINE", False)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_FP8_DECODE", False)
    gate = SimpleNamespace(weight=torch.randn(384, 8).bfloat16(), bias=torch.randn(384), bias_vl=torch.randn(384))
    weights = SimpleNamespace(gate=gate,
                              experts=SimpleNamespace(w13_q16=None, w2_q16=None, w13_s16=None, w2_s16=None),
                              shared_experts=SimpleNamespace(w1=None, w2=None, w3=None))
    seen = {}

    def project(value, weight):
        assert value.dtype == weight.dtype == torch.bfloat16
        return torch.nn.functional.linear(value.float(), weight.float())

    def select(scores, _text, _image, _mask):
        seen["scores"] = scores
        return torch.zeros(1, 6, dtype=torch.int32), torch.ones(1, 6)

    def experts(value, *_args):
        return value

    monkeypatch.setattr(
        torch.ops, "custom_op",
        SimpleNamespace(custom_deepseek_v41_bf16_linear_f32_gaudi2=project,
                        custom_deepseek_v41_router_top6_gaudi2=select,
                        custom_deepseek_v41_mxfp4_prepared_moe_k128_bf16_gaudi2=experts))
    monkeypatch.setattr(program, "linear", lambda value, _: torch.zeros_like(value))
    moe = program.PreparedMoE(weights, 6, True, torch.empty(0), lambda value: value)
    value = torch.randn(1, 8).bfloat16()
    assert torch.equal(moe(value, torch.tensor([False])), value)
    expected = torch.nn.functional.softplus(project(value, gate.weight)).sqrt()
    assert torch.equal(seen["scores"], expected)


def test_head_greedy_and_logprob_share_fp32_projection(monkeypatch):
    calls = []
    weight = torch.randn(8, 16).bfloat16()

    def project(hidden, w):
        assert hidden.dtype == w.dtype == torch.bfloat16 and w is weight
        calls.append(hidden.clone())
        return torch.nn.functional.linear(hidden.float(), w.float())

    monkeypatch.setattr(torch.ops, "custom_op", SimpleNamespace(custom_deepseek_v41_bf16_linear_f32_gaudi2=project))
    stage = object.__new__(program.PreparedStage)
    torch.nn.Module.__init__(stage)
    stage.bf16_head, stage.pp_rank, stage.tp_rank = True, 1, 0
    stage.weights = SimpleNamespace(head=SimpleNamespace(weight=weight))
    stage.all_gather = lambda x, dim: torch.cat((x, x), dim=dim)
    for _ in range(2):
        hidden = torch.randn(1, 16).bfloat16()
        logits = stage.logits(hidden)
        assert logits.dtype == torch.float32
        assert torch.equal(stage.sample_greedy(hidden), logits.argmax(-1, keepdim=True))
        assert torch.log_softmax(logits, -1).dtype == torch.float32
    assert len(calls) == 4


def test_precision_change_invalidates_before_input_copy():
    from vllm_gaudi.ops.tp2_graph_inputs import FixedDecodeInputs

    hidden = torch.ones(1, 4)
    positions = torch.tensor([3])
    first = dict(hidden_states=hidden,
                 positions=positions,
                 metadata=SimpleNamespace(),
                 state_generation=(7, "channel-layout-a"))
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, [[hidden, positions]])
    assert bindings.updates(first) == []
    changed = dict(first, positions=torch.tensor([9]), state_generation=(7, "channel-layout-b"))
    assert bindings.updates(changed) is None
    assert positions.item() == 3
    assert bindings.updates(dict(first, state_generation=(8, "channel-layout-a"))) is None


def test_shared_gate_up_prepares_one_weight_and_preserves_math(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_SHARED_GATE_UP", True)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_EXPERT_FUSED_QUANT", False)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_ROUTER_TOP6", False)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_EXPERT_K128", False)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_EXPERT_COORD_PIPELINE", False)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_FP8_DECODE", False)
    monkeypatch.setattr(program, "quantize_activation", lambda value: value)
    w1 = torch.randn(6, 8).bfloat16()
    w3 = torch.randn(6, 8).bfloat16()
    w2 = torch.randn(8, 6).bfloat16()
    shared = SimpleNamespace(w1=SimpleNamespace(weight=w1, scale=None),
                             w3=SimpleNamespace(weight=w3, scale=None),
                             w2=SimpleNamespace(weight=w2, scale=None))
    weights = SimpleNamespace(shared_experts=shared)
    moe = program.PreparedMoE(weights, 6, True, torch.empty(0), lambda value: value)
    value = torch.randn(2, 8).bfloat16()
    gate = torch.nn.functional.linear(value, w1).float().clamp(max=10.0)
    up = torch.nn.functional.linear(value, w3).float().clamp(-10.0, 10.0)
    expected = torch.nn.functional.linear((torch.nn.functional.silu(gate) * up).bfloat16(), w2)
    moe.prepare_shared_gate_up_weight()
    assert moe.shared_gate_up_weight.shape == (12, 8)
    assert shared.w1.weight.device.type == shared.w3.weight.device.type == "meta"
    assert torch.equal(moe.shared_expert(value), expected)
    moe.release_shared_gate_up_weight()
    assert moe.shared_gate_up_weight is None
