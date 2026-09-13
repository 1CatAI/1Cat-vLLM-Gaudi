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
