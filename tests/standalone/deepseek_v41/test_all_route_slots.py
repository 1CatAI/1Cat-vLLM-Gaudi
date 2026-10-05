# SPDX-License-Identifier: Apache-2.0
"""C1 scheduling must retain route/shared precision and wider-bucket dispatch."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.models.deepseek_v41_program import PreparedMoE


@pytest.mark.parametrize('decode', [False, True])
def test_public_moe_forward_passes_the_serving_decode_flag(decode):
    value = torch.ones(1, 6, dtype=torch.bfloat16)
    gate = SimpleNamespace(bias=torch.zeros(6), bias_vl=torch.zeros(6))
    seen = []

    def body(*args, **kwargs):
        seen.append(kwargs)
        return value

    moe = SimpleNamespace(weights=SimpleNamespace(gate=gate, experts=None), router_top6=False,
                          topk=6, n256=True, n256_fp8=True, n256_fused_reduce=True,
                          prefill_grouped=False, prefill_mxfp4=False,
                          _router_logits=lambda *args: torch.ones(1, 6),
                          shared_expert=lambda *args: torch.zeros_like(value),
                          _forward_n256_fp8=body, reduce=lambda x: x)
    PreparedMoE.forward(moe, value, torch.zeros(1, dtype=torch.bool), decode=decode,
                        prequant=(value, torch.ones(1, 1)))
    assert seen[0]['decode'] == decode
    assert not seen[0]['ordinary_decode']


@pytest.mark.parametrize('tp_size', [2, 4])
@pytest.mark.parametrize('tokens,ordinary,decode', [(1, True, False), (1, False, True),
                                                 (1, False, False), (2, True, False), (6, True, False)])
def test_all_slots_keeps_shared_rounding_and_non_c1_dispatch(monkeypatch, tp_size, tokens, ordinary, decode):
    tensor = torch.ones(1)
    experts = SimpleNamespace(**{key: tensor for key in (
        'w13_q16', 'w2_q16', 'w13_s16', 'w2_s16', 'w13_fp8_channel', 'w2_fp8_channel')})
    moe = SimpleNamespace(weights=SimpleNamespace(experts=experts), lookup=tensor,
                          tensor_parallel_size=tp_size, batch_expert_reuse=False, batch_route_pack=False,
                          concurrent_moe_rows=0, n256_fused=True, n256_fused_reduce=True, normal_scales=True,
                          all_route_slots=True, batch_w13_horizontal=False, N256_PREFILL_TILE=128,
                          feature_silu=False, expert_pipeline3=False, expert_streamed_sat=False, expert_w2_three_routes=False, token_wide_experts=False)
    calls = []
    routed = torch.tensor([[1., -1., 0., -0., 256., -256.]], dtype=torch.bfloat16).expand(tokens, -1).clone()
    shared = torch.tensor([[0.00390625, 0.00390625, -0., -0., 0.5, -0.5]], dtype=torch.bfloat16)

    def invoke(kind, *args):
        calls.append(kind)
        assert args[1].dtype == torch.int32 and args[2].dtype == torch.float32
        return (routed.float() + args[-2].float()).bfloat16() if kind == 'shared' else routed

    ops = torch.ops.custom_op
    for kind, name in (
        ('slots', 'custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_slots_fp8_gaudi2'),
        ('shared', 'custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_shared_prefetch_w2_fp8_gaudi2'),
        ('batch', 'custom_deepseek_v41_expert_n256_moe_prequant_fused_reduce_fp8_gaudi2'),
    ):
        monkeypatch.setattr(ops, name, lambda *args, kind=kind: invoke(kind, *args), raising=False)
    value = torch.ones(tokens, 6, dtype=torch.bfloat16)
    ids = torch.arange(6).expand(tokens, -1)
    prequant = value, torch.ones(tokens, 1)
    with_shared = shared if tokens == 1 else None
    output = PreparedMoE._forward_n256_fp8(
        moe, value, ids, torch.ones_like(ids).float(), ordinary_decode=ordinary, decode=decode,
        prequant=prequant, shared=with_shared,
    )
    expected = routed if with_shared is None else (routed.float() + shared.float()).bfloat16()
    assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    assert calls == ['slots' if tokens == 1 and (ordinary or decode) else 'shared' if tokens == 1 else 'batch']
