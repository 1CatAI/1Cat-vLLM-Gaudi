# SPDX-License-Identifier: Apache-2.0
"""Check that joint preparation retains the shared producer and reference router."""
import torch
from torch import nn

from vllm_gaudi.models.deepseek_v41_program import PreparedMoE


def test_joint_weight_retains_shared_columns_and_reference(monkeypatch):
    monkeypatch.setenv('VLLM_HPU_DSV41_DSPARK_ROUTER_SHARED', '1')
    torch.manual_seed(42)
    owner = PreparedMoE.__new__(PreparedMoE)
    nn.Module.__init__(owner)
    owner.topk = 6
    owner.weights = nn.Module()
    owner.weights.gate = nn.Module()
    router = (torch.randn(384, 5120) * .02).bfloat16().float()
    owner.weights.gate.register_buffer('weight', router)
    shared = (torch.randn(1280, 5120) * .02).to(torch.float8_e4m3fn)
    owner.register_buffer('shared_gate_up_weight', shared)
    owner.shared_gate_up_channel = torch.ones(1, 5, 256, dtype=torch.bfloat16)
    owner.prepare_router_shared_weight()
    assert owner.router_shared_weight.shape == (1792, 5120)
    assert owner.router_shared_columns == 1280
    assert owner.router_shared_channel.shape == (1, 384)
    assert torch.equal(owner.router_shared_weight[:1280].view(torch.uint8), shared.view(torch.uint8))
    assert torch.count_nonzero(owner.router_shared_weight[-128:].float()) == 0
    assert owner.weights.gate.weight is router
    assert torch.isfinite(owner.router_shared_weight.float()).all()
    assert owner.router_shared_weight.float().abs().max() <= 240
    for rows in (2, 5, 6):
        q = torch.randn(rows, 5120).to(torch.float8_e4m3fn).float()
        reference = q @ shared.float().t()
        joined = q @ owner.router_shared_weight.float().t()
        torch.testing.assert_close(joined[:, :1280], reference, rtol=1e-5, atol=1e-5)


def test_decode_flag_does_not_exclude_native_c6(monkeypatch):
    from types import SimpleNamespace

    class JointReached(Exception):
        pass

    class ReferenceReached(Exception):
        pass

    def joint(*args):
        raise JointReached

    def reference(*args):
        raise ReferenceReached

    owner = PreparedMoE.__new__(PreparedMoE)
    nn.Module.__init__(owner)
    owner.weights = nn.Module()
    owner.dspark_shared_prequant = False
    owner.router_shared_weight = torch.empty(1)
    owner._router_logits = reference
    monkeypatch.setattr(torch.ops, 'hpu', SimpleNamespace(fp8_gemm_v2=joint))
    monkeypatch.setattr(torch.ops, 'custom_op', SimpleNamespace(
        custom_deepseek_v41_dense_quant_gaudi2=lambda x: (x, torch.ones(x.shape[0], 1))))
    import pytest

    with pytest.raises(JointReached):
        owner(torch.empty(6, 5120), torch.zeros(6, dtype=torch.bool), ordinary_decode=True)
    with pytest.raises(ReferenceReached):
        owner(torch.empty(1, 5120), torch.zeros(1, dtype=torch.bool), ordinary_decode=True)
    with pytest.raises(ReferenceReached):
        owner(torch.empty(128, 5120), torch.zeros(128, dtype=torch.bool), decode=False)
