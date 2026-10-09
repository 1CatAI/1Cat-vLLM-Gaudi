# SPDX-License-Identifier: Apache-2.0
"""A collapse must not survive an intervening update or a group invocation."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.models.deepseek_v41_program import PreparedDecoderLayer, PreparedLayerGroup


class Layer(torch.nn.Module):
    def __init__(self, number, *, engram=False, enabled=True):
        super().__init__()
        self.layer = number
        self.weights = SimpleNamespace(engram=True) if engram else SimpleNamespace()
        self.mhc_interlayer_collapse = enabled
        self.draft = False
        self.used_handoff = []

    def forward(self, residual, pre, positions, image_mask, rows, *, collapse_handoff=None,
                publish_collapse=False, **kwargs):
        collapsed = None if collapse_handoff is None else collapse_handoff.pop(self.layer, None)
        self.used_handoff.append(collapsed is not None)
        if hasattr(self.weights, 'engram'):
            assert collapsed is None
            residual = (residual.float() + positions[:, None, None]).bfloat16()
        if collapsed is None:
            collapsed = (residual.float() * pre[:, :, None]).sum(1).bfloat16()
        residual = (residual.float() + collapsed[:, None, :].float() * (self.layer + 1) / 100).bfloat16()
        pre = torch.full_like(pre, (self.layer + 1) / 10)
        if publish_collapse:
            assert collapse_handoff is not None
            collapse_handoff[self.layer + 1] = (residual.float() * pre[:, :, None]).sum(1).bfloat16()
        return residual, pre, residual.mean(1) if self.layer == 38 else None


@pytest.mark.parametrize('tp,tokens,decode,numbers,engram,disabled,expected', [
    (4, 1, True, [0, 1, 2, 3], 1, None, [False, False, True, True]),
    (4, 2, True, [12, 13, 14, 15], 14, None, [False, True, False, True]),
    (4, 1, True, [4, 5, 6, 7], None, 5, [False, False, False, True]),
    (4, 1, True, [32, 34, 37, 38], None, None, [False, False, False, True]),
    (4, 6, True, [4, 5, 6, 7], None, None, [False, True, True, True]),
    (4, 6, True, [12, 13, 14, 15], 14, None, [False, True, False, True]),
    (4, 8, False, [4, 5, 6, 7], None, None, [False] * 4),
    (2, 1, True, [4, 5, 6, 7], None, None, [False] * 4),
])
@torch.inference_mode()
def test_handoff_ownership_and_fresh_invocation(monkeypatch, tp, tokens, decode, numbers, engram, disabled, expected):
    from vllm_gaudi.models import deepseek_v41_program as program
    monkeypatch.setattr(program.gaudi_envs, 'VLLM_HPU_DSV41_MHC_GATES_FUSED', False)
    identities = []
    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_bf16_identity_gaudi2',
                        lambda x: identities.append(x.shape) or x.clone(), raising=False)
    layers = [Layer(i, engram=i == engram, enabled=i != disabled) for i in numbers]
    stage = SimpleNamespace(tensor_parallel_size=tp, pp_rank=0, is_last_stage=False,
                            layers=layers, config={'text_config': {'rms_norm_eps': 1e-6}})
    group = PreparedLayerGroup(stage, 0, 4, decode=decode)
    generator = torch.Generator().manual_seed(41131)
    for iteration in range(3):
        value = torch.randn(tokens, 4, 8, generator=generator).bfloat16()
        pre = torch.rand(tokens, 4, generator=generator)
        positions = torch.arange(tokens) + iteration
        reference, ref_pre = value, pre
        for layer in layers:
            reference, ref_pre, _ = layer(reference, ref_pre, positions, None, None)
        identities.clear()
        result, result_pre, _ = group(value, pre, positions, torch.zeros(tokens, dtype=torch.int64), (None, None))
        assert torch.equal(reference, result) and torch.equal(ref_pre, result_pre)
        assert [layer.used_handoff[-1] for layer in layers] == expected
        assert len(identities) == (3 - sum(expected) if tp == 4 and decode else 0)


def test_engram_rejects_stale_precomputed_input():
    layer = PreparedDecoderLayer.__new__(PreparedDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.weights = SimpleNamespace(engram=True)
    layer.layer = 14
    with pytest.raises(RuntimeError, match='Engram update invalidates'):
        layer(torch.ones(1, 4, 5120), None, None, None, collapse_handoff={14: torch.ones(1)})
