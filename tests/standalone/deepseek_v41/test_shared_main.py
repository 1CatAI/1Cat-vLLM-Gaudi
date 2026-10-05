# SPDX-License-Identifier: Apache-2.0
"""Do not reuse selected rows across ownership changes, tokens, or requests."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.models.deepseek_v41_program import PreparedLayerGroup
from vllm_gaudi.ops.deepseek_v41_shared_main import shared_main_attention


@pytest.fixture
def native_reference(monkeypatch):
    def publish(q, swa, main, selected, positions, pages, sink, scale, lengths, ratio):
        rows = main.index_select(0, selected.flatten().long()).clone()
        return q + rows.sum() + swa.sum(), rows, torch.ones(1)

    def reuse(q, swa, rows, mask, positions, sink, scale, lengths):
        return q + rows.sum() + swa.sum()

    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_main_publish_mla_gaudi2', publish, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_main_reuse_mla_gaudi2', reuse, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_bf16_identity_gaudi2',
                        lambda x: x.clone(), raising=False)
    return publish


class Layer(torch.nn.Module):
    def __init__(self, layer, source, index, main, selected):
        super().__init__()
        self.layer = layer
        self.owner = SimpleNamespace(kv_source=source, index_source=index, ratio=1, cache=SimpleNamespace(main=main),
                                     shared=SimpleNamespace(block_table=torch.ones(1, dtype=torch.int32)),
                                     swa=torch.tensor([layer], dtype=torch.float32),
                                     weights=SimpleNamespace(attn_sink=torch.zeros(1)), scale=torch.ones(1))
        self.selected = selected

    def forward(self, residual, pre_mix, positions, image_mask, rows, *, fp8_decode, decode, selected_main,
                decode_metadata=None):
        lengths = torch.ones(1, dtype=torch.int32)
        if selected_main is None:
            output = torch.ops.custom_op.custom_deepseek_v41_main_publish_mla_gaudi2(
                residual, self.owner.swa, self.owner.cache.main, self.selected, positions,
                self.owner.shared.block_table, self.owner.weights.attn_sink, self.owner.scale, lengths, 1)[0]
        else:
            output = shared_main_attention(self.owner, residual, positions, self.selected, lengths, selected_main)
        return output, pre_mix, None


@pytest.mark.parametrize('tp,tokens,decode', [(4, 1, True), (4, 2, True), (4, 6, True),
                                             (4, 8, False), (2, 1, True)])
@torch.inference_mode()
def test_layer_group_ownership_and_each_invocation_are_exact(native_reference, monkeypatch, tp, tokens, decode):
    from vllm_gaudi.models import deepseek_v41_program as program
    monkeypatch.setattr(program.gaudi_envs, 'VLLM_HPU_DSV41_MHC_GATES_FUSED', False)
    main = torch.arange(16).float().reshape(8, 2)
    selected = torch.tensor([[1, 2]], dtype=torch.int32)
    different = torch.tensor([[3, 4]], dtype=torch.int32)
    other_main = main + 100
    layers = [Layer(20, 20, 20, main, selected), Layer(21, 20, 20, main, selected),
              Layer(24, 20, 24, main, different), Layer(28, 28, 24, other_main, different)]
    stage = SimpleNamespace(tensor_parallel_size=tp, pp_rank=0, is_last_stage=False,
                            layers=layers, config={'text_config': {'rms_norm_eps': 1e-6}})
    group = PreparedLayerGroup(stage, 0, 4, decode=decode)
    for iteration in range(3):
        # Same tensor bindings, changed contents: cached values must never
        # survive a token boundary or a reused request slot.
        main.add_(iteration + 1)
        selected.copy_(torch.tensor([[iteration + 1, iteration + 2]], dtype=torch.int32))
        value = torch.full((tokens, 4, 2), float(iteration))
        positions = torch.full((tokens,), 16384 + iteration, dtype=torch.int32)
        expected = value.clone()
        pre = torch.zeros(tokens, 4)
        for layer in layers:
            expected = layer(expected, pre, positions, torch.zeros(tokens, dtype=torch.bool), None,
                             fp8_decode=False, decode=decode, selected_main=None)[0]
        observed = group(value, pre, positions, torch.zeros(tokens, dtype=torch.int64), (None, None))[0]
        assert torch.equal(observed, expected)


def test_projected_publish_preserves_selection_workspace(monkeypatch):
    calls = []
    rows, mask = torch.ones((1, 640, 512)), torch.ones((1, 640))
    partial = torch.ones((1, 5120), dtype=torch.bfloat16)

    def publish(*args):
        calls.append(('publish', args))
        return partial, rows, mask

    def reuse(*args):
        calls.append(('reuse', args))
        assert args[2] is rows and args[3] is mask
        return partial

    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_main_publish_projection_gaudi2', publish, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_main_reuse_projection_gaudi2', reuse, raising=False)
    owner = Layer(20, 20, 20, torch.ones(1), torch.ones((1, 512), dtype=torch.int32)).owner
    owner.weights.wo_a = SimpleNamespace(weight=torch.ones(1), channel_scale=torch.ones(1))
    owner.weights.wo_b = SimpleNamespace(weight=torch.ones(1), channel_scale=torch.ones(1))
    phase = torch.ones((128, 64))
    owner._rotary_native_table = lambda: phase
    workspace = {}
    args = owner, torch.ones((1, 16, 512)), torch.tensor([0]), torch.zeros((1, 512), dtype=torch.int32), torch.tensor([640])
    for _ in range(2):
        assert shared_main_attention(*args, workspace, projection=True) is partial
    assert [name for name, _ in calls] == ['publish', 'reuse']
    assert calls[0][1][-3] is phase and calls[1][1][-3] is phase
    owner.index_source = 24
    assert shared_main_attention(*args, workspace, projection=True) is partial
    assert [name for name, _ in calls] == ['publish', 'reuse', 'publish']
