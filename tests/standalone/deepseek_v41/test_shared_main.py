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

    for name in ('main_publish_mla', 'main_publish_vector_mla', 'main_publish_vector_mask_mla'):
        monkeypatch.setattr(torch.ops.custom_op, f'custom_deepseek_v41_{name}_gaudi2', publish, raising=False)
    for name in ('main_reuse_mla', 'main_reuse_vector_mla', 'main_reuse_vector_mask_mla',
                 'main_reuse_native_codec_mla'):
        monkeypatch.setattr(torch.ops.custom_op, f'custom_deepseek_v41_{name}_gaudi2', reuse, raising=False)
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


@pytest.mark.parametrize("publish_mask", [False, True])
@pytest.mark.parametrize("hardware", [False, True])
def test_vector_mask_reuses_the_current_selection_owner(monkeypatch, publish_mask, hardware):
    from vllm_gaudi.ops import deepseek_v41_shared_main as module
    monkeypatch.setattr(module.gaudi_envs, "VLLM_HPU_DSV41_MLA_VECTOR_CODEC", True)
    monkeypatch.setattr(module.gaudi_envs, "VLLM_HPU_DSV41_MLA_VECTOR_MASK", True)
    monkeypatch.setattr(module.gaudi_envs, "VLLM_HPU_DSV41_MLA_PUBLISH_MASK", publish_mask)
    monkeypatch.setattr(module.gaudi_envs, "VLLM_HPU_DSV41_MLA_REUSE_HW_CODEC", hardware)
    rows, mask = torch.ones((1, 640, 512)), torch.ones((1, 640))
    calls = []

    def publish(*args):
        calls.append("publish")
        return args[0], rows.clone(), mask.clone()

    def reuse(q, swa, selected_rows, selected_mask, *args):
        calls.append("mask")
        assert selected_rows.shape == rows.shape and selected_mask.shape == mask.shape
        return q

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_main_publish_vector_mla_gaudi2", publish, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_main_publish_vector_mask_mla_gaudi2", publish, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_main_reuse_vector_mask_mla_gaudi2", reuse, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_main_reuse_native_codec_mla_gaudi2", reuse, raising=False)
    owner = Layer(20, 20, 20, torch.ones(1), torch.ones((1, 512), dtype=torch.int32)).owner
    q = torch.ones((1, 16, 512))
    arguments = owner, q, torch.tensor([16384]), torch.zeros((1, 512), dtype=torch.int32), torch.tensor([640])
    for token in range(2):
        workspace = {}
        assert shared_main_attention(*arguments, workspace) is q
        assert shared_main_attention(*arguments, workspace) is q
        owner.index_source += 4
        assert shared_main_attention(*arguments, workspace) is q
    assert calls == ["publish", "mask", "publish"] * 2


def test_vector_mask_requires_its_codec_parent(monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_shared_main as module
    monkeypatch.setattr(module.gaudi_envs, "VLLM_HPU_DSV41_MLA_VECTOR_CODEC", False)
    monkeypatch.setattr(module.gaudi_envs, "VLLM_HPU_DSV41_MLA_VECTOR_MASK", True)
    owner = Layer(20, 20, 20, torch.ones(1), torch.ones((1, 512), dtype=torch.int32)).owner
    with pytest.raises(ValueError, match="vector codec parent"):
        shared_main_attention(owner, torch.ones((1, 16, 512)), torch.tensor([0]),
                              torch.zeros((1, 512), dtype=torch.int32), torch.tensor([640]), {})


def test_hardware_reuse_requires_the_mask_parent(monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_shared_main as module
    monkeypatch.setattr(module.gaudi_envs, "VLLM_HPU_DSV41_MLA_VECTOR_CODEC", True)
    monkeypatch.setattr(module.gaudi_envs, "VLLM_HPU_DSV41_MLA_VECTOR_MASK", False)
    monkeypatch.setattr(module.gaudi_envs, "VLLM_HPU_DSV41_MLA_REUSE_HW_CODEC", True)
    owner = Layer(20, 20, 20, torch.ones(1), torch.ones((1, 512), dtype=torch.int32)).owner
    with pytest.raises(ValueError, match="vector mask parents"):
        shared_main_attention(owner, torch.ones((1, 16, 512)), torch.tensor([0]),
                              torch.zeros((1, 512), dtype=torch.int32), torch.tensor([640]), {})


def test_decoded_swa_reader_uses_current_owned_slot_after_rebind(monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_shared_main as module
    monkeypatch.setattr(module.gaudi_envs, 'VLLM_HPU_DSV41_MLA_DECODED_SWA', True)
    owner = Layer(20, 20, 20, torch.ones(1), torch.ones((1, 512), dtype=torch.int32)).owner
    q = torch.ones((1, 16, 512), dtype=torch.bfloat16)
    rows = torch.ones((1, 640, 512), dtype=torch.bfloat16)
    mask = torch.ones((1, 640))
    selected = torch.zeros((1, 512), dtype=torch.int32)
    first = torch.zeros((1536, 512), dtype=torch.bfloat16).narrow(0, 1024, 512)
    second = torch.ones((2048, 512), dtype=torch.bfloat16).narrow(0, 512, 512)
    received = []

    def reader(query, mirror, main, validity, *args):
        received.append(mirror)
        assert main is rows and validity is mask
        return query

    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_main_reuse_decoded_swa_mla_gaudi2',
                        reader, raising=False)
    for slot in (first, second):
        workspace = {(20, 20, 1): (rows, mask)}
        assert shared_main_attention(owner, q, torch.tensor([16384]), selected, torch.tensor([640]),
                                     workspace, decoded_swa=slot) is q
    assert received[0] is first and received[1] is second
    assert first.storage_offset() == 1024 * 512


def test_unavailable_mirror_retains_packed_reader(monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_shared_main as module
    monkeypatch.setattr(module.gaudi_envs, 'VLLM_HPU_DSV41_MLA_DECODED_SWA', True)
    monkeypatch.setattr(module.gaudi_envs, 'VLLM_HPU_DSV41_MLA_REUSE_HW_CODEC', True)
    monkeypatch.setattr(module.gaudi_envs, 'VLLM_HPU_DSV41_MLA_VECTOR_CODEC', True)
    monkeypatch.setattr(module.gaudi_envs, 'VLLM_HPU_DSV41_MLA_VECTOR_MASK', True)
    owner = Layer(20, 20, 20, torch.ones(1), torch.ones((1, 512), dtype=torch.int32)).owner
    q = torch.ones((1, 16, 512), dtype=torch.bfloat16)
    received = []

    def reader(query, cache, *args):
        received.append(cache)
        return query

    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_main_reuse_native_codec_mla_gaudi2',
                        reader, raising=False)
    workspace = {(20, 20, 1): (torch.ones((1, 640, 512)), torch.ones((1, 640)))}
    assert shared_main_attention(owner, q, torch.tensor([200000]), torch.zeros((1, 512), dtype=torch.int32),
                                 torch.tensor([640]), workspace) is q
    assert received == [owner.swa]


@pytest.mark.parametrize('shape,dtype', [((256, 512), torch.bfloat16), ((512, 512), torch.float32)])
def test_decoded_reader_rejects_an_incompatible_owned_slot(monkeypatch, shape, dtype):
    from vllm_gaudi.ops import deepseek_v41_shared_main as module
    monkeypatch.setattr(module.gaudi_envs, 'VLLM_HPU_DSV41_MLA_DECODED_SWA', True)
    owner = Layer(20, 20, 20, torch.ones(1), torch.ones((1, 512), dtype=torch.int32)).owner
    workspace = {(20, 20, 1): (torch.ones((1, 640, 512)), torch.ones((1, 640)))}
    with pytest.raises(ValueError, match='current publisher'):
        shared_main_attention(owner, torch.ones((1, 16, 512)), torch.tensor([16384]),
                              torch.zeros((1, 512), dtype=torch.int32), torch.tensor([640]),
                              workspace, decoded_swa=torch.zeros(shape, dtype=dtype))


@pytest.mark.parametrize('tokens', [1, 2, 6])
def test_register_softmax_switch_preserves_larger_decode_entrypoints(monkeypatch, tokens):
    from vllm_gaudi.ops import deepseek_v41_shared_main as module
    for flag in ('MLA_VECTOR_CODEC', 'MLA_VECTOR_MASK', 'MLA_PUBLISH_MASK',
                 'MLA_REUSE_HW_CODEC', 'MLA_DECODED_SWA'):
        monkeypatch.setattr(module.gaudi_envs, f'VLLM_HPU_DSV41_{flag}', False)
    monkeypatch.setattr(module.gaudi_envs, 'VLLM_HPU_DSV41_MLA_REGISTER_SOFTMAX', True)
    calls = []

    def publish(*args):
        calls.append(('publish', args[10:]))
        return args[0], torch.zeros((tokens, 640, 512)), torch.ones((tokens, 640))

    def reuse(*args):
        calls.append(('reuse', args[8:]))
        return args[0]

    prefix = 'custom_deepseek_v41_main' if tokens == 1 else 'custom_deepseek_v41_main_batch'
    monkeypatch.setattr(torch.ops.custom_op, prefix + '_publish_mla_gaudi2', publish, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, prefix + '_reuse_mla_gaudi2', reuse, raising=False)
    owner = Layer(20, 20, 20, torch.ones(1), torch.ones((tokens, 512), dtype=torch.int32)).owner
    q = torch.ones((tokens, 16, 512))
    args = (owner, q, torch.zeros(tokens, dtype=torch.int32),
            torch.zeros((tokens, 512), dtype=torch.int32), torch.full((tokens,), 640, dtype=torch.int32))
    workspace = {}
    assert shared_main_attention(*args, workspace) is q
    assert shared_main_attention(*args, workspace) is q
    extra = (True,) if tokens == 1 else ()
    assert calls == [('publish', extra), ('reuse', extra)]
