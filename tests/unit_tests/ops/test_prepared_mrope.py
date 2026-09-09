# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope
from vllm_gaudi.ops.prepared_mrope import make_mrope_selection, prepare_mrope_coefficients
from vllm_gaudi.ops.tp2_graph_inputs import FixedDecodeInputs


@pytest.mark.parametrize("shape", [(3, 1), (3, 1, 1)])
def test_independent_positions_and_lane_map_are_exact(shape):
    selection = make_mrope_selection(64, [11, 11, 10], "cpu")
    cache = torch.arange(257 * 64).reshape(257, 64).bfloat16()
    for step in range(257):
        positions = torch.tensor([step, (step * 7 + 1) % 257, (step * 11 + 2) % 257]).reshape(shape)
        expected = []
        for part in cache[positions.reshape(3, 1)].chunk(2, -1):
            part = apply_interleaved_rope(part, [11, 11, 10])
            expected.append(torch.cat((part, part), -1).unsqueeze(1))
        actual = prepare_mrope_coefficients(cache, positions, selection)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("dim,sections", [(128, [11, 11, 10]), (64, [10, 11, 11])])
def test_unsupported_map_rejected(dim, sections):
    with pytest.raises(RuntimeError, match="qualified rotary"):
        make_mrope_selection(dim, sections, "cpu")


def make_rotary():
    from vllm_gaudi.ops.hpu_rotary_embedding import HPUMRotaryEmbedding

    rotary = HPUMRotaryEmbedding.__new__(HPUMRotaryEmbedding)
    torch.nn.Module.__init__(rotary)
    rotary.head_size = 256
    rotary.rotary_dim = 64
    rotary.mrope_section = [11, 11, 10]
    rotary.mrope_interleaved = True
    rotary.is_neox_style = True
    rotary._hpu_prepared_mrope = True
    rotary.recompute_cos_sin = False
    rotary.register_buffer("cos_sin_cache", torch.linspace(-1, 1, 257 * 64).reshape(257, 64).bfloat16())
    rotary.register_buffer("_hpu_mrope_selection", make_mrope_selection(64, [11, 11, 10], "cpu"), persistent=False)
    return rotary


@pytest.mark.parametrize("prompt,direct,accepted", [(False, True, None), (True, True, None), (False, False, None),
                                                    (False, True, object())])
def test_model_forward_uses_shared_coefficients_only_in_ordinary_decode(monkeypatch, prompt, direct, accepted):
    import vllm.forward_context
    import habana_frameworks.torch.hpex.kernels as kernels

    calls = []

    def rotary_op(x, cosine, sine, *args):
        calls.append((cosine, sine))
        first, second = x.chunk(2, -1)
        rotated = torch.cat((-second, first), -1)
        return x * cosine + rotated * sine

    monkeypatch.setattr(kernels, "apply_rotary_pos_emb", rotary_op)
    metadata = SimpleNamespace(is_prompt=prompt, direct_gdn_state=direct, num_accepted_tokens=accepted)
    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata))
    rotary = make_rotary()
    positions = torch.tensor([[3], [7], [11]])
    query, key = torch.randn(1, 3072).bfloat16(), torch.randn(1, 512).bfloat16()
    rotary.prepare_cos_sin(positions)
    prepared = rotary.cos
    actual = rotary.forward_oot(positions, query, key)
    eligible = not prompt and direct and accepted is None
    assert (calls[0][0] is prepared) == eligible
    assert calls[0][0] is calls[1][0]
    rotary._hpu_prepared_mrope = False
    expected = rotary.forward_oot(positions, query, key)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_fixed_input_binding_observes_replacement_and_fails_before_partial_copy():
    rotary = make_rotary()
    model = torch.nn.Module()
    model.rotary = rotary
    positions = torch.tensor([[3], [7], [11]])
    rotary.prepare_cos_sin(positions)
    metadata = SimpleNamespace(is_prompt=False, direct_gdn_state=True, block_size=128)
    roots = dict(positions=positions,
                 hidden_states=torch.zeros(1, 5120),
                 residual=torch.zeros(1, 5120),
                 metadata=metadata)
    original_cos, original_sin = rotary.cos, rotary.sin
    plan = FixedDecodeInputs(model, roots, [[original_cos, original_sin]])
    assert len(plan.bindings) == 2
    for offset in (1, 19, 29):
        del rotary.cos, rotary.sin
        rotary.prepare_cos_sin(positions + offset)
        updates = plan.updates(roots)
        assert len(updates) == 2
        plan.apply(updates)
        assert torch.equal(original_cos, rotary.cos) and torch.equal(original_sin, rotary.sin)
    rotary.prepare_cos_sin(positions + 31)
    before = original_cos.clone()
    rotary.sin = torch.zeros(2, 1, 64)
    assert plan.updates(roots) is None
    assert torch.equal(before, original_cos)
