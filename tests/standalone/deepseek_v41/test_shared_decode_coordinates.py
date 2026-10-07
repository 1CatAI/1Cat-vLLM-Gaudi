# SPDX-License-Identifier: Apache-2.0
"""Cross-group metadata must refresh from device-feedback roots once per token."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.models.deepseek_v41_program import PreparedLayerGroup
from vllm_gaudi.ops.deepseek_v41_decode_coordinates import decode_coordinates_reference


class Layer(torch.nn.Module):
    def __init__(self, number, pages):
        super().__init__()
        self.layer = number
        self.attention = SimpleNamespace(shared_decode_metadata=True, shared=SimpleNamespace(block_table=pages))
        self.seen = []

    def forward(self, residual, pre_mix, positions, image_mask, rows, **kwargs):
        self.seen.append((positions, image_mask, kwargs['decode_metadata']))
        return residual, pre_mix, None


@pytest.mark.parametrize('tokens', [1, 2, 6])
@pytest.mark.parametrize('tp', [2, 4])
def test_refresh_reorder_and_shared_owned_outputs(monkeypatch, tokens, tp):
    from vllm_gaudi.models import deepseek_v41_program as program
    calls = []

    def producer(pos, ids, pages):
        calls.append(pos.clone())
        return (decode_coordinates_reference(pos, ids, pages), pos.bitwise_and(255),
                torch.full_like(pos, 640), ((ids == 129264) | (ids == 129265)).int())

    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_decode_metadata_gaudi2', producer, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_bf16_identity_gaudi2', lambda x: x.clone(), raising=False)
    monkeypatch.setattr(program.gaudi_envs, 'VLLM_HPU_DSV41_MHC_GATES_FUSED', False)
    pages = torch.arange(4096, dtype=torch.int32).flip(0)
    layers = [Layer(i, pages) for i in range(8)]
    stage = SimpleNamespace(tensor_parallel_size=tp, pp_rank=0, is_last_stage=False, layers=layers,
                            config={'text_config': {'rms_norm_eps': 1e-6}})
    first, second = (PreparedLayerGroup(stage, start, start + 4, decode=True) for start in (0, 4))
    positions = torch.tensor([127, 255, 16383, 32767, 131071, 524286][:tokens])
    ids = torch.tensor([129264, 17, 129265, 31, 53, 47][:tokens])
    residual, pre = torch.zeros(tokens, 4, 2), torch.zeros(tokens, 4)
    original_address = positions.data_ptr()
    for step in range(5):
        before = positions.clone()
        outputs, shared = first.coordinate_forward(residual, pre, positions, ids, (None, None))
        _, same = second.coordinate_forward(*outputs[:2], positions, ids, (None, None), shared)
        assert same is shared and len(calls) == step + 1
        assert positions.data_ptr() == original_address and torch.equal(positions, before)
        assert shared[0].dtype == torch.int32 and torch.equal(shared[0].long(), positions)
        for layer in layers:
            pos, mask, metadata = layer.seen[-1]
            assert pos is shared[0] and mask is shared[1]
            assert metadata[0] is shared[2] and metadata[1] is shared[3]
            assert torch.equal(metadata[0], positions.remainder(256))
        for tensor in shared:
            assert tensor.storage_offset() == 0 and tensor.is_contiguous()
        # Simulate sampled feedback and request-slot reorder in the same roots.
        positions.add_(1)
        if step == 2:
            positions.copy_(positions.flip(0))
            ids.copy_(ids.flip(0))


def test_sampling_tail_mutates_canonical_roots_not_i32_metadata(monkeypatch):
    from vllm_gaudi.models import deepseek_v41_program as program
    monkeypatch.setattr(program.gaudi_envs, 'VLLM_HPU_DSV41_MHC_GATES_FUSED', False)
    monkeypatch.setattr(program, 'final_collapse_rms_norm', lambda residual, *_: residual[:, 0])
    layer = Layer(0, torch.arange(8, dtype=torch.int32))
    stage = SimpleNamespace(tensor_parallel_size=4, pp_rank=0, is_last_stage=True, layers=[layer],
                            weights=SimpleNamespace(norm=SimpleNamespace(weight=torch.ones(2))),
                            config={'text_config': {'rms_norm_eps': 1e-6}})
    group = PreparedLayerGroup(stage, 0, 1, decode=True)
    positions, ids = torch.tensor([255]), torch.tensor([17])
    shared = (positions.int(), torch.tensor([False]), torch.tensor([255]), torch.tensor([640]),
              torch.zeros(1, 192, dtype=torch.int32))

    class Tail(torch.nn.Module):
        def forward(self, hidden, pos, token):
            assert pos is positions and token is ids
            pos.add_(1)
            token.fill_(41)
            return (token.clone(),)

    group.greedy_tail = Tail()
    group.coordinate_forward(torch.zeros(1, 4, 2), torch.zeros(1, 4), positions, ids, (None, None), shared)
    assert positions.item() == 256 and ids.item() == 41
    assert shared[0].item() == 255
