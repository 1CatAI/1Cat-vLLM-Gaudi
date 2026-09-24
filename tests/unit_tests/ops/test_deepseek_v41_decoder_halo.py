# SPDX-License-Identifier: Apache-2.0
"""Safety boundary for the experimental PP1 decoder prefill halo."""

import pytest
import torch
from torch import nn
from types import SimpleNamespace

from vllm_gaudi.ops.deepseek_v41_decoder_halo import decoder_halo_mode, decoder_halo_rows


def test_halo_covers_final_output_and_future_local_caches():
    retained = decoder_halo_rows(19)
    assert retained == 4096
    # Trace each layer's trailing 128 cache rows back through its preceding
    # causal windows. All source-layer rows must lie in the retained suffix.
    for layer in range(1, 20):
        source_span = 128 + (layer - 1) * 127
        assert source_span <= retained
    assert retained >= 1 + 19 * 127


def test_halo_uses_aligned_prefix_blocks_across_prompt_lengths():
    count = 8192
    assert [decoder_halo_mode(start, count, 32768, eligible=True)
            for start in range(0, 32768, count)] == ["prefix_only"] * 3 + ["final"]
    assert [decoder_halo_mode(start, count, 16384, eligible=True)
            for start in range(0, 16384, count)] == ["prefix_only", "final"]
    assert [
        decoder_halo_mode(start, n, 20480, eligible=True)
        for start, n in ((0, count), (count, count), (2 * count, 4096))
    ] == ["prefix_only", "prefix_only", "final"]
    assert [
        decoder_halo_mode(start, n, 20000, eligible=True)
        for start, n in ((0, count), (count, count), (2 * count, 3616))
    ] == ["prefix_only", "full", "final"]
    assert [
        decoder_halo_mode(start, n, 32769, eligible=True)
        for start, n in ((0, count), (count, count), (2 * count, count), (3 * count, count), (4 * count, 1))
    ] == ["prefix_only", "prefix_only", "prefix_only", "full", "final"]
    assert decoder_halo_mode(0, count, 32768, eligible=False) == "full"
    assert decoder_halo_mode(0, 4096, 32768, eligible=True) == "full"
    assert decoder_halo_mode(1, count, 32768, eligible=True) == "full"
    assert decoder_halo_mode(0, count, count, eligible=True) == "full"
    assert decoder_halo_mode(0, count, 32768, eligible=True, block_tokens=2048) == "full"


@pytest.mark.parametrize("start,count,prompt_tokens", [(-1, 8192, 32768), (0, 0, 32768), (24576, 8192, 32767)])
def test_halo_rejects_invalid_prompt_chunk(start, count, prompt_tokens):
    with pytest.raises(ValueError):
        decoder_halo_mode(start, count, prompt_tokens, eligible=True)


def test_final_halo_rebinds_source_selection_for_every_reuse_layer(monkeypatch):
    from vllm_gaudi.models import deepseek_v41_program as program

    class Source(nn.Module):

        def __init__(self, shared):
            super().__init__()
            self.shared = shared

        def forward(self, residual, pre_mix, positions, image_mask):
            count = positions.numel()
            self.shared.candidate_pool[:count, 0].copy_(positions)
            self.shared.topk["20"].indices[:count, 0].copy_(positions)
            return residual, pre_mix, None

    class Consumer(nn.Module):

        def __init__(self, shared):
            super().__init__()
            self.shared = shared

        def forward(self, residual, pre_mix, positions, image_mask):
            count = positions.numel()
            assert torch.equal(self.shared.candidate_pool[:count, 0], positions)
            assert torch.equal(self.shared.topk["20"].indices[:count, 0], positions)
            return residual, pre_mix, None

    shared = SimpleNamespace(candidate_pool=torch.full((8192, 1), -1, dtype=torch.int32),
                             topk={"20": SimpleNamespace(indices=torch.full((8192, 1), -1, dtype=torch.int32))},
                             prefill_main_workspace=None)
    stage = program.PreparedStage.__new__(program.PreparedStage)
    nn.Module.__init__(stage)
    stage.pp_rank, stage.dspark, stage.start, stage.stop = 1, False, 20, 40
    stage.shared = shared
    stage.layers = nn.ModuleList([Source(shared)] + [Consumer(shared) for _ in range(19)])
    stage.config = {"text_config": {"rms_norm_eps": 1e-6}}
    stage.weights = SimpleNamespace(norm=SimpleNamespace(weight=torch.ones(1)))
    monkeypatch.setattr(program, "final_collapse_rms_norm", lambda residual, pre_mix, weight, eps: residual[:, 0])

    positions = torch.arange(24576, 32768, dtype=torch.int32)
    residual = positions.float().reshape(-1, 1, 1).expand(-1, 4, 1).contiguous()
    pre_mix = torch.zeros((8192, 4), dtype=torch.float32)
    output, out_pre, aux = stage._forward_prefill_halo(residual, pre_mix, positions, positions, "final")
    assert output.shape == (8192, 1)
    assert torch.equal(output[-4096:, 0], positions[-4096:].float())
    assert out_pre.shape == (8192, 4)
    assert aux is None
