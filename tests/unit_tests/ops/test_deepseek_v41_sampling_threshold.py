# SPDX-License-Identifier: Apache-2.0
"""Coverage proof, full normalizer and unsupported-bucket contracts."""
import torch
from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet, sample_nucleus_packet


def install_coarse_selector(monkeypatch):
    emitted = []

    def threshold(scores, position, ratio, reindex, blocks, variant):
        assert (ratio, reindex, blocks, variant) == (1, 0, 0, 1)
        assert position.item() + 1 == scores.shape[-1]
        bits = (scores.view(torch.int32).long() >> 16) & 65535
        keys = torch.where(bits > 32767, (~bits) & 65535, bits ^ 32768)
        chosen = keys.argsort(dim=-1, descending=True, stable=True)[:, :512]
        cut = keys.gather(-1, chosen[:, -1:])
        metadata = torch.zeros((1, 50), dtype=torch.int32)
        metadata[:, 1:2] = (keys > cut).sum(-1, keepdim=True).int()
        emitted[:] = [chosen.int()]
        return metadata

    def emit(*args):
        return emitted[0]

    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_index_threshold_gaudi2', threshold, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_index_emit_gaudi2', emit, raising=False)


def state(columns):
    return torch.tensor([columns - 1], dtype=torch.int32), torch.zeros((1, 2048), dtype=torch.int32)


def test_certified_original_fp32_candidates_and_full_normalizer_match(monkeypatch):
    install_coarse_selector(monkeypatch)
    logits = torch.randn((1, 4096), generator=torch.Generator().manual_seed(23)) * 7
    controls = torch.tensor([[1., .95, .7, -1.]])
    original = local_nucleus_packet(logits, controls, 2, 128)
    screened = local_nucleus_packet(logits, controls, 2, 128, threshold_state=state(4096))
    assert torch.equal(original.view(torch.uint8), screened.view(torch.uint8))


def test_coarse_tie_bucket_cannot_silently_drop_true_fp32_topk(monkeypatch):
    install_coarse_selector(monkeypatch)
    logits = torch.full((1, 1024), 7.99)
    logits[:, :127] = 8.0
    # The remaining F32 ranks live inside one truncated-key bucket. The
    # existing full sampler must repair rather than trusting a partial bucket.
    controls = torch.tensor([[1., .95, .7, -1.]])
    packet = local_nucleus_packet(logits, controls, 0, 128, threshold_state=state(1024))
    assert torch.isnan(packet[:, 1]).all()
    _, covered = sample_nucleus_packet(packet, controls, tp_size=1, width=128)
    assert not covered.item()


def test_batched_sampler_retains_original_path():
    logits = torch.randn((2, 4096), generator=torch.Generator().manual_seed(24))
    controls = torch.tensor([[1., .95, .7, -1.]])
    original = local_nucleus_packet(logits, controls, 0, 128)
    with_state = local_nucleus_packet(logits, controls, 0, 128, threshold_state=state(4096))
    assert torch.equal(original.view(torch.uint8), with_state.view(torch.uint8))


def test_tail_coordinates_follow_registered_buffer_migration(monkeypatch):
    from types import SimpleNamespace
    from torch import nn
    from vllm_gaudi.models.deepseek_v41_program import PreparedGreedyTail
    from vllm_gaudi.ops import deepseek_v41_replay
    monkeypatch.setenv('VLLM_HPU_DSV41_SAMPLING_THRESHOLD', '1')
    monkeypatch.setattr(deepseek_v41_replay, 'stage_collectives', lambda *args, **kwargs: (None, None))
    head = nn.Module()
    head.register_buffer('weight', torch.empty((32320, 1), dtype=torch.bfloat16, device='meta'))
    stage = SimpleNamespace(weights=SimpleNamespace(head=head),
                            bf16_head=True,
                            tp_rank=0,
                            tensor_parallel_size=4,
                            device_sampling=True,
                            sampling_params=torch.empty((1, 3), device='meta'),
                            sampling_seed=torch.empty((1, ), dtype=torch.int32, device='meta'),
                            sampling_origin=torch.empty((1, ), dtype=torch.int32, device='meta'))
    tail = PreparedGreedyTail(stage)
    assert tail.sampling_threshold
    original = tail.sampling_selection_position
    tail.to_empty(device='cpu')
    assert tail.sampling_selection_position is not original
    assert tail.sampling_selection_position.device.type == 'cpu'
    assert tail.sampling_selection_ids.shape == (1, 2048)
    # Coordinate ownership is in registered buffers, not a stale cached tuple.
    assert not hasattr(tail, 'sampling_threshold_state')
