# SPDX-License-Identifier: Apache-2.0
"""TP4 halo must retain source publication, Engram and the scheduler row contract."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_gaudi.ops.deepseek_v41_decoder_halo import decoder_halo_mode


@pytest.mark.parametrize("prompt,chunks,expected", [
    (16384, [(0, 16384)], ["final"]),
    (32768, [(0, 16384), (16384, 16384)], ["prefix_only", "final"]),
    (20000, [(0, 16384), (16384, 3616)], ["full", "final"]),
    (20480, [(0, 16384), (16384, 4096)], ["prefix_only", "final"]),
    (32769, [(0, 16384), (16384, 16384), (32768, 1)], ["prefix_only", "full", "final"]),
    (4096, [(0, 4096)], ["full"]),
    (16384, [(0, 8192), (8192, 8192)], ["full", "full"]),
])
def test_tp4_scheduler_chunks_and_short_tail(prompt, chunks, expected):
    actual = [
        decoder_halo_mode(start, count, prompt, eligible=True, block_tokens=16384, allow_single_block=True)
        for start, count in chunks
    ]
    assert actual == expected


def make_stage(monkeypatch, *, dspark=False):
    from vllm_gaudi.models import deepseek_v41_program as program
    from vllm_gaudi.ops import deepseek_v41_prefill_sequence_state as sequence
    monkeypatch.setattr(sequence, "can_sequence_prefill_state", lambda stage, tokens: False)

    shared = SimpleNamespace(candidate_pool=torch.full((16384, 1), -1, dtype=torch.int32),
                             topk={"20": SimpleNamespace(indices=torch.full((16384, 1), -1, dtype=torch.int32))},
                             prefill_main_workspace=None)
    calls = []

    class Layer(nn.Module):

        def __init__(self, index):
            super().__init__()
            self.layer = index

        def forward(self,
                    residual,
                    pre,
                    positions,
                    image_mask,
                    engram=None,
                    *,
                    prefill_router_tokens=0,
                    prefill_sequence=False):
            self.router_tokens = prefill_router_tokens
            count = positions.numel()
            if prefill_sequence:
                assert residual.shape[0] * 4 == count
            else:
                assert residual.shape[0] == count
            calls.append((self.layer, count, engram))
            if self.layer == 20:
                shared.candidate_pool[:count, 0].copy_(positions)
                shared.topk["20"].indices[:count, 0].copy_(positions)
            if self.layer > 20:
                assert torch.equal(shared.candidate_pool[:count, 0], positions)
                assert torch.equal(shared.topk["20"].indices[:count, 0], positions)
            target = residual[-256:].mean(1) if dspark and self.layer in (37, 38, 39) else None
            return residual + 1, pre, target

    stage = program.PreparedStage.__new__(program.PreparedStage)
    nn.Module.__init__(stage)
    stage.tensor_parallel_size, stage.pp_rank, stage.dspark = 4, 0, dspark
    stage.start, stage.stop, stage.is_last_stage = 0, 40, True
    stage.shared = shared
    stage.layers = nn.ModuleList([Layer(index) for index in range(40)])
    stage.config = {"text_config": {"rms_norm_eps": 1e-6}}
    stage.weights = SimpleNamespace(norm=SimpleNamespace(weight=torch.ones(1)))
    monkeypatch.setattr(program, "final_collapse_rms_norm", lambda residual, pre, weight, eps: residual[:, 0])
    return stage, calls


@pytest.mark.parametrize("mode", ("final", "prefix_only"))
def test_dspark_halo_publishes_causal_target_context_for_every_prompt_chunk(monkeypatch, mode):
    stage, calls = make_stage(monkeypatch, dspark=True)
    positions = torch.arange(16384, 32768, dtype=torch.int32)
    residual = positions.float().reshape(-1, 1, 1).expand(-1, 4, -1).clone()
    output, pre, context = stage._forward_prefill_halo(
        residual, torch.zeros(16384, 4), positions, positions, mode, (None, None)
    )
    assert [(layer, count) for layer, count, _ in calls] == [
        (layer, 16384 if layer <= 20 else 4096) for layer in range(40)
    ]
    assert output.shape == (16384, 1) and pre.shape == (16384, 4)
    assert torch.equal(output[-4096:, 0], positions[-4096:].float() + 40)
    assert context.shape == (256, 3)
    assert torch.equal(context, positions[-256:, None].float() + torch.tensor([37, 38, 39]))


def test_owned_tp4_final_keeps_full_front_and_rebinds_source20(monkeypatch):
    from vllm_gaudi.models.deepseek_v41_program import PrefillInput
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_DECODER_HALO", "1")
    stage, calls = make_stage(monkeypatch)
    stage.prefill_halo_mode = "final"
    for start in (0, 16384):
        calls.clear()
        positions = torch.arange(start, start + 16384, dtype=torch.int32)
        residual = positions.float().reshape(-1, 1, 1).expand(-1, 4, 1).clone()
        owner = PrefillInput(residual, torch.zeros(16384, 4))
        engram = (object(), object())
        output, pre, target = stage(owner, None, positions, positions, engram)
        assert owner.residual is owner.pre_mix is None
        assert [(layer, count) for layer, count, _ in calls] == [(layer, 16384 if layer <= 20 else 4096)
                                                                 for layer in range(40)]
        assert calls[1][2] is engram[0] and calls[14][2] is engram[1]
        assert [layer.router_tokens for layer in stage.layers[21:]] == [16384] * 19
        assert output.shape == (16384, 1) and pre.shape == (16384, 4)
        assert torch.equal(output[-4096:, 0], positions[-4096:].float() + 40)
        assert torch.count_nonzero(output[:-4096]) == 0
        assert target is None


def test_tp4_prefix_publishes_all_source_rows_before_skipping_suffix(monkeypatch):
    stage, calls = make_stage(monkeypatch)
    positions = torch.arange(16384, dtype=torch.int32)
    output, _, target = stage._forward_prefill_halo(torch.zeros(16384, 4, 1), torch.zeros(16384, 4), positions,
                                                    positions, "prefix_only", (None, None))
    assert [layer for layer, _, _ in calls] == list(range(21))
    assert torch.equal(stage.shared.candidate_pool[:, 0], positions)
    assert torch.equal(stage.shared.topk["20"].indices[:, 0], positions)
    assert output.shape == (16384, 1) and target is None


def test_token_owned_final_preserves_full_positions_and_replicates_tail(monkeypatch):
    from vllm import distributed
    from vllm_gaudi.ops import deepseek_v41_prefill_sequence_state as sequence
    # Device collective ordering/content is covered by the four-rank state
    # fixture; here compare the normal stage transition against replicated B.
    stage, calls = make_stage(monkeypatch)
    positions = torch.arange(16384, dtype=torch.int32)
    residual = positions.float().reshape(-1, 1, 1).expand(-1, 4, 1).clone()
    previous = torch.ones(16384, 4)
    monkeypatch.setattr(sequence, "can_sequence_prefill_state", lambda stage, count: False)
    expected = stage._forward_prefill_halo(residual, previous, positions, positions, "final", (None, None))
    calls.clear()
    group = SimpleNamespace(rank_in_group=3, device_group=object())
    monkeypatch.setattr(distributed, "get_tp_group", lambda: group)
    monkeypatch.setattr(sequence, "can_sequence_prefill_state", lambda stage, count: count == 16384)
    transfers = []

    def replicate(local, retained, *, group):
        transfers.append((local.shape, retained))
        return local[-retained:].clone()

    monkeypatch.setattr(sequence, "replicate_owned_tail", replicate)
    actual = stage._forward_prefill_halo(residual, previous, positions, positions, "final", (None, None))
    assert len(transfers) == 2 and all(retained == 4096 for _, retained in transfers)
    assert [(layer, count) for layer, count, _ in calls] == [(layer, 16384 if layer <= 20 else 4096)
                                                             for layer in range(40)]
    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])
    assert actual[2] is None


@pytest.mark.parametrize("batch,multimodal,logprobs,capacity,expected", [
    (1, False, None, 16384, ["prefix_only", "final"]),
    (2, False, None, 16384, ["full", "full"]),
    (1, True, None, 16384, ["full", "full"]),
    (1, False, 1, 16384, ["full", "full"]),
    (1, False, None, 8192, ["full"] * 4),
])
def test_normal_runner_transactions_and_ineligible_requests(monkeypatch, batch, multimodal, logprobs, capacity,
                                                            expected):
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_DECODER_HALO", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_REINDEX_REUSE", "1")
    monkeypatch.delenv("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE", raising=False)
    monkeypatch.delenv("VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS", raising=False)
    program = SimpleNamespace(tensor_parallel_size=4, runtime_indexer=False, length=524288, prefill_halo_mode="full")
    request = SimpleNamespace(num_computed_tokens=0,
                              decode_start=32768,
                              tokens=list(range(32768)),
                              prompt=list(range(32768)),
                              mm_features=[object()] if multimodal else [],
                              sampling_params=SimpleNamespace(prompt_logprobs=logprobs))
    request.output = []
    request.token_slice = lambda start, stop: request.tokens[start:stop]
    calls = []

    def forward(req_id, chunk, start, **kwargs):
        calls.append((program.prefill_halo_mode, start, len(chunk), kwargs["search_length"]))
        return None

    runner = SimpleNamespace(round_timing_enabled=False,
                             prefill_capacity=capacity,
                             requests={"A": request},
                             _bind_request=lambda request: None,
                             _prepare_device_sampling_request=lambda request: None,
                             use_dspark=False,
                             model_config=SimpleNamespace(max_model_len=524288),
                             verify_timing=None,
                             model=SimpleNamespace(program=program, pp_rank=0, tp_rank=0),
                             _forward=forward,
                             pp=SimpleNamespace(generation=0, group=SimpleNamespace(is_last_rank=False)))
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={},
                                num_scheduled_tokens={str(i): capacity
                                                      for i in range(batch)})
    for start in range(0, 32768, capacity):
        request.num_computed_tokens = start
        V41ModelRunner._execute_request(runner, scheduled, "A", capacity)
        assert program.prefill_halo_mode == "full"
    assert [mode for mode, _, _, _ in calls] == expected
    geometry = ([(0, 16384, 16384), (16384, 16384, 32768)] if capacity == 16384 else [(0, 8192, 16384),
                                                                                      (8192, 8192, 16384),
                                                                                      (16384, 8192, 32768),
                                                                                      (24576, 8192, 32768)])
    assert [(start, count, search) for _, start, count, search in calls] == geometry


def test_halo_router_keeps_original_m_and_suffix_row_placement(monkeypatch):
    from vllm_gaudi.models.deepseek_v41_program import PreparedMoE
    observed = []

    def gate(value, weight):
        observed.append(value.clone())
        return value.float() @ weight.float().T

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_bf16_linear_f32_gaudi2", gate, raising=False)
    weight = torch.arange(6).reshape(2, 3).bfloat16()
    moe = SimpleNamespace(router_bf16_gate=True, weights=SimpleNamespace(gate=SimpleNamespace(weight=weight)))
    value = torch.arange(12).reshape(4, 3).bfloat16()
    actual = PreparedMoE._router_logits(moe, value, 8)
    assert observed[-1].shape == (8, 3)
    assert not observed[-1][:4].any()
    assert torch.equal(observed[-1][-4:], value)
    assert torch.equal(actual, value.float() @ weight.float().T)
    PreparedMoE._router_logits(moe, value)
    assert torch.equal(observed[-1], value)
    for invalid in (3, 16385):
        with pytest.raises(ValueError, match="original bounded prefill"):
            PreparedMoE._router_logits(moe, value, invalid)
