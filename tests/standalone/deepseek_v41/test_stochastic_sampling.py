# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_sampling import request_uniform, sample_probabilities


def sample(logits, temperature=1, top_p=1, uniform=0.5, top_k=-1):
    controls = torch.tensor([[temperature, top_p, uniform, top_k]], dtype=torch.float32)
    return int(sample_probabilities(torch.tensor([logits]), controls, filtered=top_p < 1 or top_k > 0)[0, 0])


def test_categorical_distribution_and_seed():
    logits = torch.log(torch.tensor([0.1, 0.2, 0.7])).tolist()
    draws = torch.tensor([request_uniform("request", 42, index) for index in range(10000)])
    controls = torch.stack((torch.ones_like(draws), torch.ones_like(draws), draws, -torch.ones_like(draws)), -1)
    selected = sample_probabilities(torch.tensor([logits]).expand(10000, -1), controls, filtered=False)
    frequencies = torch.bincount(selected.flatten(), minlength=3) / 10000
    torch.testing.assert_close(frequencies, torch.tensor([0.1, 0.2, 0.7]), atol=0.015, rtol=0)
    assert request_uniform("rank0", 42, 7) == request_uniform("rank3", 42, 7)
    assert request_uniform("one", None, 7) != request_uniform("two", None, 7)
    assert request_uniform("one", 42, 7) != request_uniform("one", 42, 8)


def test_nucleus_retains_boundary_token_and_top_k():
    logits = torch.log(torch.tensor([0.4, 0.35, 0.25])).tolist()
    assert sample(logits, top_p=0.6, uniform=0.99) == 1
    assert sample(logits, top_p=0.3, uniform=0.99) == 0
    assert sample(logits, top_k=1, uniform=0.99) == 0
    assert sample(logits, top_k=2, uniform=0.99) == 1
    assert sample(logits, temperature=1, uniform=0.99) == 2
    assert sample(logits, temperature=0, uniform=0.99) == 0
    assert sample([0., 1.], temperature=0.1, uniform=0.25) == 1
    assert sample([0., 1.], temperature=2, uniform=0.25) == 0


def test_mixed_rows_and_replayed_controls_change():
    logits = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
    controls = torch.tensor([[0., 1., 0.01, -1], [1., 1., 0.01, -1]])
    first = sample_probabilities(logits, controls, filtered=False)
    assert first.tolist() == [[2], [0]]
    controls[1, 2] = 0.99
    assert sample_probabilities(logits, controls, filtered=False).tolist() == [[2], [2]]


def test_sampling_validation_distinguishes_dspark():
    from vllm.sampling_params import SamplingParams
    from vllm_gaudi.ops.deepseek_v41_config import validate_sampling
    from vllm.exceptions import VLLMValidationError

    params = SamplingParams(temperature=1, top_p=0.95, seed=42)
    validate_sampling(params, dspark=False)
    with pytest.raises(VLLMValidationError, match="DSpark requires greedy"):
        validate_sampling(params, dspark=True)
    with pytest.raises(VLLMValidationError, match="penalties"):
        validate_sampling(SamplingParams(temperature=1, presence_penalty=0.1), dspark=False)


def test_sampling_controls_reuse_upload_and_wait_before_replacement(monkeypatch):
    from types import SimpleNamespace
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner

    calls = []

    class Event:
        def record(self):
            calls.append('record')

        def synchronize(self):
            calls.append('wait')

    monkeypatch.setattr(torch.hpu, 'Event', Event)
    runner = object.__new__(V41ModelRunner)
    runner.model = SimpleNamespace(program=SimpleNamespace(sampling_tail=True))
    runner.device = torch.device('cpu')
    runner.audit = {}
    runner.sampling_buffers = {1: (torch.zeros(1, 4), torch.zeros(1, 4))}
    request = SimpleNamespace(req_id='request', output=[],
                              sampling_params=SimpleNamespace(temperature=1., top_p=.95, top_k=-1, seed=42))
    controls = runner._prepare_sample_controls(1, (request,))
    assert torch.equal(controls, torch.tensor([[1., .95, request_uniform('request', 42, 0), -1.]]))
    assert runner._prepare_sample_controls(1, (request,)) is controls
    assert calls == ['record'] and runner.audit['sampling_control_uploads'] == 1
    request.output.append(7)
    assert runner._prepare_sample_controls(1, (request,)) is controls
    assert calls == ['record', 'wait', 'record'] and runner.audit['sampling_control_uploads'] == 2
    assert torch.equal(controls, torch.tensor([[1., .95, request_uniform('request', 42, 1), -1.]]))
    assert runner.audit['sampling_copy_wait_ns'] >= 0
    assert runner.audit['sampling_decode_uploads'] == 1
    assert runner._prepare_sample_controls(1, (request,)) is controls
    assert runner.audit['sampling_control_uploads'] == 2
    assert calls == ['record', 'wait', 'record']
    for phase in ('fill', 'copy', 'event'):
        assert runner.audit[f'sampling_decode_{phase}_ns'] >= 0
        assert runner.audit[f'sampling_decode_{phase}_cpu_ns'] >= 0
