# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from concurrent.futures import Future
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner
from vllm_gaudi.v1.worker.deepseek_v41_v2_runner import CompletionRecord, V41V2ModelRunner


def runner():
    value = object.__new__(V41ModelRunner)
    hidden = torch.ones(1, 3)
    program = SimpleNamespace(sampling_tail=True, tensor_parallel_size=4,
        weights=SimpleNamespace(head=SimpleNamespace(weight=torch.zeros(2, 3))),
        replay_owner=SimpleNamespace(latest_tail=(1, hidden, ())))
    value.model = SimpleNamespace(program=program)
    value.audit = {}
    selected = torch.tensor([[3]], dtype=torch.int32)
    value._sample_requests = Mock(return_value=selected)
    value.tp4_token_readback = Mock(return_value=(selected.cpu(), Mock()))
    return value


def test_complete_candidate_reuses_its_existing_readback():
    value = runner()
    selected = torch.tensor([[2]], dtype=torch.int32)
    done = Mock()
    request = SimpleNamespace(sampling_params=SimpleNamespace(temperature=1.))
    actual = value._resolve_sampled_marker(request, selected, selected, done)
    assert actual == (selected, selected, done)
    done.synchronize.assert_called_once()
    value._sample_requests.assert_not_called()


def test_incomplete_candidate_reuses_same_request_uniform_before_continuation():
    value = runner()
    selected = torch.tensor([[10]], dtype=torch.int32)
    done = Mock()
    request = SimpleNamespace(sampling_params=SimpleNamespace(temperature=1.), output=[4])
    actual, host, event = value._resolve_sampled_marker(request, selected, selected, done)
    assert actual.item() == 3
    args, kwargs = value._sample_requests.call_args
    assert args[1] == (request,)
    assert args[0] is value.model.program.replay_owner.latest_tail[1]
    assert kwargs['force_full'] and kwargs['replay'] is value.model.program.replay_owner
    assert request.output == [4]
    assert value.audit['sampling_fallbacks'] == 1


@pytest.mark.parametrize('marker', [-1, 16])
def test_invalid_marker_cannot_escape_as_a_token(marker):
    value = runner()
    selected = torch.tensor([[marker]], dtype=torch.int32)
    request = SimpleNamespace(sampling_params=SimpleNamespace(temperature=1.))
    with pytest.raises(RuntimeError, match='completion marker'):
        value._resolve_sampled_marker(request, selected, selected, Mock())


def test_async_completion_waits_for_resolution_before_exposing_host_token():
    future = Future()
    host = torch.tensor([[10]], dtype=torch.int32)
    done = Mock()
    record = CompletionRecord('a', 1, 3, host, done, torch.tensor([2]), future)
    host.fill_(2)
    future.set_result(None)
    assert record.token() == 2
    done.synchronize.assert_called_once()


def test_async_resolution_error_cannot_expose_an_encoded_marker():
    future = Future()
    future.set_exception(RuntimeError('fallback collective failed'))
    done = Mock()
    record = CompletionRecord('a', 1, 3, torch.tensor([[10]]), done, torch.tensor([10]), future)
    with pytest.raises(RuntimeError, match='fallback collective failed'):
        record.token()
    done.synchronize.assert_not_called()


def test_background_fallback_updates_existing_device_and_host_storage(monkeypatch):
    value = runner()
    original = torch.tensor([[10]], dtype=torch.int32)
    host = original.clone()
    done = Mock()
    stream = Mock()
    monkeypatch.setattr(torch.hpu, 'stream', lambda _: nullcontext())
    monkeypatch.setattr(torch.hpu, 'current_stream', lambda: stream)
    request = SimpleNamespace(sampling_params=SimpleNamespace(temperature=1.), output=[])
    V41V2ModelRunner._resolve_sampling_completion(value, request, original, host, done, stream)
    assert original.item() == host.item() == 3
    assert request.output == []
    stream.synchronize.assert_called_once()
