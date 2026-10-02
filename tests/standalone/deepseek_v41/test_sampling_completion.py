# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner


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
