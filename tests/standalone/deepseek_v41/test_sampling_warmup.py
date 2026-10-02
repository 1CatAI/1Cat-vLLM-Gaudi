# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner


@pytest.mark.parametrize('last', [False, True])
def test_official_sampler_warms_once_before_readiness(monkeypatch, last):
    monkeypatch.setenv('VLLM_HPU_DSV41_DEVICE_VERIFY', '0')
    monkeypatch.setattr(torch.hpu, 'synchronize', lambda: None)
    runner = object.__new__(V41ModelRunner)
    runner.model = SimpleNamespace(pp_rank=0, complete_step=Mock())
    runner.state = SimpleNamespace(clear=Mock())
    runner.audit = {'target_steps': 0}
    runner._forward = Mock(return_value=torch.ones(128, 8))
    runner.use_dspark = False
    runner.verify_prefix = None
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=last, barrier=Mock()),
                                drain=Mock(), complete_packet=Mock())
    runner.sample_target = Mock()
    runner.stochastic_samplers = {}
    calls = []
    def official(hidden, requests):
        request = requests[0]
        params = request.sampling_params
        calls.append((hidden.shape, params.temperature, params.top_p, params.seed, request.output))
        runner.stochastic_samplers[(1, params.top_p < 1)] = object()
    runner._sample_requests = official
    runner._dummy_run(128)
    runner._dummy_run(64)
    assert calls == ([(torch.Size([1, 8]), 1., .95, 42, []),
                     (torch.Size([1, 8]), 1., 1., 42, [])] if last else [])
    assert runner.sample_target.call_count == (2 if last else 0)
    assert runner.model.complete_step.call_count == 2
    assert runner.pp.group.barrier.call_count == 2
