# SPDX-License-Identifier: Apache-2.0
"""Interrupted startup must still release native plans and runner ownership."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("failure", [None, "statistics", "completion"])
def test_shutdown_retires_plans_when_optional_statistics_fail(monkeypatch, failure):
    from vllm_gaudi.ops import tp2_prepared_plan
    from vllm_gaudi.v1.worker.hpu_worker import HPUWorker

    monkeypatch.setenv("VLLM_HPU_TP2_STATIC_GROUP_PLAN", "1")
    calls = []

    def prepare():
        calls.append("prepare")
        if failure == "completion":
            raise RuntimeError("completion failed")

    def statistics(phase):
        calls.append(phase)
        if failure == "statistics":
            raise AttributeError("uninitialized decode bindings")

    worker = object.__new__(HPUWorker)
    worker.model_runner = SimpleNamespace(prepare_shutdown=prepare, shutdown_inc=lambda: calls.append("close"))
    worker._model_runner_stash = {"generation": object()}
    worker._model_runner_state_stash = {"generation": object()}
    worker._write_native_decoder_stats = statistics
    monkeypatch.setattr(tp2_prepared_plan, "shutdown_prepared_group_plans", lambda: calls.append("retire"))
    if failure == "completion":
        with pytest.raises(RuntimeError, match="completion failed"):
            worker.shutdown()
    else:
        worker.shutdown()
    assert calls == ["prepare", "shutdown", "retire", "close"]
    assert not worker._model_runner_stash
    assert not worker._model_runner_state_stash
