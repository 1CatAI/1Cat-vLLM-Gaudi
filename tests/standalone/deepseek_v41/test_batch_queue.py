# SPDX-License-Identifier: Apache-2.0
"""Advertised queue depth must match the runner's actual scratch ownership."""
from types import SimpleNamespace

from vllm_gaudi.platform import HpuPlatform


def test_request_batch_has_one_transaction_without_disabling_c1_continuation(monkeypatch):
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
                             parallel_config=SimpleNamespace(pipeline_parallel_size=2))
    monkeypatch.setenv("VLLM_HPU_DSV41_V2", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_BATCH_DECODE", "1")
    assert HpuPlatform.get_max_concurrent_batches(config) == 1
    monkeypatch.setenv("VLLM_HPU_DSV41_BATCH_DECODE", "0")
    assert HpuPlatform.get_max_concurrent_batches(config) == 3
    config.model_config.hf_config.model_type = "qwen3"
    monkeypatch.setenv("VLLM_HPU_DSV41_BATCH_DECODE", "1")
    assert HpuPlatform.get_max_concurrent_batches(config) is None
