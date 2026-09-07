# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import pytest

from vllm_gaudi.ops.deepseek_v4_config import DEFAULTS, bind_worker_cpu, configure_defaults


def config(model_type="deepseek_v4", tp=2, length=512, batch=1, eager=False):
    return NS(model_config=NS(hf_config=NS(model_type=model_type), max_model_len=length, enforce_eager=eager),
              parallel_config=NS(tensor_parallel_size=tp), scheduler_config=NS(max_num_seqs=batch))


def test_scoped_defaults_respect_explicit_configuration(monkeypatch):
    import os
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_HPU_DSV4_FLASHMLA_SPLIT_KV", "0")
    assert configure_defaults(config(), gaudi2=True)
    assert os.environ["VLLM_HPU_DSV4_FLASHMLA_SPLIT_KV"] == "0"
    assert os.environ["VLLM_HPU_DSV4_EARLY_OUTPUT_LOWERING"] == "1"
    assert not any("FUSED_QNORM_COMPRESSOR" in key or "ORDERED_C128" in key for key in DEFAULTS)


@pytest.mark.parametrize("cfg,gaudi2", [(config("qwen3"), True), (config(tp=1), True),
    (config(length=4096), True), (config(batch=2), True), (config(eager=True), True), (config(), False)])
def test_unsupported_configuration_unchanged(monkeypatch, cfg, gaudi2):
    import os
    before = dict(os.environ)
    assert not configure_defaults(cfg, gaudi2=gaudi2)
    assert dict(os.environ) == before


def test_worker_affinity_is_explicit_and_validated(monkeypatch):
    import os
    calls = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {6, 16})
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, mask: calls.append(mask))
    monkeypatch.delenv("VLLM_HPU_DSV4_WORKER_CPUS", raising=False)
    bind_worker_cpu(0)
    assert not calls
    monkeypatch.setenv("VLLM_HPU_DSV4_WORKER_CPUS", "6,16")
    bind_worker_cpu(1)
    assert calls == [{16}]
    monkeypatch.setenv("VLLM_HPU_DSV4_WORKER_CPUS", "6,99")
    with pytest.raises(ValueError, match="outside"):
        bind_worker_cpu(1)
