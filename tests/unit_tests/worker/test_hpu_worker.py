# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from types import SimpleNamespace

import torch

import vllm_gaudi.v1.worker.hpu_worker as hpu_worker_module
from vllm_gaudi.v1.worker.hpu_worker import HPUWorker


def test_init_profiler_defers_kineto_until_first_start(monkeypatch, tmp_path):
    worker = HPUWorker.__new__(HPUWorker)
    worker.vllm_config = SimpleNamespace(
        profiler_config=SimpleNamespace(
            profiler="torch",
            torch_profiler_dir=str(tmp_path),
            torch_profiler_summary_only=False,
            torch_profiler_with_stack=False,
            torch_profiler_use_gzip=False,
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=True,
            torch_profiler_with_flops=True,
        )
    )
    worker.model_runner = SimpleNamespace(profiler=SimpleNamespace())
    captured = {}
    profiler = object()

    monkeypatch.delenv("VLLM_TORCH_PROFILER_DIR", raising=False)
    monkeypatch.setattr(
        torch.profiler,
        "tensorboard_trace_handler",
        lambda path, use_gzip: captured.update(
            handler_path=path, use_gzip=use_gzip
        )
        or "trace-handler",
    )
    monkeypatch.setattr(
        torch.profiler,
        "profile",
        lambda **kwargs: captured.update(profile_kwargs=kwargs) or profiler,
    )

    worker.init_profiler()

    assert worker.profiler is None
    assert captured == {}

    worker._create_profiler()

    assert worker.profiler is profiler
    assert captured["handler_path"] == str(tmp_path)
    assert captured["use_gzip"] is False
    assert captured["profile_kwargs"]["on_trace_ready"] == "trace-handler"
    assert captured["profile_kwargs"]["record_shapes"] is True
    assert captured["profile_kwargs"]["profile_memory"] is True
    assert captured["profile_kwargs"]["with_stack"] is False
    assert captured["profile_kwargs"]["with_flops"] is True


def test_init_device_initializes_workspace_before_model_runner(monkeypatch):
    events = []
    worker = HPUWorker.__new__(HPUWorker)
    worker.vllm_config = SimpleNamespace()
    worker.parallel_config = SimpleNamespace(enable_dbo=True)
    worker.model_config = SimpleNamespace(seed=17)
    worker.rank = 1
    worker.local_rank = 0
    worker.distributed_init_method = "tcp://127.0.0.1:12345"
    worker.is_driver_worker = False

    monkeypatch.setattr(
        torch.hpu,
        "set_device",
        lambda device: events.append(("device", device)),
    )
    monkeypatch.setattr(
        hpu_worker_module,
        "init_worker_distributed_environment",
        lambda *args: events.append("distributed"),
    )
    monkeypatch.setattr(
        hpu_worker_module,
        "set_random_seed",
        lambda seed: events.append(("seed", seed)),
    )
    monkeypatch.setattr(
        hpu_worker_module,
        "init_workspace_manager",
        lambda device, num_ubatches: events.append(
            ("workspace", device, num_ubatches)
        ),
    )
    monkeypatch.setattr(
        hpu_worker_module,
        "set_current_vllm_config",
        lambda config: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        hpu_worker_module,
        "HPUModelRunner",
        lambda **kwargs: events.append(("runner", kwargs)) or object(),
    )
    monkeypatch.setattr(
        HPUWorker,
        "init_profiler",
        lambda self: events.append("profiler"),
    )

    worker.init_device()

    assert events[:4] == [
        ("device", 0),
        "distributed",
        ("seed", 17),
        ("workspace", torch.device("hpu"), 2),
    ]
    assert events[4][0] == "runner"
    assert events[4][1]["vllm_config"] is worker.vllm_config
    assert events[4][1]["is_driver_worker"] is False
    assert events[5] == "profiler"
