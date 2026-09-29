# SPDX-License-Identifier: Apache-2.0
"""A successful stop API must not be reported as a successful empty capture."""
import json
import gzip
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm_gaudi.ops import deepseek_v41_native_trace as trace


def configure(monkeypatch, tmp_path, *, skip_parse=True):
    config = {
        "GeneralSettings": {
            "values": {
                "outdir": {
                    "value": str(tmp_path)
                },
                "session": {
                    "value": "test"
                },
                "addPid": {
                    "value": True
                },
            }
        },
        "Plugins": [{
            "name": "HwTrace",
            "enable": True,
            "values": {
                "parseOptions": {
                    "skipParse": {
                        "value": skip_parse
                    }
                },
            }
        }],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("HABANA_PROF_CONFIG", str(path))
    monkeypatch.setenv("HABANA_PROFILE_WRITE_HLTV", "1")
    monkeypatch.setattr(trace, "_active", False)
    monkeypatch.setattr(trace, "_annotations_supported", None)
    monkeypatch.setattr(trace, "_scope_recorder", None)
    monkeypatch.setattr(trace.torch.hpu, "synchronize", lambda: None)
    return tmp_path / f"test_{os.getpid()}.hltv"


def test_raw_capture_requires_file_publication_after_stop(monkeypatch, tmp_path):
    output = configure(monkeypatch, tmp_path)
    calls = []

    def publish(kind, device, form, buffer, size, count):
        assert (kind, device, form, buffer, size, count) == (1, 0, 1, None, None, None)
        calls.append("publish")
        output.write_bytes(b"raw trace fixture")
        return 0

    api = SimpleNamespace(synProfilerQueryRequiredMemory=lambda *_: 0,
                          synProfilerStart=lambda *_: 0,
                          synProfilerStop=lambda *_: calls.append("stop") or 0,
                          synProfilerGetTrace=publish)
    monkeypatch.setattr(trace, "_api", lambda: api)
    recorder = trace.NativeTrace()
    recorder.start()
    recorder.stop()
    assert calls == ["stop", "publish"]
    assert output.is_file() and not recorder.running and not trace._active


def test_raw_capture_accepts_sdk_accel_filename(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr(trace, "_local_accel_name", lambda: "accel3")
    output = tmp_path / "test_accel3_1.hltv"

    def publish(*_):
        output.write_bytes(b"raw trace fixture")
        return 0

    api = SimpleNamespace(synProfilerQueryRequiredMemory=lambda *_: 0,
                          synProfilerStart=lambda *_: 0,
                          synProfilerStop=lambda *_: 0,
                          synProfilerGetTrace=publish)
    monkeypatch.setattr(trace, "_api", lambda: api)
    recorder = trace.NativeTrace()
    recorder.start()
    recorder.stop()
    assert recorder.output == output


@pytest.mark.parametrize("stale", [False, True])
def test_successful_sdk_return_without_new_bundle_is_failure(monkeypatch, tmp_path, stale):
    output = configure(monkeypatch, tmp_path)
    if stale:
        output.write_bytes(b"previous capture")
    api = SimpleNamespace(synProfilerQueryRequiredMemory=lambda *_: 0,
                          synProfilerStart=lambda *_: 0,
                          synProfilerStop=lambda *_: 0,
                          synProfilerGetTrace=lambda *_: 0)
    monkeypatch.setattr(trace, "_api", lambda: api)
    recorder = trace.NativeTrace()
    recorder.start()
    with pytest.raises(RuntimeError, match="without publishing"):
        recorder.stop()
    assert not recorder.running and not trace._active


def test_raw_capture_rejects_worker_side_event_expansion(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, skip_parse=False)
    with pytest.raises(RuntimeError, match="skipParse=true"):
        trace.NativeTrace().start()


def test_unsupported_custom_measurement_does_not_abort_kernel_capture(monkeypatch):
    calls = []
    api = SimpleNamespace(
        synProfilerGetCurrentTimeNS=lambda *_: 0,
        synProfilerAddCustomMeasurement=lambda *_: calls.append("annotation") or 18,
    )
    monkeypatch.setattr(trace, "_api", lambda: api)
    monkeypatch.setattr(trace, "_active", True)
    monkeypatch.setattr(trace, "_annotations_supported", None)
    with trace.scope("outer"), trace.scope("first"):
        calls.append("first kernel")
    with trace.scope("second"):
        calls.append("second kernel")
    assert calls == ["first kernel", "annotation", "second kernel"]
    assert trace._annotations_supported is False


def test_raw_capture_retains_cpu_scopes_and_clock_alignment_without_hpu_expansion(monkeypatch, tmp_path):
    output = configure(monkeypatch, tmp_path)
    monkeypatch.setattr(trace, "_torch_active", False)
    calls = []

    def publish(*_):
        output.write_bytes(b"raw fixture")
        calls.append("publish")
        return 0

    api = SimpleNamespace(synProfilerQueryRequiredMemory=lambda *_: 0,
                          synProfilerStart=lambda *_: 0,
                          synProfilerStop=lambda *_: calls.append("stop") or 0,
                          synProfilerGetTrace=publish,
                          synProfilerGetCurrentTimeNS=lambda *_: 0)
    monkeypatch.setattr(trace, "_api", lambda: api)
    captured = {}
    profiler = SimpleNamespace(start=lambda: calls.append("cpu_start"), stop=lambda: calls.append("cpu_stop"))
    monkeypatch.setattr(trace.torch.profiler, "profile", lambda **kw: captured.update(kw) or profiler)
    monkeypatch.setattr(trace.torch.profiler, "tensorboard_trace_handler", lambda *a, **kw: None)

    @contextmanager
    def record(label):
        calls.append(label)
        yield

    monkeypatch.setattr(trace.torch.profiler, "record_function", record)
    recorder = trace.NativeTrace(cpu_trace_dir=tmp_path / "cpu")
    recorder.start()
    with trace.scope("complete consumer"):
        pass
    recorder.stop()
    assert captured["activities"] == [trace.torch.profiler.ProfilerActivity.CPU]
    assert calls == ["cpu_start", "complete consumer", "stop", "publish", "cpu_stop"]
    assert len(recorder.metadata["clock_samples"]) == 2
    assert list((tmp_path / "cpu").glob("raw-capture-*.json"))
    assert not trace._torch_active and not trace._active


def test_scope_only_capture_preserves_nested_consumer_boundaries_without_cpu_profiler(monkeypatch, tmp_path):
    output = configure(monkeypatch, tmp_path)
    monkeypatch.setattr(trace, "_torch_active", False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Scope-only mode must not enable CPU operator profiling")

    monkeypatch.setattr(trace.torch.profiler, "profile", forbidden)
    monkeypatch.setattr(trace.torch.profiler, "record_function", forbidden)

    def publish(*args):
        output.write_bytes(b"raw scope fixture")
        return 0

    api = SimpleNamespace(synProfilerQueryRequiredMemory=lambda *_: 0,
                          synProfilerStart=lambda *_: 0,
                          synProfilerStop=lambda *_: 0,
                          synProfilerGetTrace=publish)
    monkeypatch.setattr(trace, "_api", lambda: api)
    raw0, wall0 = 10_000_000, 1_790_000_000_123_000_000

    def calibration(self, label):
        delta = 1_000_000 if label == "stop" else 0
        self.clock_samples.append(
            dict(label=label,
                 monotonic_raw_before_ns=raw0 + delta,
                 monotonic_raw_after_ns=raw0 + delta,
                 wall_before_ns=wall0 + delta,
                 wall_after_ns=wall0 + delta,
                 synapse_clock_ns=2 * (raw0 + delta)))

    monkeypatch.setattr(trace.NativeTrace, "_clock_sample", calibration)
    stamps = iter([raw0 + value for value in (1, 100, 200, 301)])
    monkeypatch.setattr(trace.time, "clock_gettime_ns", lambda *_: next(stamps))
    recorder = trace.NativeTrace(cpu_trace_dir=tmp_path / "cpu", scope_only=True)
    recorder.start()
    with trace.scope("producer-to-consumer"), pytest.raises(ValueError), trace.scope("failing inner"):
        raise ValueError("test boundary")
    recorder.stop()
    path, = (tmp_path / "cpu").glob("*.pt.trace.json.gz")
    with gzip.open(path, "rt") as stream:
        saved = json.load(stream)
    events = {row["name"]: row for row in saved["traceEvents"]}
    assert events["producer-to-consumer"]["ts"] == 123000.001
    assert events["producer-to-consumer"]["dur"] == .3
    assert events["failing inner"]["dur"] == .1
    assert recorder.metadata["cpu_trace_mode"] == "scopes_only"
    assert recorder.metadata["explicit_scope_count"] == 2
    assert not trace._active and not trace._torch_active and trace._scope_recorder is None


def test_scope_only_capture_requires_output_directory():
    with pytest.raises(ValueError, match="output directory"):
        trace.NativeTrace(scope_only=True)
