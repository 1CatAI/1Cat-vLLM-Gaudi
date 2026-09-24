# SPDX-License-Identifier: Apache-2.0
"""A successful stop API must not be reported as a successful empty capture."""
import json
import os
from types import SimpleNamespace

import pytest

from vllm_gaudi.ops import deepseek_v41_native_trace as trace


def configure(monkeypatch, tmp_path, *, skip_parse=True):
    config = {
        "GeneralSettings": {"values": {
            "outdir": {"value": str(tmp_path)},
            "session": {"value": "test"},
            "addPid": {"value": True},
        }},
        "Plugins": [{"name": "HwTrace", "enable": True, "values": {
            "parseOptions": {"skipParse": {"value": skip_parse}},
        }}],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("HABANA_PROF_CONFIG", str(path))
    monkeypatch.setenv("HABANA_PROFILE_WRITE_HLTV", "1")
    monkeypatch.setattr(trace, "_active", False)
    monkeypatch.setattr(trace, "_annotations_supported", None)
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
