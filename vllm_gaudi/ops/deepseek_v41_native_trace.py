# SPDX-License-Identifier: Apache-2.0
"""On-demand Synapse raw capture; offline parsing avoids concurrent Kineto expansion."""
import ctypes
import logging
from contextlib import contextmanager
from functools import wraps
import gzip
import json
import os
from pathlib import Path
import threading
import time

import torch

_library = None
_active = False
_torch_active = False
_annotations_supported = None
_scope_recorder = None


def set_torch_annotations(enabled):
    global _torch_active
    _torch_active = bool(enabled)


def annotations_enabled():
    return _active or _torch_active


def _local_accel_name():
    """Find the accelerator opened by this worker, for SDK raw-file naming."""
    names = set()
    for descriptor in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(descriptor)
        except OSError:
            continue
        if target.startswith("/dev/accel/accel") and target[len("/dev/accel/accel"):].isdigit():
            names.add(Path(target).name)
    return next(iter(names)) if len(names) == 1 else None


def prefill_span(name):
    """Mark host submission scope without synchronizing or altering a graph."""

    def decorate(function):

        @wraps(function)
        def invoke(*args, **kwargs):
            if torch.compiler.is_compiling():
                return function(*args, **kwargs)
            first = args[0] if isinstance(args[0], torch.Tensor) else args[1]
            if kwargs.get("decode", False) or first.shape[0] <= 6:
                return function(*args, **kwargs)
            layer = getattr(args[0], "layer", "unknown")
            from vllm_gaudi.ops.deepseek_v41_prefill_event_trace import span
            with span(name, layer, first.shape[0]), scope(f"v41::prefill::{name}::layer{layer}::C{first.shape[0]}"):
                return function(*args, **kwargs)

        return invoke

    return decorate


def _api():
    global _library
    if _library is None:
        # Resolve the already-loaded, fingerprint-checked runtime. No alternate
        # system Synapse library or process-local ABI bypass is introduced.
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if line.split() and Path(line.split()[-1]).name == "libSynapse.so"
        }
        if len(paths) != 1:
            raise RuntimeError(f"Native trace requires one loaded Synapse library, found {len(paths)}")
        _library = ctypes.CDLL(paths.pop(), mode=os.RTLD_NOW | os.RTLD_NOLOAD)
        _library.synProfilerGetCurrentTimeNS.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        _library.synProfilerQueryRequiredMemory.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        _library.synProfilerSetUserBuffer.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        for name in ("synProfilerStart", "synProfilerStop"):
            getattr(_library, name).argtypes = [ctypes.c_int, ctypes.c_uint32]
        _library.synProfilerGetTrace.argtypes = [
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        _library.synProfilerAddCustomMeasurement.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    return _library


def _check(status, name):
    if status:
        raise RuntimeError(f"{name} failed with Synapse status {status}")


class NativeTrace:

    def __init__(self, cpu_trace_dir=None, *, scope_only=False):
        if scope_only and cpu_trace_dir is None:
            raise ValueError("Scope-only capture requires a trace output directory")
        self.buffer = None
        self.running = False
        self.output = None
        self._directory = None
        self._session = None
        self._accel_name = None
        self.capture_cpu = cpu_trace_dir is not None
        self.cpu_trace_dir = Path(cpu_trace_dir) if self.capture_cpu else None
        self.cpu_profiler = None
        self.clock_samples = []
        self.metadata = None
        self.scope_only = scope_only
        self.scope_events = []

    def _export_scopes(self):
        first, last = self.clock_samples
        raw0 = (first["monotonic_raw_before_ns"] + first["monotonic_raw_after_ns"]) // 2
        raw1 = (last["monotonic_raw_before_ns"] + last["monotonic_raw_after_ns"]) // 2
        # Explicit scopes and the SDK's calibrated device events already use
        # MONOTONIC_RAW. Keep that common clock instead of stretching durations
        # to a wall clock that may step during NTP correction.
        base = raw0 // 1_000_000_000 * 1_000_000_000
        events = [dict(name=label, ph="X", cat="user_annotation", pid=os.getpid(), tid=tid,
                       ts=(start - base) / 1000, dur=(end - start) / 1000)
                  for start, end, tid, label in self.scope_events]
        events.append(dict(name="NativeTrace scope-only capture", ph="X", cat="trace_boundary",
                           pid=os.getpid(), tid=0, ts=(raw0 - base) / 1000, dur=(raw1 - raw0) / 1000))
        path = self.cpu_trace_dir / f"native_{os.getpid()}.{time.time_ns()}.pt.trace.json.gz"
        with gzip.open(path, "wt") as stream:
            json.dump(dict(schemaVersion=1, baseTimeNanoseconds=base, traceEvents=events,
                           clockDomain="CLOCK_MONOTONIC_RAW",
                           recorder="explicit scopes only; no torch CPU operator profiling"), stream)

    def _clock_sample(self, label):
        clock = ctypes.c_uint64()
        before = time.time_ns()
        raw_before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        _check(_api().synProfilerGetCurrentTimeNS(ctypes.byref(clock)), "trace clock calibration")
        raw_after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        after = time.time_ns()
        self.clock_samples.append(dict(label=label, synapse_clock_ns=clock.value,
                                       monotonic_raw_before_ns=raw_before, monotonic_raw_after_ns=raw_after,
                                       wall_before_ns=before, wall_after_ns=after))

    def _published_files(self):
        paths = [self._directory / f"{self._session}_{os.getpid()}.hltv"]
        if self._accel_name:
            paths.extend(self._directory.glob(f"{self._session}_{self._accel_name}_*.hltv"))
        return {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths if path.is_file()}

    def _prepare_output(self):
        config_path = os.environ.get("HABANA_PROF_CONFIG")
        if not config_path or os.environ.get("HABANA_PROFILE_WRITE_HLTV") != "1":
            raise RuntimeError("Raw trace requires HABANA_PROF_CONFIG and HABANA_PROFILE_WRITE_HLTV=1")
        config = json.loads(Path(config_path).read_text())
        plugins = [
            plugin for plugin in config.get("Plugins", ()) if plugin.get("name") == "HwTrace" and plugin.get("enable")
        ]
        if len(plugins) != 1 or not plugins[0]["values"]["parseOptions"]["skipParse"]["value"]:
            raise RuntimeError("Raw trace requires one enabled HwTrace plugin with skipParse=true")
        settings = config["GeneralSettings"]["values"]
        if not settings["addPid"]["value"]:
            raise RuntimeError("Raw trace requires addPid=true to isolate worker output")
        directory = Path(settings["outdir"]["value"])
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._session = settings["session"]["value"]
        self._accel_name = _local_accel_name()
        self._output_before = self._published_files()

    def start(self):
        global _active, _annotations_supported, _scope_recorder
        if self.running:
            raise RuntimeError("Native trace is already running")
        self._prepare_output()
        torch.hpu.synchronize()
        if self.capture_cpu:
            self.cpu_trace_dir.mkdir(parents=True, exist_ok=True)
        if self.capture_cpu and not self.scope_only:
            self.cpu_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False, with_stack=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.cpu_trace_dir), use_gzip=True))
            self.cpu_profiler.start()
            set_torch_annotations(True)
        api = _api()
        required = ctypes.c_uint32()
        _check(api.synProfilerQueryRequiredMemory(0, ctypes.byref(required)), "trace memory query")
        if required.value:
            self.buffer = torch.empty(required.value, dtype=torch.uint8, device="hpu")
            _check(api.synProfilerSetUserBuffer(0, self.buffer.data_ptr()), "trace memory binding")
        _check(api.synProfilerStart(1, 0), "trace start")
        if self.capture_cpu:
            self._clock_sample("start")
        _annotations_supported = None
        self.running = _active = True
        _scope_recorder = self if self.scope_only else None

    def stop(self):
        global _active, _scope_recorder
        if not self.running:
            raise RuntimeError("Native trace is not running")
        torch.hpu.synchronize()
        if self.capture_cpu:
            self._clock_sample("stop")
        api = _api()
        _check(api.synProfilerStop(1, 0), "trace stop")
        self.running = _active = False
        _scope_recorder = None
        # A null size requests file publication. Unlike the size-query and
        # caller-buffer forms used by Kineto, this honors the configured raw
        # output/skipParse mode without materializing all events in Python.
        # Stop alone does not publish a bundle with every supported SDK.
        _check(api.synProfilerGetTrace(1, 0, 1, None, None, None), "trace file publication")
        published = [
            path for path, state in self._published_files().items()
            if state[0] > 0 and state != self._output_before.get(path)
        ]
        if not published or (len(published) != 1 and not self.capture_cpu):
            raise RuntimeError("Profiler stopped without publishing the worker's raw trace bundle")
        self.output = published[0]
        if self.cpu_profiler is not None:
            self.cpu_profiler.stop()
            set_torch_annotations(False)
        if self.scope_only:
            self._export_scopes()
        self.metadata = dict(pid=os.getpid(), raw_files=[str(path) for path in published],
                             raw_directory=str(self._directory), session=self._session,
                             clock_samples=self.clock_samples, capture_cpu=self.capture_cpu,
                             cpu_trace_mode="scopes_only" if self.scope_only else "torch_cpu",
                             scope_clock_domain="CLOCK_MONOTONIC_RAW" if self.scope_only else "wall",
                             explicit_scope_count=len(self.scope_events),
                             stop_complete_wall_ns=time.time_ns())
        if self.cpu_trace_dir is not None:
            destination = self.cpu_trace_dir / f"raw-capture-{os.getpid()}-{time.time_ns()}.json"
            destination.write_text(json.dumps(self.metadata, indent=2) + "\n")


@contextmanager
def scope(label):
    global _annotations_supported
    recorder = _scope_recorder
    if _active and recorder is not None:
        start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        tid = threading.get_native_id()
        try:
            yield
        finally:
            end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            recorder.scope_events.append((start, end, tid, label))
        return
    if _torch_active:
        with torch.profiler.record_function(label):
            yield
        return
    if not _active or _annotations_supported is False:
        yield
        return
    api = _api()
    start = ctypes.c_uint64()
    _check(api.synProfilerGetCurrentTimeNS(ctypes.byref(start)), "scope clock")
    try:
        yield
    finally:
        if _annotations_supported is not False:
            status = api.synProfilerAddCustomMeasurement(label.encode(), start.value)
            if status == 18:  # synUnsupported: raw kernel events remain usable.
                _annotations_supported = False
                logging.getLogger(__name__).warning(
                    "Synapse custom measurements are unsupported; continuing raw kernel capture without annotations")
            else:
                _check(status, "scope annotation")
                _annotations_supported = True
