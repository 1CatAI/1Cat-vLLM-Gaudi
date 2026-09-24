# SPDX-License-Identifier: Apache-2.0
"""On-demand Synapse raw capture; offline parsing avoids concurrent Kineto expansion."""
import ctypes
import logging
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path

import torch

_library = None
_active = False
_annotations_supported = None


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
        paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if line.split() and Path(line.split()[-1]).name == "libSynapse.so"}
        if len(paths) != 1:
            raise RuntimeError(f"Native trace requires one loaded Synapse library, found {len(paths)}")
        _library = ctypes.CDLL(paths.pop(), mode=os.RTLD_NOW | os.RTLD_NOLOAD)
        _library.synProfilerGetCurrentTimeNS.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        _library.synProfilerQueryRequiredMemory.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        _library.synProfilerSetUserBuffer.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        for name in ("synProfilerStart", "synProfilerStop"):
            getattr(_library, name).argtypes = [ctypes.c_int, ctypes.c_uint32]
        _library.synProfilerGetTrace.argtypes = [
            ctypes.c_int, ctypes.c_uint32, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        _library.synProfilerAddCustomMeasurement.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    return _library


def _check(status, name):
    if status:
        raise RuntimeError(f"{name} failed with Synapse status {status}")


class NativeTrace:
    def __init__(self):
        self.buffer = None
        self.running = False
        self.output = None
        self._directory = None
        self._session = None
        self._accel_name = None

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
        plugins = [plugin for plugin in config.get("Plugins", ())
                   if plugin.get("name") == "HwTrace" and plugin.get("enable")]
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
        global _active, _annotations_supported
        if self.running:
            raise RuntimeError("Native trace is already running")
        self._prepare_output()
        torch.hpu.synchronize()
        api = _api()
        required = ctypes.c_uint32()
        _check(api.synProfilerQueryRequiredMemory(0, ctypes.byref(required)), "trace memory query")
        if required.value:
            self.buffer = torch.empty(required.value, dtype=torch.uint8, device="hpu")
            _check(api.synProfilerSetUserBuffer(0, self.buffer.data_ptr()), "trace memory binding")
        _check(api.synProfilerStart(1, 0), "trace start")
        _annotations_supported = None
        self.running = _active = True

    def stop(self):
        global _active
        if not self.running:
            raise RuntimeError("Native trace is not running")
        torch.hpu.synchronize()
        api = _api()
        _check(api.synProfilerStop(1, 0), "trace stop")
        self.running = _active = False
        # A null size requests file publication. Unlike the size-query and
        # caller-buffer forms used by Kineto, this honors the configured raw
        # output/skipParse mode without materializing all events in Python.
        # Stop alone does not publish a bundle with every supported SDK.
        _check(api.synProfilerGetTrace(1, 0, 1, None, None, None), "trace file publication")
        published = [path for path, state in self._published_files().items()
                     if state[0] > 0 and state != self._output_before.get(path)]
        if len(published) != 1:
            raise RuntimeError("Profiler stopped without publishing the worker's raw trace bundle")
        self.output = published[0]


@contextmanager
def scope(label):
    global _annotations_supported
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
