# SPDX-License-Identifier: Apache-2.0
"""Owned component replay via the same native program APIs used by serving."""
import ctypes as C
import json
import os
from pathlib import Path
import torch


class Sync(C.Structure):
    _fields_ = [("index", C.c_uint32), ("target", C.c_uint64)]


class Info(C.Structure):
    _fields_ = [("state", C.c_int), ("segments", C.c_uint64), ("replays", C.c_uint64),
                ("completion_index", C.c_uint32), ("completion_target", C.c_uint64),
                ("program_bytes", C.c_uint64), ("arc_bytes", C.c_uint64)]


class ComponentReplay:
    def __init__(self, compiled, inputs, *, single_recipe=False, single_segment=True, prepare=None):
        if not os.environ.get("DSV41_RUN_EVIDENCE"):
            raise RuntimeError("Component capture requires an owned diagnostic run")
        self.runtime = C.CDLL("libSynapse.so", mode=os.RTLD_NOW | os.RTLD_NOLOAD)
        probe = C.CDLL(os.environ["DSV41_COMPONENT_REPLAY_PROBE"])
        probe.dsv41GroupedDiagnosticDefaultStream.restype = C.c_uint64
        self.accessor = probe.dsv41GroupedDiagnosticDefaultStream
        specifications = {
            "Create": [C.POINTER(C.c_void_p), C.c_void_p],
            "BeginCapture": [C.c_void_p], "EndCapture": [C.c_void_p], "AbortCapture": [C.c_void_p],
            "GetInfo": [C.c_void_p, C.POINTER(Info)], "Destroy": [C.c_void_p],
            "PreparePlan": [C.c_void_p, C.POINTER(C.c_uint32), C.c_uint64, C.c_uint32, C.c_void_p, C.c_void_p],
            "ReplayPlan": [C.c_void_p, C.POINTER(Sync), C.POINTER(C.c_uint64), C.c_uint64],
            "BeginReplay": [C.c_void_p],
            "ReplaySegment": [C.c_void_p, C.c_uint64, C.POINTER(Sync), C.c_uint8, C.POINTER(Sync)],
        }
        if single_recipe:
            specifications.update(PrepareOrderedReplay=[C.c_void_p], ReplayOrdered=[C.c_void_p])
        self.api = {}
        for name, arguments in specifications.items():
            method = getattr(self.runtime, "synNativeComputeGraph" + name)
            method.argtypes, method.restype = arguments, C.c_int
            self.api[name] = method
        self.handle = C.c_void_p()
        self.inputs, self.compiled, self.outputs = inputs, compiled, None
        self.single_recipe = single_recipe
        self.single_segment = single_segment
        torch.hpu.synchronize()
        self.check("Create", C.byref(self.handle), C.c_void_p(self.accessor()))
        self.check("BeginCapture", self.handle)
        capturing = True
        try:
            self.outputs = compiled(*inputs)
            self.accessor()  # Join ordinary Bridge capture submissions.
            self.check("EndCapture", self.handle)
            capturing = False
            torch.hpu.synchronize()
            captured_info = self.info()
            (Path(os.environ["DSV41_RUN_EVIDENCE"]) / "capture-info.json").write_text(
                json.dumps(captured_info, indent=2) + "\n")
            if prepare is not None:
                prepare(self.runtime, self.handle, captured_info["segments"])
            if single_segment:
                if captured_info["segments"] != 1:
                    raise RuntimeError("Component must compile to one retained recipe")
            elif single_recipe:
                if self.info()["segments"] != 1:
                    raise RuntimeError("Component reference must compile to exactly one production recipe")
                self.check("PrepareOrderedReplay", self.handle)
            else:
                self.check("PreparePlan", self.handle, None, 0, 0, None, None)
        except BaseException:
            if capturing:
                self.check("AbortCapture", self.handle)
            self.close()
            raise
        self.completion, self.statistics = Sync(), (C.c_uint64 * 12)()

    def check(self, name, *args):
        status = self.api[name](*args)
        if status:
            raise RuntimeError(f"Native component {name} failed: {status}")

    def replay(self):
        # The runtime enqueues its prepared job through Stream::addJob, so
        # preceding input DMA and following consumers retain real dependencies.
        if self.single_segment:
            self.check("BeginReplay", self.handle)
            self.check("ReplaySegment", self.handle, 0, None, 1, C.byref(self.completion))
        elif self.single_recipe:
            self.check("ReplayOrdered", self.handle)
        else:
            self.check("ReplayPlan", self.handle, C.byref(self.completion), self.statistics, len(self.statistics))

    def info(self):
        info = Info()
        self.check("GetInfo", self.handle, C.byref(info))
        return {name: getattr(info, name) for name, _ in info._fields_}

    def close(self):
        if self.handle:
            torch.hpu.synchronize()
            self.check("Destroy", self.handle)
            self.handle = C.c_void_p()
