# SPDX-License-Identifier: Apache-2.0
"""Measure one cached four-layer preparation without another reference arm."""
import functools
from pathlib import Path
import time

import torch

import habana_frameworks.torch.dynamo.compile_backend._recipe_compiler_C as sdk
from vllm_gaudi.ops import tp2_prepared_plan as prepared
from vllm_gaudi.ops.deepseek_v41_replay import StageVariant, _Snapshot
from vllm_gaudi.compilation import deepseek_v41_frontend_cache as front

_stats = {}
_skipped = []


def wrap(owner, name, label):
    original = getattr(owner, name)

    @functools.wraps(original)
    def measured(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            row = _stats.setdefault(label, dict(calls=0, seconds=0., max_seconds=0.))
            row['calls'] += 1
            row['seconds'] += elapsed
            row['max_seconds'] = max(row['max_seconds'], elapsed)

    try:
        setattr(owner, name, measured)
    except (TypeError, AttributeError):
        _skipped.append(label)


for name in ('graph_compile', 'graph_launch', 'batch_empty'):
    wrap(sdk, name, 'SDK.' + name)
wrap(torch.hpu, 'synchronize', 'HPU.synchronize')
wrap(StageVariant, '__init__', 'StageVariant.constructor')
wrap(_Snapshot, '__init__', 'Snapshot.constructor')
wrap(_Snapshot, 'restore', 'Snapshot.restore')
wrap(front, 'cached_group_entry', 'Frontend.identity_and_index')

_original_runtime = prepared._runtime
_patched = False


def runtime():
    global _patched
    bridge, backend = _original_runtime()
    if not _patched:
        for name in ('add_compute', 'prepare'):
            wrap(bridge.PreparedGroupPlan, name, 'PreparedGroupPlan.' + name)
        for name in dir(bridge):
            cls = getattr(bridge, name)
            if isinstance(cls, type) and 'Graph' in name and hasattr(cls, 'instantiate'):
                wrap(cls, 'instantiate', name + '.instantiate')
        _patched = True
    return bridge, backend


prepared._runtime = runtime
script = Path(__file__).with_name('check_deepseek_v41_request_c6_batch.py')
source = script.read_text()
old = 'arm_order = (1, 0) if args.candidate == "startup_frontend" and args.frontend_first else (0, 1)'
assert old in source
source = source.replace(old, 'arm_order = (1,)')
marker = '                if plan not in _native_entries:\n'
assert marker in source
source = source.replace(
    marker, '''                if args.candidate == "startup_frontend":
                    if plan not in _native_entries:
                        raise RuntimeError("Startup probe did not retain the native replay")
                    repeated = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        additional = StageVariant(program, *values, plan_engram,
                                                  native_input=False, fused_text_io=False)
                        repeated.append(time.perf_counter() - t0)
                        if [entry.identity for entry in additional.compiled.chunks] != [entry.identity for entry in plan.compiled.chunks]:
                            raise RuntimeError("Metadata memoization changed the compute identity")
                    report["repeated_constructor_seconds"] = repeated
                    report.update(status="startup_execution_diagnosed", host_execution_timings=_stats,
                                  host_execution_timing_skipped=_skipped,
                                  diagnostic_only=True, performance_gain_qualified=False)
                    return
''' + marker, 1)
namespace = dict(__name__='__main__', __file__=str(script), _stats=_stats, _skipped=_skipped)
exec(compile(source, str(script), 'exec'), namespace)
