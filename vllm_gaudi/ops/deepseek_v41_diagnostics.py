# SPDX-License-Identifier: Apache-2.0
"""Opt-in host ranges without stack capture or changes to execution ordering."""
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import os

_context = ContextVar("dsv41_diagnostic_context", default=None)


def trace_phase(function):
    # Return the original function when disabled: no timed-path wrapper,
    # context lookup, event or per-call environment lookup is introduced.
    if os.environ.get("VLLM_HPU_DSV41_PHASE_TRACE", "0") != "1":
        return function

    @wraps(function)
    def recorded(self, *args, **kwargs):
        import torch
        if function.__name__ == "execute_model":
            scheduled = args[0] if args else kwargs["scheduled"]
            request = next(iter(scheduled.num_scheduled_tokens), "")
            rank = torch.distributed.get_rank()
            packed = self.pp.packed
            _context.set(
                dict(rank=rank,
                     stage=rank // 2,
                     generation=self.pp.generation + 1,
                     request=hashlib.sha256(request.encode()).hexdigest()[:16],
                     next_packet_slot=packed.generation % len(packed.packets) if packed else None))
        fields = dict(_context.get() or {},
                      phase=function.__qualname__,
                      segment=None,
                      collective=None,
                      completion_event=None)
        # Missing native identifiers stay null; host context does not prove
        # ownership of a device stall or completion of a collective.
        name = "dsv41_phase:" + json.dumps(fields, separators=(",", ":"), sort_keys=True)
        with torch.profiler.record_function(name):
            return function(self, *args, **kwargs)

    return recorded
