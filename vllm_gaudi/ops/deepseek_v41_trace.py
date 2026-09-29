# SPDX-License-Identifier: Apache-2.0
"""On-demand prefill scopes; never add synchronization or compiled nodes."""
from functools import wraps

import torch

_active = False


def enable(enabled):
    global _active
    _active = bool(enabled)


def prefill_scope(name):

    def decorate(function):

        @wraps(function)
        def call(self, *args, **kwargs):
            if torch.compiler.is_compiling() or not _active:
                return function(self, *args, **kwargs)
            # C1 native replay has its own stage/segment markers. These host
            # scopes are for the large-M path, not per-token replay callbacks.
            if args and isinstance(args[0], torch.Tensor) and args[0].shape[0] <= 6:
                return function(self, *args, **kwargs)
            label = f"v41::prefill::{name}::L{getattr(self, 'layer', 'unknown')}"
            with torch.profiler.record_function(label):
                return function(self, *args, **kwargs)

        return call

    return decorate
