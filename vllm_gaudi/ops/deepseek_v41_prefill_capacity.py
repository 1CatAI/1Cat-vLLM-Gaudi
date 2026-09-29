# SPDX-License-Identifier: Apache-2.0
"""Bound prefill storage and execution by the normal scheduler configuration."""
import os

MAX_PREFILL_TOKENS = 16384
DEFAULT_PREFILL_TOKENS = 8192
PREFILL_COMPUTE_BUCKETS = (16384, 8192, 4096, 2048, 1024, 512, 256, 128)


def prefill_capacity(scheduler_tokens, tensor_parallel_size=4):
    if scheduler_tokens < 1:
        raise ValueError("Prefill scheduler capacity must be positive")
    maximum = MAX_PREFILL_TOKENS if tensor_parallel_size == 4 else DEFAULT_PREFILL_TOKENS
    return min(int(scheduler_tokens), maximum)


def prefill_compute_buckets(capacity=DEFAULT_PREFILL_TOKENS):
    if not 1 <= capacity <= MAX_PREFILL_TOKENS:
        raise ValueError("Prefill execution capacity must be between 1 and 16384")
    # An explicit debug cap may reduce execution tiles. Normal serving derives
    # its maximum from the scheduler, without a per-optimization opt-in.
    override = os.getenv("VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS")
    maximum = capacity if override is None else int(override)
    if override is not None and maximum not in (4096, 8192, 16384):
        raise ValueError("V4.1 Prefill compute capacity must be 4096, 8192 or 16384 tokens")
    return tuple(size for size in PREFILL_COMPUTE_BUCKETS if size <= min(maximum, capacity))
