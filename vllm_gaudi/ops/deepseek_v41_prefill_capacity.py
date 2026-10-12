# SPDX-License-Identifier: Apache-2.0
"""Bound prefill storage and execution by the normal scheduler configuration."""
import os

MAX_PREFILL_TOKENS = 16384
DEFAULT_PREFILL_TOKENS = 8192
PREFILL_COMPUTE_BUCKETS = (16384, 8192, 4096, 2048, 1024, 512, 256, 128)


def prefill_target_tokens(scheduler_config):
    """Separate target compute/storage from additional parallel-draft inputs."""
    scheduled = scheduler_config.max_num_scheduled_tokens
    return scheduler_config.max_num_batched_tokens if scheduled is None else scheduled


def reserve_dspark_input_slots(config):
    """Keep the requested target budget while reserving separate draft inputs.

    The scheduler deducts parallel-draft input slots even for prompt chunks.
    Target and draft execute in separate prepared plans; a full target chunk
    must not become an unaligned remainder when speculation is enabled.
    Configuration can run again after serialization, so retain an explicit
    compute budget rather than adding the reservation repeatedly.
    """
    scheduler = config.scheduler_config
    target = prefill_target_tokens(scheduler)
    slots = config.speculative_config.max_num_new_slots_for_drafting
    scheduler.max_num_scheduled_tokens = target
    scheduler.max_num_batched_tokens = max(scheduler.max_num_batched_tokens, target + scheduler.max_num_seqs * slots)


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
