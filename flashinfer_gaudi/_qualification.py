# SPDX-License-Identifier: Apache-2.0
"""Offline paired performance qualification; this never enables production ops."""

from __future__ import annotations

import math
import random
import statistics


def qualify(sessions: list[dict],
            *,
            correctness: bool,
            whole_operation_native: bool,
            minimum_speedup: float = 1.05,
            bootstrap_samples: int = 5000) -> dict:
    """Use session-clustered, paired resampling instead of cherry-picked minima.

    Every session must contain at least 15 interleaved reference/candidate
    waves. Three independent process sessions are required for qualification.
    Callers must record unsuccessful correctness tests without benchmarking.
    """
    if not math.isfinite(minimum_speedup) or minimum_speedup < 1.05:
        raise ValueError("The qualification threshold cannot be less than 1.05.")
    if bootstrap_samples < 1000:
        raise ValueError("At least 1000 bootstrap samples are required.")
    result = {
        "schema_version": 1,
        "qualified": False,
        "minimum_speedup": minimum_speedup,
        "correctness": correctness,
        "whole_operation_native": whole_operation_native
    }
    if not correctness or not whole_operation_native:
        return {**result, "reason": "correctness_or_native_gate_failed"}
    if len(sessions) < 3:
        return {**result, "reason": "at_least_three_process_sessions_required"}
    if len({session["session_id"] for session in sessions}) != len(sessions):
        raise ValueError("Process session identities must be distinct.")
    pairs = []
    for session in sessions:
        reference, candidate = session["reference_ms"], session["candidate_ms"]
        if len(reference) != len(candidate) or len(reference) < 15:
            raise ValueError("Each process session requires at least 15 paired waves.")
        if not all(math.isfinite(value) and value > 0 for value in (*reference, *candidate)):
            raise ValueError("Timings must be finite and positive.")
        pairs.append(list(zip(reference, candidate)))

    def ratio(waves):
        return statistics.median(pair[0] for pair in waves) / statistics.median(pair[1] for pair in waves)

    speedup = ratio([pair for session in pairs for pair in session])
    rng = random.Random(31)
    estimates = []
    for _ in range(bootstrap_samples):
        sampled = []
        for session in rng.choices(pairs, k=len(pairs)):
            sampled.extend(rng.choices(session, k=len(session)))
        estimates.append(ratio(sampled))
    estimates.sort()
    lower, upper = estimates[int(bootstrap_samples * .025)], estimates[int(bootstrap_samples * .975)]
    qualified = speedup >= minimum_speedup and lower > 1.0
    return {
        **result, "qualified": qualified,
        "speedup": speedup,
        "confidence_interval_95": [lower, upper],
        "process_sessions": len(sessions),
        "reason": "passed" if qualified else "performance_gate_failed"
    }
