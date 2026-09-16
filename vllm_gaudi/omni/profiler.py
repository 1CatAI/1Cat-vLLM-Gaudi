# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch profiler integration for vLLM Omni workers on HPU."""

from __future__ import annotations

from typing import Any

import torch
from vllm_omni.profiler.omni_torch_profiler import OmniTorchProfilerWrapper


class HPUOmniTorchProfilerWrapper(OmniTorchProfilerWrapper):
    """Collect CPU and Synapse hardware events through ``torch.profiler``."""

    def _get_default_activities(self) -> list[str]:
        return ["CPU", "HPU"]

    def _create_profiler(
        self,
        profiler_config: Any,
        activities: list[str],
    ):
        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "HPU": torch.profiler.ProfilerActivity.HPU,
        }
        unsupported = sorted(set(activities) - activity_map.keys())
        if unsupported:
            raise ValueError(f"unsupported HPU profiler activities: {unsupported}")
        return torch.profiler.profile(
            activities=[activity_map[activity] for activity in activities],
            record_shapes=profiler_config.torch_profiler_record_shapes,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_stack=profiler_config.torch_profiler_with_stack,
            with_flops=profiler_config.torch_profiler_with_flops,
            on_trace_ready=self._on_trace_ready,
        )


__all__ = ["HPUOmniTorchProfilerWrapper"]
