# SPDX-License-Identifier: Apache-2.0
"""Separate control processes from the V4.1 worker CPU pools at startup."""

import os
from pathlib import Path

from vllm_gaudi import envs
from vllm_gaudi.ops.deepseek_v4_config import parse_cpu_set


def control_cpu_set(role, allowed):
    if role not in ("engine", "api"):
        raise ValueError("Unknown V4.1 control process role")
    groups = {"engine": parse_cpu_set(envs.VLLM_HPU_DSV41_ENGINE_CPUS or ""),
              "api": parse_cpu_set(envs.VLLM_HPU_DSV41_API_CPUS or "")}
    workers = parse_cpu_set(os.environ["VLLM_HPU_DSV4_WORKER_CPUS"])
    workers.update(parse_cpu_set(os.environ["VLLM_HPU_DSV4_WORKER_HELPER_CPUS"].replace(";", ",")))
    owners = {}
    for owner, group in {"workers": workers, **groups}.items():
        for cpu in group:
            siblings = parse_cpu_set(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text())
            for sibling in siblings:
                if sibling in owners and owners[sibling] != owner:
                    raise ValueError("V4.1 control and worker CPUs share a core")
                owners[sibling] = owner
    if not groups[role] <= allowed:
        raise ValueError("V4.1 control CPUs are outside the launch affinity")
    return groups[role]


def bind_control_process(role):
    if not envs.VLLM_HPU_DSV41_ISOLATE_CONTROL:
        return
    if not envs.VLLM_HPU_DSV41_GRAPH_REPLAY:
        raise ValueError("V4.1 control CPU isolation requires its graph runner")
    selected = control_cpu_set(role, os.sched_getaffinity(0))
    for task in Path(f"/proc/{os.getpid()}/task").iterdir():
        try:
            os.sched_setaffinity(int(task.name), selected)
        except ProcessLookupError:
            continue
    from vllm_gaudi.extension.logger import logger as init_logger
    init_logger().info("V4.1 %s control affinity pid=%d CPUs=%s", role, os.getpid(), sorted(selected))
