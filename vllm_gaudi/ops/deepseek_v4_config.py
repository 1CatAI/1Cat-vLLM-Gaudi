# SPDX-License-Identifier: Apache-2.0
"""Defaults for the bounded source-integrated DeepSeek V4 Gaudi2 TP2 path."""

import os
from pathlib import Path


_initial_worker_affinity = None


DEFAULTS = {
    "VLLM_HPU_DSV4_BF16_SCORE_PROJECTION": "1",
    "VLLM_HPU_DSV4_BF16_ATTN_WEIGHT_CACHE": "1",
    "VLLM_HPU_DSV4_COMPILED_ATTN_FRONTEND": "1",
    "VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP": "1",
    "VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN": "1",
    "VLLM_HPU_DSV4_FLASHMLA_SPLIT_KV": "1",
    "VLLM_HPU_DSV4_FLASHMLA_TILED": "1",
    "VLLM_HPU_DSV4_FLASHMLA_SPLITS": "4",
    "VLLM_HPU_DSV4_DIRECT_DECODE_DISPATCH": "1",
    "VLLM_HPU_DSV4_TPC_SAVE_COMPRESS_NORM_C4": "1",
    "VLLM_HPU_DSV4_TPC_ORDERED_COMPRESSOR": "1",
    "VLLM_HPU_DSV4_TPC_QNORM_ROPE_KV_PACK": "1",
    "VLLM_HPU_DSV4_TPC_DEQUANT_GATHER": "1",
    "VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN": "1",
    "VLLM_HPU_DSV4_COMPILE_CHUNK_SIZE": "8",
    "VLLM_HPU_DSV4_PACKED_DECODE_METADATA": "1",
    "VLLM_HPU_DSV4_Q1_METADATA_FASTPATH": "1",
    "VLLM_HPU_FP8_JIT_DYNAMIC_QUANT": "1",
    "VLLM_HPU_DSV4_EARLY_OUTPUT_LOWERING": "1",
}


def configure_defaults(config, *, gaudi2):
    model = config.model_config
    if (not gaudi2 or model is None or model.hf_config.model_type != "deepseek_v4"
            or model.max_model_len > 512 or config.parallel_config.tensor_parallel_size != 2
            or config.scheduler_config.max_num_seqs != 1 or model.enforce_eager):
        return False
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)
    library_dir = Path(__file__).resolve().parents[1] / "lib"
    libraries = sorted(library_dir.glob("hpu_dsv4_sparse_attn_pt2*.so"))
    if len(libraries) == 1:
        os.environ.setdefault("VLLM_HPU_DSV4_TPC_OP_LIBRARY", str(libraries[0]))
        kernel = library_dir / "libdeepseek_v4_gaudi2_kernels.so"
        if kernel.is_file():
            os.environ.setdefault("GC_KERNEL_PATH", str(kernel))
    return True


def bind_worker_cpu(rank):
    global _initial_worker_affinity
    configured = os.environ.get("VLLM_HPU_DSV4_WORKER_CPUS")
    if configured is None:
        return
    cpus = [int(cpu) for cpu in configured.split(",")]
    if rank >= len(cpus) or len(set(cpus)) != len(cpus):
        raise ValueError("VLLM_HPU_DSV4_WORKER_CPUS requires one distinct CPU per rank")
    if _initial_worker_affinity is None:
        _initial_worker_affinity = os.sched_getaffinity(0)
    if cpus[rank] not in _initial_worker_affinity:
        raise ValueError("DeepSeek V4 worker CPU is outside the process affinity")
    os.sched_setaffinity(0, {cpus[rank]})


def parse_cpu_set(value):
    result = set()
    for field in value.split(","):
        bounds = field.strip().split("-")
        if len(bounds) == 1 and bounds[0]:
            first = last = int(bounds[0])
        elif len(bounds) == 2 and all(bounds):
            first, last = map(int, bounds)
        else:
            raise ValueError("CPU sets require comma-separated CPU IDs or inclusive ranges")
        if first < 0 or last < first:
            raise ValueError("Invalid CPU range")
        result.update(range(first, last + 1))
    if not result:
        raise ValueError("CPU set cannot be empty")
    return result


def bind_worker_helpers(rank):
    """Bind initialized runtime pools once, outside timed model execution."""
    configured = os.environ.get("VLLM_HPU_DSV4_WORKER_HELPER_CPUS")
    if configured is None:
        return
    mains = [int(cpu) for cpu in os.environ.get("VLLM_HPU_DSV4_WORKER_CPUS", "").split(",") if cpu]
    helpers = [parse_cpu_set(value) for value in configured.split(";")]
    if len(helpers) != len(mains) or rank >= len(mains) or _initial_worker_affinity is None:
        raise ValueError("Helper affinity requires one CPU set per configured worker rank")
    reserved = set(mains)
    for cpu in mains:
        siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        reserved.update(parse_cpu_set(siblings.read_text().strip()))
    for index, group in enumerate(helpers):
        if not group <= _initial_worker_affinity or group & reserved:
            raise ValueError("Helper CPUs must be allowed and exclude every main CPU and SMT sibling")
        if any(group & previous for previous in helpers[:index]):
            raise ValueError("Worker helper CPU sets must be disjoint")
    pid = os.getpid()
    count = 0
    for task in Path(f"/proc/{pid}/task").iterdir():
        tid = int(task.name)
        if tid == pid:
            continue
        try:
            os.sched_setaffinity(tid, helpers[rank])
            count += 1
        except ProcessLookupError:
            pass
    from vllm_gaudi.extension.logger import logger as init_logger
    init_logger().info("DeepSeek V4 worker affinity rank=%d main=%d helpers=%s threads=%d",
                       rank, mains[rank], sorted(helpers[rank]), count)
