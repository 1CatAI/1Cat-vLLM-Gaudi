# SPDX-License-Identifier: Apache-2.0
"""Defaults for the qualified short-context DeepSeek V4 Gaudi2 TP2 path."""

import os
from pathlib import Path


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
    configured = os.environ.get("VLLM_HPU_DSV4_WORKER_CPUS")
    if configured is None:
        return
    cpus = [int(cpu) for cpu in configured.split(",")]
    if rank >= len(cpus) or len(set(cpus)) != len(cpus):
        raise ValueError("VLLM_HPU_DSV4_WORKER_CPUS requires one distinct CPU per rank")
    if cpus[rank] not in os.sched_getaffinity(0):
        raise ValueError("DeepSeek V4 worker CPU is outside the process affinity")
    os.sched_setaffinity(0, {cpus[rank]})
