# SPDX-License-Identifier: Apache-2.0
"""Launch the source-integrated Gaudi2 TP2 short-context decode profile."""

import argparse
import os
from pathlib import Path
import sys


def prepare_environment():
    library_dir = Path(__file__).resolve().parents[1] / "lib"
    kernel = library_dir / "libdeepseek_v4_gaudi2_kernels.so"
    extensions = sorted(library_dir.glob("hpu_dsv4_sparse_attn_pt2*.so"))
    if not kernel.is_file() or len(extensions) != 1:
        raise RuntimeError("Build the DeepSeek V4 native libraries with tools/build_deepseek_v4.py first")
    existing = os.environ.get("GC_KERNEL_PATH")
    if existing and existing not in (str(kernel), "/usr/lib/habanalabs/libtpc_kernels.so"):
        raise RuntimeError("This profile needs its combined kernel database; conflicting GC_KERNEL_PATH is unchanged")
    # One ordinary runtime library includes both the custom and stock GUIDs.
    # Bridge and Synapse see the same value; no delayed imports or hooks.
    os.environ["GC_KERNEL_PATH"] = str(kernel)
    os.environ.setdefault("VLLM_HPU_DSV4_TPC_OP_LIBRARY", str(extensions[0]))
    defaults = {
        "PT_HPU_LAZY_MODE": "0", "PT_HPU_ENABLE_EAGER_CACHE": "0",
        "PT_HPU_EAGER_PIPELINE_ENABLE": "1", "PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE": "1",
        "RUNTIME_SCALE_PATCHING": "1", "PT_HPU_WEIGHT_SHARING": "0",
        "VLLM_HPU_FORCE_CHANNEL_FP8": "0", "PT_HPU_ENABLE_FUSED_SDPA_SINK": "1",
        "VLLM_T_COMPILE_REGIONAL_COMPILATION": "1", "VLLM_KV_CACHE_LAYOUT": "BLHNC",
        "VLLM_GRAPH_RESERVED_MEM": "0.1", "VLLM_BUCKETING_STRATEGY": "lin",
    }
    for bucket, value in (("PROMPT_BS", "1"), ("PROMPT_QUERY", "64"), ("PROMPT_CTX", "0"),
                          ("DECODE_BS", "1"), ("DECODE_BLOCK", "1")):
        for bound in ("MIN", "STEP", "MAX"):
            defaults[f"VLLM_{bucket}_BUCKET_{bound}"] = "1" if bucket == "PROMPT_CTX" and bound == "STEP" else value
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--worker-cpus", help="One allowed CPU per rank, e.g. 6,16; omitted leaves affinity unchanged")
    args, extra = parser.parse_known_args()
    prepare_environment()
    if args.worker_cpus:
        os.environ["VLLM_HPU_DSV4_WORKER_CPUS"] = args.worker_cpus
    sys.argv = ["vllm", "serve", args.model, "--host", args.host, "--port", str(args.port),
                "--dtype", "bfloat16", "--max-model-len", "512", "--generation-config", "vllm",
                "--tensor-parallel-size", "2", "--gpu-memory-utilization", "0.09", "--kv-cache-dtype", "fp8",
                "--no-enable-prefix-caching", "--max-num-batched-tokens", "512", "--max-num-seqs", "1",
                "--async-scheduling", *extra]
    from vllm.entrypoints.cli.main import main as serve
    serve()


if __name__ == "__main__":
    main()
