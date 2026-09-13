# SPDX-License-Identifier: Apache-2.0
"""Launch the prepared V4.1 TP2 x PP2 profile through the normal vLLM CLI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def prepare_native_libraries():
    configured_library = os.environ.get("VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR")
    library_dir = (Path(configured_library).resolve() if configured_library
                   else Path(__file__).resolve().parents[1] / "lib")
    kernel = library_dir / "libdeepseek_v4_gaudi2_kernels.so"
    extensions = list(library_dir.glob("hpu_dsv4_sparse_attn_pt2*.so"))
    if not kernel.is_file() or len(extensions) != 1:
        raise RuntimeError("Build the prepared V4.1 native libraries first")
    if configured_library:
        manifest = json.loads((library_dir / "deepseek_v4_build.json").read_text())
        for library in (kernel, extensions[0]):
            if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["binaries"].get(library.name):
                raise RuntimeError(f"V4.1 native binary differs from its build manifest: {library.name}")
    configured = os.environ.get("GC_KERNEL_PATH", str(kernel))
    if configured not in (str(kernel), "/usr/lib/habanalabs/libtpc_kernels.so"):
        raise RuntimeError("The V4.1 launch profile requires its combined kernel database")
    os.environ["GC_KERNEL_PATH"] = str(kernel)
    os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"] = str(extensions[0])


def prepare_environment():
    prepare_native_libraries()
    defaults = {
        "PT_HPU_LAZY_MODE": "0", "PT_HPU_ENABLE_LAZY_COLLECTIVES": "0",
        "PT_HPU_EAGER_PIPELINE_ENABLE": "1", "PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE": "1",
        "PT_HPU_ENABLE_EAGER_CACHE": "0", "PT_HPU_WEIGHT_SHARING": "0", "RUNTIME_SCALE_PATCHING": "0",
        "PT_HPU_POOL_MEM_ACQUIRE_PERC": "95", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0", "VLLM_GRAPH_RESERVED_MEM": "0.1",
        "VLLM_HPU_FORCE_CHANNEL_FP8": "0", "OMP_NUM_THREADS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-audit")
    args, extra = parser.parse_known_args()
    prepare_environment()
    from vllm_gaudi import envs
    speculative = (["--speculative-config", '{"method":"dspark","num_speculative_tokens":5}']
                   if envs.VLLM_HPU_DSV41_DSPARK else [])
    loader = {} if args.checkpoint_audit is None else {"checkpoint_audit": args.checkpoint_audit}
    sys.argv = ["vllm", "serve", args.model, "--host", args.host, "--port", str(args.port),
                "--dtype", "bfloat16", "--max-model-len", "512", "--generation-config", "vllm",
                "--tensor-parallel-size", "2", "--pipeline-parallel-size", "2", "--max-num-seqs", "1",
                "--max-num-batched-tokens", "512", "--load-format", "dsv41_prepared",
                "--model-loader-extra-config", json.dumps(loader), "--mm-encoder-tp-mode", "data",
                "--no-enable-prefix-caching", "--no-async-scheduling", "--block-size", "512",
                *speculative, *extra]
    from vllm.entrypoints.cli.main import main as serve
    serve()


if __name__ == "__main__":
    main()
