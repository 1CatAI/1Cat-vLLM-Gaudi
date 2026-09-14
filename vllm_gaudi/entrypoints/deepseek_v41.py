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
    library_dir = (Path(configured_library).resolve() if configured_library else Path(__file__).resolve().parents[1] /
                   "lib")
    kernel = library_dir / "libdeepseek_v4_gaudi2_kernels.so"
    extensions = list(library_dir.glob("hpu_dsv4_sparse_attn_pt2*.so"))
    if not kernel.is_file() or len(extensions) != 1:
        raise RuntimeError("Build the prepared V4.1 native libraries first")
    if configured_library:
        manifest = json.loads((library_dir / "deepseek_v4_build.json").read_text())
        for library in (kernel, extensions[0]):
            if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["binaries"].get(library.name):
                raise RuntimeError(f"V4.1 native binary differs from its build manifest: {library.name}")
        host_manifest = json.loads((library_dir / "deepseek_v41_build.json").read_text())
        hosts = list(library_dir.glob("dsv41_host_gather*.so"))
        if (len(hosts) != 1 or host_manifest.get("host_gather_abi_version") != 1
                or host_manifest.get("host_c1_abi_version") != 1
                or host_manifest.get("host_gather_packed_output_version") != 1
                or host_manifest.get("host_gather_profiling_version") != 1
                or hashlib.sha256(hosts[0].read_bytes()).hexdigest() != host_manifest["binaries"].get(hosts[0].name)):
            raise RuntimeError("V4.1 host gather differs from its isolated build manifest")
        # The worker imports this exact file through deepseek_v41_host after
        # the HPU environment is initialized and validates the live ABI there.
    configured = os.environ.get("GC_KERNEL_PATH", str(kernel))
    if configured not in (str(kernel), "/usr/lib/habanalabs/libtpc_kernels.so"):
        raise RuntimeError("The V4.1 launch profile requires its combined kernel database")
    os.environ["GC_KERNEL_PATH"] = str(kernel)
    os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"] = str(extensions[0])


def prepare_environment():
    prepare_native_libraries()
    defaults = {
        "PT_HPU_LAZY_MODE": "0",
        "PT_HPU_ENABLE_LAZY_COLLECTIVES": "0",
        "PT_HPU_EAGER_PIPELINE_ENABLE": "1",
        "PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE": "1",
        "PT_HPU_ENABLE_EAGER_CACHE": "0",
        "PT_HPU_WEIGHT_SHARING": "0",
        "RUNTIME_SCALE_PATCHING": "0",
        "PT_HPU_POOL_MEM_ACQUIRE_PERC": "95",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "VLLM_GRAPH_RESERVED_MEM": "0.1",
        "VLLM_HPU_FORCE_CHANNEL_FP8": "0",
        "OMP_NUM_THREADS": "1",
        # This is distinct from the API/engine drain timeout. Workers must be
        # allowed to export an active trace and retire their native resources.
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "60",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main():
    settings_path = Path.home() / ".config/1cat-vllm/deepseek-v41.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    leased_run = bool(os.environ.get("DSV41_RUN_EVIDENCE"))
    runtime_profile = (os.environ.get("DSV41_RUNTIME_PROFILE") if leased_run else None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default=settings.get("model"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-audit")
    parser.add_argument("--runtime-profile", default=runtime_profile or settings.get("runtime_profile"))
    args, extra = parser.parse_known_args()
    if args.model is None:
        parser.error("A prepared model directory is required")
    if args.runtime_profile:
        profile_path = Path(args.runtime_profile).resolve()
        profile = json.loads(profile_path.read_text())
        identity = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        if os.environ.get("DSV41_SERVING_RUNTIME") != identity:
            for item in profile.get("additional_libraries", []) + profile.get("configuration_files", []):
                with Path(item["path"]).open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != item["sha256"]:
                    raise RuntimeError(f"Serving runtime fingerprint changed: {item['path']}")
            environment = dict(os.environ)
            environment.update(profile["environment"])
            # ``tools/run_deepseek_v41.py`` leases a concrete device set and
            # writes per-acquisition paths after loading the profile.  The
            # persistent user config contains defaults for an ordinary
            # install; letting it overwrite those launcher-owned values can
            # make the child try to lock a different (already busy) device
            # set.  Preserve only the values the launcher is responsible for
            # when its evidence marker is present; user settings continue to
            # override the version-locked profile for normal starts.
            launcher_keys = (
                "HABANA_VISIBLE_MODULES",
                "HLS_MODULE_ID",
                "HABANA_LOGS",
                "VLLM_HPU_DSV4_WORKER_CPUS",
                "VLLM_HPU_DSV4_WORKER_HELPER_CPUS",
                "VLLM_HPU_TP2_PLAN_DUMP_DIR",
                "VLLM_TORCH_PROFILER_DIR",
                "DSV41_RUN_EVIDENCE",
                "DSV41_RUNTIME_PROFILE",
                "PT_HPU_RECIPE_CACHE_CONFIG",
            )
            launcher_values = ({
                key: os.environ[key]
                for key in launcher_keys if key in os.environ
            } if leased_run else {})
            environment.update(settings.get("environment", {}))
            # A leased evidence run must use the exact profile it was
            # fingerprinted against.  The persistent user runtime file may
            # contain an older direct-exchange ceiling (or bridge path), and
            # letting it overwrite the candidate makes workers fail before
            # model load.  Ordinary starts have no DSV41_RUN_EVIDENCE marker
            # and keep the existing user-settings precedence.
            if leased_run:
                environment.update(profile["environment"])
            environment.update(launcher_values)
            environment["DSV41_RUNTIME_PROFILE"] = str(profile_path)
            environment["DSV41_SERVING_RUNTIME"] = identity
            os.execve(sys.executable, [sys.executable, "-m", "vllm_gaudi.entrypoints.deepseek_v41", *sys.argv[1:]],
                      environment)
    # The evidence launcher already owns the module leases, NUMA affinity and
    # optional recipe cache. Acquiring them again here would either deadlock
    # on its locks or replace them with the ordinary installation defaults.
    if settings and not leased_run:
        from vllm_gaudi.entrypoints.serving_resources import prepare_serving_resources
        prepare_serving_resources(settings, args.model, extra)
    prepare_environment()
    loader = {} if args.checkpoint_audit is None else {"checkpoint_audit": args.checkpoint_audit}
    trace_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR")
    if trace_dir and not any(value.startswith("--profiler-config") for value in extra):
        # The engine registers /start_profile only from ProfilerConfig. The
        # legacy directory variable alone configures workers, not the API.
        extra += [
            "--profiler-config",
            json.dumps({
                "profiler": "torch",
                "torch_profiler_dir": trace_dir,
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": True
            })
        ]
    if not any(value.startswith("--shutdown-timeout") for value in extra):
        # Native programs and host staging need normal worker teardown. The
        # engine's zero-second default kills its child before cleanup runs.
        extra += ["--shutdown-timeout", "120"]
    if not any(value.startswith("--reasoning-parser") for value in extra):
        extra += ["--reasoning-parser", "deepseek_v41"]
    if not any(value.startswith("--served-model-name") for value in extra):
        extra += ["--served-model-name", "DeepSeek-V4.1-Flash"]
    sys.argv = [
        "vllm", "serve", args.model, "--host", args.host, "--port",
        str(args.port), "--dtype", "bfloat16", "--max-model-len", "1048576", "--generation-config", "vllm",
        "--override-generation-config", '{"temperature":0}', "--tensor-parallel-size", "2", "--pipeline-parallel-size",
        "2", "--max-num-seqs", "32", "--max-num-batched-tokens", "8192", "--enable-chunked-prefill", "--load-format",
        "dsv41_prepared", "--model-loader-extra-config",
        json.dumps(loader), "--mm-encoder-tp-mode", "data", "--no-enable-prefix-caching", "--no-async-scheduling",
        "--speculative-config", '{"method":"dspark","num_speculative_tokens":5}', *extra
    ]
    from vllm.entrypoints.cli.main import main as serve
    serve()


if __name__ == "__main__":
    main()
