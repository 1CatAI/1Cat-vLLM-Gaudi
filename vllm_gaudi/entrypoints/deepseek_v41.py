# SPDX-License-Identifier: Apache-2.0
"""Launch the prepared V4.1 TP2 x PP2 profile through the normal vLLM CLI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

_C1_FASTPATH_DEFAULTS = {
    # This is the quality-qualified numerical profile used by the prepared
    # V4.1 deployment.  Individual switches remain explicit below so a
    # diagnostic override can disable one component before process startup.
    "VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS": "1",
    "VLLM_HPU_DSV41_PREPARED_SHARDS": "1",
    "VLLM_HPU_DSV41_ENGRAM_HOST_TABLE": "1",
    "VLLM_HPU_DSV41_GRAPH_REPLAY": "1",
    "VLLM_HPU_DSV41_VISION": "1",
    "VLLM_HPU_DSV41_QUANT_ROUNDTRIP": "1",
    "VLLM_HPU_DSV41_PACKED_ATTENTION": "1",
    "VLLM_HPU_DSV41_FIXED_POSITIONS": "1",
    "VLLM_HPU_DSV41_PACKED_PP": "1",
    "VLLM_HPU_DSV41_TPC_MHC": "1",
    "VLLM_HPU_DSV41_DIRECT_TOKEN_IDS": "1",
    "VLLM_HPU_DSV41_NATIVE_PP_COPY": "1",
    "VLLM_HPU_DSV41_PREPARED_OUTPUT": "1",
    "VLLM_HPU_DSV41_BOUNDED_ATTENTION": "1",
    "VLLM_HPU_DSV41_SWA_PACK_WRITE": "1",
    "VLLM_HPU_DSV41_FP4_CACHE_WRITE": "1",
    "VLLM_HPU_DSV41_NATIVE_ROPE": "1",
    "VLLM_HPU_DSV41_C1_INDICES": "1",
    "VLLM_HPU_DSV41_SELECTED_VALID_ONLY": "1",
    "VLLM_HPU_DSV41_TP_MHC_OVERLAP": "1",
    "VLLM_HPU_DSV41_ENGRAM_NATIVE_C1": "1",
    "VLLM_HPU_DSV41_ENGRAM_C1_PACKET": "1",
    "VLLM_HPU_DSV41_DECODED_KV_STATE": "1",
    # The long-context model keeps its canonical packed page pool.  Decode
    # selected rows directly from that pool instead of gathering, expanding
    # and materializing a per-layer BF16 cache before MLA.
    "VLLM_HPU_DSV41_PAGED_SELECTED_KV": "1",
    "VLLM_HPU_DSV41_SHARED_PREFIX_KV": "1",
    "VLLM_HPU_DSV41_FUSED_PREFIX_LAYOUT": "1",
    "VLLM_HPU_DSV41_NATIVE_KV_PACK": "1",
    # Keep the scheduler-owned packed pages canonical for the full 1M
    # lifetime, while retaining the active <=512-token prefix in the exact
    # decoded form already produced by the same quantizing writer.
    "VLLM_HPU_DSV41_PAGED_DECODED_KV_STATE": "1",
    # Use one fixed-capacity native CSA2 graph whose device-valued position
    # bounds the real scan.  This keeps long conversations on the same replay
    # path and avoids the generic per-bucket PyTorch score/top-k chain.
    "VLLM_HPU_DSV41_RUNTIME_INDEXER": "1",
    "VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH": "1",
    "VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT": "1",
    "VLLM_HPU_DSV41_V2_EARLY_INPUT_COMMIT": "1",
    "VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX": "1",
    "VLLM_HPU_DSV41_V2_DEVICE_ENGRAM": "1",
    "VLLM_HPU_DSV41_V2": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_HPU_DSV41_EXPERT_K128": "1",
    # Reuse decoded expert weights across each scheduler prompt chunk while
    # C1 decode continues to consume the same resident N256 allocation.  The
    # stock packed-MXFP4 prefill operator leaks its internal packed dtype into
    # later Synapse recipes on this runtime, so it is deliberately excluded.
    "VLLM_HPU_DSV41_PREFILL_GROUPED": "1",
    # Keep route metadata on HPU and make the descriptor shape depend only on
    # the scheduler tile.  Compact route output bounds the shared workspace to
    # real top-6 rows instead of the occupancy-dependent host implementation.
    "VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES": "1",
    "VLLM_HPU_DSV41_PREFILL_ROUTE_OUTPUT": "1",
    "VLLM_HPU_DSV41_PREFILL_EXPERT_ROWS": "128",
    "VLLM_HPU_DSV41_PREFILL_COMPACT_CANDIDATES": "1",
    "VLLM_HPU_DSV41_PREFILL_MLA_ROWS": "64",
    "VLLM_HPU_DSV41_COMPRESSOR_FUSED_INPUT": "1",
    "VLLM_HPU_TP2_NATIVE_JOINT_PLAN": "1",
    "VLLM_HPU_TP2_PREPARED_COMM": "1",
    "VLLM_HPU_TP2_STATIC_GROUP_PLAN": "1",
}

_NUMERIC_FASTPATH_DEFAULTS = {
    "VLLM_HPU_DSV41_WO_A_FP8": "1",
    # Keep the exact group-32 BF16 roundtrip required by wo_b inside the
    # wo_a scale producer.  This removes one HBM intermediate without
    # changing any model-visible arithmetic boundary.
    "VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP": "1",
    "VLLM_HPU_DSV41_ROUTER_TOP6": "1",
    "VLLM_HPU_DSV41_BF16_LM_HEAD": "1",
    "VLLM_HPU_DSV41_MLA_MME": "1",
    "VLLM_HPU_DSV41_EXPERT_N256_FP8": "1",
    "VLLM_HPU_DSV41_EXPERT_FUSED_QUANT": "1",
    "VLLM_HPU_DSV41_QKV_FUSED_INPUT": "1",
    "VLLM_HPU_DSV41_ATTN_FUSED_NORM": "1",
    "VLLM_HPU_DSV41_SHARED_GATE_UP": "1",
    "VLLM_HPU_DSV41_BF16_ROUTER_GATE": "1",
    "VLLM_HPU_DSV41_MHC_GATES_FUSED": "1",
    "VLLM_HPU_DSV41_MHC_CONTROL_RRMS": "1",
    "VLLM_HPU_DSV41_ATTN_DENSE_FP8": "1",
    "VLLM_HPU_DSV41_ENGRAM_FP8": "1",
    "VLLM_HPU_DSV41_Q_SCALE_ROPE": "1",
    "VLLM_HPU_DSV41_EXPERT_FUSED_REDUCE": "1",
}

_SIDECARS = {
    "wo_a_fp8": ("VLLM_HPU_DSV41_WO_A_FP8", "VLLM_HPU_DSV41_WO_A_FP8_SIDECAR"),
    "attention_dense_fp8": ("VLLM_HPU_DSV41_ATTN_DENSE_FP8", "VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR"),
    "engram_fp8": ("VLLM_HPU_DSV41_ENGRAM_FP8", "VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR"),
}


def _enabled(value):
    return str(value).strip().lower() in ("1", "true")


def prepare_default_fastpaths(model, sidecars=None):
    """Enable the frozen-reference C1/V2 profile from one aggregate switch."""
    if not _enabled(os.environ.get("VLLM_HPU_DSV41_DEFAULT_FASTPATHS", "1")):
        return
    os.environ.setdefault("VLLM_HPU_DSV41_DSPARK", "0")
    if _enabled(os.environ["VLLM_HPU_DSV41_DSPARK"]):
        return
    selected_v2 = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    selected_adapter = os.environ.get("VLLM_HPU_DSV41_V2")
    if selected_v2 is not None and selected_adapter is None:
        os.environ["VLLM_HPU_DSV41_V2"] = "1" if _enabled(selected_v2) else "0"
    elif selected_adapter is not None and selected_v2 is None:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if _enabled(selected_adapter) else "0"
    for key, value in _C1_FASTPATH_DEFAULTS.items():
        os.environ.setdefault(key, value)
    if _enabled(os.environ.get("VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS", "0")):
        for key, value in _NUMERIC_FASTPATH_DEFAULTS.items():
            os.environ.setdefault(key, value)
    model = Path(model).resolve()
    configured = sidecars or {}
    for name, (enabled_key, path_key) in _SIDECARS.items():
        if not _enabled(os.environ.get(enabled_key, "0")) or os.environ.get(path_key):
            continue
        candidates = []
        if configured.get(name):
            candidates.append(Path(configured[name]).expanduser())
        candidates.extend((model / "sidecars" / name, model.parent / f"{model.name}-{name}"))
        directory = next((candidate.resolve() for candidate in candidates if candidate.is_dir()), None)
        if directory is None:
            raise RuntimeError(
                f"Default V4.1 fast paths require the {name} sidecar; prepare {model / 'sidecars' / name} "
                f"or disable {enabled_key}")
        os.environ[path_key] = str(directory)


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
                or host_manifest.get("host_c1_abi_version") != 2
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


def load_native_operators(required=()):
    """Load the fingerprinted V4.1 extension before model/Dynamo imports.

    Spawned workers do not inherit PyTorch's process-local operator registry.
    Loading on demand from the model constructor was also brittle: another
    extension could already have populated part of ``custom_op`` and make a
    single-symbol guard skip the selected library.  The worker therefore
    loads the exact manifest-checked extension once after binding its HPU and
    validates every operator needed by the selected execution profile.
    """
    prepare_native_libraries()
    import torch

    library = os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"]
    torch.ops.load_library(library)
    baseline = (
        "custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2",
        "custom_deepseek_v41_paged_attention_bf16_gaudi2",
    )
    names = tuple(dict.fromkeys((*baseline, *required)))
    missing = [name for name in names if not hasattr(torch.ops.custom_op, name)]
    if missing:
        raise RuntimeError(
            f"V4.1 native extension {library} is missing required operators: "
            + ", ".join(missing))
    return library


def prepare_environment(model=None, sidecars=None):
    if model is not None:
        prepare_default_fastpaths(model, sidecars)
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
    parser.add_argument("--n256-prepared-dir", type=Path,
                        help="Runtime-layout expert cache from prepare_deepseek_v41_n256.py")
    parser.add_argument("--runtime-profile", default=runtime_profile or settings.get("runtime_profile"))
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=512)
    v2 = parser.add_mutually_exclusive_group()
    v2.add_argument("--v2", dest="v2", action="store_true", help="Use the V2 HPU scheduling/completion adapter")
    v2.add_argument("--no-v2", dest="v2", action="store_false", help="Use the synchronous V4.1 model runner")
    parser.set_defaults(v2=None)
    args, extra = parser.parse_known_args()
    if args.v2 is not None:
        selected = "1" if args.v2 else "0"
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = selected
        os.environ["VLLM_HPU_DSV41_V2"] = selected
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
                # Candidate evidence runs build their native extension and
                # kernel database in an isolated directory.  Keep those
                # fingerprinted launcher selections across the runtime-profile
                # re-exec instead of silently restoring the profile's older
                # build.  prepare_native_libraries() validates both files
                # against the candidate build manifest before workers start.
                "GC_KERNEL_PATH",
                "VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR",
                "VLLM_HPU_DSV4_TPC_OP_LIBRARY",
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
    n256_directory = args.n256_prepared_dir or settings.get("n256_prepared_dir")
    if n256_directory:
        directory = Path(n256_directory).resolve()
        if not (directory / "manifest.json").is_file():
            raise ValueError("Runtime N256 preparation has not published a complete manifest")
        os.environ["VLLM_HPU_DSV41_N256_PREPARED_DIR"] = str(directory)
    prepare_environment(args.model, settings.get("sidecars"))
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
    from vllm_gaudi import envs as gaudi_envs
    speculative = (["--speculative-config", '{"method":"dspark","num_speculative_tokens":5}']
                   if gaudi_envs.VLLM_HPU_DSV41_DSPARK else [])
    scheduling = "--async-scheduling" if gaudi_envs.VLLM_HPU_DSV41_V2 else "--no-async-scheduling"
    sys.argv = [
        "vllm", "serve", args.model, "--host", args.host, "--port",
        str(args.port), "--dtype", "bfloat16", "--max-model-len", str(args.max_model_len),
        "--generation-config", "vllm", "--tensor-parallel-size", "2", "--pipeline-parallel-size", "2",
        "--max-num-seqs", str(args.max_num_seqs), "--max-num-batched-tokens",
        str(args.max_num_batched_tokens), "--load-format", "dsv41_prepared", "--model-loader-extra-config",
        json.dumps(loader), "--mm-encoder-tp-mode", "data", "--no-enable-prefix-caching", scheduling,
        "--block-size", str(args.block_size), *speculative, *extra
    ]
    from vllm.entrypoints.cli.main import main as serve
    serve()


if __name__ == "__main__":
    main()
