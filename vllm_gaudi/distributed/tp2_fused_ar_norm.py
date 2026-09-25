# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental TP2 all-reduce, residual-add, and RMSNorm fusion."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import threading

import torch
import torch.distributed as dist

from vllm.distributed import get_pp_group, get_tp_group, tensor_model_parallel_all_reduce

from vllm_gaudi.extension.kernels import rms_norm
from vllm_gaudi.extension.runtime import get_config

_RUNTIME_ATTR = "_vllm_gaudi_tp2_fused_ar_norm_runtime"
_PP_RUNTIME_ATTR = "_vllm_gaudi_pp_direct_exchange_runtime"
_BRIDGE_MODULE = "tp2_fused_ar_norm_bridge"
_DIRECT_ALGORITHM_ENV = "VLLM_HPU_TP2_FUSED_AR_NORM_DIRECT_ALGORITHM"
_TENSOR_IDS_ENV = "VLLM_HPU_TP2_FUSED_AR_NORM_TENSOR_IDS"
_OUTPUT_POOL_ATTR = "_vllm_gaudi_tp2_fused_ar_norm_output_pool"
_VALIDATED_WIDTHS_ATTR = "_vllm_gaudi_tp2_fused_ar_norm_validated_widths"
_output_pool_lock = threading.Lock()
# The fused boundary wins through the 16-token decode bucket on Gaudi2, while
# the stock compiled path is faster for the 32-token bucket.
_MAX_FUSED_DECODE_TOKENS = 20
_library = torch.library.Library("vllm_gaudi", "FRAGMENT")
_library.define("tp2_allreduce_residual_rms_norm(Tensor partial, Tensor residual, "
                "Tensor weight, float epsilon) -> (Tensor, Tensor)")
_library.define("tp2_allreduce_residual_rms_norm_out(Tensor partial, Tensor residual, "
                "Tensor weight, Tensor(a!) reduced, Tensor(b!) normalized, "
                "Tensor(c!) residual_out, Tensor(d!) inverse_rms, float epsilon) -> ()")
_library.define("tp2_exchange_peer(Tensor partial) -> Tensor")
_library.define("tp2_exchange_peer_scheduled(Tensor partial, Tensor[] ready_outputs) -> Tensor")
_library.define("pp_exchange_peer(Tensor partial, Tensor(a!) peer) -> Tensor(a!)")
# Functional variant owns its output and neither aliases nor mutates inputs.
# The eager entry above retains the caller-owned in-place wire contract.
_library.define("pp_exchange_peer_graph(Tensor partial) -> Tensor")
_library.define("tp2_allreduce_plain(Tensor partial) -> Tensor")


def _exceeds_fused_decode_token_limit(partial: torch.Tensor, weight: torch.Tensor) -> bool:
    return partial.numel() // weight.numel() > _MAX_FUSED_DECODE_TOKENS


def _verify_prepared_runtime(binary: Path) -> None:
    """The internal GraphExec accessor must match its qualified runtime ABI."""
    import hashlib
    import json

    manifest = binary.with_suffix(".abi.json")
    if not manifest.is_file():
        raise RuntimeError("Rebuild prepared TP2 plans with a runtime ABI manifest")
    metadata = json.loads(manifest.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if (metadata.get("schema") != 1 or metadata.get("torch_version") != torch.__version__
            or metadata.get("binary_sha256") != digest(binary)):
        raise RuntimeError("Prepared TP2 binary/Torch fingerprint differs from its build manifest")
    loaded = {
        Path(row.split(maxsplit=5)[-1]).resolve()
        for row in Path("/proc/self/maps").read_text().splitlines()
        if len(row.split(maxsplit=5)) == 6 and row.split(maxsplit=5)[-1].startswith("/")
    }
    for runtime in metadata["eager_runtime"]:
        path = Path(runtime["path"])
        if path not in loaded or digest(path) != runtime["sha256"]:
            raise RuntimeError("Prepared TP2 GraphExec runtime changed; rebuild and requalify")


def _load_bridge(path: Path):
    loaded = sys.modules.get(_BRIDGE_MODULE)
    if loaded is not None:
        loaded_path = Path(loaded.__file__).resolve()
        if loaded_path != path:
            raise RuntimeError(f"{_BRIDGE_MODULE} is already loaded from {loaded_path}, not {path}")
        return loaded

    spec = importlib.util.spec_from_file_location(_BRIDGE_MODULE, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load TP2 fused bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def initialize_tp2_fused_ar_norm_runtime() -> None:
    """Load the native bridge and bind it to the TP process group."""
    from vllm_gaudi.extension.logger import logger as init_logger
    log = init_logger()
    if getattr(torch, _RUNTIME_ATTR, None) is not None:
        return
    if not dist.is_initialized():
        raise RuntimeError("TP2 fused all-reduce requires initialized torch.distributed")

    tp_group = get_tp_group().device_group
    tp_size = dist.get_world_size(group=tp_group)
    v41 = os.environ.get("VLLM_HPU_DSV41_GRAPH_REPLAY") == "1"
    from vllm.distributed import get_pp_group
    v41_world = v41 and dist.get_world_size() == 4 and get_pp_group().world_size == 2
    if tp_size != 2 or (dist.get_world_size() != 2 and not v41_world):
        raise RuntimeError("TP2 fused all-reduce currently requires a two-rank, TP-only process world")
    direct_algorithm = os.environ.get(_DIRECT_ALGORITHM_ENV, "hccl").strip().lower()
    if direct_algorithm not in ("hccl", "tp2-exchange"):
        raise RuntimeError(f"{_DIRECT_ALGORITHM_ENV} must be 'hccl' or 'tp2-exchange'")
    use_tensor_ids = os.environ.get(_TENSOR_IDS_ENV, "0")
    if use_tensor_ids not in ("0", "1"):
        raise RuntimeError(f"{_TENSOR_IDS_ENV} must be '0' or '1'")
    required_environment = {
        "PT_HPU_LAZY_MODE": "0",
        "PT_HPU_EAGER_PIPELINE_ENABLE": "1",
        "PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE": "1",
    }
    if direct_algorithm == "tp2-exchange":
        # The legacy TP2 tensors fit the original 163840-element ceiling.
        # V4.1's single C6 PP wire is 122976 BF16 elements, so only that
        # explicitly opted-in PP path raises the HCL limit.  Do not change
        # the contract of existing TP-only profiles.
        direct_max_count = ("262144" if os.environ.get("VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE") == "1" else "163840")
        if v41 and os.environ.get("VLLM_HPU_DSV41_BATCH_DECODE") == "1":
            direct_max_count = "327680"
        required_environment.update({
            "HCCL_PRIM_COLLECTIVE_MASK": "0",
            "HCL_TP2_DIRECT_NIC_RS_AR": "0",
            "HCL_TP2_DIRECT_NIC_EXCHANGE": "0",
            "HCL_TP2_DIRECT_INPLACE_STAGING": "0",
            "HCL_TP2_DEDICATED_DIRECT_API": "1",
            "HCL_TP2_DIRECT_MAX_COUNT": direct_max_count,
            "HCL_TP2_PRUNE_SCALEOUT_STREAMS": "1",
            "RUNTIME_SCALE_PATCHING": "0",
        })
    invalid = [
        f"{name}={os.environ.get(name)!r} (expected {expected!r})" for name, expected in required_environment.items()
        if os.environ.get(name) != expected
    ]
    lazy_collectives = os.environ.get("PT_HPU_ENABLE_LAZY_COLLECTIVES")
    if ((direct_algorithm == "tp2-exchange" and lazy_collectives != "0")
            or (direct_algorithm == "hccl" and lazy_collectives not in ("0", "1"))):
        invalid.append(f"PT_HPU_ENABLE_LAZY_COLLECTIVES={lazy_collectives!r} "
                       f"(invalid for {direct_algorithm!r})")
    if invalid:
        raise RuntimeError("TP2 fused all-reduce requires eager current-stream execution: " + ", ".join(invalid))

    if direct_algorithm == "tp2-exchange":
        expected_hcl_value = os.environ.get("VLLM_HPU_TP2_EXPECT_HCL_LIBRARY")
        if not expected_hcl_value:
            raise RuntimeError("VLLM_HPU_TP2_EXPECT_HCL_LIBRARY is required for TP2 exchange")
        expected_hcl = Path(expected_hcl_value).expanduser().resolve()
        loaded_hcl = None
        for mapping in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
            candidate = mapping.rsplit(maxsplit=1)[-1]
            if candidate.endswith("/libhcl.so"):
                loaded_hcl = Path(candidate).resolve()
                break
        if loaded_hcl != expected_hcl:
            raise RuntimeError(f"Loaded libhcl.so={loaded_hcl!s}, expected {expected_hcl!s}")

    bridge_value = os.environ.get("VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE")
    if not bridge_value:
        raise RuntimeError("VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE must point to the native bridge")
    bridge_path = Path(bridge_value).expanduser().resolve()
    if not bridge_path.is_file():
        raise RuntimeError(f"TP2 fused bridge does not exist: {bridge_path}")

    # Ensure ProcessGroupEagerHCCL owns an initialized communicator before the
    # bridge requests its current-stream handle.
    probe = torch.ones(128, dtype=torch.bfloat16, device="hpu")
    log.info("TP2 prepared runtime: initializing communicator")
    dist.all_reduce(probe, group=tp_group)
    torch.hpu.synchronize()
    if not torch.equal(probe.cpu(), torch.full((128, ), 2, dtype=torch.bfloat16, device="cpu")):
        raise RuntimeError("TP2 HCCL process-group initialization failed")

    log.info("TP2 prepared runtime: loading native adapter")
    bridge = _load_bridge(bridge_path)
    bridge.set_use_tensor_ids(use_tensor_ids == "1")
    backend = tp_group._get_backend(torch.device("hpu"))
    communicator_id = bridge.communicator_id(backend)
    setattr(torch, _RUNTIME_ATTR, (bridge, backend, communicator_id))
    from vllm_gaudi import envs

    if envs.VLLM_HPU_DSV41_PP_DIRECT_EXCHANGE:
        pp_group = get_pp_group().device_group
        if not v41_world or dist.get_world_size(group=pp_group) != 2:
            raise RuntimeError("V4.1 PP direct exchange requires TP2×PP2")
        if not hasattr(bridge, "pp_exchange_peer_current_stream"):
            raise RuntimeError("Rebuild the V4.1 bridge with the PP direct-exchange entry")
        # Initialise the PP communicator once, before any request state is
        # written.  The direct steady path does not call a generic collective
        # or wait on a host Work object.
        pp_probe = torch.ones(128, dtype=torch.bfloat16, device="hpu")
        dist.all_reduce(pp_probe, group=pp_group)
        torch.hpu.synchronize()
        if not torch.equal(pp_probe.cpu(), torch.full((128, ), 2, dtype=torch.bfloat16, device="cpu")):
            raise RuntimeError("V4.1 PP HCCL process-group initialization failed")
        pp_backend = pp_group._get_backend(torch.device("hpu"))
        pp_communicator_id = bridge.communicator_id(pp_backend)
        setattr(torch, _PP_RUNTIME_ATTR, (bridge, pp_backend, pp_group, pp_communicator_id))

    if envs.VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT:
        if torch.hpu.get_device_name().upper().replace(" ", "") != "GAUDI2":
            raise RuntimeError("Native TP2 quant is currently limited to Gaudi2")
        if not hasattr(torch.ops.custom_op, "tp2_dynamic_quant"):
            raise RuntimeError("Rebuild the TP2 bridge with the qualified dynamic quant kernel adapter")

    if envs.VLLM_HPU_TP2_PREPARED_COMM:
        if not hasattr(bridge, "set_prepared_communication"):
            raise RuntimeError("Rebuild the TP2 bridge with prepared communication descriptors")
        bridge.set_prepared_communication(True)

    if envs.VLLM_HPU_GDN_DIRECT_STATE_UPDATE:
        if torch.hpu.get_device_name().upper().replace(" ", "") != "GAUDI2":
            raise RuntimeError("Direct GDN state update is currently limited to Gaudi2")
        if not hasattr(torch.ops.custom_op, "gdn_state_update_out"):
            raise RuntimeError("Rebuild the TP2 bridge with the direct GDN state epilogue")
        from vllm_gaudi.ops.gdn_state_update import register_gdn_state_update_pass

        register_gdn_state_update_pass()
    if envs.VLLM_HPU_TP2_STATIC_GROUP_PLAN:
        log.info("TP2 prepared runtime: checking runtime fingerprints")
        _verify_prepared_runtime(bridge_path)
        if not envs.VLLM_HPU_TP2_PREPARED_COMM or not (envs.VLLM_HPU_GDN_DIRECT_STATE_UPDATE
                                                       or envs.VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH
                                                       or envs.VLLM_HPU_DSV41_GRAPH_REPLAY):
            raise RuntimeError("Prepared TP2 groups require direct state update and prepared communication")
        from vllm_gaudi.ops.tp2_prepared_plan import register_tp2_prepared_group_pass

        register_tp2_prepared_group_pass()
    if (envs.VLLM_HPU_NATIVE_DECODE_GRAPH or envs.VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH
            or envs.VLLM_HPU_DSV41_GRAPH_REPLAY):
        if ((envs.VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH or envs.VLLM_HPU_DSV41_GRAPH_REPLAY)
                and torch.hpu.get_device_name().upper().replace(" ", "") != "GAUDI2"):
            raise RuntimeError("V4 native decode currently requires Gaudi2")
        required = ("NativeDecodeGraph", "native_decode_graph_available")
        if envs.VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH or envs.VLLM_HPU_DSV41_GRAPH_REPLAY:
            required += ("record_native_completion", "copy_sampled_tokens_to_host")
        if envs.VLLM_HPU_DSV41_DEVICE_COMMIT:
            required += ("copy_integer_record_to_host", )
        if envs.VLLM_HPU_DSV41_NATIVE_PP_COPY:
            required += ("copy_c1_pipeline_tensors", )
        if envs.VLLM_HPU_DSV41_V2_DEVICE_ENGRAM:
            required += ("DeviceEngramProducer", )
        if not all(hasattr(bridge, name) for name in required) or not bridge.native_decode_graph_available():
            raise RuntimeError(
                "VLLM_HPU_NATIVE_DECODE_GRAPH requires the version-locked Synapse and HCL native replay APIs; "
                "no fallback was selected")
    log.info("TP2 prepared runtime: initialization complete")


def _resolve_runtime():
    runtime = getattr(torch, _RUNTIME_ATTR, None)
    if runtime is None:
        raise RuntimeError("TP2 fused all-reduce runtime is not initialized")
    return runtime


def _allocate_outputs(partial: torch.Tensor, weight: torch.Tensor):
    reduced = torch.empty_like(partial)
    normalized = torch.empty_like(partial)
    residual_out = torch.empty_like(partial)
    inverse_rms = torch.empty((*partial.shape[:-1], 1), dtype=torch.float32, device=partial.device)
    return reduced, normalized, residual_out, inverse_rms


def _output_pool_slot_count() -> int | None:
    if os.environ.get("VLLM_HPU_TP2_FUSED_AR_NORM_OUTPUT_POOL", "0") != "1":
        return None
    try:
        slot_count = int(os.environ.get("VLLM_HPU_TP2_FUSED_AR_NORM_OUTPUT_POOL_SLOTS", "256"))
    except ValueError as error:
        raise RuntimeError("VLLM_HPU_TP2_FUSED_AR_NORM_OUTPUT_POOL_SLOTS must be an integer") from error
    if not 2 <= slot_count <= 256:
        raise RuntimeError("VLLM_HPU_TP2_FUSED_AR_NORM_OUTPUT_POOL_SLOTS must be between 2 and 256")
    return slot_count


def _pooled_output(key, slot_count: int, allocate):
    with _output_pool_lock:
        pool = getattr(torch, _OUTPUT_POOL_ATTR, None)
        if pool is None:
            pool = {}
            setattr(torch, _OUTPUT_POOL_ATTR, pool)
        bucket = pool.setdefault(key, {"cursor": 0, "slots": []})
        index = bucket["cursor"] % slot_count
        bucket["cursor"] += 1
        # Prepare the full existing pool once. No collective or warm request
        # is needed to pay these allocations before steady decode begins.
        prepare_all = os.environ.get("VLLM_HPU_TP2_PREPARED_COMM", "0") == "1"
        required = slot_count if prepare_all and key[0] == "fused" and key[2] == (1, 5120) else index + 1
        while len(bucket["slots"]) < required:
            bucket["slots"].append(allocate())
        return bucket["slots"][index]


def _direct_outputs(partial: torch.Tensor, weight: torch.Tensor):
    slot_count = _output_pool_slot_count()
    if slot_count is None:
        return _allocate_outputs(partial, weight)
    key = ("fused", str(partial.device), tuple(partial.shape), partial.dtype, weight.numel(), slot_count)
    return _pooled_output(key, slot_count, lambda: _allocate_outputs(partial, weight))


def _native_probe_valid(actual, expected, launches: int) -> bool:
    """Fail closed on missing execution, nonfinite values or incorrect outputs."""
    if launches != 1 or len(actual) != 2 or len(expected) != 2:
        return False
    for output, reference in zip(actual, expected):
        if output.shape != reference.shape or output.dtype != reference.dtype:
            return False
        if not bool(torch.isfinite(output).all() and torch.isfinite(reference).all()):
            return False
    return (torch.equal(actual[1], expected[1])
            and torch.allclose(actual[0].float(), expected[0].float(), atol=0.015625, rtol=0.015625))


def validate_tp2_gemma_fusion_runtime(hidden_size: int) -> None:
    """Check the loaded Bridge/extension pair before enabling Gemma fusion.

    The extension depends on a Bridge collective-output metadata ABI. Merely
    importing its schema does not establish that the loaded backend executes it.
    This startup-only check also exercises graph-produced effective weights and
    changing allocations. It never adds synchronization to the decode path.
    """
    validated_widths = getattr(torch, _VALIDATED_WIDTHS_ATTR, set())
    if hidden_size in validated_widths:
        return

    bridge, _backend, _communicator_id = _resolve_runtime()
    if not callable(getattr(bridge, "collective_launch_count", None)):
        raise RuntimeError("Rebuild the TP2 extension with native execution counters")
    tp_group = get_tp_group().device_group
    rank = dist.get_rank(group=tp_group)

    def native(partial, residual, raw_weight):
        return torch.ops.vllm_gaudi.tp2_allreduce_residual_rms_norm(
            partial,
            residual,
            raw_weight + 1.0,
            1e-6,
        )

    def exchange_native(partial, residual, raw_weight):
        return _tp2_exchange_residual_rms_norm(partial, residual, raw_weight + 1.0, 1e-6)

    weight_index = torch.arange(hidden_size)
    # Compile both the normal one-token decode recipe and the two-token target
    # verification recipe used by one-token speculative decoding.  Synapse
    # graph compilation is not safe to defer until the custom op is invoked
    # from an eager-pipeline launch thread while the full model is resident.
    for rows in (1, 2):
        candidate = native if rows == 1 else exchange_native
        compiled = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
        index = torch.arange(rows * hidden_size)
        shape = (rows, hidden_size)
        for replay in range(3):
            p0 = ((((index * 17 + replay * 7) % 127) - 63).float() / 64).to(torch.bfloat16)
            p1 = ((((index * 17 + 29 + replay * 7) % 127) - 63).float() / 64).to(torch.bfloat16)
            residual = ((((index * 11 + replay * 5) % 113) - 56).float() / 32).to(torch.bfloat16)
            weight = ((((weight_index * 3 + replay * 5) % 31) - 15).float() / 128).to(torch.bfloat16)
            expected_residual = (residual + (p0 + p1)).reshape(shape)
            rf = expected_residual.float()
            expected_norm = (rf * torch.rsqrt(rf.square().mean(-1, keepdim=True) + 1e-6) * (weight + 1.0).float()).to(
                torch.bfloat16)
            inputs = ((p0 if rank == 0 else p1).reshape(shape).to("hpu"), residual.reshape(shape).to("hpu"),
                      weight.to("hpu"))
            if replay == 0:
                # Populate the native recipe cache outside torch.compile.  If
                # the C++ custom op first calls synGraphCompile while the HPU
                # backend is compiling the shape-specialized outer graph,
                # Synapse can fail the nested compilation with synFail.
                before = bridge.collective_launch_count()
                eager_actual = tuple(output.cpu() for output in candidate(*inputs))
                eager_valid = _native_probe_valid(eager_actual, (expected_norm, expected_residual),
                                                  bridge.collective_launch_count() - before)
                eager_failure = torch.tensor([int(not eager_valid)], dtype=torch.int32, device="hpu")
                dist.all_reduce(eager_failure, group=tp_group)
                if eager_failure.cpu().item():
                    raise RuntimeError(
                        f"TP2 Gemma native fusion eager validation failed for {rows} decode token(s); use a "
                        "matching patched Bridge/extension or disable VLLM_HPU_TP2_GEMMA_FUSED_AR_NORM. No "
                        "model reductions were changed.")
            before = bridge.collective_launch_count()
            actual = tuple(output.cpu() for output in compiled(*inputs))
            valid = _native_probe_valid(actual, (expected_norm, expected_residual),
                                        bridge.collective_launch_count() - before)
            failure = torch.tensor([int(not valid)], dtype=torch.int32, device="hpu")
            dist.all_reduce(failure, group=tp_group)
            if failure.cpu().item():
                raise RuntimeError(
                    f"TP2 Gemma native fusion validation failed for {rows} decode token(s); use a matching "
                    "patched Bridge/extension or disable VLLM_HPU_TP2_GEMMA_FUSED_AR_NORM. No model reductions "
                    "were changed.")
    setattr(torch, _VALIDATED_WIDTHS_ATTR, {*validated_widths, hidden_size})


def _tp2_fused_ar_norm_impl(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bridge, backend = _resolve_runtime()[:2]
    reduced, normalized, residual_out, inverse_rms = _direct_outputs(partial, weight)
    return bridge.allreduce_residual_rms_norm_current_stream(
        backend,
        partial,
        residual,
        weight,
        reduced,
        normalized,
        residual_out,
        inverse_rms,
        epsilon,
    )


def _tp2_fused_ar_norm_fake(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del residual, weight, epsilon
    return torch.empty_like(partial), torch.empty_like(partial)


def _tp2_fused_ar_norm_out_impl(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    reduced: torch.Tensor,
    normalized: torch.Tensor,
    residual_out: torch.Tensor,
    inverse_rms: torch.Tensor,
    epsilon: float,
) -> None:
    bridge, backend = _resolve_runtime()[:2]
    bridge.allreduce_residual_rms_norm_current_stream(
        backend,
        partial,
        residual,
        weight,
        reduced,
        normalized,
        residual_out,
        inverse_rms,
        epsilon,
    )


def _tp2_fused_ar_norm_out_fake(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    reduced: torch.Tensor,
    normalized: torch.Tensor,
    residual_out: torch.Tensor,
    inverse_rms: torch.Tensor,
    epsilon: float,
) -> None:
    del partial, residual, weight, reduced, normalized, residual_out, inverse_rms, epsilon


_library.impl(
    "tp2_allreduce_residual_rms_norm",
    _tp2_fused_ar_norm_impl,
    dispatch_key="HPU",
)
_library._register_fake(
    "tp2_allreduce_residual_rms_norm",
    _tp2_fused_ar_norm_fake,
)
_library.impl(
    "tp2_allreduce_residual_rms_norm_out",
    _tp2_fused_ar_norm_out_impl,
    dispatch_key="HPU",
)
_library._register_fake(
    "tp2_allreduce_residual_rms_norm_out",
    _tp2_fused_ar_norm_out_fake,
)


def _tp2_exchange_peer_impl(partial: torch.Tensor) -> torch.Tensor:
    bridge, backend = _resolve_runtime()[:2]
    slot_count = _output_pool_slot_count()
    if slot_count is None:
        peer = torch.empty_like(partial)
    else:
        key = ("exchange-peer", str(partial.device), tuple(partial.shape), partial.dtype, slot_count)
        peer = _pooled_output(key, slot_count, lambda: torch.empty_like(partial))
    return bridge.tp2_exchange_peer_current_stream(backend, partial, peer)


def _tp2_exchange_peer_fake(partial: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(partial)


_library.impl("tp2_exchange_peer", _tp2_exchange_peer_impl, dispatch_key="HPU")
_library._register_fake("tp2_exchange_peer", _tp2_exchange_peer_fake)


def _tp2_exchange_peer_scheduled_impl(partial, ready_outputs):
    # These outputs belong to the preceding compute recipe, but are consumed
    # after the collective. They are not part of the communication payload.
    del ready_outputs
    return _tp2_exchange_peer_impl(partial)


def _tp2_exchange_peer_scheduled_fake(partial, ready_outputs):
    del ready_outputs
    return torch.empty_like(partial)


_library.impl("tp2_exchange_peer_scheduled", _tp2_exchange_peer_scheduled_impl, dispatch_key="HPU")
_library._register_fake("tp2_exchange_peer_scheduled", _tp2_exchange_peer_scheduled_fake)


def _pp_exchange_peer_impl(partial: torch.Tensor, peer: torch.Tensor) -> torch.Tensor:
    """Exchange a fixed BF16 PP payload on the PP communicator's stream.

    The existing TP2 bridge is intentionally reused, but the backend is
    resolved from ``get_pp_group``.  This prevents a PP boundary from using
    the TP communicator while retaining the direct HCL path and its ABI
    validation.  ``peer`` is caller-owned so graph/replay can keep its address
    stable and no per-token allocation is introduced.
    """
    runtime = getattr(torch, _PP_RUNTIME_ATTR, None)
    if runtime is None:
        raise RuntimeError("V4.1 PP direct exchange runtime was not prepared")
    bridge, backend, pp_group, _ = runtime
    if get_pp_group().device_group is not pp_group:
        raise RuntimeError("V4.1 PP communicator changed after direct-exchange preparation")
    if partial.dtype != torch.bfloat16 or peer.dtype != torch.bfloat16:
        raise RuntimeError("V4.1 PP direct exchange requires BF16 payloads")
    if partial.device.type != "hpu" or peer.device != partial.device:
        raise RuntimeError("V4.1 PP direct exchange requires matching HPU payloads")
    if partial.shape != peer.shape or not partial.is_contiguous() or not peer.is_contiguous():
        raise RuntimeError("V4.1 PP direct exchange payloads must be equal contiguous tensors")
    return bridge.pp_exchange_peer_current_stream(backend, partial, peer)


def _pp_exchange_peer_fake(partial: torch.Tensor, peer: torch.Tensor) -> torch.Tensor:
    return peer


_library.impl("pp_exchange_peer", _pp_exchange_peer_impl, dispatch_key="HPU")
_library._register_fake("pp_exchange_peer", _pp_exchange_peer_fake)


def _pp_exchange_peer_graph_impl(partial: torch.Tensor) -> torch.Tensor:
    peer = torch.empty_like(partial)
    return _pp_exchange_peer_impl(partial, peer)


def _pp_exchange_peer_graph_fake(partial: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(partial)


_library.impl("pp_exchange_peer_graph", _pp_exchange_peer_graph_impl, dispatch_key="HPU")
_library._register_fake("pp_exchange_peer_graph", _pp_exchange_peer_graph_fake)


def _tp2_allreduce_plain_impl(partial: torch.Tensor) -> torch.Tensor:
    bridge, backend, _ = _resolve_runtime()
    if partial.shape != (1, 4096) or partial.dtype != torch.bfloat16 or not partial.is_contiguous():
        raise RuntimeError("V4 native AllReduce requires contiguous BF16 [1, 4096]")
    reduced = torch.empty_like(partial)
    return bridge.tp2_allreduce_plain_current_stream(backend, partial, reduced)


_library.impl("tp2_allreduce_plain", _tp2_allreduce_plain_impl, dispatch_key="HPU")
_library._register_fake("tp2_allreduce_plain", _tp2_exchange_peer_fake)


def _tp2_exchange_residual_rms_norm(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    peer = torch.ops.vllm_gaudi.tp2_exchange_peer(partial)
    reduced = partial + peer
    residual_out = residual + reduced.reshape(residual.shape)
    fused_rms_norm = rms_norm()
    if fused_rms_norm is None:
        variance = residual_out.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normalized = residual_out * torch.rsqrt(variance + epsilon).to(residual_out.dtype)
        return normalized * weight, residual_out
    flat = residual_out.reshape(-1, weight.numel()).unsqueeze(0)
    normalized = fused_rms_norm.apply(flat, weight, epsilon).reshape(partial.shape)
    return normalized, residual_out


def _rejection_reason(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    is_prompt: bool,
    max_bytes: int,
) -> str | None:
    if is_prompt:
        return "prefill"
    if partial.device.type != "hpu":
        return "non-HPU input"
    if partial.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        return "activation dtype"
    if weight.dtype != torch.bfloat16:
        return "weight dtype"
    if partial.dim() < 2 or partial.numel() == 0:
        return "activation shape"
    if partial.shape != residual.shape:
        return "residual shape"
    if weight.dim() != 1 or partial.shape[-1] != weight.numel():
        return "weight shape"
    if _exceeds_fused_decode_token_limit(partial, weight):
        return "decode token count"
    if not partial.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        return "non-contiguous tensor"
    if max_bytes <= 0 or partial.numel() * partial.element_size() > max_bytes:
        return "payload size"
    return None


def _fallback(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced = tensor_model_parallel_all_reduce(partial)
    residual_out = residual + reduced.reshape(residual.shape)
    fused_rms_norm = rms_norm()
    if fused_rms_norm is None:
        variance = residual_out.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normalized = residual_out * torch.rsqrt(variance + epsilon).to(residual_out.dtype)
        return normalized * weight, residual_out
    normalized = fused_rms_norm.apply(residual_out, weight, epsilon)
    return normalized.reshape(partial.shape), residual_out


def tp2_allreduce_residual_rms_norm(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    *,
    is_prompt: bool,
    allow_fused: bool = True,
    prefer_direct: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the fused decode path, or preserve stock HCCL semantics."""
    from vllm_gaudi import envs

    compiled_consumer = envs.VLLM_HPU_TP2_COMPILED_CONSUMER_NORM and not is_prompt
    if compiled_consumer and (not allow_fused or not envs.VLLM_HPU_TP2_NATIVE_JOINT_PLAN):
        raise RuntimeError("Compiled TP2 consumer norm requires fused decode and the native joint plan")
    if not allow_fused:
        return _fallback(partial, residual, weight, epsilon)
    max_bytes = get_config().tp2_fused_ar_norm_max_bytes
    reason = _rejection_reason(
        partial,
        residual,
        weight,
        is_prompt=is_prompt,
        max_bytes=max_bytes,
    )
    if reason is not None:
        if compiled_consumer:
            raise RuntimeError(f"Compiled TP2 consumer norm rejected before exchange: {reason}")
        return _fallback(partial, residual, weight, epsilon)
    if compiled_consumer:
        if partial.shape != (1, 5120) or rms_norm() is None:
            raise RuntimeError("Compiled TP2 consumer norm requires C1/5120 and the native HPU RMSNorm")
        from vllm_gaudi.ops.tp2_consumer_norm import tp2_consumer_norm

        peer = torch.ops.vllm_gaudi.tp2_exchange_peer(partial)
        return tp2_consumer_norm(partial, peer, residual, weight, epsilon, rms_norm().apply)
    if prefer_direct:
        if partial.numel() != weight.numel():
            return _tp2_exchange_residual_rms_norm(partial, residual, weight, epsilon)
        return torch.ops.vllm_gaudi.tp2_allreduce_residual_rms_norm(
            partial,
            residual,
            weight,
            epsilon,
        )
    communicator_id = _resolve_runtime()[2]
    packed, _inverse_rms = torch.ops.hccl.tp2_allreduce_residual_rms_norm(
        partial,
        residual,
        weight,
        epsilon,
        communicator_id,
    )
    return packed[0], packed[1]
