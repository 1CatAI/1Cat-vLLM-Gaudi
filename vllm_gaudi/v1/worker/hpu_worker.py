# SPDX-License-Identifier: Apache-2.0
"""A GPU worker class."""
import contextlib
import gc
import math
import os
import queue
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, cast

import torch
import torch.distributed
import torch.nn as nn
import habana_frameworks.torch as htorch
from vllm.tasks import SupportedTask
from vllm_gaudi.extension.debug import init_debug_logger
from vllm_gaudi.extension.defragmentation import OnlineDefragmenter
from vllm_gaudi.extension.profiler import (HabanaMemoryProfiler, format_bytes, setup_profiler)
from vllm_gaudi.extension.runtime import get_config

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (ensure_model_parallel_initialized, init_distributed_environment)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.utils.torch_utils import (STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size, set_random_seed)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheSpec, MambaSpec)
from vllm.v1.outputs import (DraftTokenIds, AsyncModelRunnerOutput, ModelRunnerOutput)
from vllm.v1.worker.utils import bind_kv_cache
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_gaudi.extension.bucketing.common import HPUBucketingManager
from vllm_gaudi.utils import is_fake_hpu
from vllm_gaudi.v1.worker.hpu_model_runner import (HPUModelRunner, _GDN_MAMBA_TYPES, _rebind_moe_expert_weights)
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()
_QUANT_CONFIG_UNCHANGED = object()

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import GrammarOutput, SchedulerOutput


def setup_step_profiler(steps):
    if steps is None:
        return None
    step_start, step_end = steps
    active = step_end - step_start + 1
    return setup_profiler(warmup=0, active=active)


class HPUWorker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):

        # TODO: use WorkerBase.__init__(self, vllm_config=vllm_config)
        self._apply_vllm_config(vllm_config)

        self.local_rank = local_rank
        self.rank = rank
        self.parallel_config.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]

        self.gc_track_recompiles = get_config().track_graph_compilation and not get_config().high_level_profiler_enabled
        self.step = 0
        self.profile_steps = get_config().VLLM_PROFILE_STEPS
        self.step_profiler = setup_step_profiler(self.profile_steps)
        self.step_debug = init_debug_logger('steps')

        self.model_sleeping = False
        self.model_runner: HPUModelRunner | None = None
        self.kv_cache_sleeping = False
        self.kv_cache_config = None
        self._model_runner_stash: dict[tuple[object, ...], HPUModelRunner] = {}
        self._model_runner_state_stash: dict[tuple[object, ...], dict[str, Any]] = {}

    def _apply_vllm_config(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

    def _runner_stash_key(self, vllm_config: VllmConfig) -> tuple[object, ...]:
        compile_cfg = vllm_config.compilation_config
        return (
            vllm_config.model_config.model,
            vllm_config.model_config.dtype,
            vllm_config.model_config.enforce_eager,
            vllm_config.model_config.max_model_len,
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.cache_config.block_size,
            tuple(getattr(compile_cfg, "compile_ranges_endpoints", ()) or ()),
            tuple(getattr(compile_cfg, "compile_sizes", ()) or ()),
        )

    def init_profiler(self):
        """Record profiler configuration without touching Kineto.

        Constructing ``torch.profiler.profile`` before HPU graph warmup changes
        Dynamo/Synapse graph partitioning.  In particular, mutable custom ops
        can then fail graph compilation even though the same graph compiles and
        runs normally without profiling.  Keep profiler construction lazy so
        graph capture finishes before the /start_profile request arrives.
        """
        profiler_config = self.vllm_config.profiler_config
        legacy_profiler_dir = os.getenv('VLLM_TORCH_PROFILER_DIR')
        torch_profiler_dir = (legacy_profiler_dir
                              or (profiler_config.torch_profiler_dir if profiler_config.profiler == 'torch' else None))
        self.profiler_summary_only = getattr(profiler_config, "torch_profiler_summary_only", False)
        self.torch_profiler_dir = torch_profiler_dir
        self.profiler = None
        self._profiler_running = False
        if torch_profiler_dir:
            if legacy_profiler_dir:
                logger.warning("VLLM_TORCH_PROFILER_DIR is deprecated!")
            logger.info("Profiling enabled. Traces will be saved to: %s", torch_profiler_dir)
            logger.info("Profiler summary-only mode: %s", self.profiler_summary_only)

    def _create_profiler(self) -> None:
        if self.profiler is not None:
            return
        if not self.torch_profiler_dir:
            raise RuntimeError("Profiler is not enabled.")

        profiler_config = self.vllm_config.profiler_config
        if os.getenv('VLLM_PROFILER_ENABLED') == 'full':
            fn = self.model_runner.profiler.full_trace_handler  # type: ignore[union-attr]
            with_stack = False
        else:
            fn = torch.profiler.tensorboard_trace_handler
            with_stack = profiler_config.torch_profiler_with_stack
        trace_handler = (None if self.profiler_summary_only else fn(
            self.torch_profiler_dir,
            use_gzip=profiler_config.torch_profiler_use_gzip,
        ))
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.HPU,
            ],
            record_shapes=profiler_config.torch_profiler_record_shapes,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_stack=with_stack,
            with_flops=profiler_config.torch_profiler_with_flops,
            on_trace_ready=trace_handler,
        )

    def start_profile(self):
        if self._profiler_running:
            raise RuntimeError("Profiler is already running.")
        self._create_profiler()
        high_level_profiler = self.model_runner.profiler  # type: ignore[union-attr]
        with high_level_profiler.record_event('internal', 'start_profiler'):
            # Clean up the queue
            while True:
                try:
                    high_level_profiler.profiling_trace_events.get_nowait()
                except queue.Empty:
                    break
            self.profiler.start()
            # Keep ownership even if command recapture or audit fails below.
            self._profiler_running = True
            self._refresh_native_profiler_commands()
            from vllm_gaudi.ops.tp2_runtime_profile import verify_loaded_profile_libraries
            logger.info("Profiler runtime libraries: %s", verify_loaded_profile_libraries())
            if hasattr(self.model_runner, "trace_enabled"):
                self.model_runner.trace_enabled = True
                host = self.model_runner.model.engram_host
                if host is not None:
                    host.set_profiling(True)
            # Graph-local counters belong to the newly created generation.
            # Sampling before retirement made start/stop subtraction invalid.
            self._write_native_decoder_stats("profile-start")

    def stop_profile(self, *, refresh_native=True):
        if self.profiler is None or not self._profiler_running:
            raise RuntimeError("Profiler is not running.")
        self.profiler.stop()
        self._profiler_running = False
        self._write_profiler_summary()
        self._write_native_decoder_stats("profile-stop")
        if hasattr(self.model_runner, "trace_enabled"):
            self.model_runner.trace_enabled = False
            host = self.model_runner.model.engram_host
            if host is not None and host.profile_records is not None:
                from pathlib import Path
                host.export_profile(Path(self.torch_profiler_dir) / f"rank{self.rank}-engram-profile.json")
                host.set_profiling(False)
        if refresh_native:
            self._refresh_native_profiler_commands()
        # Every acquisition owns its profiler/trace sink until export ends.
        self.profiler = None

    @staticmethod
    def _refresh_native_profiler_commands():
        from vllm_gaudi import envs as gaudi_envs
        if gaudi_envs.VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH or gaudi_envs.VLLM_HPU_DSV41_GRAPH_REPLAY:
            from vllm_gaudi.ops.tp2_prepared_plan import recapture_native_decoder_programs
            recapture_native_decoder_programs()

    def _write_native_decoder_stats(self, phase):
        from vllm_gaudi import envs as gaudi_envs
        if not (gaudi_envs.VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH or gaudi_envs.VLLM_HPU_DSV41_GRAPH_REPLAY):
            return
        from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats
        import json
        from pathlib import Path
        torch.hpu.synchronize()
        stats = prepared_group_stats()
        from vllm_gaudi.ops.tp2_runtime_profile import verify_loaded_profile_libraries
        stats["profile_libraries"] = verify_loaded_profile_libraries()
        stats.update(rank=self.rank, phase=phase)
        if self.model_runner is not None:
            stats["capture_snapshot_bytes"] = sum(
                decoder.capture_bytes for module in self.model_runner.get_model().modules()
                if (decoder := getattr(module, "_hpu_native_decoder", None)) is not None)
            stats["allocated_bytes"] = torch.hpu.memory_allocated()
            stats["peak_allocated_bytes"] = torch.hpu.max_memory_allocated()
            if gaudi_envs.VLLM_HPU_DSV41_PREPARED_SHARDS:
                owner = self.model_runner.model.program.replay_owner
                stats["capture_snapshot_bytes"] = sum(variant.capture_bytes for variant in owner.variants.values())
                stats["v41"] = self.model_runner.audit
                stats["state_bytes"] = self.model_runner.state.allocated_bytes
                stats["pp"] = {key: getattr(self.model_runner.pp, key) for key in ("sends", "receives", "commits")}
                if gaudi_envs.VLLM_HPU_DSV41_NATIVE_PP_COPY:
                    from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime
                    bridge, _, _ = _resolve_runtime()
                    stats["pp"]["native_dma_batches_tensors_bytes"] = (bridge.gdn_state_dma_counts())
                host = self.model_runner.model.engram_host
                stats["engram"] = None if host is None else host.audit
                stats["engram_residency"] = None if host is None else host.residency()
        logger.info("Native decoder statistics: %s", json.dumps(stats))
        if self.torch_profiler_dir:
            destination = Path(self.torch_profiler_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"rank{self.rank}-native-{phase}.json").write_text(json.dumps(stats, indent=2))

    def _write_profiler_summary(self) -> None:
        if not self.profiler_summary_only:
            return
        averages = self.profiler.key_averages()
        tables = []
        for sort_key in (
                'self_hpu_time_total',
                'self_device_time_total',
                'self_cpu_time_total',
        ):
            try:
                tables.append(f"Sorted by {sort_key}\n" + averages.table(sort_by=sort_key, row_limit=200))
            except (AttributeError, KeyError, RuntimeError) as exc:
                tables.append(f"Unable to sort by {sort_key}: {exc}")
        summary_path = os.path.join(
            self.torch_profiler_dir,
            f"operator-summary-rank{self.rank}.txt",
        )
        os.makedirs(self.torch_profiler_dir, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as summary_file:
            summary_file.write('\n\n'.join(tables))
        logger.info("Profiler operator summary written to %s", summary_path)

    def init_device(self):
        # HCCL is imported before the multiprocessing worker receives its
        # local rank, so its import-time device selection cannot honor
        # HABANA_VISIBLE_MODULES. Bind explicitly before the first HPU
        # allocation to keep each worker on its assigned visible module.
        device_index = self.local_rank if self.local_rank >= 0 else 0
        from vllm_gaudi.ops.deepseek_v41_config import is_v41
        if is_v41(self.vllm_config):
            # Spawn reimports Python modules; the parent's namespace-package
            # search path is not inherited with the environment/config pickle.
            from vllm_gaudi.entrypoints.deepseek_v41 import prepare_native_libraries
            prepare_native_libraries()
            modules = os.environ["HABANA_VISIBLE_MODULES"].split(",")
            if len(modules) != 4 or device_index >= len(modules):
                raise RuntimeError("V4.1 requires four explicitly assigned HPU modules")
            os.environ["HLS_MODULE_ID"] = modules[device_index]
            if "{rank}" in os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", ""):
                os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].replace(
                    "{rank}", str(self.rank))
            if os.environ.get("GRAPH_VISUALIZATION") == "1":
                from pathlib import Path
                graph_dir = Path(os.environ["GRAPH_VISUALIZATION_DIR"]) / f"rank{self.rank}"
                graph_dir.mkdir(parents=True, exist_ok=True)
                os.environ["GRAPH_VISUALIZATION_DIR"] = str(graph_dir)
                os.environ["PT_HPU_GRAPH_DUMP_PREFIX"] = str(graph_dir)
        torch.hpu.set_device(device_index)
        self.device = torch.device("hpu")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank, self.distributed_init_method, self.local_rank)
        from vllm_gaudi.ops.tp2_runtime_profile import verify_loaded_profile_libraries
        logger.info("Worker runtime companion libraries: %s", verify_loaded_profile_libraries())
        # Set random seed.
        set_random_seed(self.model_config.seed)
        num_ubatches = 2 if self.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)
        with set_current_vllm_config(self.vllm_config):
            self.model_runner = self._create_model_runner()
        self.init_profiler()

    def _create_model_runner(self):
        from vllm_gaudi.ops.deepseek_v41_config import is_v41
        runner_class = HPUModelRunner
        if is_v41(self.vllm_config):
            from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner
            runner_class = V41ModelRunner
        return runner_class(vllm_config=self.vllm_config, is_driver_worker=self.is_driver_worker)

    def shutdown(self):
        from vllm_gaudi import envs as gaudi_envs

        def phase(name):
            if gaudi_envs.VLLM_HPU_DSV41_GRAPH_REPLAY:
                logger.info("V4.1 worker shutdown: %s", name)

        phase("begin")
        if getattr(self, "_profiler_running", False):
            phase("stop active profiler")
            # Export the live sink before retiring recipes/communicators. A
            # failed start_profile can leave this owner active as well.
            self.stop_profile(refresh_native=False)
            phase("profiler stopped")
        if gaudi_envs.VLLM_HPU_TP2_STATIC_GROUP_PLAN:
            self._write_native_decoder_stats("shutdown")
            from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans

            phase("retire native programs")
            shutdown_prepared_group_plans()
            phase("native programs retired")
        self._model_runner_stash.clear()
        self._model_runner_state_stash.clear()
        if self.model_runner is not None:
            phase("close runner")
            getattr(self.model_runner, 'shutdown_inc', lambda: None)()
        phase("complete")

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()  # type: ignore[union-attr]

    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()  # type: ignore[union-attr]

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()  # type: ignore[union-attr]

    def unload_model(self) -> dict[str, float | None]:
        """Stash the current HPUModelRunner (weights already on CPU from sleep)
        so its compiled ModuleCacher graph dict survives across model switches.
        On a subsequent load_model() for the same model the runner is restored
        directly, skipping warmup_graphs entirely.
        """
        with HabanaMemoryProfiler() as m:
            if self.model_runner is not None:
                runner_config = getattr(self.model_runner, "vllm_config", self.vllm_config)
                stash_key = self._runner_stash_key(runner_config)
                logger.info("[HPUWorker] Stashing runner for model: %s", runner_config.model_config.model)
                self._model_runner_stash[stash_key] = self.model_runner
                self._model_runner_state_stash[stash_key] = {
                    "vllm_config": runner_config,
                    "model_sleeping": self.model_sleeping,
                    "kv_cache_sleeping": self.kv_cache_sleeping,
                    "kv_cache_config": self.kv_cache_config,
                }
                self.model_runner = None
                HPUBucketingManager.deactivate()
            # Preserve previous KV cache metadata in stash for rollback.
            self.model_sleeping = False
            self.kv_cache_sleeping = False
            gc.collect()
            with contextlib.suppress(Exception):
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            with contextlib.suppress(Exception):
                torch.hpu.synchronize()
        msg = f"Stashing model runner took {m.get_summary_string()}"
        logger.info(msg)

        memory_after_stash_mb = self.get_hpu_used_memory_mb()

        return {
            "stash_memory_after_mb": memory_after_stash_mb,
        }

    def load_model(
        self,
        vllm_config: Optional[VllmConfig] = None,
        quant_config_path: Optional[str] | object = _QUANT_CONFIG_UNCHANGED,
    ) -> None:
        """Load a model. If vllm_config is provided, update config and rebuild runner.

        If a runner was previously stashed for this model (weights on CPU from
        a prior sleep→unload cycle) it is restored directly and weights are
        moved back to HPU, skipping the expensive warmup_graphs phase.

        Args:
            vllm_config: Optional new VllmConfig to apply before loading.
            quant_config_path: Optional path to INC FP8 calibration JSON.
        """
        if quant_config_path is not _QUANT_CONFIG_UNCHANGED:
            if quant_config_path is not None:
                quant_config_path_str = cast(str, quant_config_path)
                os.environ["QUANT_CONFIG"] = quant_config_path_str
                logger.info("QUANT_CONFIG=%s", quant_config_path_str)
            else:
                os.environ.pop("QUANT_CONFIG", None)
                logger.info("QUANT_CONFIG cleared")
        else:
            logger.info("QUANT_CONFIG unchanged: %s", os.environ.get("QUANT_CONFIG"))

        if vllm_config is not None:
            self._apply_vllm_config(vllm_config)

            stash_key = self._runner_stash_key(vllm_config)
            if stash_key in self._model_runner_stash:
                # Runner is alive with compiled graph cache intact;
                # weights are on CPU — just move them back to HPU.
                self.restore_stashed_model(vllm_config=vllm_config, restore_kv_cache=False)
                self.kv_cache_sleeping = False
                return

            with set_current_vllm_config(vllm_config):
                self.model_runner = self._create_model_runner()
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()  # type: ignore[union-attr]

        self.model_sleeping = False
        self.kv_cache_sleeping = False

    def restore_stashed_model(
        self,
        vllm_config: Optional[VllmConfig] = None,
        restore_kv_cache: bool = True,
    ) -> dict[str, bool]:
        """Restore a previously stashed runner and optionally wake its state.

        This is primarily used as a rollback path when model reconfigure fails
        after unload_model().
        """
        target_config = vllm_config or self.vllm_config
        stash_key = self._runner_stash_key(target_config)

        if stash_key not in self._model_runner_stash:
            logger.warning("[HPUWorker] No stashed runner found for rollback key=%s", stash_key)
            return {"restored": False}

        self.model_runner = self._model_runner_stash.pop(stash_key)
        stashed_state = self._model_runner_state_stash.pop(stash_key, {})

        bucketing_manager = getattr(self.model_runner, "bucketing_manager", None)
        if bucketing_manager is not None:
            bucketing_manager.activate()

        restored_config = stashed_state.get("vllm_config", getattr(self.model_runner, "vllm_config", target_config))
        self._apply_vllm_config(restored_config)

        self.model_sleeping = bool(stashed_state.get("model_sleeping", True))
        self.kv_cache_sleeping = bool(stashed_state.get("kv_cache_sleeping", False))
        self.kv_cache_config = stashed_state.get("kv_cache_config", None)

        wake_tags: list[str] = []
        if self.model_sleeping:
            wake_tags.append("weights")
        if restore_kv_cache and self.kv_cache_sleeping and self.kv_cache_config is not None:
            wake_tags.append("kv_cache")

        if wake_tags:
            self.wake_up(tags=wake_tags)

        if not restore_kv_cache:
            # gaudi_reconfigure_engine will recreate KV cache with the new config.
            self.kv_cache_sleeping = False

        logger.info("[HPUWorker] Restored stashed runner for model: %s", restored_config.model_config.model)
        return {"restored": True}

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        kv_caches: dict[str, torch.Tensor] = {}
        kv_cache_spec = self.model_runner.get_kv_cache_spec()  # type: ignore[union-attr]
        single_kv_block_size_bytes = 0
        for layer_name, layer_spec in kv_cache_spec.items():
            if self.model_runner.uses_framework_kv_cache_layout(layer_name):
                kv_caches[layer_name] = self.model_runner.allocate_framework_kv_cache_layer(
                    layer_spec,
                    num_blocks=1,
                )
                single_kv_block_size_bytes += layer_spec.page_size_bytes
            elif isinstance(layer_spec, FullAttentionSpec):
                dtype = layer_spec.dtype
                if dtype == torch.float8_e4m3fn and os.environ.get('QUANT_CONFIG', None) is not None and \
                    os.environ.get('VLLM_DYNAMIC_KV_QUANT', None) is not None and not self.model_config.use_mla:
                    create_dynamic_scales = True
                else:
                    create_dynamic_scales = False

                # Create dummy KV cache tensors with proper shapes for profiling
                num_blocks = 1  # Use single block for profiling
                block_size = layer_spec.block_size
                num_kv_heads = layer_spec.num_kv_heads
                head_size = layer_spec.head_size

                attn_backend = self.model_runner.attn_backend  # type: ignore[union-attr]
                kv_cache_shape = attn_backend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
                kv_scales_shape = kv_cache_shape[:-1] + (1, )

                hpu_k_cache = torch.zeros(kv_cache_shape, dtype=dtype, device='hpu')
                hpu_v_cache = None if self.model_config.use_mla else torch.zeros(
                    kv_cache_shape, dtype=dtype, device='hpu')

                hpu_k_scales = torch.ones(kv_scales_shape, dtype=torch.bfloat16,
                                          device='hpu') if create_dynamic_scales else None
                if create_dynamic_scales:
                    hpu_v_scales = (torch.ones(kv_scales_shape, dtype=torch.bfloat16, device='hpu'),
                                    torch.ones([num_blocks, num_kv_heads, head_size],
                                               dtype=torch.bfloat16,
                                               device='hpu'))
                else:
                    hpu_v_scales = None

                kv_caches[layer_name] = (hpu_k_cache, hpu_v_cache, hpu_k_scales, hpu_v_scales)

                single_kv_block_size_bytes += layer_spec.page_size_bytes

            elif isinstance(layer_spec, MambaSpec):
                dtype0 = layer_spec.dtypes[0]
                dtype1 = layer_spec.dtypes[1]

                # Use an empty tensor instead of `None`` to force Dynamo to pass
                # it by reference, rather by specializing on the value ``None``.
                hpu_ssm_cache = torch.tensor([], dtype=dtype0, device='hpu')
                hpu_conv_cache = torch.tensor([], dtype=dtype1, device='hpu')
                hpu_ssm_scales = torch.tensor([], dtype=dtype0, device='hpu')
                hpu_conv_scales = torch.tensor([], dtype=dtype1, device='hpu')

                kv_caches[layer_name] = (hpu_ssm_cache, hpu_conv_cache, hpu_ssm_scales, hpu_conv_scales)

                single_kv_block_size_bytes += layer_spec.page_size_bytes
            else:
                raise NotImplementedError

        runner_kv_caches: list[torch.Tensor] = []
        if kv_caches and all(self.model_runner.uses_framework_kv_cache_layout(name) for name in kv_caches):
            self.model_runner.bind_framework_kv_caches(kv_caches, runner_kv_caches)
        else:
            bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, runner_kv_caches)

        if is_fake_hpu():
            fake_hpu_cache_alloc = 4 * 2**30  # take 4 GiB flat on fake hpu
            return fake_hpu_cache_alloc
        with HabanaMemoryProfiler() as m:
            self.model_runner.profile_run(initialize_only=True)  # type: ignore[union-attr]
            torch.hpu.synchronize()
        msg = ("Model profiling run "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        # At this point we should've allocated the maximum workspace for all
        # recipes we will use the extra memory for graphs/blocks
        free_hpu_memory = torch.hpu.mem_get_info()[0]

        dummy_block_headroom = single_kv_block_size_bytes
        explicit_kv_cache_size = self.cache_config.kv_cache_memory_bytes
        if explicit_kv_cache_size is not None:
            if explicit_kv_cache_size > free_hpu_memory:
                raise ValueError("Requested KV cache memory "
                                 f"({format_bytes(explicit_kv_cache_size)}) exceeds free HPU "
                                 f"memory after profiling ({format_bytes(free_hpu_memory)}). "
                                 "Decrease --kv-cache-memory-bytes.")
            if explicit_kv_cache_size <= dummy_block_headroom:
                raise ValueError("Requested KV cache memory "
                                 f"({format_bytes(explicit_kv_cache_size)}) must exceed the "
                                 "HPU dummy-block reservation "
                                 f"({format_bytes(dummy_block_headroom)}).")

            # Match core vLLM's explicit-memory contract: an explicit byte
            # budget takes precedence over gpu_memory_utilization. HPU keeps
            # one extra padding block outside the scheduler-visible cache, so
            # subtract that fixed allocation below before returning.
            cache_size_bytes = int(explicit_kv_cache_size)
            self.model_runner.mem_margin = max(  # type: ignore[union-attr]
                0, free_hpu_memory - cache_size_bytes)
            msg = (f"Free device memory: {format_bytes(free_hpu_memory)}, "
                   f"using explicit KV cache budget {format_bytes(cache_size_bytes)} "
                   "(--kv-cache-memory-bytes; gpu_memory_utilization ignored), "
                   f"{format_bytes(dummy_block_headroom)} reserved for KV cache dummy "
                   f"block, {format_bytes(cache_size_bytes - dummy_block_headroom)} "
                   "reserved for usable KV cache")
        else:
            try:
                graph_reserved_mem = (float(os.environ.get('VLLM_GRAPH_RESERVED_MEM', '0.1'))
                                      if not self.model_config.enforce_eager else 0)
            except ValueError:
                graph_reserved_mem = 0.0 if self.model_config.enforce_eager else 0.1
                logger.warning("Invalid VLLM_GRAPH_RESERVED_MEM value, using default %s", graph_reserved_mem)
            graph_headroom = 1 - graph_reserved_mem
            available_hpu_memory = free_hpu_memory * self.cache_config.gpu_memory_utilization
            hpu_memory_margin = free_hpu_memory * (1 - self.cache_config.gpu_memory_utilization)
            self.model_runner.mem_margin = hpu_memory_margin  # type: ignore[union-attr]
            cache_size_bytes = available_hpu_memory * graph_headroom
            graph_headroom_bytes = available_hpu_memory * (1 - graph_headroom)
            msg = (f"Free device memory: {format_bytes(free_hpu_memory)}, "
                   f"{format_bytes(available_hpu_memory)} usable "
                   f"(gpu_memory_utilization={self.cache_config.gpu_memory_utilization}),"
                   f" {format_bytes(graph_headroom_bytes)} reserved for HPUGraphs "
                   f"(VLLM_GRAPH_RESERVED_MEM={graph_reserved_mem}), "
                   f"{format_bytes(dummy_block_headroom)} reserved for KV cache dummy "
                   f"block {format_bytes(cache_size_bytes - dummy_block_headroom)} "
                   "reserved for usable KV cache")

        logger.info(msg)

        # Clear the dummy KV cache to free up memory
        kv_caches = {}
        forward_context = self.vllm_config.compilation_config.static_forward_context
        for layer_name in forward_context:
            forward_context[layer_name].kv_cache = None
        runner_kv_caches = []
        gc.collect()
        available = cache_size_bytes - dummy_block_headroom
        if getattr(self.model_runner, "serving_workspace_reserve", 0):
            available -= self.model_runner.serving_workspace_reserve
            logger.info("Reserved %s for long-context CSA2 and concurrent request working state",
                        format_bytes(self.model_runner.serving_workspace_reserve))

        # For hybrid models (attention + recurrent layers), the GPU
        # backend shares a single raw buffer across spec types via
        # as_strided, but HPU allocates separate tensors per spec
        # (torch.compile can't handle as_strided mixed-dtype views).
        # Reduce reported memory so the scheduler computes fewer
        # num_blocks that fit the HPU separate-allocation model.
        has_attn = any(isinstance(s, FullAttentionSpec) for s in kv_cache_spec.values())
        has_gdn = any(isinstance(s, MambaSpec) and s.mamba_type in _GDN_MAMBA_TYPES for s in kv_cache_spec.values())
        has_standard_mamba = any(
            isinstance(s, MambaSpec) and s.mamba_type not in _GDN_MAMBA_TYPES for s in kv_cache_spec.values())
        compact_gdn = os.environ.get("VLLM_COMPACT_GDN", "0").strip().lower() in ("1", "true")
        if has_attn and has_gdn and not compact_gdn:
            # When compact GDN is OFF, GDN state scales with num_blocks
            # just like ATN.  GPU shares one raw buffer via as_strided,
            # but HPU allocates separate tensors per spec type, so the
            # total per-block cost is real_attn + real_mamba (not
            # max(real_attn, real_mamba)).  Reduce reported memory so
            # the scheduler computes fewer num_blocks that fit.
            # When compact GDN is ON, GDN state is a small fixed
            # allocation (max_reqs * num_groups + 2), independent of
            # num_blocks, so no adjustment is needed.
            padded_page = next(iter(kv_cache_spec.values())).page_size_bytes
            real_attn = next(s.real_page_size_bytes for s in kv_cache_spec.values() if isinstance(s, FullAttentionSpec))
            real_mamba = next(
                sum(math.prod(sh) * get_dtype_size(dt) for sh, dt in zip(s.shapes, s.dtypes))
                for s in kv_cache_spec.values() if isinstance(s, MambaSpec) and s.mamba_type in _GDN_MAMBA_TYPES)
            total_real = real_attn + real_mamba
            if total_real > padded_page:
                factor = padded_page / total_real
                adjusted = int(available * factor)
                logger.info(
                    "HPU hybrid cache: reducing available KV cache "
                    "memory by %.1f%% (factor=%.3f) for separate "
                    "per-spec allocations (padded_page=%s, "
                    "real_attn=%s, real_mamba=%s).", (1 - factor) * 100, factor, format_bytes(padded_page),
                    format_bytes(real_attn), format_bytes(real_mamba))
                available = adjusted

        if has_attn and has_standard_mamba:
            # Standard Mamba2 + ATN hybrids (e.g. Granite): the
            # naive_mamba_cache_sharing path allocates independent
            # tensors per layer type, so the real per-block cost is
            # attn_page + mamba_state (not max(attn, mamba)).
            attn_page_size = next(s.page_size_bytes for s in kv_cache_spec.values() if isinstance(s, FullAttentionSpec))
            mamba_state_per_block = next(
                sum(math.prod(sh) * get_dtype_size(dt) for sh, dt in zip(s.shapes, s.dtypes))
                for s in kv_cache_spec.values() if isinstance(s, MambaSpec) and s.mamba_type not in _GDN_MAMBA_TYPES)
            if attn_page_size > 0:
                ratio = attn_page_size / (attn_page_size + mamba_state_per_block)
                adjusted = int(available * ratio)
                logger.info(
                    "Hybrid model (standard Mamba2 + ATN): adjusted "
                    "usable KV cache from %s to %s (attn_page=%d, "
                    "mamba_state=%d, ratio=%.3f)", format_bytes(available), format_bytes(adjusted), attn_page_size,
                    mamba_state_per_block, ratio)
                available = adjusted

        # Core vLLM's determine_available_memory contract returns an int
        # (bytes). Most models route through get_num_blocks() which casts to
        # int, but gemma4's heterogeneous KV layout (interleaved sliding/full
        # layers with different page_size_bytes) collapses into a single
        # UniformTypeKVCacheSpecs group whose num_blocks math
        # (`available_memory // page_size_bytes`) does NOT cast. A float
        # available_memory then yields a float num_blocks, which the
        # scheduler's BlockPool rejects via `isinstance(num_gpu_blocks, int)`.
        # Coerce to int here so every downstream path (uniform AND
        # heterogeneous) receives an integer byte budget.
        return int(available)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        with HabanaMemoryProfiler() as m:
            self.kv_cache_config = kv_cache_config
            self.model_runner.initialize_kv_cache(kv_cache_config)  # type: ignore[union-attr]
            self.kv_cache_sleeping = False
            torch.hpu.synchronize()
        if len(self.model_runner.kv_caches) > 0:  # type: ignore[union-attr]
            # Find the first ATN layer's tensor shape for a meaningful
            # block count (compact GDN layers have a much smaller dim-0).
            alloc_blocks = None
            for kv in self.model_runner.kv_caches:  # type: ignore[union-attr]
                t = kv[0] if not isinstance(kv[0], tuple) else kv[0][0]
                dim0 = t.shape[0]
                if alloc_blocks is None or dim0 > alloc_blocks:
                    alloc_blocks = dim0
            msg = (
                f"Usable num_blocks: {kv_cache_config.num_blocks}, "
                f"actual allocated num_blocks (max across layers): "
                f"{alloc_blocks} "
                f"(_PAD_BLOCK_ID={self.model_runner._PAD_BLOCK_ID}, "  # type: ignore[union-attr]
                f"_PAD_SLOT_ID={self.model_runner._PAD_SLOT_ID})")  # type: ignore[union-attr]
            logger.info(msg)
        msg = ("Initializing cache engine "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        self.compile_or_warm_up_model()

    def compile_or_warm_up_model(self) -> CompilationTimes:
        if self.model_config.hf_config.model_type in ("deepseek_v4", "deepseek_v41"):
            from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
            bind_worker_cpu(self.rank)
        # Don't run the warmup if the model is already warmed up
        if not getattr(self.model_runner, 'graphed_buckets', None):
            self.model_runner.warmup_model()  # type: ignore[union-attr]
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        if self.model_config.hf_config.model_type in ("deepseek_v4", "deepseek_v41"):
            from vllm_gaudi.ops.deepseek_v4_config import bind_worker_helpers
            bind_worker_helpers(self.rank)

        if self.model_config.hf_config.model_type == "deepseek_v41":
            self._write_native_decoder_stats("ready")

        return CompilationTimes(
            language_model=self.vllm_config.compilation_config.compilation_time,
            encoder=self.vllm_config.compilation_config.encoder_compilation_time,
        )

    def sample_tokens(self, grammar_output: "GrammarOutput|None") -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)  # type: ignore[union-attr]

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        if self.step_debug:
            self.step_debug(f'step={self.step}')
        if self.step_profiler and self.step == self.profile_steps[0]:
            self.step_profiler.start()
        with track_graph_compile('HPUWorker.execute_model') \
                if self.gc_track_recompiles \
                else contextlib.nullcontext():
            output = self.model_runner.execute_model(scheduler_output)  # type: ignore[union-attr]
        # TODO(woosuk): Send the output to the engine process.
        if self.step_profiler:
            if self.step >= self.profile_steps[0]:
                self.step_profiler.step()
            if self.step == self.profile_steps[1]:
                self.step_profiler.stop()
                self.step_profiler = None
                raise RuntimeError('Step profiling finished!')
        self.step += 1
        # NOTE(Harish): removed "if self.rank == 0 else None" for KV_connector enabling with TP>1
        # referred to Gpu Model Runner, KV connector aggregation expects valid output from all ranks
        return output

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()  # type: ignore[union-attr]

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.model_runner.take_draft_token_ids()  # type: ignore[union-attr]

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        if is_start:
            self.start_profile()
        else:
            self.stop_profile()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)  # type: ignore[union-attr]

    def synchronize_device(self) -> None:
        """Block until in-flight HPU work completes.

        Overrides WorkerBase, whose default calls torch.accelerator.synchronize(). HPU registers as a
        torch accelerator but rejects a device-wide multi-stream wait, so the base default raises
        RuntimeError once EngineCore broadcasts "synchronize_device" on every pause (e.g. LLM.sleep()).
        """
        if is_fake_hpu():
            return
        torch.hpu.synchronize()

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        # Upstream now consumes handshake metadata via
        # set_xfer_handshake_metadata_pp_aware(), which expects keys to be
        # (pp_rank, tp_rank) tuples. Returning a flat {tp_rank: metadata} dict
        # made it unpack an int and raise TypeError during EngineCore init.
        pp_rank = get_pp_group().rank_in_group
        tp_rank = get_tp_group().rank_in_group
        return {(pp_rank, tp_rank): metadata}

    def get_hpu_used_memory_mb(self) -> float | None:
        """Return currently used HPU memory in MB for this worker."""
        if is_fake_hpu():
            return None
        try:
            torch.hpu.synchronize()
            free_bytes, total_bytes = torch.hpu.mem_get_info()
            return (total_bytes - free_bytes) / (1024**2)
        except Exception:
            return None

    def sleep(self, level: int = 1) -> None:
        """Put the worker into sleep mode to reduce memory usage. Unlike GPU workers that use custom
        memory allocators, HPU workers use a simpler approach of moving model to CPU and clearing KV cache.
        Args:
            level (int): Sleep level (kept for interface compatibility, always performs level 1 operations)
        """

        if level == 2:
            logger.warning("Currently, HPU does not support level 2 sleep mode. Performing level 1 operations")
        assert not htorch.utils.internal.is_lazy(
        ) or self.model_config.enforce_eager, "Sleep mode is supported only for torch.compile mode"

        # Handle model - if model was loaded move it to CPU
        if self.model_sleeping:
            logger.warning("Model is already in a sleep mode, skipping moving it to CPU")
        elif self.model_runner is None or not hasattr(self.model_runner, "model") or self.model_runner.model is None:
            logger.warning("Model was not loaded yet, skipping moving it to CPU")
        else:
            with HabanaMemoryProfiler() as m:
                self.model_runner.model.to("cpu")
                # Re-derive MoeMatmul.weight slices from the now-CPU params
                _rebind_moe_expert_weights(self.model_runner.model)
                gc.collect()
                torch.hpu.synchronize()
            msg = f"Moving model to CPU for sleep mode took {m.get_summary_string()}"
            logger.info(msg)
            self.model_sleeping = True

        # Handle KV cache - discard it
        if self.kv_cache_sleeping:
            logger.warning("KV cache has already been discarded by calling sleep method and it has not been "
                           "reinitialized by calling wake up method yet, skipping discarding it again")
        elif self.kv_cache_config is None:
            logger.warning("KV cache has not been initialized yet, skipping discarding it")
        else:
            with HabanaMemoryProfiler() as m:
                self.model_runner.defragmenter = None
                self.model_runner.kv_caches = []
                forward_context = self.vllm_config.compilation_config.static_forward_context
                for layer_name in forward_context:
                    forward_context[layer_name].kv_cache = None
                gc.collect()
                torch.hpu.synchronize()
            msg = f"Discarding KV cache for sleep mode took {m.get_summary_string()}"
            logger.info(msg)
            self.kv_cache_sleeping = True

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the worker from sleep mode.
        It can move the model back to HPU and/or reinitialize KV cache.

        Args:
            tags: Optional list of tags (kept for interface compatibility)
        """
        assert not htorch.utils.internal.is_lazy(
        ) or self.model_config.enforce_eager, "Sleep mode is supported only for torch.compile mode"

        if tags is None:
            tags = ["weights", "kv_cache"]

        # Handle model - if model was loaded, move it back to HPU
        if "weights" in tags:
            if not self.model_sleeping:
                logger.warning("Model is not in a sleep mode, skipping moving it to HPU")
            elif self.model_runner is None or not hasattr(self.model_runner,
                                                          "model") or self.model_runner.model is None:
                logger.warning("Model was not loaded yet, skipping moving it to HPU")
            else:
                with HabanaMemoryProfiler() as m:
                    self.model_runner.model.to(self.vllm_config.device_config.device)
                    # Re-derive MoeMatmul.weight slices from the now-moved
                    # parent FusedMoE registered params (w13_weight/w2_weight).
                    _rebind_moe_expert_weights(self.model_runner.model)
                    gc.collect()
                    torch.hpu.synchronize()
                msg = f"Waking up model, moving it back to HPU took {m.get_summary_string()}"
                logger.info(msg)
                self.model_sleeping = False

        # Handle KV cache - reinitialize it
        if "kv_cache" in tags:
            if not self.kv_cache_sleeping:
                logger.warning("KV cache is not in a sleep mode, skipping reinitializing it")
            elif self.kv_cache_config is None:
                logger.warning("KV cache config is empty, skipping reinitializing KV cache")
            else:
                with HabanaMemoryProfiler() as m:
                    self.model_runner.initialize_kv_cache(self.kv_cache_config)
                    self.model_runner.defragmenter = OnlineDefragmenter(self.model_runner.kv_caches,
                                                                        self.model_runner.block_size)
                    gc.collect()
                    torch.hpu.synchronize()
                msg = f"Waking up KV cache, reinitializing it took {m.get_summary_string()}"
                logger.info(msg)
                self.kv_cache_sleeping = False


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    parallel_config = vllm_config.parallel_config
    """Initialize the distributed environment."""
    init_distributed_environment(parallel_config.world_size, rank, distributed_init_method, local_rank, backend='hccl')

    dummy_tensor_hpu = torch.ones(1).to('hpu')
    torch.distributed.all_reduce(dummy_tensor_hpu)
    assert dummy_tensor_hpu.item() == parallel_config.world_size * parallel_config.data_parallel_size
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size)


@contextmanager
def track_graph_compile(name: str):
    from habana_frameworks.torch.hpu.metrics import metric_localcontext
    with metric_localcontext("graph_compilation") as gc:
        yield
        htorch.hpu.synchronize()
    if gc.stats()[0][1] != 0:
        msg = f"[{name}] graph compilation detected: {gc.stats()}"
        logger.warning(msg)
