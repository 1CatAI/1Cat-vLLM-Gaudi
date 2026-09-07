# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU quantization overrides for vLLM's hardware-agnostic model path."""

from importlib import import_module
import logging
import os
from pathlib import Path

import torch

import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi import envs
from vllm_gaudi.extension.kernels import rms_norm as load_hpu_rms_norm
from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpMXFP4
from vllm_gaudi.ops.deepseek_v4_flashmla import (
    HPUFlashMLABackend,
    build_hpu_flashmla_decode_plan,
    flashmla_output_buffer_is_compatible,
    parse_hpu_flashmla_backend,
)
from vllm_gaudi.ops.hpu_fused_moe import _normalize_moe_activation

try:
    from vllm.model_executor.hw_agnostic.kernels.linear import (
        _POSSIBLE_FP8_BLOCK_KERNELS, )
    from vllm.model_executor.hw_agnostic.custom_op import CustomOp
    from vllm.model_executor.hw_agnostic.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
        Fp8BlockScaledMMLinearKernel, )
    from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
        FusedMoEQuantConfig,
        mxfp4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.hw_agnostic.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase, )
    from vllm.model_executor.hw_agnostic.layers.fused_moe.gate_linear import (
        GateLinear as HwAgnosticGateLinear, )
    from vllm.model_executor.hw_agnostic.layers.fused_moe.runner.moe_runner import (
        MoERunner as HwAgnosticMoERunner, )
    from vllm.model_executor.hw_agnostic.layers.fused_moe.runner.shared_experts import (
        SharedExpertsOrder, )
    from vllm.model_executor.hw_agnostic.layers.fused_moe.routed_experts import (
        RoutedExperts, )
    from vllm.model_executor.hw_agnostic.quantization import (
        fp8_linear_method as hw_fp8_module, )
    from vllm.model_executor.hw_agnostic.quantization import (
        mxfp4_moe_method as hw_mxfp4_module, )
    from vllm.model_executor.hw_agnostic.quantization.fp8_linear_method import (
        Fp8LinearMethod as HwAgnosticFp8LinearMethod, )
    from vllm.model_executor.hw_agnostic.quantization.mxfp4_moe_method import (
        Mxfp4MoEMethod as HwAgnosticMxfp4MoEMethod, )
    from vllm.models.deepseek_v4.hw_agnostic.attention._profile import (
        dsv4_profile_stage, )
    from vllm.models.deepseek_v4.hw_agnostic.layers.mhc import (
        MHCFusedPostPreOp, )
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.platforms import PlatformEnum
    from vllm.utils.torch_utils import direct_register_custom_op
except ModuleNotFoundError:
    # The hardware-agnostic DeepSeek V4 stack is newer than some supported
    # vLLM revisions. Keep the plugin importable with those revisions.
    pass
else:
    logger = logging.getLogger(__name__)
    _DSV4_TPC_OPS_LOADED = False
    _DSV4_COMPILE_ONLY_ACTIVE = False
    _DSV4_QNORM_TPC_PREWARMED = False
    _DSV4_QNORM_PATHS_LOGGED: set[str] = set()

    def _ensure_dsv4_tpc_ops_loaded() -> None:
        global _DSV4_TPC_OPS_LOADED
        if _DSV4_TPC_OPS_LOADED:
            return
        if hasattr(
                torch.ops.custom_op,
                "custom_deepseek_v4_dual_dequant_gather_bf16_gaudi2",
        ):
            _DSV4_TPC_OPS_LOADED = True
            return
        library = envs.VLLM_HPU_DSV4_TPC_OP_LIBRARY
        if not library:
            raise RuntimeError("VLLM_HPU_DSV4_TPC_OP_LIBRARY must point to the "
                               "DeepSeek V4 TPC PyTorch registration library")
        if not os.path.isfile(library):
            raise FileNotFoundError(f"DeepSeek V4 TPC operator library not found: {library}")
        torch.ops.load_library(library)
        _DSV4_TPC_OPS_LOADED = True

    def _hpu_dsv4_is_compile_only() -> bool:
        from habana_frameworks.torch.internal import bridge_config

        return (_DSV4_COMPILE_ONLY_ACTIVE or os.getenv("_VLLM_HPU_DSV4_COMPILE_ONLY_ACTIVE") == "1"
                or bool(bridge_config.get_pt_compile_only_mode()))

    def _set_hpu_dsv4_compile_only(active: bool) -> bool:
        global _DSV4_COMPILE_ONLY_ACTIVE

        previous = _DSV4_COMPILE_ONLY_ACTIVE
        _DSV4_COMPILE_ONLY_ACTIVE = active
        return previous

    def _set_hpu_dsv4_qnorm_tpc_prewarmed(active: bool) -> bool:
        global _DSV4_QNORM_TPC_PREWARMED

        previous = _DSV4_QNORM_TPC_PREWARMED
        _DSV4_QNORM_TPC_PREWARMED = active
        return previous

    def _log_hpu_dsv4_qnorm_path_once(path: str) -> None:
        if os.getenv("VLLM_HPU_DSV4_LOG_QNORM_PATH") != "1":
            return
        if path in _DSV4_QNORM_PATHS_LOGGED:
            return
        _DSV4_QNORM_PATHS_LOGGED.add(path)
        logger.warning("DeepSeek V4 qnorm path=%s", path)

    @CustomOp.register_oot(name="MHCFusedPostPreOp")
    class HPUMHCFusedPostPreOp(MHCFusedPostPreOp):
        """Gaudi2 decode fast path matching the CUDA TileLang fusion boundary."""

        def forward_oot(
            self,
            x: torch.Tensor,
            residual: torch.Tensor,
            post_layer_mix: torch.Tensor,
            comb_res_mix: torch.Tensor,
            fn: torch.Tensor,
            hc_scale: torch.Tensor,
            hc_base: torch.Tensor,
            rms_eps: float,
            hc_pre_eps: float,
            hc_sinkhorn_eps: float,
            hc_post_mult_value: float,
            sinkhorn_repeat: int,
            n_splits: int = 1,
            tile_n: int = 1,
            norm_weight: torch.Tensor | None = None,
            norm_eps: float = 0.0,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            norm_supported = ((norm_weight is None and norm_eps == 0.0)
                              or (norm_weight is not None and norm_weight.dtype == torch.bfloat16
                                  and norm_weight.shape == (4096, ) and abs(norm_eps - 1.0e-6) < 1.0e-12
                                  and norm_weight.is_contiguous()))
            eligible = (envs.VLLM_HPU_DSV4_TPC_MHC and x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
                        and x.ndim == 2 and x.shape == (1, 4096) and residual.shape == (1, 4, 4096)
                        and post_layer_mix.shape == (1, 4, 1) and comb_res_mix.shape == (1, 4, 4)
                        and fn.dtype == torch.float32 and fn.shape == (24, 16384) and hc_scale.dtype == torch.float32
                        and hc_scale.shape == (3, ) and hc_base.dtype == torch.float32 and hc_base.shape == (24, )
                        and abs(rms_eps - 1.0e-6) < 1.0e-12 and abs(hc_pre_eps - 1.0e-6) < 1.0e-12
                        and abs(hc_sinkhorn_eps - 1.0e-6) < 1.0e-12 and abs(hc_post_mult_value - 2.0) < 1.0e-12
                        and sinkhorn_repeat == 20 and n_splits == 1 and tile_n == 1 and norm_supported
                        and x.is_contiguous() and residual.is_contiguous() and post_layer_mix.is_contiguous()
                        and comb_res_mix.is_contiguous() and fn.is_contiguous() and hc_scale.is_contiguous()
                        and hc_base.is_contiguous())
            if eligible:
                _ensure_dsv4_tpc_ops_loaded()
                stage = ("mhc_fused_post_pre_tpc" if norm_weight is None else "mhc_fused_post_pre_norm_tpc")
                with dsv4_profile_stage(stage):
                    residual_cur, residual_f32, rrms = (torch.ops.custom_op.custom_deepseek_v4_mhc_post_prepare_gaudi2(
                        x,
                        residual,
                        post_layer_mix,
                        comb_res_mix,
                    ))
                    raw_mixes = torch.matmul(residual_f32, fn.t())
                    if norm_weight is None:
                        post_mix_cur, comb_mix_flat, layer_input_cur = (
                            torch.ops.custom_op.custom_deepseek_v4_mhc_pre_emit_gaudi2(
                                residual_cur,
                                raw_mixes,
                                rrms,
                                hc_scale,
                                hc_base,
                            ))
                    else:
                        post_mix_cur, comb_mix_flat, layer_input_cur = (
                            torch.ops.custom_op.custom_deepseek_v4_mhc_pre_emit_norm_gaudi2(
                                residual_cur,
                                raw_mixes,
                                rrms,
                                hc_scale,
                                hc_base,
                                norm_weight,
                            ))
                    comb_mix_cur = comb_mix_flat.view(-1, 4, 4)
                    return (
                        residual_cur,
                        post_mix_cur,
                        comb_mix_cur,
                        layer_input_cur,
                    )
            return MHCFusedPostPreOp.forward_native(
                self,
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                tile_n,
                norm_weight,
                norm_eps,
            )

    _HPU_MOE_STAGE_DEBUG = (os.getenv("VLLM_HPU_MOE_STAGE_DEBUG") == "1")
    _HPU_MOE_QUALITY_DEBUG = (os.getenv("VLLM_HPU_MOE_QUALITY_DEBUG") == "1")
    _HPU_MOE_QUALITY_MAX_CALLS = int(os.getenv("VLLM_HPU_MOE_QUALITY_MAX_CALLS", "3"))
    _HPU_MOE_QUALITY_CALL_INDEX = 0

    @torch.compiler.disable
    def _log_hpu_moe_stage(stage: str) -> None:
        logger.warning("HPU MoE stage=%s", stage)

    @torch.compiler.disable
    def _dump_hpu_moe_quality(
        call_index: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        shared_output: torch.Tensor | None,
        routed_output: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> None:
        if get_tensor_model_parallel_rank() != 0:
            return
        torch.hpu.synchronize()
        tensors = {
            "hidden_states": hidden_states.detach().float().cpu(),
            "router_logits": router_logits.detach().float().cpu(),
            "topk_ids": topk_ids.detach().cpu(),
            "topk_weights": topk_weights.detach().float().cpu(),
            "routed_output": routed_output.detach().float().cpu(),
        }
        if input_ids is not None:
            tensors["input_ids"] = input_ids.detach().cpu()
        if shared_output is not None:
            tensors["shared_output"] = shared_output.detach().float().cpu()

        def token_norms(name: str) -> list[float]:
            value = tensors[name]
            return torch.linalg.vector_norm(value.reshape(value.shape[0], -1), dim=-1).tolist()

        logger.warning(
            "HPU MoE quality call=%d hidden_norms=%s router_range=(%.6g, %.6g) "
            "input_ids=%s topk_ids=%s topk_weights=%s shared_norms=%s "
            "routed_norms=%s",
            call_index,
            token_norms("hidden_states"),
            float(tensors["router_logits"].min()),
            float(tensors["router_logits"].max()),
            tensors["input_ids"].tolist() if input_ids is not None else None,
            tensors["topk_ids"].tolist(),
            tensors["topk_weights"].tolist(),
            token_norms("shared_output") if shared_output is not None else None,
            token_norms("routed_output"),
        )
        dump_dir = os.getenv("VLLM_HPU_MOE_QUALITY_DUMP_DIR")
        if dump_dir:
            path = Path(dump_dir)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(tensors, path / f"moe_call_{call_index:02d}_tp0.pt")

    class HPUHwAgnosticFp8BlockLinearKernel(Fp8BlockScaledMMLinearKernel):
        """Kernel-selection shim; HPU execution is handled by the method."""

        @classmethod
        def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
            return True, None

        def apply_weights(self, layer, x, bias=None):
            raise NotImplementedError("HPU uses HPUHwAgnosticFp8LinearMethod.apply() directly")

        def apply_block_scaled_mm(self, A, B, As, Bs):
            raise NotImplementedError("HPU uses HPUHwAgnosticFp8LinearMethod.apply() directly")

    _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT] = [HPUHwAgnosticFp8BlockLinearKernel]

    _ORIGINAL_GATE_LINEAR_FORWARD = HwAgnosticGateLinear.forward

    def _hpu_gate_linear_forward(
        self,
        x: torch.Tensor,
    ):
        if self.allow_router_gemm and x.dtype == torch.bfloat16:
            weight_fp32 = getattr(self, "_hpu_router_weight_fp32", None)
            if weight_fp32 is None:
                raise RuntimeError("HPU router FP32 weight cache is not initialized")
            return torch.nn.functional.linear(x.float(), weight_fp32), None
        return _ORIGINAL_GATE_LINEAR_FORWARD(self, x)

    HwAgnosticGateLinear.forward = _hpu_gate_linear_forward

    _ORIGINAL_HW_AGNOSTIC_MOE_RUNNER_FORWARD = (HwAgnosticMoERunner.forward)

    def _hpu_hw_agnostic_moe_runner_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global _HPU_MOE_QUALITY_CALL_INDEX

        if self.moe_config.dp_size != 1 or self.moe_config.pcp_size > 1:
            return _ORIGINAL_HW_AGNOSTIC_MOE_RUNNER_FORWARD(
                self,
                hidden_states,
                router_logits,
                input_ids,
            )

        hidden_states, shared_experts_input = (self.apply_routed_input_transform(hidden_states))
        (
            hidden_states,
            og_hidden_dim_pre_xform,
            og_hidden_dim_post_xform,
        ) = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )

        self.routed_experts._ensure_moe_quant_config_init()
        with dsv4_profile_stage("moe_shared_sync"):
            self._maybe_sync_shared_experts_stream(shared_experts_input)
        if self.gate is not None:
            with dsv4_profile_stage("moe_gate"):
                router_logits, _ = self.gate(hidden_states)

        if _HPU_MOE_QUALITY_DEBUG:
            quality_call_index = _HPU_MOE_QUALITY_CALL_INDEX
            _HPU_MOE_QUALITY_CALL_INDEX += 1
            quality_debug_this_call = (quality_call_index < _HPU_MOE_QUALITY_MAX_CALLS)
        else:
            # A changing Python global becomes a Dynamo guard and recompiles
            # every decoder layer even when quality diagnostics are disabled.
            quality_call_index = -1
            quality_debug_this_call = False
        if _HPU_MOE_STAGE_DEBUG or quality_debug_this_call:
            _log_hpu_moe_stage("shared_start")
            assert self._shared_experts is not None
            shared_layer = self._shared_experts._layer
            _log_hpu_moe_stage("shared_gate_up_start")
            gate_up, _ = shared_layer.gate_up_proj(shared_experts_input)
            _log_hpu_moe_stage("shared_gate_up_done")
            shared_hidden = shared_layer.act_fn(gate_up)
            _log_hpu_moe_stage("shared_act_done")
            shared_output, _ = shared_layer.down_proj(shared_hidden)
            _log_hpu_moe_stage("shared_down_done")
            self._shared_experts._output = shared_output
            _log_hpu_moe_stage("shared_done")
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )
            _log_hpu_moe_stage("router_done")
            fused_output = self.routed_experts(
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=self._shared_experts,
                shared_experts_input=shared_experts_input,
            )
            _log_hpu_moe_stage("routed_done")
            shared_output = (self._shared_experts.output if self._shared_experts is not None else None)
            if quality_debug_this_call:
                _dump_hpu_moe_quality(
                    quality_call_index,
                    hidden_states,
                    router_logits,
                    topk_ids,
                    topk_weights,
                    shared_output,
                    fused_output,
                    input_ids,
                )
        else:
            with dsv4_profile_stage("moe_shared_start"):
                self._maybe_apply_shared_experts(
                    shared_experts_input,
                    SharedExpertsOrder.NO_OVERLAP,
                )
            with dsv4_profile_stage("moe_router"):
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    topk_indices_dtype=self._quant_method.topk_indices_dtype,
                    input_ids=input_ids,
                )
            with dsv4_profile_stage("moe_routed_experts"):
                fused_output = self.routed_experts(
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts=self._shared_experts,
                    shared_experts_input=shared_experts_input,
                )
            with dsv4_profile_stage("moe_shared_finish"):
                self._maybe_apply_shared_experts(
                    shared_experts_input,
                    SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
                )
                shared_output = (self._shared_experts.output if self._shared_experts is not None else None)
        with dsv4_profile_stage("moe_combine"):
            result = self._maybe_combine(shared_output, fused_output)
        if isinstance(result, tuple):
            shared_output, fused_output = result
        else:
            shared_output, fused_output = None, result

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        shared_output, fused_output = (self._maybe_apply_routed_scale_to_output(
            shared_output,
            fused_output,
        ))
        fused_output = self.apply_routed_output_transform(fused_output)
        if shared_output is not None:
            fused_output = shared_output + fused_output

        with dsv4_profile_stage("moe_all_reduce"):
            return self._maybe_reduce_final_output(
                fused_output,
                og_hidden_dim_post_xform,
            )

    HwAgnosticMoERunner.forward = (_hpu_hw_agnostic_moe_runner_forward)

    class HPUHwAgnosticFp8LinearMethod(HwAgnosticFp8LinearMethod):
        """Run hardware-agnostic FP8 linear layers through HPU kernels."""

        def create_weights(self, *args, **kwargs) -> None:
            if hpu_ops.is_hpu_gaudi2:
                kwargs["weight_loader"] = hpu_ops.gaudi_weight_wrapper(kwargs.get("weight_loader"))
            # Gaudi2 cannot allocate torch.float8_e8m0fnu tensors. Load the
            # checkpoint's E8M0 scales into FP32 parameters instead; PyTorch's
            # weight loader converts the encoded values during the copy.
            is_scale_e8m0 = self.is_scale_e8m0
            self.is_scale_e8m0 = False
            try:
                super().create_weights(*args, **kwargs)
            finally:
                self.is_scale_e8m0 = is_scale_e8m0

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            layer.quant_config = self.quant_config
            if self.block_quant:
                hpu_ops.fp8_block_linear_postprocess_weights(layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
                return
            super().process_weights_after_loading(layer)

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if not self.block_quant:
                return super().apply(layer, x, bias)

            cached_weight = getattr(
                layer,
                "_hpu_block_fp8_weight_dequant_transposed",
                None,
            )
            dsv4_decode_cache = getattr(
                layer,
                "_hpu_dsv4_bf16_attention_cache",
                False,
            )
            if cached_weight is not None and (not dsv4_decode_cache or (x.ndim == 2 and x.shape[0] == 1)):
                output = torch.matmul(x, cached_weight)
                if bias is not None:
                    output = output + bias
                return output

            assert self.quant_config.weight_block_size is not None
            return hpu_ops.apply_block_fp8_linear_hpu(
                input=x,
                layer=layer,
                block_size=self.quant_config.weight_block_size,
                bias=bias,
                do_unpad=True,
                force_channel_fp8=envs.VLLM_HPU_FORCE_CHANNEL_FP8,
            )

    class HPUHwAgnosticMxfp4MoEMethod(HwAgnosticMxfp4MoEMethod):
        """Run hardware-agnostic MXFP4 experts with the native HPU MoE op."""

        MXFP4_BLOCK_SIZE = 32

        def __init__(self, moe):
            FusedMoEMethodBase.__init__(self, moe)
            self.weight_dtype = "mxfp4"

        @property
        def supports_eplb(self) -> bool:
            return False

        @property
        def topk_indices_dtype(self) -> torch.dtype:
            return torch.int32

        def process_weights_after_loading(self, layer: RoutedExperts) -> None:
            if (envs.VLLM_HPU_DSV4_TPC_MXFP4_GATHER or envs.VLLM_HPU_DSV4_TPC_MXFP4_INDEXED):
                _ensure_dsv4_tpc_ops_loaded()
            w13 = layer.w13_weight
            w2 = layer.w2_weight
            w13_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale
            w13_bias = getattr(layer, "w13_bias", None)
            w2_bias = getattr(layer, "w2_bias", None)
            self._validate_weight_shapes(
                w13,
                w2,
                w13_scale,
                w2_scale,
                w13_bias,
                w2_bias,
            )

            weight_dump_dir = os.getenv("VLLM_HPU_MXFP4_WEIGHT_DUMP_DIR")
            weight_dump_layer = os.getenv("VLLM_HPU_MXFP4_WEIGHT_DUMP_LAYER", "")
            if (weight_dump_dir and weight_dump_layer in layer.layer_name and get_tensor_model_parallel_rank() == 0):
                expert_ids = [
                    int(value) for value in os.getenv(
                        "VLLM_HPU_MXFP4_WEIGHT_DUMP_EXPERTS",
                        "8,93,163,184",
                    ).split(",") if value
                ]
                torch.hpu.synchronize()
                payload = {
                    "layer_name": layer.layer_name,
                    "expert_ids": expert_ids,
                    "w13": w13.data[expert_ids].cpu(),
                    "w2": w2.data[expert_ids].cpu(),
                    "w13_scale": w13_scale.data[expert_ids].cpu(),
                    "w2_scale": w2_scale.data[expert_ids].cpu(),
                }
                dump_dir = Path(weight_dump_dir)
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / "mxfp4_layer1_tp0_loaded.pt"
                torch.save(payload, dump_path)
                logger.warning("Dumped loaded MXFP4 weights to %s", dump_path)

            num_experts = layer.local_num_experts
            ep_rank = layer.moe_config.moe_parallel_config.ep_rank
            ep_shift = ep_rank * num_experts
            has_bias = w13_bias is not None and w2_bias is not None

            layer.moe_op = VllmMixtureOfExpertsOpMXFP4(
                layer.global_num_experts,
                num_experts,
                ep_shift,
                ep_shift + num_experts - 1,
                block_size=self.MXFP4_BLOCK_SIZE,
                has_bias=has_bias,
            )

            for expert_id in range(num_experts):
                layer.moe_op.w13_list[expert_id].set_weight(w13.data[expert_id])
                layer.moe_op.w13_list[expert_id].set_scale(w13_scale.data[expert_id])
                layer.moe_op.w2_list[expert_id].set_weight(w2.data[expert_id])
                layer.moe_op.w2_list[expert_id].set_scale(w2_scale.data[expert_id])
                if has_bias:
                    layer.moe_op.w13_list[expert_id].set_bias(w13_bias.data[expert_id])
                    layer.moe_op.w2_list[expert_id].set_bias(w2_bias.data[expert_id])

            layer.moe_op.set_stacked_weights(
                w13.data,
                w2.data,
                w13_scale.data,
                w2_scale.data,
            )
            layer.moe_op._cache_weight_lists()
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
            return mxfp4_w4a16_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            )

        def apply(
            self,
            layer: RoutedExperts,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            input_shape = x.shape
            if x.ndim != 2:
                x = x.reshape(-1, x.shape[-1])
            if topk_ids.ndim != 2:
                topk_ids = topk_ids.reshape(-1, topk_ids.shape[-1])
            if topk_weights.ndim != 2:
                topk_weights = topk_weights.reshape(-1, topk_weights.shape[-1])
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = topk_weights.to(x.dtype)

            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
                activation=_normalize_moe_activation(layer.activation),
            )
            if len(input_shape) == 2:
                return output
            return output.view(*input_shape)

    hw_fp8_module.Fp8LinearMethod = HPUHwAgnosticFp8LinearMethod
    hw_mxfp4_module.Mxfp4MoEMethod = HPUHwAgnosticMxfp4MoEMethod

    # The DeepSeek quant config can be imported while the worker is still
    # starting, before general plugins run. Patch its bound symbols as well as
    # the source modules so both import orders select the HPU implementations.
    from vllm.models.deepseek_v4.hw_agnostic.quantization import (
        quant_config as deepseek_v4_quant_module, )

    deepseek_v4_quant_module.Fp8LinearMethod = HPUHwAgnosticFp8LinearMethod
    deepseek_v4_quant_module.Mxfp4MoEMethod = HPUHwAgnosticMxfp4MoEMethod

    def _hpu_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        hpu_fused_rms_norm = load_hpu_rms_norm()
        if hpu_fused_rms_norm is not None:
            return hpu_fused_rms_norm.apply(x, weight, eps)

        x_fp32 = x.float()
        variance = x_fp32.square().mean(dim=-1, keepdim=True)
        return (x_fp32 * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)

    def _hpu_fused_q_kv_rmsnorm(
        qr: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert qr.ndim == 2 and kv.ndim == 2
        assert qr.shape[0] == kv.shape[0]
        if qr.shape[0] == 0:
            return torch.empty_like(qr), torch.empty_like(kv)
        return (
            _hpu_rms_norm(qr, q_weight, eps),
            _hpu_rms_norm(kv, kv_weight, eps),
        )

    def _hpu_dsv4_inv_rope_einsum(
        rotary_emb: torch.nn.Module,
        o: torch.Tensor,
        positions: torch.Tensor,
        rope_head_dim: int,
        n_local_groups: int,
        o_lora_rank: int,
        wo_a: torch.nn.Module,
    ) -> torch.Tensor:
        cached_weight = getattr(wo_a, "_hpu_dsv4_wo_a_bmm_weight", None)
        if (cached_weight is None or o.ndim != 3 or o.shape[0] != 1 or cached_weight.shape[0] != n_local_groups
                or cached_weight.shape[1] != o_lora_rank):
            return _ORIGINAL_DEEPSEEK_V4_INV_ROPE_EINSUM(
                rotary_emb,
                o,
                positions,
                rope_head_dim,
                n_local_groups,
                o_lora_rank,
                wo_a,
            )

        with dsv4_profile_stage("o_proj_inv_rope"):
            o_ref = _DEEPSEEK_V4_INV_ROPE_MODULE._apply_inv_rope_ref(
                rotary_emb,
                o,
                positions,
                rope_head_dim,
            ).to(torch.bfloat16)
        o_ref = o_ref.view(o.shape[0], n_local_groups, -1)
        if cached_weight.shape[2] != o_ref.shape[2]:
            return _ORIGINAL_DEEPSEEK_V4_INV_ROPE_EINSUM(
                rotary_emb,
                o,
                positions,
                rope_head_dim,
                n_local_groups,
                o_lora_rank,
                wo_a,
            )
        with dsv4_profile_stage("o_proj_wo_a_bmm"):
            return torch.bmm(
                o_ref.transpose(0, 1),
                cached_weight.transpose(1, 2),
            ).transpose(0, 1)

    def _hpu_deepseek_score_projection(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if envs.VLLM_HPU_DSV4_BF16_SCORE_PROJECTION:
            return torch.mm(hidden_states, weight.T).float()
        return torch.mm(hidden_states.float(), weight.T.float())

    _DEEPSEEK_V4_Q_LORA_RANK = 1024

    def _hpu_dsv4_frontend_base(
        hidden_states: torch.Tensor,
        fused_qkv_weight_t: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qr_kv = torch.matmul(hidden_states, fused_qkv_weight_t)
        qr, kv = qr_kv.split(
            (_DEEPSEEK_V4_Q_LORA_RANK, _DEEPSEEK_V4_HEAD_DIM),
            dim=-1,
        )
        qr = torch.ops.hpu.rms_norm(qr, q_norm_weight, 1e-6, None, False)[0]
        kv = torch.ops.hpu.rms_norm(kv, kv_norm_weight, 1e-6, None, False)[0]
        q = torch.matmul(qr, q_weight_t)
        return q, qr, kv

    def _hpu_dsv4_native_fp8_linear(
        value: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        value_fp8, value_scale = torch.ops.hpu.cast_to_fp8_just_in_time(
            value,
            [1, value.shape[-1]],
            out_dtype=torch.float8_e4m3fn,
            scale_dtype=torch.float32,
        )
        return torch.ops.hpu.fp8_gemm_v2(
            A=value_fp8,
            trans_A=False,
            B=weight,
            trans_B=True,
            D=None,
            out_dtype=torch.bfloat16,
            A_scale_inv=value_scale,
            B_scale_inv=weight_scale,
            bias=None,
            accumulate=False,
        )

    def _hpu_dsv4_native_frontend_base(
        hidden_states: torch.Tensor,
        fused_qkv_weight: torch.Tensor,
        fused_qkv_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight: torch.Tensor,
        q_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qr_kv = _hpu_dsv4_native_fp8_linear(
            hidden_states,
            fused_qkv_weight,
            fused_qkv_scale,
        )
        qr, kv = qr_kv.split(
            (_DEEPSEEK_V4_Q_LORA_RANK, _DEEPSEEK_V4_HEAD_DIM),
            dim=-1,
        )
        qr = torch.ops.hpu.rms_norm(qr, q_norm_weight, 1e-6, None, False)[0]
        kv = torch.ops.hpu.rms_norm(kv, kv_norm_weight, 1e-6, None, False)[0]
        q = _hpu_dsv4_native_fp8_linear(qr, q_weight, q_scale)
        return q, qr, kv

    def _hpu_dsv4_native_frontend_no_scores(
        hidden_states: torch.Tensor,
        fused_qkv_weight: torch.Tensor,
        fused_qkv_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight: torch.Tensor,
        q_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _hpu_dsv4_native_frontend_base(
            hidden_states,
            fused_qkv_weight,
            fused_qkv_scale,
            q_norm_weight,
            kv_norm_weight,
            q_weight,
            q_scale,
        )

    def _hpu_dsv4_native_frontend_compressor(
        hidden_states: torch.Tensor,
        fused_qkv_weight: torch.Tensor,
        fused_qkv_scale: torch.Tensor,
        compressor_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight: torch.Tensor,
        q_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        q, qr, kv = _hpu_dsv4_native_frontend_base(
            hidden_states,
            fused_qkv_weight,
            fused_qkv_scale,
            q_norm_weight,
            kv_norm_weight,
            q_weight,
            q_scale,
        )
        compressor_score = torch.mm(hidden_states, compressor_weight.T).float()
        return q, qr, kv, compressor_score

    def _hpu_dsv4_native_frontend_compressor_indexer(
        hidden_states: torch.Tensor,
        fused_qkv_weight: torch.Tensor,
        fused_qkv_scale: torch.Tensor,
        compressor_weight: torch.Tensor,
        indexer_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight: torch.Tensor,
        q_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        q, qr, kv = _hpu_dsv4_native_frontend_base(
            hidden_states,
            fused_qkv_weight,
            fused_qkv_scale,
            q_norm_weight,
            kv_norm_weight,
            q_weight,
            q_scale,
        )
        compressor_score = torch.mm(hidden_states, compressor_weight.T).float()
        indexer_score = torch.mm(hidden_states, indexer_weight.T).float()
        return q, qr, kv, compressor_score, indexer_score

    def _hpu_dsv4_native_frontend_compressor_bf16(
        hidden_states: torch.Tensor,
        fused_qkv_weight: torch.Tensor,
        fused_qkv_scale: torch.Tensor,
        compressor_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight: torch.Tensor,
        q_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        q, qr, kv = _hpu_dsv4_native_frontend_base(
            hidden_states,
            fused_qkv_weight,
            fused_qkv_scale,
            q_norm_weight,
            kv_norm_weight,
            q_weight,
            q_scale,
        )
        compressor_score = torch.mm(hidden_states, compressor_weight.T)
        return q, qr, kv, compressor_score

    def _hpu_dsv4_native_frontend_compressor_indexer_bf16(
        hidden_states: torch.Tensor,
        fused_qkv_weight: torch.Tensor,
        fused_qkv_scale: torch.Tensor,
        compressor_weight: torch.Tensor,
        indexer_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight: torch.Tensor,
        q_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        q, qr, kv = _hpu_dsv4_native_frontend_base(
            hidden_states,
            fused_qkv_weight,
            fused_qkv_scale,
            q_norm_weight,
            kv_norm_weight,
            q_weight,
            q_scale,
        )
        compressor_score = torch.mm(hidden_states, compressor_weight.T)
        indexer_score = torch.mm(hidden_states, indexer_weight.T)
        return q, qr, kv, compressor_score, indexer_score

    def _hpu_dsv4_frontend_no_scores(
        hidden_states: torch.Tensor,
        fused_qkv_weight_t: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _hpu_dsv4_frontend_base(
            hidden_states,
            fused_qkv_weight_t,
            q_norm_weight,
            kv_norm_weight,
            q_weight_t,
        )

    def _hpu_dsv4_frontend_compressor(
        hidden_states: torch.Tensor,
        fused_qkv_weight_t: torch.Tensor,
        compressor_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight_t: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        q, qr, kv = _hpu_dsv4_frontend_base(
            hidden_states,
            fused_qkv_weight_t,
            q_norm_weight,
            kv_norm_weight,
            q_weight_t,
        )
        compressor_score = torch.mm(hidden_states, compressor_weight.T).float()
        return q, qr, kv, compressor_score

    def _hpu_dsv4_frontend_compressor_indexer(
        hidden_states: torch.Tensor,
        fused_qkv_weight_t: torch.Tensor,
        compressor_weight: torch.Tensor,
        indexer_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight_t: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        q, qr, kv = _hpu_dsv4_frontend_base(
            hidden_states,
            fused_qkv_weight_t,
            q_norm_weight,
            kv_norm_weight,
            q_weight_t,
        )
        compressor_score = torch.mm(hidden_states, compressor_weight.T).float()
        indexer_score = torch.mm(hidden_states, indexer_weight.T).float()
        return q, qr, kv, compressor_score, indexer_score

    def _hpu_dsv4_frontend_compressor_bf16(
        hidden_states: torch.Tensor,
        fused_qkv_weight_t: torch.Tensor,
        compressor_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight_t: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        q, qr, kv = _hpu_dsv4_frontend_base(
            hidden_states,
            fused_qkv_weight_t,
            q_norm_weight,
            kv_norm_weight,
            q_weight_t,
        )
        compressor_score = torch.mm(hidden_states, compressor_weight.T)
        return q, qr, kv, compressor_score

    def _hpu_dsv4_frontend_compressor_indexer_bf16(
        hidden_states: torch.Tensor,
        fused_qkv_weight_t: torch.Tensor,
        compressor_weight: torch.Tensor,
        indexer_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        q_weight_t: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        q, qr, kv = _hpu_dsv4_frontend_base(
            hidden_states,
            fused_qkv_weight_t,
            q_norm_weight,
            kv_norm_weight,
            q_weight_t,
        )
        compressor_score = torch.mm(hidden_states, compressor_weight.T)
        indexer_score = torch.mm(hidden_states, indexer_weight.T)
        return q, qr, kv, compressor_score, indexer_score

    def _compile_hpu_dsv4_frontend(function):
        return torch.compile(
            function,
            backend="hpu_backend",
            fullgraph=True,
            dynamic=False,
        )

    _HPU_DSV4_FRONTEND_NO_SCORES = _compile_hpu_dsv4_frontend(_hpu_dsv4_frontend_no_scores)
    _HPU_DSV4_FRONTEND_COMPRESSOR = _compile_hpu_dsv4_frontend(_hpu_dsv4_frontend_compressor)
    _HPU_DSV4_FRONTEND_COMPRESSOR_INDEXER = (_compile_hpu_dsv4_frontend(_hpu_dsv4_frontend_compressor_indexer))
    _HPU_DSV4_FRONTEND_COMPRESSOR_BF16 = _compile_hpu_dsv4_frontend(_hpu_dsv4_frontend_compressor_bf16)
    _HPU_DSV4_FRONTEND_COMPRESSOR_INDEXER_BF16 = (
        _compile_hpu_dsv4_frontend(_hpu_dsv4_frontend_compressor_indexer_bf16))
    _HPU_DSV4_NATIVE_FRONTEND_NO_SCORES = _compile_hpu_dsv4_frontend(_hpu_dsv4_native_frontend_no_scores)
    _HPU_DSV4_NATIVE_FRONTEND_COMPRESSOR = (_compile_hpu_dsv4_frontend(_hpu_dsv4_native_frontend_compressor))
    _HPU_DSV4_NATIVE_FRONTEND_COMPRESSOR_INDEXER = (
        _compile_hpu_dsv4_frontend(_hpu_dsv4_native_frontend_compressor_indexer))
    _HPU_DSV4_NATIVE_FRONTEND_COMPRESSOR_BF16 = (_compile_hpu_dsv4_frontend(_hpu_dsv4_native_frontend_compressor_bf16))
    _HPU_DSV4_NATIVE_FRONTEND_COMPRESSOR_INDEXER_BF16 = (
        _compile_hpu_dsv4_frontend(_hpu_dsv4_native_frontend_compressor_indexer_bf16))

    def _hpu_dsv4_cached_weight(layer) -> torch.Tensor | None:
        return getattr(
            layer,
            "_hpu_block_fp8_weight_dequant_transposed",
            None,
        )

    def _hpu_dsv4_cached_native_weight(layer, ) -> tuple[torch.Tensor, torch.Tensor] | None:
        weight = getattr(layer, "_hpu_dsv4_native_fp8_weight", None)
        scale = getattr(
            layer,
            "_hpu_dsv4_native_fp8_weight_scale",
            None,
        )
        if weight is None or scale is None:
            return None
        return weight, scale

    def _hpu_compiled_attention_frontend(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
    ] | None:
        if (not envs.VLLM_HPU_DSV4_COMPILED_ATTN_FRONTEND or hidden_states.ndim != 2 or hidden_states.shape[0] != 1
                or self.eps != 1e-6):
            return None

        use_native_fp8 = envs.VLLM_HPU_DSV4_NATIVE_FP8_ATTN_FRONTEND
        use_bf16_compressor = (envs.VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS)
        if use_native_fp8:
            fused_qkv = _hpu_dsv4_cached_native_weight(self.fused_wqa_wkv)
            q_projection = _hpu_dsv4_cached_native_weight(self.wq_b)
            if fused_qkv is None or q_projection is None:
                return None
            common = (hidden_states, *fused_qkv)
            norms_and_q = (
                self.q_norm.weight.data,
                self.kv_norm.weight.data,
                *q_projection,
            )
            no_scores_fn = _HPU_DSV4_NATIVE_FRONTEND_NO_SCORES
            compressor_fn = (_HPU_DSV4_NATIVE_FRONTEND_COMPRESSOR_BF16
                             if use_bf16_compressor else _HPU_DSV4_NATIVE_FRONTEND_COMPRESSOR)
            compressor_indexer_fn = (_HPU_DSV4_NATIVE_FRONTEND_COMPRESSOR_INDEXER_BF16
                                     if use_bf16_compressor else _HPU_DSV4_NATIVE_FRONTEND_COMPRESSOR_INDEXER)
        else:
            fused_qkv_weight_t = _hpu_dsv4_cached_weight(self.fused_wqa_wkv)
            q_weight_t = _hpu_dsv4_cached_weight(self.wq_b)
            if fused_qkv_weight_t is None or q_weight_t is None:
                return None
            common = (hidden_states, fused_qkv_weight_t)
            norms_and_q = (
                self.q_norm.weight.data,
                self.kv_norm.weight.data,
                q_weight_t,
            )
            no_scores_fn = _HPU_DSV4_FRONTEND_NO_SCORES
            compressor_fn = (_HPU_DSV4_FRONTEND_COMPRESSOR_BF16
                             if use_bf16_compressor else _HPU_DSV4_FRONTEND_COMPRESSOR)
            compressor_indexer_fn = (_HPU_DSV4_FRONTEND_COMPRESSOR_INDEXER_BF16
                                     if use_bf16_compressor else _HPU_DSV4_FRONTEND_COMPRESSOR_INDEXER)
        if self.compressor is None:
            q, qr, kv = no_scores_fn(
                *common,
                *norms_and_q,
            )
            return q, qr, kv, None, None

        compressor_weight = self.compressor.fused_wkv_wgate.weight
        if self.indexer is None:
            q, qr, kv, compressor_score = (compressor_fn(
                *common,
                compressor_weight,
                *norms_and_q,
            ))
            return q, qr, kv, compressor_score, None

        if _hpu_can_skip_short_indexer_cache(self.indexer):
            q, qr, kv, compressor_score = compressor_fn(
                *common,
                compressor_weight,
                *norms_and_q,
            )
            return q, qr, kv, compressor_score, None

        if not _hpu_can_skip_short_indexer(self.indexer):
            return None
        indexer_weight = (self.indexer.compressor.fused_wkv_wgate.weight)
        q, qr, kv, compressor_score, indexer_score = (compressor_indexer_fn(
            *common,
            compressor_weight,
            indexer_weight,
            *norms_and_q,
        ))
        return q, qr, kv, compressor_score, indexer_score

    def _hpu_attn_gemm_parallel_execute(self, hidden_states):
        with dsv4_profile_stage("projection_fused_qkv"):
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)

        kv_score = None
        if self.compressor is not None:
            with dsv4_profile_stage("projection_compressor_score"):
                kv_score = _hpu_deepseek_score_projection(
                    hidden_states,
                    self.compressor.fused_wkv_wgate.weight,
                )

        indexer_weights = None
        indexer_kv_score = None
        if (self.indexer is not None and not _hpu_can_skip_short_indexer_cache(self.indexer)):
            if not _hpu_can_skip_short_indexer(self.indexer):
                with dsv4_profile_stage("projection_indexer_weights"):
                    indexer_weights, _ = self.indexer.weights_proj(hidden_states)
            with dsv4_profile_stage("projection_indexer_score"):
                indexer_kv_score = _hpu_deepseek_score_projection(
                    hidden_states,
                    self.indexer.compressor.fused_wkv_wgate.weight,
                )

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def _hpu_dsv4_can_order_compressor_decode(
        mla_attn,
        q: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: dict,
    ) -> bool:
        if not envs.VLLM_HPU_DSV4_TPC_ORDERED_COMPRESSOR:
            return False
        if mla_attn.compress_ratio not in (4, 128):
            return False
        if (mla_attn.compress_ratio == 128 and not envs.VLLM_HPU_DSV4_TPC_ORDERED_C128_COMPRESSOR):
            return False
        swa_metadata = attn_metadata.get(mla_attn.swa_cache_layer.prefix)
        mla_metadata = attn_metadata.get(mla_attn.prefix)
        if swa_metadata is None or mla_metadata is None:
            return False
        if (swa_metadata.num_prefills != 0 or swa_metadata.num_decodes <= 0 or swa_metadata.num_decode_tokens != 1):
            return False
        if (swa_metadata.decode_swa_indices is None or swa_metadata.decode_swa_lens is None):
            return False
        if not _hpu_dsv4_can_direct_decode_dispatch(q, output, swa_only=False):
            return False
        if mla_attn.compress_ratio == 4:
            return (mla_attn.topk_indices_buffer is not None and swa_metadata.token_to_req_indices is not None
                    and swa_metadata.is_valid_token is not None)
        return (mla_metadata.c128a_global_decode_topk_indices is not None
                and mla_metadata.c128a_decode_topk_lens is not None)

    def _hpu_dsv4_qnorm_compressor_recipe(*inputs):
        return (torch.ops.custom_op.custom_deepseek_v4_qnorm_compressor_c4_f32_gaudi2(*inputs))

    _HPU_DSV4_QNORM_COMPRESSOR_RECIPE = torch.compile(
        _hpu_dsv4_qnorm_compressor_recipe,
        backend="hpu_backend",
        fullgraph=True,
        dynamic=False,
    )

    def _hpu_dsv4_can_fuse_qnorm_compressor(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: dict,
    ) -> bool:
        if not envs.VLLM_HPU_DSV4_FUSED_QNORM_COMPRESSOR:
            return False
        compressor = self.compressor
        mla_attn = self.mla_attn
        if not (compressor is not None and compressor.compress_ratio == 4 and mla_attn.compress_ratio == 4
                and q.ndim == 3 and q.shape[0] == 1 and q.shape[-1] == _DEEPSEEK_V4_HEAD_DIM
                and q.dtype == torch.bfloat16 and kv.ndim == 2 and kv.shape == (1, _DEEPSEEK_V4_HEAD_DIM)
                and kv.dtype == torch.bfloat16 and kv_score.ndim == 2 and kv_score.shape[0] == 1
                and kv_score.dtype == torch.float32 and compressor.state_cache.kv_cache_storage.dtype == torch.uint8
                and _hpu_dsv4_can_direct_decode_dispatch(q, output, swa_only=False)):
            return False

        static_eligible = getattr(
            self,
            "_hpu_dsv4_qnorm_compressor_static_eligible",
            None,
        )
        if static_eligible is None:
            static_eligible = bool(
                envs.VLLM_HPU_DSV4_TPC_QNORM_ROPE_KV_PACK and envs.VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN
                and envs.VLLM_HPU_DSV4_TPC_SAVE_COMPRESS_NORM_C4 and envs.VLLM_HPU_DSV4_TPC_ORDERED_COMPRESSOR
                and not envs.VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS and not envs.VLLM_HPU_DSV4_TPC_MIXED_COMPRESS_INPUTS
                and _DSV4_QNORM_TPC_PREWARMED and self.eps == 1e-6)
            object.__setattr__(
                self,
                "_hpu_dsv4_qnorm_compressor_static_eligible",
                static_eligible,
            )
        if not static_eligible:
            return False
        if self.indexer is not None and not (_hpu_can_skip_short_indexer_cache(self.indexer)
                                             and _hpu_can_skip_decode_topk_fill(self.indexer)):
            return False

        swa_metadata = attn_metadata.get(mla_attn.swa_cache_layer.prefix)
        mla_metadata = attn_metadata.get(mla_attn.prefix)
        state_metadata = attn_metadata.get(compressor.state_cache.prefix)
        k_cache_metadata = attn_metadata.get(compressor.k_cache_prefix)
        if any(value is None for value in (
                swa_metadata,
                mla_metadata,
                state_metadata,
                k_cache_metadata,
        )):
            return False
        metadata_key = (
            id(swa_metadata),
            id(mla_metadata),
            id(state_metadata),
            id(k_cache_metadata),
        )
        metadata_cache = getattr(
            self,
            "_hpu_dsv4_qnorm_compressor_metadata_cache",
            None,
        )
        if metadata_cache is not None and metadata_key in metadata_cache:
            return True
        eligible = bool(swa_metadata.num_prefills == 0 and swa_metadata.num_decodes > 0
                        and swa_metadata.num_decode_tokens == 1 and swa_metadata.slot_mapping.shape[0] == 1
                        and swa_metadata.decode_swa_indices is not None and swa_metadata.decode_swa_lens is not None
                        and swa_metadata.token_to_req_indices is not None and swa_metadata.is_valid_token is not None
                        and swa_metadata.seq_lens is not None and state_metadata.token_to_req_indices is not None
                        and state_metadata.slot_mapping.shape[0] == 1 and k_cache_metadata.slot_mapping.shape[0] == 1
                        and mla_attn.topk_indices_buffer is not None)
        if eligible:
            if metadata_cache is None:
                metadata_cache = set()
                object.__setattr__(
                    self,
                    "_hpu_dsv4_qnorm_compressor_metadata_cache",
                    metadata_cache,
                )
            metadata_cache.add(metadata_key)
        return eligible

    def _hpu_dsv4_fused_qnorm_compressor_decode(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressor = self.compressor
        assert compressor is not None
        mla_attn = self.mla_attn
        swa_metadata = attn_metadata[mla_attn.swa_cache_layer.prefix]
        state_metadata = attn_metadata[compressor.state_cache.prefix]
        k_cache_metadata = attn_metadata[compressor.k_cache_prefix]
        k_cache_layer = compressor._static_forward_context[compressor.k_cache_prefix]
        assert swa_metadata.is_valid_token is not None
        assert state_metadata.token_to_req_indices is not None

        _ensure_dsv4_tpc_ops_loaded()
        compressor_kv, compressor_score = kv_score.split(
            [
                compressor.coff * compressor.head_dim,
                compressor.coff * compressor.head_dim,
            ],
            dim=-1,
        )
        norm_weight = getattr(
            compressor.norm,
            "_hpu_dsv4_weight_fp32",
            None,
        )
        if norm_weight is None:
            norm_weight = _hpu_dsv4_fp32_contiguous(compressor.norm.weight)
        q = q.contiguous()
        _, compressor_completion = (_HPU_DSV4_QNORM_COMPRESSOR_RECIPE(
            q,
            kv.contiguous(),
            mla_attn.swa_cache_layer.kv_cache_storage,
            mla_attn.swa_cache_layer.kv_cache_geometry,
            _hpu_dsv4_i32_contiguous(swa_metadata.slot_mapping[:1]),
            _hpu_dsv4_i32_contiguous(positions[:1]),
            _hpu_dsv4_fp32_contiguous(self.rotary_emb.cos_sin_cache),
            compressor.state_cache.kv_cache_storage,
            compressor.state_cache.kv_cache_geometry,
            k_cache_layer.kv_cache_geometry,
            compressor_kv,
            compressor_score,
            compressor.ape,
            _hpu_dsv4_i32_contiguous(state_metadata.slot_mapping[:1]),
            _hpu_dsv4_i32_contiguous(state_metadata.token_to_req_indices[:1]),
            _hpu_dsv4_i32_contiguous(state_metadata.block_table),
            norm_weight,
            compressor._rms_norm_eps_tensor,
            _hpu_dsv4_i32_contiguous(k_cache_metadata.slot_mapping[:1]),
            _hpu_dsv4_i32_contiguous(swa_metadata.is_valid_token[:1]),
        ))
        return q, compressor_completion

    def _hpu_dsv4_can_fuse_compressor_flashmla(
        self,
        q: torch.Tensor,
        kv_score: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: dict,
    ) -> bool:
        if not (envs.VLLM_HPU_DSV4_FUSED_COMPRESSOR_FLASHMLA and envs.VLLM_HPU_DSV4_TPC_SAVE_COMPRESS_NORM_C4
                and envs.VLLM_HPU_DSV4_FLASHMLA_SPLIT_KV and envs.VLLM_HPU_DSV4_FLASHMLA_TILED
                and not envs.VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS and not envs.VLLM_HPU_DSV4_TPC_MIXED_COMPRESS_INPUTS
                and self.compressor is not None and self.compressor.compress_ratio == 4 and self.mla_attn.compress_ratio
                == 4 and kv_score.shape[0] == 1 and self.compressor.state_cache.kv_cache_storage.dtype == torch.uint8
                and _hpu_dsv4_can_direct_decode_dispatch(q, output, swa_only=False)):
            return False
        if self.indexer is not None and not (_hpu_can_skip_short_indexer_cache(self.indexer)
                                             and _hpu_can_skip_decode_topk_fill(self.indexer)):
            return False

        mla_attn = self.mla_attn
        swa_metadata = attn_metadata.get(mla_attn.swa_cache_layer.prefix)
        mla_metadata = attn_metadata.get(mla_attn.prefix)
        state_metadata = attn_metadata.get(self.compressor.state_cache.prefix)
        k_cache_metadata = attn_metadata.get(self.compressor.k_cache_prefix)
        if any(value is None for value in (
                swa_metadata,
                mla_metadata,
                state_metadata,
                k_cache_metadata,
        )):
            return False
        if (swa_metadata.num_prefills != 0 or swa_metadata.num_decodes <= 0 or swa_metadata.num_decode_tokens != 1
                or swa_metadata.decode_swa_indices is None or swa_metadata.decode_swa_lens is None
                or swa_metadata.token_to_req_indices is None or swa_metadata.is_valid_token is None
                or swa_metadata.seq_lens is None or state_metadata.token_to_req_indices is None
                or state_metadata.slot_mapping.shape[0] != 1 or mla_attn.topk_indices_buffer is None):
            return False
        split_count = envs.VLLM_HPU_DSV4_FLASHMLA_SPLITS
        return (1 <= split_count <= 16 and mla_attn.topk_indices_buffer.shape[1] >= split_count)

    def _hpu_dsv4_fused_compressor_flashmla_decode(
        self,
        q: torch.Tensor,
        kv_score: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: dict,
    ) -> None:
        compressor = self.compressor
        assert compressor is not None
        mla_attn = self.mla_attn
        state_metadata = attn_metadata[compressor.state_cache.prefix]
        k_cache_metadata = attn_metadata[compressor.k_cache_prefix]
        k_cache_layer = compressor._static_forward_context[compressor.k_cache_prefix]
        swa_metadata = attn_metadata[mla_attn.swa_cache_layer.prefix]
        mla_metadata = attn_metadata[mla_attn.prefix]
        assert state_metadata.token_to_req_indices is not None
        assert swa_metadata.token_to_req_indices is not None
        assert swa_metadata.is_valid_token is not None
        assert swa_metadata.seq_lens is not None
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None
        assert mla_attn.topk_indices_buffer is not None

        _ensure_dsv4_tpc_ops_loaded()
        kv, score = kv_score.split(
            [
                compressor.coff * compressor.head_dim,
                compressor.coff * compressor.head_dim,
            ],
            dim=-1,
        )
        norm_weight = getattr(
            compressor.norm,
            "_hpu_dsv4_weight_fp32",
            None,
        )
        if norm_weight is None:
            norm_weight = _hpu_dsv4_fp32_contiguous(compressor.norm.weight)
        topk_shape_buffer = mla_attn.topk_indices_buffer[:1]
        split_count = envs.VLLM_HPU_DSV4_FLASHMLA_SPLITS
        split_shape_buffer = topk_shape_buffer[0, :split_count]
        swa_indices = swa_metadata.decode_swa_indices
        swa_idx_2d = (swa_indices.squeeze(1) if swa_indices.dim() == 3 else swa_indices)

        with dsv4_profile_stage("compressor_flashmla_fused_recipe"):
            attention_output, _, _, _ = (torch.ops.custom_op.custom_deepseek_v4_compress_flashmla_c4_f32_gaudi2(
                compressor.state_cache.kv_cache_storage,
                compressor.state_cache.kv_cache_geometry,
                k_cache_layer.kv_cache_geometry,
                kv,
                score,
                compressor.ape,
                _hpu_dsv4_i32_contiguous(positions[:1]),
                _hpu_dsv4_i32_contiguous(state_metadata.slot_mapping[:1]),
                _hpu_dsv4_i32_contiguous(state_metadata.token_to_req_indices[:1]),
                _hpu_dsv4_i32_contiguous(state_metadata.block_table),
                norm_weight,
                compressor._rms_norm_eps_tensor,
                _hpu_dsv4_fp32_contiguous(self.rotary_emb.cos_sin_cache),
                _hpu_dsv4_i32_contiguous(k_cache_metadata.slot_mapping[:1]),
                q.contiguous(),
                _hpu_dsv4_i32_contiguous(topk_shape_buffer),
                _hpu_dsv4_i32_contiguous(split_shape_buffer),
                _hpu_dsv4_i32_contiguous(swa_metadata.token_to_req_indices[:1]),
                _hpu_dsv4_i32_contiguous(mla_metadata.block_table[:swa_metadata.num_decodes]),
                _hpu_dsv4_i32_contiguous(swa_metadata.is_valid_token[:1]),
                _hpu_dsv4_i32_contiguous(swa_metadata.seq_lens),
                mla_attn.swa_cache_layer.kv_cache_storage,
                mla_attn.swa_cache_layer.kv_cache_geometry,
                _hpu_dsv4_i32_contiguous(swa_idx_2d),
                _hpu_dsv4_i32_contiguous(swa_metadata.decode_swa_lens),
                _hpu_dsv4_fp32_contiguous(mla_attn.attn_sink),
                output,
            ))
            output.copy_(attention_output)

    def _hpu_dsv4_execute_attention_core(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
        frontend: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        q, qr, kv, kv_score, indexer_kv_score = frontend
        q = q.view(-1, self.n_local_heads, self.head_dim)
        use_fused_qnorm_compressor = False
        compressor_dependency = None
        if self.compressor is not None:
            assert kv_score is not None
            use_fused_qnorm_compressor = (_hpu_dsv4_can_fuse_qnorm_compressor(
                self,
                q,
                kv,
                kv_score,
                out,
                attn_metadata,
            ))
        if use_fused_qnorm_compressor:
            with dsv4_profile_stage("qnorm_compressor_fused_recipe"):
                q, compressor_dependency = (_hpu_dsv4_fused_qnorm_compressor_decode(
                    self,
                    q,
                    kv,
                    kv_score,
                    positions,
                    attn_metadata,
                ))
        else:
            with dsv4_profile_stage("qnorm_rope_kv_write"):
                q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        use_fused_compressor_flashmla = False
        use_noclone_compressor = use_fused_qnorm_compressor
        if self.compressor is not None:
            assert kv_score is not None
            if not use_fused_qnorm_compressor:
                use_fused_compressor_flashmla = (_hpu_dsv4_can_fuse_compressor_flashmla(
                    self,
                    q,
                    kv_score,
                    out,
                    attn_metadata,
                ))
                use_noclone_compressor = (not use_fused_compressor_flashmla and _hpu_dsv4_can_order_compressor_decode(
                    self.mla_attn,
                    q,
                    out,
                    attn_metadata,
                ))
            if not (use_fused_qnorm_compressor or use_fused_compressor_flashmla):
                with dsv4_profile_stage("compressor"):
                    if use_noclone_compressor:
                        compressor_dependency = self.compressor(
                            kv_score,
                            positions,
                            self.rotary_emb,
                            use_noclone=True,
                        )
                    else:
                        self.compressor(kv_score, positions, self.rotary_emb)

        if self.indexer is not None:
            if _hpu_can_skip_short_indexer_cache(self.indexer):
                if not _hpu_can_skip_decode_topk_fill(self.indexer):
                    with dsv4_profile_stage("indexer_short_topk_fill"):
                        _hpu_materialize_short_indexer_topk(self.indexer, positions)
            else:
                assert indexer_kv_score is not None
                with dsv4_profile_stage("indexer"):
                    self.indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        None,
                        positions,
                        self.indexer_rotary_emb,
                    )

        with dsv4_profile_stage("mla_attention"):
            if use_fused_compressor_flashmla:
                assert kv_score is not None
                _hpu_dsv4_fused_compressor_flashmla_decode(
                    self,
                    q,
                    kv_score,
                    positions,
                    out,
                    attn_metadata,
                )
                return
            if use_noclone_compressor:
                if compressor_dependency is None:
                    raise RuntimeError("No-clone Compressor did not return its "
                                       "device dependency token")
                mla_metadata = attn_metadata.get(self.mla_attn.prefix)
                swa_metadata = attn_metadata.get(self.mla_attn.swa_cache_layer.prefix)
                if swa_metadata is None or not (_hpu_dsv4_try_direct_decode_dispatch(
                        self.mla_attn,
                        q,
                        self.mla_attn.kv_cache_storage,
                        self.mla_attn.kv_cache_geometry,
                        swa_metadata,
                        mla_metadata,
                        False,
                        out,
                        compressor_dependency=compressor_dependency,
                )):
                    raise RuntimeError("No-clone Compressor lost its ordered decode path")
                return
            self.mla_attn(q, kv, positions, output=out)

    def _hpu_materialize_short_indexer_topk(
        indexer,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        assert indexer.topk_indices_buffer is not None
        return _hpu_fill_short_context_topk_indices(
            indexer.topk_indices_buffer,
            positions,
            indexer.topk_tokens,
            indexer.compress_ratio,
        )

    def _hpu_deepseek_v4_attention_impl(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return _ORIGINAL_DEEPSEEK_V4_ATTENTION_IMPL(self, hidden_states, positions, out)

        with dsv4_profile_stage("compiled_frontend"):
            frontend = _hpu_compiled_attention_frontend(self, hidden_states)
        if frontend is None:
            return _ORIGINAL_DEEPSEEK_V4_ATTENTION_IMPL(self, hidden_states, positions, out)
        _hpu_dsv4_execute_attention_core(self, hidden_states, positions, out, frontend)

    def _hpu_inline_attention_frontend(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
    ] | None:
        fused_qkv_weight_t = _hpu_dsv4_cached_weight(self.fused_wqa_wkv)
        q_weight_t = _hpu_dsv4_cached_weight(self.wq_b)
        if fused_qkv_weight_t is None or q_weight_t is None:
            return None
        common = (hidden_states, fused_qkv_weight_t)
        norms_and_q = (
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            q_weight_t,
        )
        use_bf16_scores = envs.VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS
        if self.compressor is None:
            q, qr, kv = _hpu_dsv4_frontend_no_scores(*common, *norms_and_q)
            return q, qr, kv, None, None

        compressor_weight = self.compressor.fused_wkv_wgate.weight
        if self.indexer is None:
            frontend_fn = (_hpu_dsv4_frontend_compressor_bf16 if use_bf16_scores else _hpu_dsv4_frontend_compressor)
            q, qr, kv, compressor_score = frontend_fn(*common, compressor_weight, *norms_and_q)
            return q, qr, kv, compressor_score, None

        if _hpu_can_skip_short_indexer_cache(self.indexer):
            frontend_fn = (_hpu_dsv4_frontend_compressor_bf16 if use_bf16_scores else _hpu_dsv4_frontend_compressor)
            q, qr, kv, compressor_score = frontend_fn(*common, compressor_weight, *norms_and_q)
            return q, qr, kv, compressor_score, None

        if not _hpu_can_skip_short_indexer(self.indexer):
            return None
        frontend_fn = (_hpu_dsv4_frontend_compressor_indexer_bf16
                       if use_bf16_scores else _hpu_dsv4_frontend_compressor_indexer)
        q, qr, kv, compressor_score, indexer_score = frontend_fn(
            *common,
            compressor_weight,
            self.indexer.compressor.fused_wkv_wgate.weight,
            *norms_and_q,
        )
        return q, qr, kv, compressor_score, indexer_score

    def _hpu_deepseek_v4_attention_preprojected(
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
        q: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor | None,
        indexer_kv_score: torch.Tensor | None,
        layer_name: str,
    ) -> None:
        forward_context = get_forward_context()
        self = forward_context.no_compile_layers[layer_name]
        if not isinstance(forward_context.attn_metadata, dict):
            return _ORIGINAL_DEEPSEEK_V4_ATTENTION_IMPL(self, hidden_states, positions, out)
        _hpu_dsv4_execute_attention_core(
            self,
            hidden_states,
            positions,
            out,
            (q, qr, kv, kv_score, indexer_kv_score),
        )

    def _hpu_deepseek_v4_attention_preprojected_fake(
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
        q: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor | None,
        indexer_kv_score: torch.Tensor | None,
        layer_name: str,
    ) -> None:
        return None

    direct_register_custom_op(
        op_name="hpu_deepseek_v4_attention_preprojected",
        op_func=_hpu_deepseek_v4_attention_preprojected,
        mutates_args=["out"],
        fake_impl=_hpu_deepseek_v4_attention_preprojected_fake,
    )

    def _hpu_deepseek_v4_attention_frontend_impl(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        frontend = _hpu_inline_attention_frontend(self, hidden_states)
        if frontend is None:
            return _hpu_deepseek_v4_attention_impl(self, hidden_states, positions, out)
        q, qr, kv, kv_score, indexer_kv_score = frontend
        torch.ops.vllm.hpu_deepseek_v4_attention_preprojected(
            hidden_states,
            positions,
            out,
            q,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            self.layer_name,
        )

    def _hpu_can_skip_short_indexer(indexer) -> bool:
        return (envs.VLLM_HPU_DSV4_SHORT_INDEXER_SKIP and indexer.max_model_len <= indexer.topk_tokens)

    def _hpu_can_skip_short_indexer_cache(indexer) -> bool:
        return (envs.VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP and _hpu_can_skip_short_indexer(indexer)
                and _hpu_is_decode_only_attention(indexer))

    def _hpu_is_decode_only_attention(indexer) -> bool:
        if (not getattr(indexer, "prefix", "").endswith(".indexer") or not is_forward_context_available()):
            return False
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return False
        parent_prefix = indexer.prefix.removesuffix(".indexer")
        swa_metadata = attn_metadata.get(f"{parent_prefix}.swa_cache")
        return (swa_metadata is not None and getattr(swa_metadata, "num_decodes", 0) > 0
                and getattr(swa_metadata, "num_prefills", 0) == 0)

    def _hpu_can_skip_decode_topk_fill(indexer) -> bool:
        if (not (envs.VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN or envs.VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN)
                or not _hpu_can_skip_short_indexer(indexer) or indexer.vllm_config.scheduler_config.max_num_seqs != 1):
            return False
        return _hpu_is_decode_only_attention(indexer)

    def _hpu_fill_short_context_topk_indices(
        output: torch.Tensor,
        positions: torch.Tensor,
        topk_tokens: int,
        compress_ratio: int,
    ) -> torch.Tensor:
        num_tokens = positions.numel()
        offsets = torch.arange(
            topk_tokens,
            dtype=output.dtype,
            device=output.device,
        ).view(1, -1)
        num_compressed = ((positions.flatten().to(output.dtype) + 1).div(compress_ratio,
                                                                         rounding_mode="floor").view(-1, 1))
        output[:num_tokens, :topk_tokens].copy_(
            torch.where(
                offsets < num_compressed,
                offsets,
                torch.full_like(offsets, -1),
            ))
        return output

    def _hpu_deepseek_v4_indexer_forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor | None,
        positions: torch.Tensor,
        rotary_emb,
    ) -> torch.Tensor:
        if not _hpu_can_skip_short_indexer(self):
            return _ORIGINAL_DEEPSEEK_V4_INDEXER_FORWARD(
                self,
                hidden_states,
                qr,
                compressed_kv_score,
                indexer_weights,
                positions,
                rotary_emb,
            )

        if not _hpu_can_skip_short_indexer_cache(self):
            with dsv4_profile_stage("indexer_k_compressor"):
                self.compressor(compressed_kv_score, positions, rotary_emb)
        assert self.topk_indices_buffer is not None
        with dsv4_profile_stage("indexer_short_topk_fill"):
            return _hpu_fill_short_context_topk_indices(
                self.topk_indices_buffer,
                positions,
                self.topk_tokens,
                self.compress_ratio,
            )

    _DEEPSEEK_V4_HEAD_DIM = 512
    _DEEPSEEK_V4_FP8_DIM = 448
    _DEEPSEEK_V4_ROPE_DIM = 64
    _DEEPSEEK_V4_QUANT_BLOCK = 64
    _DEEPSEEK_V4_SCALE_DIM = 8
    _DEEPSEEK_V4_TOKEN_DATA_BYTES = 576
    _DEEPSEEK_V4_FP8_MAX = 448.0

    def _hpu_dsv4_i32_contiguous(value: torch.Tensor, ) -> torch.Tensor:
        value = value if value.dtype == torch.int32 else value.to(torch.int32)
        if not value.is_contiguous():
            value = value.contiguous()
        if (os.getenv("PT_HPU_LAZY_MODE", "0") == "1" and getattr(value, "_base", None) is not None):
            value = value.clone(memory_format=torch.contiguous_format)
        return value

    def _hpu_dsv4_preserve_i32_dependency(
        value: torch.Tensor,
        dependency: torch.Tensor | None,
    ) -> torch.Tensor:
        value = _hpu_dsv4_i32_contiguous(value)
        if dependency is None:
            return value
        dependency_i32 = _hpu_dsv4_i32_contiguous(dependency.flatten()[:value.shape[0]])
        return torch.where(
            dependency_i32 <= 1,
            value,
            torch.zeros_like(value),
        )

    def _hpu_dsv4_pad_token_metadata(
        value: torch.Tensor,
        num_tokens: int,
        pad_value: int,
    ) -> torch.Tensor:
        value = value.flatten()
        if value.shape[0] < num_tokens:
            value = torch.nn.functional.pad(
                value,
                (0, num_tokens - value.shape[0]),
                value=pad_value,
            )
        return _hpu_dsv4_i32_contiguous(value[:num_tokens])

    def _hpu_dsv4_fp32_contiguous(value: torch.Tensor, ) -> torch.Tensor:
        value = value if value.dtype == torch.float32 else value.float()
        if not value.is_contiguous():
            value = value.contiguous()
        if (os.getenv("PT_HPU_LAZY_MODE", "0") == "1" and getattr(value, "_base", None) is not None):
            value = value.clone(memory_format=torch.contiguous_format)
        return value

    def _hpu_dsv4_bf16_contiguous(value: torch.Tensor, ) -> torch.Tensor:
        value = value if value.dtype == torch.bfloat16 else value.to(torch.bfloat16)
        if not value.is_contiguous():
            value = value.contiguous()
        if (os.getenv("PT_HPU_LAZY_MODE", "0") == "1" and getattr(value, "_base", None) is not None):
            value = value.clone(memory_format=torch.contiguous_format)
        return value

    def _hpu_dsv4_flashmla_splitkv_decode(
        q: torch.Tensor,
        compressed_storage: torch.Tensor,
        compressed_geometry: torch.Tensor,
        topk_shape_buffer: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        is_valid_token: torch.Tensor,
        seq_lens: torch.Tensor,
        swa_storage: torch.Tensor,
        swa_geometry: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        attn_sink: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        split_count = envs.VLLM_HPU_DSV4_FLASHMLA_SPLITS
        if split_count < 1 or split_count > 16:
            raise ValueError("VLLM_HPU_DSV4_FLASHMLA_SPLITS must be in [1, 16]; "
                             f"got {split_count}")
        if topk_shape_buffer.shape[1] < split_count:
            raise ValueError("FlashMLA split shape exceeds top-k workspace width: "
                             f"{split_count} > {topk_shape_buffer.shape[1]}")
        split_shape_buffer = _hpu_dsv4_i32_contiguous(topk_shape_buffer[0, :split_count])
        use_tiled = envs.VLLM_HPU_DSV4_FLASHMLA_TILED
        op_name = ("custom_deepseek_v4_flashmla_splitkv_tiled_fp8_gaudi2"
                   if use_tiled else "custom_deepseek_v4_flashmla_splitkv_fp8_gaudi2")
        profile_stage = ("decode_flashmla_splitkv_tiled_fused" if use_tiled else "decode_flashmla_splitkv_fused")
        with dsv4_profile_stage(profile_stage):
            flashmla_op = getattr(torch.ops.custom_op, op_name)
            attention_output, stats, _, _ = flashmla_op(
                q.contiguous(),
                compressed_storage,
                compressed_geometry,
                _hpu_dsv4_i32_contiguous(topk_shape_buffer),
                split_shape_buffer,
                _hpu_dsv4_i32_contiguous(token_to_req_indices),
                _hpu_dsv4_i32_contiguous(block_table),
                _hpu_dsv4_i32_contiguous(is_valid_token),
                _hpu_dsv4_i32_contiguous(seq_lens),
                swa_storage,
                swa_geometry,
                _hpu_dsv4_i32_contiguous(swa_indices),
                _hpu_dsv4_i32_contiguous(swa_lens),
                _hpu_dsv4_fp32_contiguous(attn_sink),
                output,
            )
            output.copy_(attention_output)
            return stats

    def _hpu_encode_e4m3fn_bytes(values: torch.Tensor) -> torch.Tensor:
        abs_values = values.abs().clamp(max=_DEEPSEEK_V4_FP8_MAX)
        is_subnormal = abs_values < 2.0**-6
        sub_mantissa = torch.round(abs_values * 512.0).to(torch.int32).clamp(0, 8)
        sub_code = torch.where(
            sub_mantissa == 8,
            torch.full_like(sub_mantissa, 8),
            sub_mantissa,
        )

        safe_values = abs_values.clamp_min(2.0**-6)
        exponent = torch.floor(torch.log2(safe_values))
        mantissa = torch.round((safe_values / torch.exp2(exponent) - 1.0) * 8.0).to(torch.int32)
        carry = mantissa == 8
        exponent = exponent.to(torch.int32) + carry.to(torch.int32)
        mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
        exponent_bits = (exponent + 7).clamp(1, 15)
        mantissa = torch.where(
            exponent_bits == 15,
            mantissa.clamp_max(6),
            mantissa,
        )
        normal_code = exponent_bits * 8 + mantissa
        magnitude = torch.where(is_subnormal, sub_code, normal_code)
        sign = torch.signbit(values).to(torch.int32) * 128
        return (magnitude + sign).to(torch.uint8)

    def _hpu_decode_e4m3fn_bytes(encoded: torch.Tensor) -> torch.Tensor:
        code = encoded.to(torch.int32)
        sign = torch.where((code & 128) != 0, -1.0, 1.0)
        magnitude = code & 127
        exponent = magnitude >> 3
        mantissa = magnitude & 7
        subnormal = mantissa.float() * (2.0**-9)
        normal = ((1.0 + mantissa.float() / 8.0) * torch.exp2(exponent.float() - 7.0))
        return torch.where(exponent == 0, subnormal, normal) * sign

    def _hpu_storage_view(tensor: torch.Tensor) -> torch.Tensor:
        element_size = tensor.element_size()
        storage_bytes = tensor.untyped_storage().nbytes()
        if storage_bytes % element_size != 0:
            raise ValueError(f"Storage size {storage_bytes} is not aligned to "
                             f"{tensor.dtype} ({element_size} bytes)")
        return torch.as_strided(
            tensor,
            (storage_bytes // element_size, ),
            (1, ),
            0,
        )

    def _hpu_uint8_storage_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.uint8:
            raise TypeError(f"Expected a uint8 cache, got {tensor.dtype}")
        return _hpu_storage_view(tensor)

    def _hpu_apply_pairwise_rope(
        x: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        from habana_frameworks.torch.hpex.kernels import (
            RotaryPosEmbeddingMode,
            apply_rotary_pos_emb,
        )

        positions = positions.flatten()
        cos_sin = cos_sin_cache.index_select(0, positions).view(positions.numel(), 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = torch.repeat_interleave(cos, 2, dim=-1, output_size=cos_sin.shape[-1])
        sin = torch.repeat_interleave(sin, 2, dim=-1, output_size=cos_sin.shape[-1])
        return apply_rotary_pos_emb(
            x,
            cos,
            sin,
            None,
            0,
            RotaryPosEmbeddingMode.PAIRWISE,
        ).to(x.dtype)

    def _hpu_quantize_and_insert_k_cache(
        k: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int = 64,
        is_ue8m0: bool = True,
        cache_storage: torch.Tensor | None = None,
        cache_geometry: torch.Tensor | None = None,
    ) -> None:
        assert k.ndim == 2 and k.shape[1] == _DEEPSEEK_V4_HEAD_DIM
        assert k.dtype == torch.bfloat16
        assert is_ue8m0, "Only UE8M0 cache quantization is supported."

        num_tokens = slot_mapping.shape[0]
        k = k[:num_tokens]
        fp8_source = k[:, :_DEEPSEEK_V4_FP8_DIM].float().view(
            num_tokens,
            _DEEPSEEK_V4_FP8_DIM // _DEEPSEEK_V4_QUANT_BLOCK,
            _DEEPSEEK_V4_QUANT_BLOCK,
        )
        block_max = fp8_source.abs().amax(dim=-1).clamp_min(1e-4)
        exponent = torch.ceil(torch.log2(block_max / _DEEPSEEK_V4_FP8_MAX))
        scale = torch.exp2(exponent).unsqueeze(-1)
        fp8_bytes = _hpu_encode_e4m3fn_bytes((fp8_source / scale).clamp(
            -_DEEPSEEK_V4_FP8_MAX,
            _DEEPSEEK_V4_FP8_MAX,
        )).view(
            num_tokens,
            _DEEPSEEK_V4_FP8_DIM,
        )
        bf16_bytes = (k[:, _DEEPSEEK_V4_FP8_DIM:].contiguous().view(torch.uint8))
        token_data = torch.cat((fp8_bytes, bf16_bytes), dim=-1)

        encoded_scales = (exponent + 127.0).clamp(0.0, 255.0).to(torch.uint8)
        scale_data = torch.cat(
            (
                encoded_scales,
                torch.zeros(
                    (num_tokens, 1),
                    dtype=torch.uint8,
                    device=k.device,
                ),
            ),
            dim=-1,
        )

        if k_cache.ndim == 4 and k_cache.shape[1] == 1:
            k_cache = k_cache.squeeze(1)
        block_stride = k_cache.stride(0)
        if cache_storage is None:
            cache_storage = _hpu_uint8_storage_view(k_cache)
        elif cache_storage.dtype != torch.uint8 or cache_storage.ndim != 1:
            raise ValueError("Pre-bound KV cache storage must be flat uint8")
        del cache_geometry
        base_offset = k_cache.storage_offset()

        slots = slot_mapping.to(torch.int64)
        dummy_slot = k_cache.shape[0] * block_size - 1
        slots = torch.where(
            slots >= 0,
            slots,
            torch.full_like(slots, dummy_slot),
        )
        block_indices = slots.div(block_size, rounding_mode="floor")
        positions = slots.remainder(block_size)
        data_offsets = (base_offset + block_indices.unsqueeze(1) * block_stride +
                        positions.unsqueeze(1) * _DEEPSEEK_V4_TOKEN_DATA_BYTES + torch.arange(
                            _DEEPSEEK_V4_TOKEN_DATA_BYTES,
                            dtype=torch.int64,
                            device=k_cache.device,
                        ).unsqueeze(0))
        scale_offsets = (base_offset + block_indices.unsqueeze(1) * block_stride +
                         block_size * _DEEPSEEK_V4_TOKEN_DATA_BYTES + positions.unsqueeze(1) * _DEEPSEEK_V4_SCALE_DIM +
                         torch.arange(
                             _DEEPSEEK_V4_SCALE_DIM,
                             dtype=torch.int64,
                             device=k_cache.device,
                         ).unsqueeze(0))
        cache_storage[data_offsets] = token_data
        cache_storage[scale_offsets] = scale_data

    def _hpu_dequantize_and_gather_k_cache(
        out: torch.Tensor,
        k_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: torch.Tensor | None,
        block_table: torch.Tensor,
        block_size: int,
        offset: int,
    ) -> None:
        num_reqs = seq_lens.shape[0]
        if num_reqs == 0:
            return

        if k_cache.ndim == 4 and k_cache.shape[1] == 1:
            k_cache = k_cache.squeeze(1)
        block_stride = k_cache.stride(0)
        cache_storage = _hpu_uint8_storage_view(k_cache)
        base_offset = k_cache.storage_offset()
        gather_lens = seq_lens if gather_lens is None else gather_lens
        max_tokens = min(
            out.shape[1] - offset,
            block_table.shape[1] * block_size,
        )
        if max_tokens <= 0:
            return

        token_offsets = torch.arange(
            max_tokens,
            dtype=torch.int64,
            device=seq_lens.device,
        )
        gather_lens_i64 = gather_lens.to(torch.int64)
        positions = (seq_lens.to(torch.int64).unsqueeze(1) - gather_lens_i64.unsqueeze(1) + token_offsets.unsqueeze(0))
        valid = token_offsets.unsqueeze(0) < gather_lens_i64.unsqueeze(1)
        safe_positions = positions.clamp_min(0)
        logical_blocks = (safe_positions.div(block_size, rounding_mode="floor").clamp_max(block_table.shape[1] - 1))
        block_numbers = torch.gather(
            block_table[:num_reqs],
            1,
            logical_blocks,
        ).to(torch.int64)
        valid = valid & (block_numbers >= 0)
        block_numbers = block_numbers.clamp_min(0)
        positions_in_block = safe_positions.remainder(block_size)

        data_offsets = (base_offset + block_numbers.unsqueeze(-1) * block_stride +
                        positions_in_block.unsqueeze(-1) * _DEEPSEEK_V4_TOKEN_DATA_BYTES + torch.arange(
                            _DEEPSEEK_V4_TOKEN_DATA_BYTES,
                            dtype=torch.int64,
                            device=k_cache.device,
                        ))
        scale_offsets = (base_offset + block_numbers.unsqueeze(-1) * block_stride +
                         block_size * _DEEPSEEK_V4_TOKEN_DATA_BYTES +
                         positions_in_block.unsqueeze(-1) * _DEEPSEEK_V4_SCALE_DIM + torch.arange(
                             _DEEPSEEK_V4_SCALE_DIM,
                             dtype=torch.int64,
                             device=k_cache.device,
                         ))
        token_data = cache_storage[data_offsets]
        scale_data = cache_storage[scale_offsets]

        fp8_values = _hpu_decode_e4m3fn_bytes(token_data[..., :_DEEPSEEK_V4_FP8_DIM]).view(
            num_reqs,
            max_tokens,
            _DEEPSEEK_V4_FP8_DIM // _DEEPSEEK_V4_QUANT_BLOCK,
            _DEEPSEEK_V4_QUANT_BLOCK,
        )
        scales = torch.exp2(scale_data[..., :-1].float() - 127.0).unsqueeze(-1)
        fp8_values = (fp8_values * scales).to(torch.bfloat16).view(
            num_reqs,
            max_tokens,
            _DEEPSEEK_V4_FP8_DIM,
        )
        bf16_values = (token_data[..., _DEEPSEEK_V4_FP8_DIM:].contiguous().view(torch.bfloat16))
        gathered = torch.cat((fp8_values, bf16_values), dim=-1)

        target = out[:num_reqs, offset:offset + max_tokens]
        target.copy_(torch.where(valid.unsqueeze(-1), gathered, target))

    def _hpu_dequant_gather_slots(
        out: torch.Tensor,
        cache: torch.Tensor,
        indices: torch.Tensor,
        cache_block_size: int,
        cache_storage: torch.Tensor | None = None,
        cache_geometry: torch.Tensor | None = None,
    ) -> None:
        """Gather fp8_ds_mla slots without relying on a Triton runtime."""
        total_slots = indices.shape[0]
        if total_slots == 0:
            return

        if cache.ndim == 4 and cache.shape[1] == 1:
            cache = cache.squeeze(1)
        if cache.ndim != 3:
            raise ValueError("fp8_ds_mla cache must have shape [blocks, tokens, bytes], "
                             f"got {tuple(cache.shape)}")

        if envs.VLLM_HPU_DSV4_TPC_DEQUANT_GATHER:
            _ensure_dsv4_tpc_ops_loaded()
            if cache_storage is None or cache_geometry is None:
                raise ValueError("DeepSeek V4 TPC decode requires pre-bound cache storage "
                                 "and geometry")
            gathered = (torch.ops.custom_op.custom_deepseek_v4_dequant_gather_bf16_gaudi2(
                cache_storage,
                cache_geometry,
                _hpu_dsv4_i32_contiguous(indices),
            ))
            out.copy_(gathered)
            return

        required_page_bytes = cache_block_size * (_DEEPSEEK_V4_TOKEN_DATA_BYTES + _DEEPSEEK_V4_SCALE_DIM)
        page_bytes = cache.shape[1] * cache.shape[2]
        if page_bytes < required_page_bytes:
            raise ValueError(f"fp8_ds_mla page has {page_bytes} bytes, "
                             f"requires at least {required_page_bytes}")
        slots = indices.to(torch.int64)
        valid = (slots >= 0) & (slots < cache.shape[0] * cache_block_size)
        safe_slots = slots.clamp(
            min=0,
            max=cache.shape[0] * cache_block_size - 1,
        )
        block_indices = safe_slots.div(cache_block_size, rounding_mode="floor")
        positions = safe_slots.remainder(cache_block_size)
        block_stride = cache.stride(0)
        if cache_storage is None:
            cache_storage = _hpu_uint8_storage_view(cache)
        base_offset = cache.storage_offset()

        data_columns = torch.arange(
            _DEEPSEEK_V4_TOKEN_DATA_BYTES,
            dtype=torch.int64,
            device=cache.device,
        )
        data_offsets = (base_offset + block_indices.unsqueeze(1) * block_stride +
                        positions.unsqueeze(1) * _DEEPSEEK_V4_TOKEN_DATA_BYTES + data_columns.unsqueeze(0))
        token_data = cache_storage[data_offsets]

        scale_columns = torch.arange(
            _DEEPSEEK_V4_SCALE_DIM,
            dtype=torch.int64,
            device=cache.device,
        )
        scale_offsets = (base_offset + block_indices.unsqueeze(1) * block_stride +
                         cache_block_size * _DEEPSEEK_V4_TOKEN_DATA_BYTES +
                         positions.unsqueeze(1) * _DEEPSEEK_V4_SCALE_DIM + scale_columns.unsqueeze(0))
        scale_data = cache_storage[scale_offsets]

        fp8_values = _hpu_decode_e4m3fn_bytes(token_data[:, :_DEEPSEEK_V4_FP8_DIM]).view(
            total_slots,
            _DEEPSEEK_V4_FP8_DIM // _DEEPSEEK_V4_QUANT_BLOCK,
            _DEEPSEEK_V4_QUANT_BLOCK,
        )
        scales = torch.exp2(scale_data[:, :-1].float() - 127.0).unsqueeze(-1)
        fp8_values = (fp8_values * scales).to(torch.bfloat16).view(total_slots, _DEEPSEEK_V4_FP8_DIM)
        bf16_values = (token_data[:, _DEEPSEEK_V4_FP8_DIM:].contiguous().view(torch.bfloat16))
        gathered = torch.cat((fp8_values, bf16_values), dim=-1)
        out.copy_(torch.where(
            valid.unsqueeze(-1),
            gathered,
            torch.zeros_like(gathered),
        ))

    def _hpu_get_masked_input_and_mask(
        input_: torch.Tensor,
        org_vocab_start_index: int,
        org_vocab_end_index: int,
        num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
        added_vocab_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
        added_offset = (added_vocab_start_index - (org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding)
        valid_offset = org_vocab_start_index * org_vocab_mask + (added_offset * added_vocab_mask)
        vocab_mask = org_vocab_mask | added_vocab_mask
        masked_input = vocab_mask * (input_ - valid_offset)
        return masked_input, ~vocab_mask

    def _hpu_compute_global_topk_indices_and_lens(
        topk_indices: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        is_valid_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = topk_indices.shape[0]
        local_indices = topk_indices.to(torch.int64)
        valid = local_indices >= 0
        safe_local = local_indices.clamp_min(0)
        logical_blocks = safe_local.div(block_size, rounding_mode="floor").clamp_max(block_table.shape[1] - 1)
        request_indices = token_to_req_indices[:num_tokens].to(torch.int64)
        request_tables = block_table.index_select(0, request_indices)
        block_numbers = torch.gather(
            request_tables,
            1,
            logical_blocks,
        ).to(torch.int64)
        slot_ids = (block_numbers * block_size + safe_local.remainder(block_size))
        global_indices = torch.where(
            valid,
            slot_ids.to(topk_indices.dtype),
            torch.full_like(topk_indices, -1),
        )
        topk_lens = valid.sum(dim=-1).to(torch.int32)
        topk_lens = torch.where(
            is_valid_token[:num_tokens].to(torch.bool),
            topk_lens,
            torch.zeros_like(topk_lens),
        )
        return global_indices, topk_lens

    def _hpu_combine_topk_swa_indices(
        topk_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: torch.Tensor,
        window_size: int,
        compress_ratio: int,
        topk: int,
        M: int,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_indices = topk_indices.reshape(topk_indices.shape[0], -1)
        num_tokens = topk_indices.shape[0]
        num_reqs = seq_lens.shape[0]
        combined_topk = ((topk + window_size + 127) // 128 * 128)
        if topk > topk_indices.shape[1]:
            raise ValueError(f"topk={topk} exceeds index width "
                             f"{topk_indices.shape[1]}")

        query_start_loc = query_start_loc[:num_reqs + 1].to(torch.int64)
        query_start_loc = query_start_loc - query_start_loc[:1]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        request_indices = torch.repeat_interleave(
            torch.arange(
                num_reqs,
                dtype=torch.int64,
                device=topk_indices.device,
            ),
            query_lens,
            output_size=num_tokens,
        )
        token_indices = torch.arange(
            num_tokens,
            dtype=torch.int64,
            device=topk_indices.device,
        )
        positions = (seq_lens[request_indices].to(torch.int64) - query_lens[request_indices] + token_indices -
                     query_start_loc[request_indices])
        topk_lens = ((positions + 1).div(compress_ratio, rounding_mode="floor").clamp(max=topk))
        swa_lens = (positions + 1).clamp(max=window_size)
        gather_starts = (seq_lens[request_indices].to(torch.int64) - gather_lens[request_indices].to(torch.int64))

        columns = torch.arange(
            combined_topk,
            dtype=torch.int64,
            device=topk_indices.device,
        ).unsqueeze(0)
        topk_mask = columns < topk_lens.unsqueeze(1)
        if topk_indices.shape[1] > 0:
            safe_columns = columns.clamp(max=topk_indices.shape[1] - 1).expand(num_tokens, -1)
            topk_values = torch.gather(
                topk_indices,
                1,
                safe_columns,
            ).to(torch.int64)
            topk_values = (topk_values + request_indices.unsqueeze(1) * M)
        else:
            topk_values = torch.zeros(
                (num_tokens, combined_topk),
                dtype=torch.int64,
                device=topk_indices.device,
            )

        swa_offsets = columns - topk_lens.unsqueeze(1)
        swa_mask = ((swa_offsets >= 0) & (swa_offsets < swa_lens.unsqueeze(1)))
        swa_values = (request_indices.unsqueeze(1) * M + N + swa_offsets + positions.unsqueeze(1) -
                      swa_lens.unsqueeze(1) + 1 - gather_starts.unsqueeze(1))
        combined_indices = torch.where(
            topk_mask,
            topk_values,
            torch.where(
                swa_mask,
                swa_values,
                torch.full_like(columns, -1),
            ),
        ).to(torch.int32)
        combined_lens = (topk_lens + swa_lens).to(torch.int32)
        return combined_indices, combined_lens

    def _hpu_bf16_mla_sparse_interface(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        sm_scale: float,
        d_v: int = 512,
        block_dpe: int = 64,
        attn_sink: torch.Tensor | None = None,
        topk_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del block_dpe
        num_tokens, num_heads_q, dim_qk = q.shape
        seq_kv, num_heads_kv, kv_dim = kv.shape
        if dim_qk != kv_dim:
            raise ValueError(f"q dim {dim_qk} does not match kv dim {kv_dim}")
        if num_heads_kv != 1:
            raise ValueError(f"Only one KV head is supported, got {num_heads_kv}")
        if d_v > kv_dim:
            raise ValueError(f"d_v={d_v} exceeds kv dim {kv_dim}")
        if indices.shape[:2] != (num_tokens, num_heads_kv):
            raise ValueError("indices must have shape "
                             f"[{num_tokens}, {num_heads_kv}, topk]")

        if topk_length is not None:
            if topk_length.ndim != 1 or topk_length.shape[0] != num_tokens:
                raise ValueError("topk_length must have shape "
                                 f"[{num_tokens}], got {tuple(topk_length.shape)}")
            if (envs.VLLM_HPU_DSV4_FLASHMLA_PREFILL and attn_sink is not None and dim_qk == 512 and d_v == 512
                    and indices.shape[-1] <= envs.VLLM_HPU_DSV4_FLASHMLA_PREFILL_MAX_WIDTH):
                _ensure_dsv4_tpc_ops_loaded()
                length_op = getattr(
                    torch.ops.custom_op,
                    "custom_deepseek_v4_sparse_attn_bf16_lengths_gaudi2",
                    None,
                )
                if length_op is not None:
                    with dsv4_profile_stage("prefill_flashmla_sparse_tpc"):
                        scale = torch.full(
                            (1, ),
                            sm_scale,
                            dtype=torch.float32,
                            device=q.device,
                        )
                        return length_op(
                            q.contiguous(),
                            kv[:, 0].contiguous(),
                            indices[:, 0].to(torch.int32).contiguous(),
                            attn_sink[:num_heads_q].float().contiguous(),
                            scale,
                            topk_length.to(torch.int32).contiguous(),
                        )

        if (envs.VLLM_HPU_DSV4_TPC_SPARSE_ATTN and attn_sink is not None and dim_qk == 512 and d_v == 512
                and indices.shape[-1] <= envs.VLLM_HPU_DSV4_TPC_SPARSE_ATTN_MAX_WIDTH):
            _ensure_dsv4_tpc_ops_loaded()
            with dsv4_profile_stage("sparse_tpc"):
                scale = torch.full((1, ), sm_scale, dtype=torch.float32, device=q.device)
                return (torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(
                    q.contiguous(),
                    kv[:, 0].contiguous(),
                    indices[:, 0].to(torch.int32).contiguous(),
                    attn_sink[:num_heads_q].float().contiguous(),
                    scale,
                ))

        with dsv4_profile_stage("sparse_select_kv"):
            sparse_indices = indices[:, 0].to(torch.int64)
            valid = (sparse_indices >= 0) & (sparse_indices < seq_kv)
            safe_indices = sparse_indices.clamp(min=0, max=max(seq_kv - 1, 0))
            selected_kv = torch.index_select(
                kv[:, 0],
                0,
                safe_indices.flatten(),
            ).view(
                num_tokens,
                sparse_indices.shape[1],
                dim_qk,
            )
        if envs.VLLM_HPU_DSV4_FUSED_SDPA:
            with dsv4_profile_stage("sparse_fused_sdpa"):
                attention_bias = torch.where(
                    valid.view(num_tokens, 1, 1, sparse_indices.shape[1]),
                    torch.zeros((), dtype=q.dtype, device=q.device),
                    torch.full((), -float("inf"), dtype=q.dtype, device=q.device),
                )
                out, sdpa_max, inv_sumexp, _ = (torch.ops.hpu.sdpa_recomp_fwd(
                    q.unsqueeze(2),
                    selected_kv.unsqueeze(1),
                    selected_kv[..., :d_v].unsqueeze(1),
                    attention_bias,
                    0.0,
                    sm_scale,
                    False,
                    True,
                    "fp32",
                    None,
                    "right",
                ))
                kv_lse = sdpa_max - torch.log(inv_sumexp)
                if attn_sink is None:
                    merged_lse = kv_lse
                    max_logits = sdpa_max
                else:
                    sink = attn_sink[:num_heads_q].float().view(1, num_heads_q, 1, 1)
                    merged_lse = torch.logaddexp(kv_lse, sink)
                    max_logits = torch.maximum(sdpa_max, sink)
                    out = out * torch.exp(kv_lse - merged_lse).to(out.dtype)

                has_valid = valid.any(dim=-1).view(num_tokens, 1, 1, 1)
                out = torch.where(has_valid, out, torch.zeros_like(out))
                if attn_sink is not None:
                    merged_lse = torch.where(has_valid, merged_lse, sink.expand_as(merged_lse))
                    max_logits = torch.where(has_valid, max_logits, sink.expand_as(max_logits))
                else:
                    invalid_stat = torch.full_like(merged_lse, -float("inf"))
                    merged_lse = torch.where(has_valid, merged_lse, invalid_stat)
                    max_logits = torch.where(has_valid, max_logits, invalid_stat)
                return (
                    out.squeeze(2),
                    max_logits.squeeze(-1).squeeze(-1),
                    merged_lse.squeeze(-1).squeeze(-1),
                )

        with dsv4_profile_stage("sparse_qk"):
            logits = torch.matmul(
                q,
                selected_kv.transpose(1, 2),
            ).float()
        with dsv4_profile_stage("sparse_scale_mask"):
            logits = logits * sm_scale
            logits = torch.where(
                valid.unsqueeze(1),
                logits,
                torch.full_like(logits, -float("inf")),
            )

        with dsv4_profile_stage("sparse_sink"):
            if attn_sink is None:
                logits_with_sink = logits
            else:
                sink = attn_sink[:num_heads_q].float().view(1, num_heads_q, 1)
                logits_with_sink = torch.cat((logits, sink.expand(num_tokens, -1, -1)), dim=-1)

        with dsv4_profile_stage("sparse_stats"):
            max_logits = logits_with_sink.amax(dim=-1)
            softmax_lse = torch.logsumexp(logits_with_sink, dim=-1)
        with dsv4_profile_stage("sparse_softmax"):
            probabilities = torch.softmax(logits_with_sink, dim=-1,
                                          dtype=torch.float32)[..., :sparse_indices.shape[1]].to(q.dtype)
        with dsv4_profile_stage("sparse_pv"):
            out = torch.matmul(
                probabilities,
                selected_kv[..., :d_v],
            ).view(num_tokens, num_heads_q, d_v)
        return out, max_logits, softmax_lse

    def _hpu_qnorm_rope_kv_fp8_insert(
        q: torch.Tensor,
        kv: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        eps: float,
        block_size: int,
        swa_kv_cache_storage: torch.Tensor | None = None,
        swa_kv_cache_geometry: torch.Tensor | None = None,
    ) -> None:
        assert q.ndim == 3 and q.shape[-1] == _DEEPSEEK_V4_HEAD_DIM
        assert kv.ndim == 2 and kv.shape[-1] == _DEEPSEEK_V4_HEAD_DIM

        tpc_requested = envs.VLLM_HPU_DSV4_TPC_QNORM_ROPE_KV_PACK
        compile_only = tpc_requested and _hpu_dsv4_is_compile_only()
        if (tpc_requested
                # Synapse cannot materialize this mutating op as a cold recipe in
                # compile-only mode. Once an eager prewarm has compiled the exact
                # cache geometry, keep capture and replay on the same TPC path.
                and (not compile_only or _DSV4_QNORM_TPC_PREWARMED) and swa_kv_cache_storage is not None and
                swa_kv_cache_geometry is not None and eps == 1e-6):
            _log_hpu_dsv4_qnorm_path_once("tpc")
            _ensure_dsv4_tpc_ops_loaded()
            num_tokens = q.shape[0]
            # Prompt graphs pad Q to the static bucket while framework
            # metadata retains only real tokens. Invalid slots keep the
            # extra rows from mutating KV cache; position zero is safe for
            # those otherwise-unused Q rows.
            positions_actual = _hpu_dsv4_pad_token_metadata(
                positions,
                num_tokens,
                0,
            )
            slots_actual = _hpu_dsv4_pad_token_metadata(
                slot_mapping,
                num_tokens,
                -1,
            )
            if envs.VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN:
                _mutation_token = (torch.ops.custom_op.custom_deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2(
                    q,
                    kv,
                    swa_kv_cache_storage,
                    swa_kv_cache_geometry,
                    slots_actual,
                    positions_actual,
                    cos_sin_cache.contiguous(),
                ))
            else:
                _mutation_token = (torch.ops.custom_op.custom_deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2(
                    q,
                    kv,
                    swa_kv_cache_storage,
                    swa_kv_cache_geometry,
                    slots_actual,
                    positions_actual,
                    cos_sin_cache.contiguous(),
                ))
            return

        if tpc_requested:
            _log_hpu_dsv4_qnorm_path_once(
                "stock_compile_only_cold" if compile_only and not _DSV4_QNORM_TPC_PREWARMED else "stock_ineligible")

        q_fp32 = q.float()
        q_normed = (q_fp32 * torch.rsqrt(q_fp32.square().mean(dim=-1, keepdim=True) + eps))
        q_roped = _hpu_apply_pairwise_rope(
            q_normed[..., _DEEPSEEK_V4_FP8_DIM:],
            positions,
            cos_sin_cache,
        )
        q.copy_(torch.cat(
            (
                q_normed[..., :_DEEPSEEK_V4_FP8_DIM],
                q_roped,
            ),
            dim=-1,
        ).to(q.dtype))

        kv_roped = _hpu_apply_pairwise_rope(
            kv[:, _DEEPSEEK_V4_FP8_DIM:].view(kv.shape[0], 1, _DEEPSEEK_V4_ROPE_DIM),
            positions,
            cos_sin_cache,
        ).view(kv.shape[0], _DEEPSEEK_V4_ROPE_DIM)
        kv_roped = torch.cat(
            (kv[:, :_DEEPSEEK_V4_FP8_DIM], kv_roped),
            dim=-1,
        )
        _hpu_quantize_and_insert_k_cache(
            kv_roped,
            swa_kv_cache,
            slot_mapping,
            block_size=block_size,
            cache_storage=swa_kv_cache_storage,
            cache_geometry=swa_kv_cache_geometry,
        )

    def _hpu_mme_sparse_decode_fp8_result(
        q: torch.Tensor,
        kv_cache_storage: torch.Tensor | None,
        kv_cache_geometry: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
        topk_lens: torch.Tensor | None,
        swa_kv_cache_storage: torch.Tensor,
        swa_kv_cache_geometry: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        attn_sink: torch.Tensor,
        softmax_scale: float,
        local_topk_shape_buffer: torch.Tensor | None = None,
        local_topk_token_to_req_indices: torch.Tensor | None = None,
        local_topk_block_table: torch.Tensor | None = None,
        local_topk_is_valid: torch.Tensor | None = None,
        local_topk_seq_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        swa_idx_2d = (swa_indices.squeeze(1) if swa_indices.dim() == 3 else swa_indices)
        swa_slots = _hpu_dsv4_i32_contiguous(swa_idx_2d[0])
        topk_slots = None
        if topk_indices is not None:
            topk_idx_2d = (topk_indices.squeeze(1) if topk_indices.dim() == 3 else topk_indices)
            topk_slots = _hpu_dsv4_i32_contiguous(topk_idx_2d[0])

        local_topk_inputs = (
            local_topk_shape_buffer,
            local_topk_token_to_req_indices,
            local_topk_block_table,
            local_topk_is_valid,
            local_topk_seq_lens,
        )
        has_local_topk = all(value is not None for value in local_topk_inputs)
        if topk_slots is None:
            selected = (torch.ops.custom_op.custom_deepseek_v4_dequant_gather_bf16_gaudi2(
                swa_kv_cache_storage,
                swa_kv_cache_geometry,
                swa_slots,
            ))
        elif has_local_topk:
            assert kv_cache_storage is not None
            assert kv_cache_geometry is not None
            assert local_topk_shape_buffer is not None
            assert local_topk_token_to_req_indices is not None
            assert local_topk_block_table is not None
            assert local_topk_is_valid is not None
            assert local_topk_seq_lens is not None
            selected = (torch.ops.custom_op.custom_deepseek_v4_local_dual_dequant_gather_bf16_gaudi2(
                kv_cache_storage,
                kv_cache_geometry,
                _hpu_dsv4_i32_contiguous(local_topk_shape_buffer[0]),
                _hpu_dsv4_i32_contiguous(local_topk_token_to_req_indices),
                _hpu_dsv4_i32_contiguous(local_topk_block_table),
                _hpu_dsv4_i32_contiguous(local_topk_is_valid),
                _hpu_dsv4_i32_contiguous(local_topk_seq_lens),
                swa_kv_cache_storage,
                swa_kv_cache_geometry,
                swa_slots,
            ))
        else:
            assert kv_cache_storage is not None
            assert kv_cache_geometry is not None
            selected = (torch.ops.custom_op.custom_deepseek_v4_dual_dequant_gather_bf16_gaudi2(
                kv_cache_storage,
                kv_cache_geometry,
                topk_slots,
                swa_kv_cache_storage,
                swa_kv_cache_geometry,
                swa_slots,
            ))

        swa_columns = torch.arange(
            swa_slots.shape[0],
            dtype=torch.int32,
            device=q.device,
        )
        swa_valid = swa_columns < swa_lens[0]
        if topk_slots is None:
            valid = swa_valid
        else:
            assert topk_lens is not None
            topk_columns = torch.arange(
                topk_slots.shape[0],
                dtype=torch.int32,
                device=q.device,
            )
            valid = torch.cat((topk_columns < topk_lens[0], swa_valid), dim=0)
        attention_bias = torch.where(
            valid.view(1, 1, 1, -1),
            torch.zeros((), dtype=q.dtype, device=q.device),
            torch.full((), -float("inf"), dtype=q.dtype, device=q.device),
        )
        result, _, _, _ = torch.ops.hpu.sdpa_recomp_fwd(
            q.unsqueeze(2),
            selected.view(1, 1, selected.shape[0], q.shape[2]),
            selected.view(1, 1, selected.shape[0], q.shape[2]),
            attention_bias,
            0.0,
            softmax_scale,
            False,
            False,
            "fp32",
            None,
            "right",
            [-1, -1],
            attn_sink[:q.shape[1]].float().contiguous(),
        )
        return result.squeeze(2)

    def _hpu_sparse_decode_fp8(
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        kv_cache_storage: torch.Tensor | None,
        kv_cache_geometry: torch.Tensor | None,
        swa_kv_cache: torch.Tensor,
        swa_kv_cache_storage: torch.Tensor,
        swa_kv_cache_geometry: torch.Tensor,
        swa_only: bool,
        topk_indices: torch.Tensor | None,
        topk_lens: torch.Tensor | None,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        attn_sink: torch.Tensor,
        softmax_scale: float,
        head_dim: int,
        nope_head_dim: int,
        rope_head_dim: int,
        out: torch.Tensor,
        local_topk_token_to_req_indices: torch.Tensor | None = None,
        local_topk_block_table: torch.Tensor | None = None,
        local_topk_is_valid: torch.Tensor | None = None,
        local_topk_block_size: int | None = None,
        local_topk_seq_lens: torch.Tensor | None = None,
    ) -> None:
        expected_scale = _DEEPSEEK_V4_HEAD_DIM**-0.5
        local_topk_inputs = (
            local_topk_token_to_req_indices,
            local_topk_block_table,
            local_topk_is_valid,
        )
        has_local_topk = all(value is not None for value in local_topk_inputs)
        # Match FlashMLA's decode contract: the query can be unpadded while
        # the caller-owned output workspace is padded for another backend.
        # Only the first q.shape[1] heads are consumed by the output
        # projection, and the TPC glue already accepts output_head_count >=
        # query_head_count.  Keep the packed-cache path ahead of the gathered
        # MME fallback so enabling both capabilities selects the FlashMLA-like
        # direct path rather than materializing a global BF16 K/V tensor.
        tpc_eligible = (not swa_only and q.shape[0] == 1 and q.dtype == torch.bfloat16
                        and q.shape[-1] == _DEEPSEEK_V4_HEAD_DIM and head_dim == _DEEPSEEK_V4_HEAD_DIM
                        and nope_head_dim == _DEEPSEEK_V4_FP8_DIM and rope_head_dim == _DEEPSEEK_V4_ROPE_DIM
                        and abs(softmax_scale - expected_scale) < 1e-12 and kv_cache_storage is not None
                        and kv_cache_geometry is not None and topk_indices is not None
                        and (has_local_topk or topk_lens is not None) and flashmla_output_buffer_is_compatible(q, out))
        mme_eligible = (
            envs.VLLM_HPU_DSV4_TPC_QNORM_ROPE_KV_PACK and q.shape[0] == 1 and q.dtype == torch.bfloat16
            and q.shape[-1] == _DEEPSEEK_V4_HEAD_DIM and head_dim == _DEEPSEEK_V4_HEAD_DIM
            and nope_head_dim == _DEEPSEEK_V4_FP8_DIM and rope_head_dim == _DEEPSEEK_V4_ROPE_DIM
            and abs(softmax_scale - expected_scale) < 1e-12 and flashmla_output_buffer_is_compatible(q, out)
            and (swa_only or
                 (kv_cache_storage is not None and kv_cache_geometry is not None and topk_indices is not None and
                  (has_local_topk or topk_lens is not None))))
        decode_plan = build_hpu_flashmla_decode_plan(
            requested_backend=envs.VLLM_HPU_DSV4_ATTENTION_BACKEND,
            swa_only=swa_only,
            has_local_topk=has_local_topk,
            uses_sequential_topk=local_topk_seq_lens is not None,
            tpc_enabled=envs.VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN,
            mme_enabled=envs.VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN,
            tpc_eligible=tpc_eligible,
            mme_eligible=mme_eligible,
        )
        use_paged_tpc = (decode_plan.backend == HPUFlashMLABackend.FLASHMLA)
        use_mme_paged = decode_plan.backend == HPUFlashMLABackend.MME
        if use_mme_paged:
            _ensure_dsv4_tpc_ops_loaded()
            local_topk_shape_buffer = None
            if has_local_topk:
                assert topk_indices is not None
                assert local_topk_token_to_req_indices is not None
                assert local_topk_block_table is not None
                assert local_topk_is_valid is not None
                assert local_topk_block_size is not None
                local_topk_2d = (topk_indices.squeeze(1) if topk_indices.dim() == 3 else topk_indices)
                max_mappable_width = (local_topk_block_table.shape[1] * local_topk_block_size)
                local_topk_2d = local_topk_2d[:, :max_mappable_width]
                if local_topk_seq_lens is not None:
                    request_indices = (local_topk_token_to_req_indices.to(torch.int64))
                    compressed_lens = (local_topk_seq_lens.index_select(0, request_indices).to(local_topk_2d.dtype) //
                                       4)
                    compressed_lens = compressed_lens.clamp(max=local_topk_2d.shape[1])
                    topk_lens = torch.where(
                        local_topk_is_valid.to(torch.bool),
                        compressed_lens,
                        torch.zeros_like(compressed_lens),
                    ).to(torch.int32)
                    local_topk_shape_buffer = local_topk_2d
                    topk_indices = local_topk_2d.view(local_topk_2d.shape[0], 1, -1)
                else:
                    mapped_indices, topk_lens = (_hpu_compute_global_topk_indices_and_lens(
                        local_topk_2d,
                        local_topk_token_to_req_indices,
                        local_topk_block_table,
                        local_topk_block_size,
                        local_topk_is_valid,
                    ))
                    topk_indices = mapped_indices.view(mapped_indices.shape[0], 1, -1)
            with dsv4_profile_stage("decode_mme_math"):
                mme_result = _hpu_mme_sparse_decode_fp8_result(
                    q=q,
                    kv_cache_storage=kv_cache_storage,
                    kv_cache_geometry=kv_cache_geometry,
                    topk_indices=None if swa_only else topk_indices,
                    topk_lens=None if swa_only else topk_lens,
                    swa_kv_cache_storage=swa_kv_cache_storage,
                    swa_kv_cache_geometry=swa_kv_cache_geometry,
                    swa_indices=swa_indices,
                    swa_lens=swa_lens,
                    attn_sink=attn_sink,
                    softmax_scale=softmax_scale,
                    local_topk_shape_buffer=local_topk_shape_buffer,
                    local_topk_token_to_req_indices=(local_topk_token_to_req_indices
                                                     if local_topk_shape_buffer is not None else None),
                    local_topk_block_table=(local_topk_block_table if local_topk_shape_buffer is not None else None),
                    local_topk_is_valid=(local_topk_is_valid if local_topk_shape_buffer is not None else None),
                    local_topk_seq_lens=(local_topk_seq_lens if local_topk_shape_buffer is not None else None),
                )
            with dsv4_profile_stage("decode_mme_output_copy"):
                out[:, :q.shape[1]].copy_(mme_result)
                if out.shape[1] > q.shape[1]:
                    out[:, q.shape[1]:].zero_()
            return

        if use_paged_tpc:
            _ensure_dsv4_tpc_ops_loaded()
            topk_idx_2d = (topk_indices.squeeze(1) if topk_indices.dim() == 3 else topk_indices)
            swa_idx_2d = (swa_indices.squeeze(1) if swa_indices.dim() == 3 else swa_indices)
            if has_local_topk:
                assert local_topk_token_to_req_indices is not None
                assert local_topk_block_table is not None
                assert local_topk_is_valid is not None
                if local_topk_seq_lens is not None:
                    with dsv4_profile_stage("decode_paged_sequential_fp8_tpc"):
                        if envs.VLLM_HPU_DSV4_FLASHMLA_SPLIT_KV:
                            _score_debug = _hpu_dsv4_flashmla_splitkv_decode(
                                q,
                                kv_cache_storage,
                                kv_cache_geometry,
                                topk_idx_2d,
                                local_topk_token_to_req_indices,
                                local_topk_block_table,
                                local_topk_is_valid,
                                local_topk_seq_lens,
                                swa_kv_cache_storage,
                                swa_kv_cache_geometry,
                                swa_idx_2d,
                                swa_lens,
                                attn_sink,
                                out,
                            )
                        else:
                            sequential_op = (
                                torch.ops.custom_op.custom_deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2
                                if envs.VLLM_HPU_DSV4_TPC_PAIR_HEADS and q.shape[1] % 2 == 0 else
                                torch.ops.custom_op.custom_deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2)
                            _score_debug = sequential_op(
                                q.contiguous(),
                                kv_cache_storage,
                                kv_cache_geometry,
                                _hpu_dsv4_i32_contiguous(topk_idx_2d),
                                _hpu_dsv4_i32_contiguous(local_topk_token_to_req_indices),
                                _hpu_dsv4_i32_contiguous(local_topk_block_table),
                                _hpu_dsv4_i32_contiguous(local_topk_is_valid),
                                _hpu_dsv4_i32_contiguous(local_topk_seq_lens),
                                swa_kv_cache_storage,
                                swa_kv_cache_geometry,
                                _hpu_dsv4_i32_contiguous(swa_idx_2d),
                                _hpu_dsv4_i32_contiguous(swa_lens),
                                _hpu_dsv4_fp32_contiguous(attn_sink),
                                out,
                            )
                else:
                    with dsv4_profile_stage("decode_paged_local_fp8_tpc"):
                        _score_debug = (torch.ops.custom_op.custom_deepseek_v4_paged_sparse_attn_local_fp8_gaudi2(
                            q.contiguous(),
                            kv_cache_storage,
                            kv_cache_geometry,
                            _hpu_dsv4_i32_contiguous(topk_idx_2d),
                            _hpu_dsv4_i32_contiguous(local_topk_token_to_req_indices),
                            _hpu_dsv4_i32_contiguous(local_topk_block_table),
                            _hpu_dsv4_i32_contiguous(local_topk_is_valid),
                            swa_kv_cache_storage,
                            swa_kv_cache_geometry,
                            _hpu_dsv4_i32_contiguous(swa_idx_2d),
                            _hpu_dsv4_i32_contiguous(swa_lens),
                            _hpu_dsv4_fp32_contiguous(attn_sink),
                            out,
                        ))
            else:
                assert topk_lens is not None
                with dsv4_profile_stage("decode_paged_fp8_tpc"):
                    _score_debug = (torch.ops.custom_op.custom_deepseek_v4_paged_sparse_attn_fp8_gaudi2(
                        q.contiguous(),
                        kv_cache_storage,
                        kv_cache_geometry,
                        _hpu_dsv4_i32_contiguous(topk_idx_2d),
                        _hpu_dsv4_i32_contiguous(topk_lens),
                        swa_kv_cache_storage,
                        swa_kv_cache_geometry,
                        _hpu_dsv4_i32_contiguous(swa_idx_2d),
                        _hpu_dsv4_i32_contiguous(swa_lens),
                        _hpu_dsv4_fp32_contiguous(attn_sink),
                        out,
                    ))
            return

        if has_local_topk:
            assert topk_indices is not None
            assert local_topk_token_to_req_indices is not None
            assert local_topk_block_table is not None
            assert local_topk_is_valid is not None
            assert local_topk_block_size is not None
            local_topk_2d = (topk_indices.squeeze(1) if topk_indices.dim() == 3 else topk_indices)
            if local_topk_seq_lens is not None:
                request_indices = local_topk_token_to_req_indices.to(torch.int64)
                compressed_lens = local_topk_seq_lens.index_select(0, request_indices).to(local_topk_2d.dtype) // 4
                offsets = torch.arange(
                    local_topk_2d.shape[1],
                    dtype=local_topk_2d.dtype,
                    device=local_topk_2d.device,
                ).view(1, -1)
                local_topk_2d = torch.where(
                    offsets < compressed_lens.view(-1, 1),
                    offsets,
                    torch.full_like(offsets, -1),
                )
            mapped_indices, topk_lens = (_hpu_compute_global_topk_indices_and_lens(
                local_topk_2d,
                local_topk_token_to_req_indices,
                local_topk_block_table,
                local_topk_block_size,
                local_topk_is_valid,
            ))
            topk_indices = mapped_indices.view(mapped_indices.shape[0], 1, -1)

        _ORIGINAL_DEEPSEEK_V4_SPARSE_DECODE_FP8(
            q=q,
            kv_cache=kv_cache,
            kv_cache_storage=kv_cache_storage,
            kv_cache_geometry=kv_cache_geometry,
            swa_kv_cache=swa_kv_cache,
            swa_kv_cache_storage=swa_kv_cache_storage,
            swa_kv_cache_geometry=swa_kv_cache_geometry,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
            head_dim=head_dim,
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            out=out,
        )

    _hpu_sparse_decode_fp8.supports_local_topk_mapping = envs.VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN or envs.VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN
    _hpu_sparse_decode_fp8.supports_sequential_topk = (
        envs.VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN
        or envs.VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN) and envs.VLLM_HPU_DSV4_SHORT_INDEXER_SKIP
    _hpu_sparse_decode_fp8.supports_unpadded_q = parse_hpu_flashmla_backend(
        envs.VLLM_HPU_DSV4_ATTENTION_BACKEND) != HPUFlashMLABackend.LEGACY and (
            envs.VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN
            or envs.VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN and envs.VLLM_HPU_DSV4_TPC_QNORM_ROPE_KV_PACK)

    def _hpu_dsv4_can_direct_decode_dispatch(
        q: torch.Tensor,
        output: torch.Tensor,
        *,
        swa_only: bool,
    ) -> bool:
        backend = parse_hpu_flashmla_backend(envs.VLLM_HPU_DSV4_ATTENTION_BACKEND)
        return (envs.VLLM_HPU_DSV4_DIRECT_DECODE_DISPATCH and envs.VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN
                and backend in (HPUFlashMLABackend.AUTO, HPUFlashMLABackend.FLASHMLA) and q.shape[0] == 1
                and q.dtype == torch.bfloat16 and q.shape[-1] == _DEEPSEEK_V4_HEAD_DIM
                and flashmla_output_buffer_is_compatible(q, output))

    def _hpu_dsv4_try_direct_decode_dispatch(
        self,
        q: torch.Tensor,
        kv_cache_storage: torch.Tensor | None,
        kv_cache_geometry: torch.Tensor | None,
        swa_metadata,
        attn_metadata,
        swa_only: bool,
        output: torch.Tensor,
        compressor_dependency: torch.Tensor | None = None,
    ) -> bool:
        if not _hpu_dsv4_can_direct_decode_dispatch(q, output, swa_only=swa_only):
            return False

        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        if num_decode_tokens != 1 or (not swa_only and attn_metadata is None):
            return False
        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        if swa_indices is None or swa_lens is None:
            return False
        swa_idx_2d = (swa_indices.squeeze(1) if swa_indices.dim() == 3 else swa_indices)

        if swa_only:
            _ensure_dsv4_tpc_ops_loaded()
            with (dsv4_profile_stage("decode_fp8_sparse"), dsv4_profile_stage("decode_swa_only_fp8_tpc")):
                swa_op = (torch.ops.custom_op.custom_deepseek_v4_paged_swa_attn_fp8_gaudi2)
                swa_op(
                    q.contiguous(),
                    self.swa_cache_layer.kv_cache_storage,
                    self.swa_cache_layer.kv_cache_geometry,
                    _hpu_dsv4_i32_contiguous(swa_idx_2d),
                    _hpu_dsv4_i32_contiguous(swa_lens),
                    _hpu_dsv4_i32_contiguous(swa_idx_2d),
                    _hpu_dsv4_i32_contiguous(swa_lens),
                    _hpu_dsv4_i32_contiguous(swa_lens),
                    self.swa_cache_layer.kv_cache_storage,
                    self.swa_cache_layer.kv_cache_geometry,
                    _hpu_dsv4_i32_contiguous(swa_idx_2d),
                    _hpu_dsv4_i32_contiguous(swa_lens),
                    _hpu_dsv4_fp32_contiguous(self.attn_sink),
                    output,
                )
            return True

        if kv_cache_storage is None or kv_cache_geometry is None:
            return False

        _ensure_dsv4_tpc_ops_loaded()
        if self.compress_ratio == 128:
            if compressor_dependency is None:
                return False
            topk_indices = (attn_metadata.c128a_global_decode_topk_indices)
            topk_lens = attn_metadata.c128a_decode_topk_lens
            if topk_indices is None or topk_lens is None:
                return False
            topk_idx_2d = (topk_indices.squeeze(1) if topk_indices.dim() == 3 else topk_indices)
            ordered_topk_lens = _hpu_dsv4_preserve_i32_dependency(topk_lens[:1], compressor_dependency)
            with (dsv4_profile_stage("decode_fp8_sparse"), dsv4_profile_stage("decode_paged_fp8_tpc")):
                torch.ops.custom_op.custom_deepseek_v4_paged_sparse_attn_fp8_gaudi2(
                    q.contiguous(),
                    kv_cache_storage,
                    kv_cache_geometry,
                    _hpu_dsv4_i32_contiguous(topk_idx_2d),
                    ordered_topk_lens,
                    self.swa_cache_layer.kv_cache_storage,
                    self.swa_cache_layer.kv_cache_geometry,
                    _hpu_dsv4_i32_contiguous(swa_idx_2d),
                    _hpu_dsv4_i32_contiguous(swa_lens),
                    _hpu_dsv4_fp32_contiguous(self.attn_sink),
                    output,
                )
            return True
        if self.compress_ratio != 4:
            return False

        with dsv4_profile_stage("decode_topk_mapping"):
            if (self.topk_indices_buffer is None or swa_metadata.token_to_req_indices is None
                    or swa_metadata.is_valid_token is None):
                return False
            topk_idx_2d = self.topk_indices_buffer[:1]
            token_to_req = swa_metadata.token_to_req_indices[:1]
            block_table = attn_metadata.block_table[:num_decodes]
            is_valid = _hpu_dsv4_preserve_i32_dependency(
                swa_metadata.is_valid_token[:1],
                compressor_dependency,
            )
            use_sequential = (self.indexer is not None and self.indexer.max_model_len <= self.indexer.topk_tokens
                              and swa_metadata.seq_lens is not None)

        with dsv4_profile_stage("decode_fp8_sparse"):
            if use_sequential:
                with dsv4_profile_stage("decode_paged_sequential_fp8_tpc"):
                    if envs.VLLM_HPU_DSV4_FLASHMLA_SPLIT_KV:
                        _hpu_dsv4_flashmla_splitkv_decode(
                            q,
                            kv_cache_storage,
                            kv_cache_geometry,
                            topk_idx_2d,
                            token_to_req,
                            block_table,
                            is_valid,
                            swa_metadata.seq_lens,
                            self.swa_cache_layer.kv_cache_storage,
                            self.swa_cache_layer.kv_cache_geometry,
                            swa_idx_2d,
                            swa_lens,
                            self.attn_sink,
                            output,
                        )
                    else:
                        sequential_op = (
                            torch.ops.custom_op.custom_deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2
                            if envs.VLLM_HPU_DSV4_TPC_PAIR_HEADS and q.shape[1] % 2 == 0 else
                            torch.ops.custom_op.custom_deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2)
                        sequential_op(
                            q.contiguous(),
                            kv_cache_storage,
                            kv_cache_geometry,
                            _hpu_dsv4_i32_contiguous(topk_idx_2d),
                            _hpu_dsv4_i32_contiguous(token_to_req),
                            _hpu_dsv4_i32_contiguous(block_table),
                            _hpu_dsv4_i32_contiguous(is_valid),
                            _hpu_dsv4_i32_contiguous(swa_metadata.seq_lens),
                            self.swa_cache_layer.kv_cache_storage,
                            self.swa_cache_layer.kv_cache_geometry,
                            _hpu_dsv4_i32_contiguous(swa_idx_2d),
                            _hpu_dsv4_i32_contiguous(swa_lens),
                            _hpu_dsv4_fp32_contiguous(self.attn_sink),
                            output,
                        )
                return True
            with dsv4_profile_stage("decode_paged_local_fp8_tpc"):
                torch.ops.custom_op.custom_deepseek_v4_paged_sparse_attn_local_fp8_gaudi2(
                    q.contiguous(),
                    kv_cache_storage,
                    kv_cache_geometry,
                    _hpu_dsv4_i32_contiguous(topk_idx_2d),
                    _hpu_dsv4_i32_contiguous(token_to_req),
                    _hpu_dsv4_i32_contiguous(block_table),
                    _hpu_dsv4_i32_contiguous(is_valid),
                    self.swa_cache_layer.kv_cache_storage,
                    self.swa_cache_layer.kv_cache_geometry,
                    _hpu_dsv4_i32_contiguous(swa_idx_2d),
                    _hpu_dsv4_i32_contiguous(swa_lens),
                    _hpu_dsv4_fp32_contiguous(self.attn_sink),
                    output,
                )
            return True

    def _hpu_dsv4_mla_forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        kv_cache_storage: torch.Tensor | None,
        kv_cache_geometry: torch.Tensor | None,
        swa_metadata,
        attn_metadata,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        if _hpu_dsv4_try_direct_decode_dispatch(
                self,
                q,
                kv_cache_storage,
                kv_cache_geometry,
                swa_metadata,
                attn_metadata,
                swa_only,
                output,
        ):
            return
        return _ORIGINAL_DEEPSEEK_V4_MLA_FORWARD_DECODE(
            self,
            q,
            kv_cache,
            kv_cache_storage,
            kv_cache_geometry,
            swa_metadata,
            attn_metadata,
            swa_only,
            output,
        )

    def _hpu_dsv4_mla_forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata,
        swa_metadata,
    ) -> None:
        if not envs.VLLM_HPU_DSV4_FLASHMLA_PREFILL:
            return _ORIGINAL_DEEPSEEK_V4_MLA_FORWARD_PREFILL(
                self,
                q,
                positions,
                compressed_k_cache,
                swa_k_cache,
                output,
                attn_metadata,
                swa_metadata,
            )

        del positions
        from vllm.v1.worker.workspace import current_workspace_manager

        swa_only = attn_metadata is None
        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        if (seq_lens is None or gather_lens is None or query_start_loc_cpu is None or query_start_loc is None):
            raise RuntimeError("Sparse prefill metadata is incomplete")
        prefill_token_base = query_start_loc_cpu[num_decodes]

        with dsv4_profile_stage("prefill_flashmla_metadata"):
            if not swa_only:
                if self.compress_ratio == 4:
                    if self.topk_indices_buffer is None:
                        raise RuntimeError("C4 sparse prefill requires top-k indices")
                    topk_indices = self.topk_indices_buffer[num_decode_tokens:][:num_prefill_tokens]
                else:
                    topk_indices = attn_metadata.c128a_prefill_topk_indices
                    if topk_indices is None:
                        raise RuntimeError("C128 sparse prefill metadata is missing")
                top_k = topk_indices.shape[-1]
                N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
            else:
                if self.topk_indices_buffer is None:
                    raise RuntimeError("SWA-only prefill requires an index workspace")
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                top_k = 0
                N = 0

            M = N + self.window_size + self.max_num_batched_tokens
            prefill_chunk_size = (_ORIGINAL_DEEPSEEK_V4_MLA_FORWARD_PREFILL.__globals__.get("PREFILL_CHUNK_SIZE", 4))
            num_chunks = (num_prefills + prefill_chunk_size - 1) // prefill_chunk_size

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(((prefill_chunk_size, M, q.shape[-1]), torch.bfloat16), )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * prefill_chunk_size
            chunk_end = min(chunk_start + prefill_chunk_size, num_prefills)
            chunk_size = chunk_end - chunk_start

            if not swa_only:
                if compressed_k_cache is None:
                    raise RuntimeError("Compressed sparse prefill requires its KV cache")
                block_table = attn_metadata.block_table[num_decodes:]
                with dsv4_profile_stage("prefill_compressed_cache_gather"):
                    _hpu_dequantize_and_gather_k_cache(
                        kv[:chunk_size],
                        compressed_k_cache,
                        seq_lens=(seq_lens[chunk_start:chunk_end] // self.compress_ratio),
                        gather_lens=None,
                        block_table=block_table[chunk_start:chunk_end],
                        block_size=(attn_metadata.block_size // self.compress_ratio),
                        offset=0,
                    )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            with dsv4_profile_stage("prefill_swa_cache_gather"):
                _hpu_dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    swa_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    block_table=swa_block_table[chunk_start:chunk_end],
                    block_size=swa_metadata.block_size,
                    offset=N,
                )

            query_start = (query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base)
            query_end = (query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base)
            query_metadata_start = num_decodes + chunk_start
            query_metadata_end = num_decodes + chunk_end + 1
            with dsv4_profile_stage("prefill_sparse_index_merge"):
                combined_indices, combined_lens = (_hpu_combine_topk_swa_indices(
                    topk_indices[query_start:query_end],
                    query_start_loc[query_metadata_start:query_metadata_end],
                    seq_lens[chunk_start:chunk_end],
                    gather_lens[chunk_start:chunk_end],
                    self.window_size,
                    self.compress_ratio,
                    top_k,
                    M,
                    N,
                ))

            kv_ws = kv[:chunk_size].reshape(-1, 1, q.shape[-1])
            out_chunk, _, _ = _hpu_bf16_mla_sparse_interface(
                q=q[query_start:query_end],
                kv=kv_ws,
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                d_v=q.shape[-1],
                block_dpe=0,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
            )
            with dsv4_profile_stage("prefill_output_copy"):
                output[query_start:query_end].copy_(out_chunk)

    def _hpu_fused_indexer_q_rope_quant(
        positions: torch.Tensor,
        index_q: torch.Tensor,
        index_q_cos_sin_cache: torch.Tensor,
        index_weights: torch.Tensor,
        index_weights_softmax_scale: float,
        index_weights_head_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.flatten().to(torch.int64)
        num_tokens, _, head_dim = index_q.shape
        half_rot_dim = index_q_cos_sin_cache.shape[-1] // 2
        rot_dim = half_rot_dim * 2
        nope_dim = head_dim - rot_dim
        if nope_dim < 0:
            raise ValueError(f"Indexer head dim {head_dim} is smaller than "
                             f"RoPE dim {rot_dim}")

        cos_sin = index_q_cos_sin_cache.index_select(0, positions).float()
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.view(num_tokens, 1, half_rot_dim)
        sin = sin.view(num_tokens, 1, half_rot_dim)

        rot = index_q[..., nope_dim:].float().view(num_tokens, index_q.shape[1], half_rot_dim, 2)
        even = rot[..., 0]
        odd = rot[..., 1]
        rotated = torch.stack(
            (
                even * cos - odd * sin,
                odd * cos + even * sin,
            ),
            dim=-1,
        ).view(num_tokens, index_q.shape[1], rot_dim)
        rotated = rotated.to(torch.bfloat16).float()

        quant_source = (torch.cat(
            (index_q[..., :nope_dim].float(), rotated),
            dim=-1,
        ) if nope_dim else rotated)
        amax = quant_source.abs().amax(dim=-1, keepdim=True)
        scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-4) / _DEEPSEEK_V4_FP8_MAX)))
        # Gaudi's native E4M3 cast uses 240 as the largest finite value,
        # while this model's Triton path uses NVIDIA E4M3FN (max 448).
        # The logits path immediately converts Q to BF16, so encode/decode
        # the model format explicitly and hand it the identical BF16 values.
        index_q_fp8 = _hpu_decode_e4m3fn_bytes(_hpu_encode_e4m3fn_bytes(quant_source / scale)).to(torch.bfloat16)
        weights_out = (index_weights.float() * scale.squeeze(-1) * index_weights_softmax_scale *
                       index_weights_head_scale)
        return index_q_fp8, weights_out

    def _hpu_fp8_paged_mqa_logits(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        max_model_len: int,
    ) -> torch.Tensor:
        batch_size, next_n, _, dim = q.shape
        if next_n != 1:
            raise NotImplementedError("HPU DeepSeek V4 indexer currently supports next_n=1")

        if kv_cache.ndim == 4 and kv_cache.shape[2] == 1:
            kv_cache = kv_cache.squeeze(2)
        elif kv_cache.ndim == 4 and kv_cache.shape[1] == 1:
            kv_cache = kv_cache.squeeze(1)
        if kv_cache.ndim != 3:
            raise ValueError("Indexer cache must have shape [blocks, tokens, bytes], "
                             f"got {tuple(kv_cache.shape)}")

        block_size = kv_cache.shape[1]
        page_bytes = block_size * (dim + 4)
        cache_storage = _hpu_uint8_storage_view(kv_cache)
        base_offset = kv_cache.storage_offset()
        max_pages = block_tables.shape[1]
        pages = block_tables[:batch_size, :max_pages].to(torch.int64)
        page_offsets = (base_offset + pages.unsqueeze(-1) * kv_cache.stride(0) + torch.arange(
            page_bytes,
            dtype=torch.int64,
            device=kv_cache.device,
        ).view(1, 1, -1))
        cache_pages = cache_storage[page_offsets]

        value_bytes = cache_pages[..., :block_size * dim]
        cache_values = _hpu_decode_e4m3fn_bytes(value_bytes).view(
            batch_size,
            max_pages * block_size,
            dim,
        )
        scale_bytes = cache_pages[..., block_size * dim:]
        cache_scales = scale_bytes.contiguous().view(torch.float32).reshape(batch_size, max_pages * block_size)

        q_fp32 = q[:, 0].float()
        scores = torch.einsum("btd,bhd->bth", cache_values, q_fp32)
        scores = (torch.relu(scores) * weights[:batch_size].unsqueeze(1)).sum(dim=-1)
        scores = scores * cache_scales

        if context_lens.ndim > 1:
            context_lens = context_lens.squeeze(-1)
        padded_seq_len = max_pages * block_size
        positions = torch.arange(
            padded_seq_len,
            dtype=context_lens.dtype,
            device=q.device,
        )
        scores = torch.where(
            positions.unsqueeze(0) < context_lens.unsqueeze(1),
            scores,
            torch.full_like(scores, float("-inf")),
        )

        logits = torch.full(
            (batch_size, max_model_len),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        write_width = min(padded_seq_len, max_model_len)
        logits[:, :write_width].copy_(scores[:, :write_width])
        return logits

    def _hpu_save_partial_states(
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        positions: torch.Tensor,
        state_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
        state_width: int,
        compress_ratio: int,
        pdl_kwargs: dict | None = None,
    ) -> None:
        del pdl_kwargs
        num_actual = slot_mapping.shape[0]
        assert kv.shape[-1] == state_width
        assert score.shape[-1] == state_width

        if (envs.VLLM_HPU_DSV4_TPC_SAVE_PARTIAL_STATES
                and state_width <= envs.VLLM_HPU_DSV4_TPC_SAVE_PARTIAL_STATES_MAX_WIDTH):
            _ensure_dsv4_tpc_ops_loaded()
            if state_cache.ndim == 4 and state_cache.shape[1] == 1:
                state_cache = state_cache.squeeze(1)
            state_storage = _hpu_storage_view(state_cache)
            state_geometry = torch.tensor(
                (
                    state_cache.storage_offset(),
                    state_cache.stride(0),
                    state_cache.stride(1),
                    block_size,
                    state_cache.shape[0],
                ),
                dtype=torch.int32,
                device=state_cache.device,
            )
            (torch.ops.custom_op.custom_deepseek_v4_save_partial_states_f32_gaudi2(
                state_storage,
                state_geometry,
                kv[:num_actual],
                score[:num_actual],
                ape,
                positions[:num_actual].to(torch.int32).contiguous(),
                slot_mapping.to(torch.int32).contiguous(),
            ))
            return

        positions = positions[:num_actual]
        ape_rows = ape.index_select(
            0,
            positions.remainder(compress_ratio).to(torch.int64),
        )
        packed_state = torch.cat(
            (
                kv[:num_actual],
                score[:num_actual] + ape_rows,
            ),
            dim=-1,
        )

        slots = slot_mapping.to(torch.int64)
        dummy_slot = state_cache.shape[0] * block_size - 1
        slots = torch.where(
            slots >= 0,
            slots,
            torch.full_like(slots, dummy_slot),
        )
        block_indices = slots.div(block_size, rounding_mode="floor")
        positions_in_block = slots.remainder(block_size)
        if state_cache.ndim == 4 and state_cache.shape[1] == 1:
            state_cache = state_cache.squeeze(1)
        block_stride = state_cache.stride(0)
        state_storage = _hpu_storage_view(state_cache)
        base_offset = state_cache.storage_offset()
        state_offsets = (base_offset + block_indices.unsqueeze(1) * block_stride +
                         positions_in_block.unsqueeze(1) * state_cache.stride(1) + torch.arange(
                             packed_state.shape[1],
                             dtype=torch.int64,
                             device=state_cache.device,
                         ).unsqueeze(0))
        state_storage[state_offsets] = packed_state

    def _hpu_store_indexer_compressed_cache(
        values: torch.Tensor,
        scales: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        head_dim = values.shape[-1]
        if kv_cache.ndim == 4 and kv_cache.shape[1] == 1:
            kv_cache = kv_cache.squeeze(1)
        block_size = kv_cache.shape[1]
        block_stride = kv_cache.stride(0)
        cache_storage = _hpu_uint8_storage_view(kv_cache)
        base_offset = kv_cache.storage_offset()

        slots = slot_mapping.to(torch.int64)
        dummy_slot = kv_cache.shape[0] * block_size - 1
        slots = torch.where(
            slots >= 0,
            slots,
            torch.full_like(slots, dummy_slot),
        )
        block_indices = slots.div(block_size, rounding_mode="floor")
        positions_in_block = slots.remainder(block_size)
        value_offsets = (base_offset + block_indices.unsqueeze(1) * block_stride +
                         positions_in_block.unsqueeze(1) * head_dim + torch.arange(
                             head_dim,
                             dtype=torch.int64,
                             device=kv_cache.device,
                         ).unsqueeze(0))
        scale_offsets = (base_offset + block_indices.unsqueeze(1) * block_stride + block_size * head_dim +
                         positions_in_block.unsqueeze(1) * 4 + torch.arange(
                             4,
                             dtype=torch.int64,
                             device=kv_cache.device,
                         ).unsqueeze(0))
        cache_storage[value_offsets] = values
        cache_storage[scale_offsets] = scales.float().contiguous().view(torch.uint8)

    def _hpu_insert_c4_packed_payload(
        packed: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_slot_mapping: torch.Tensor,
        valid: torch.Tensor,
        head_dim: int,
        cache_storage: torch.Tensor | None = None,
        cache_geometry: torch.Tensor | None = None,
    ) -> None:
        if kv_cache.ndim == 4 and kv_cache.shape[1] == 1:
            kv_cache = kv_cache.squeeze(1)
        block_size = kv_cache.shape[1]
        block_stride = kv_cache.stride(0)
        if cache_storage is None:
            cache_storage = _hpu_uint8_storage_view(kv_cache)
        elif cache_storage.dtype != torch.uint8 or cache_storage.ndim != 1:
            raise ValueError("Pre-bound KV cache storage must be flat uint8")
        del cache_geometry
        base_offset = kv_cache.storage_offset()
        data_bytes = 576 if head_dim == 512 else 128
        scale_bytes = 8 if head_dim == 512 else 4

        slots = kv_slot_mapping.to(torch.int64)
        dummy_slot = kv_cache.shape[0] * block_size - 1
        slots = torch.where(
            valid,
            slots,
            torch.full_like(slots, dummy_slot),
        )
        block_indices = slots.div(block_size, rounding_mode="floor")
        positions_in_block = slots.remainder(block_size)
        data_offsets = (base_offset + block_indices.unsqueeze(1) * block_stride +
                        positions_in_block.unsqueeze(1) * data_bytes + torch.arange(
                            data_bytes,
                            dtype=torch.int64,
                            device=kv_cache.device,
                        ).unsqueeze(0))
        scale_offsets = (base_offset + block_indices.unsqueeze(1) * block_stride + block_size * data_bytes +
                         positions_in_block.unsqueeze(1) * scale_bytes + torch.arange(
                             scale_bytes,
                             dtype=torch.int64,
                             device=kv_cache.device,
                         ).unsqueeze(0))
        cache_storage[data_offsets] = packed[:, :data_bytes]
        cache_storage[scale_offsets] = packed[:, data_bytes:data_bytes + scale_bytes]

    def _hpu_finish_c4_compressor_store(
        self,
        normed: torch.Tensor,
        positions: torch.Tensor,
        boundary: torch.Tensor,
        rotary_emb,
        kv_cache: torch.Tensor,
        kv_slot_mapping: torch.Tensor,
    ) -> None:
        compressed_positions = (positions.to(torch.int64).div(self.compress_ratio, rounding_mode="floor") *
                                self.compress_ratio)
        roped = _hpu_apply_pairwise_rope(
            normed[:, self.nope_head_dim:].view(normed.shape[0], 1, self.rope_head_dim),
            compressed_positions,
            rotary_emb.cos_sin_cache,
        ).view(normed.shape[0], self.rope_head_dim)
        normed_roped = torch.cat((normed[:, :self.nope_head_dim], roped), dim=-1).to(torch.bfloat16)
        kv_slots = torch.where(
            boundary & (kv_slot_mapping >= 0),
            kv_slot_mapping,
            torch.full_like(kv_slot_mapping, -1),
        )

        if self.head_dim == _DEEPSEEK_V4_HEAD_DIM:
            _hpu_quantize_and_insert_k_cache(
                normed_roped,
                kv_cache,
                kv_slots,
                block_size=kv_cache.shape[1],
            )
            return

        if self.head_dim != 128:
            raise ValueError(f"Unsupported DeepSeek V4 compressor head_dim={self.head_dim}")
        quant_input = normed_roped.float()
        block_max = quant_input.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
        exponent = torch.ceil(torch.log2(block_max / _DEEPSEEK_V4_FP8_MAX))
        scales = torch.exp2(exponent)
        encoded = _hpu_encode_e4m3fn_bytes((quant_input / scales).clamp(
            -_DEEPSEEK_V4_FP8_MAX,
            _DEEPSEEK_V4_FP8_MAX,
        ))
        _hpu_store_indexer_compressed_cache(
            encoded,
            scales,
            kv_cache,
            kv_slots,
        )

    def _hpu_deepseek_compressor_forward_c4(
        self,
        kv_score: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
        *,
        use_noclone: bool = False,
    ) -> torch.Tensor | None:
        if (not envs.VLLM_HPU_DSV4_TPC_SAVE_COMPRESS_NORM_C4 or self.compress_ratio not in (4, 128)
                or kv_score.shape[0] != 1 or self.state_cache.kv_cache_storage.dtype != torch.uint8):
            return _ORIGINAL_DEEPSEEK_COMPRESSOR_FORWARD(self, kv_score, positions, rotary_emb)

        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return _ORIGINAL_DEEPSEEK_COMPRESSOR_FORWARD(self, kv_score, positions, rotary_emb)

        state_metadata = attn_metadata[self.state_cache.prefix]
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        if token_to_req_indices is None or slot_mapping.shape[0] != 1:
            return _ORIGINAL_DEEPSEEK_COMPRESSOR_FORWARD(self, kv_score, positions, rotary_emb)

        _ensure_dsv4_tpc_ops_loaded()
        kv, score = kv_score.split([self.coff * self.head_dim, self.coff * self.head_dim], dim=-1)
        positions_actual = positions[:1]
        slots_actual = slot_mapping[:1]
        k_cache_metadata = attn_metadata[self.k_cache_prefix]
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        use_bf16_inputs = (envs.VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS and kv.dtype == torch.bfloat16
                           and score.dtype == torch.bfloat16)
        use_mixed_inputs = (not use_bf16_inputs and envs.VLLM_HPU_DSV4_TPC_MIXED_COMPRESS_INPUTS
                            and kv.dtype == torch.float32 and score.dtype == torch.float32)
        use_noclone = (use_noclone and envs.VLLM_HPU_DSV4_TPC_ORDERED_COMPRESSOR and not use_bf16_inputs
                       and not use_mixed_inputs)
        compressor_op = (
            torch.ops.custom_op.custom_deepseek_v4_save_compress_norm_c4_f32_noclone_gaudi2 if use_noclone else
            (torch.ops.custom_op.custom_deepseek_v4_save_compress_norm_c4_bf16_gaudi2 if use_bf16_inputs else
             (torch.ops.custom_op.custom_deepseek_v4_save_compress_norm_c4_mixed_gaudi2
              if use_mixed_inputs else torch.ops.custom_op.custom_deepseek_v4_save_compress_norm_c4_f32_gaudi2)))
        if use_bf16_inputs or use_mixed_inputs:
            norm_weight = _hpu_dsv4_bf16_contiguous(self.norm.weight)
        else:
            norm_weight = getattr(
                self.norm,
                "_hpu_dsv4_weight_fp32",
                None,
            )
            if norm_weight is None:
                norm_weight = _hpu_dsv4_fp32_contiguous(self.norm.weight)
        compressor_inputs = (
            self.state_cache.kv_cache_storage,
            self.state_cache.kv_cache_geometry,
            k_cache_layer.kv_cache_geometry,
            kv,
            score,
            self.ape,
            _hpu_dsv4_i32_contiguous(positions_actual),
            _hpu_dsv4_i32_contiguous(slots_actual),
            _hpu_dsv4_i32_contiguous(token_to_req_indices[:1]),
            _hpu_dsv4_i32_contiguous(state_metadata.block_table),
            norm_weight,
            self._rms_norm_eps_tensor,
            _hpu_dsv4_fp32_contiguous(rotary_emb.cos_sin_cache),
            _hpu_dsv4_i32_contiguous(k_cache_metadata.slot_mapping[:1]),
        )
        return compressor_op(*compressor_inputs)

    def _hpu_compress_norm_rope_store(
        state_cache: torch.Tensor,
        num_actual: int,
        token_to_req_indices: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        state_width: int,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        k_cache_metadata,
        pdl_kwargs: dict,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        overlap: bool,
        rms_norm_weight: torch.Tensor,
        rms_norm_eps: float,
        quant_block: int,
        token_stride: int,
        scale_dim: int,
    ) -> None:
        del pdl_kwargs, quant_block, token_stride, scale_dim
        positions = positions[:num_actual]
        state_slots = slot_mapping[:num_actual].to(torch.int64)
        request_indices = token_to_req_indices[:num_actual].to(torch.int64)
        boundary = ((state_slots >= 0) & ((positions + 1).remainder(compress_ratio) == 0))

        coff = 1 + int(overlap)
        window_size = coff * compress_ratio
        window_offsets = torch.arange(
            window_size,
            dtype=torch.int64,
            device=positions.device,
        )
        state_positions = (positions.to(torch.int64).unsqueeze(1) - window_size + 1 + window_offsets.unsqueeze(0))
        valid = boundary.unsqueeze(1) & (state_positions >= 0)
        safe_positions = state_positions.clamp_min(0)
        logical_blocks = (safe_positions.div(block_size, rounding_mode="floor").clamp_max(block_table.shape[1] - 1))
        request_block_tables = block_table.index_select(0, request_indices)
        block_numbers = torch.gather(
            request_block_tables,
            1,
            logical_blocks,
        ).to(torch.int64)
        valid = valid & (block_numbers >= 0)
        block_numbers = block_numbers.clamp_min(0)
        positions_in_block = safe_positions.remainder(block_size)
        state_heads = (window_offsets >= compress_ratio).to(torch.int64)
        if state_cache.ndim == 4 and state_cache.shape[1] == 1:
            state_cache = state_cache.squeeze(1)
        state_storage = _hpu_storage_view(state_cache)
        base_offset = state_cache.storage_offset()
        head_offsets = torch.arange(
            head_dim,
            dtype=torch.int64,
            device=state_cache.device,
        )
        common_offsets = (base_offset + block_numbers.unsqueeze(-1) * state_cache.stride(0) +
                          positions_in_block.unsqueeze(-1) * state_cache.stride(1) +
                          state_heads.view(1, -1, 1) * head_dim + head_offsets.view(1, 1, -1))
        scores = state_storage[common_offsets + state_width]
        scores = torch.where(
            valid.unsqueeze(-1),
            scores,
            torch.full_like(scores, -1e30),
        )
        weights = torch.softmax(scores, dim=1)
        values = state_storage[common_offsets]
        values = torch.where(
            valid.unsqueeze(-1),
            values,
            torch.zeros_like(values),
        )
        compressed = (values * weights).sum(dim=1)

        variance = compressed.square().mean(dim=-1, keepdim=True)
        normed = (compressed * torch.rsqrt(variance + rms_norm_eps) * rms_norm_weight.float())
        compressed_positions = (positions.to(torch.int64).div(compress_ratio, rounding_mode="floor") * compress_ratio)
        nope_head_dim = head_dim - rope_head_dim
        roped = _hpu_apply_pairwise_rope(
            normed[:, nope_head_dim:].view(num_actual, 1, rope_head_dim),
            compressed_positions,
            cos_sin_cache,
        ).view(num_actual, rope_head_dim)
        normed_roped = torch.cat(
            (normed[:, :nope_head_dim], roped),
            dim=-1,
        ).to(torch.bfloat16)

        kv_slots = k_cache_metadata.slot_mapping[:num_actual].to(torch.int64)
        kv_slots = torch.where(
            boundary & (kv_slots >= 0),
            kv_slots,
            torch.full_like(kv_slots, -1),
        )
        if head_dim == _DEEPSEEK_V4_HEAD_DIM:
            _hpu_quantize_and_insert_k_cache(
                normed_roped,
                kv_cache,
                kv_slots,
                block_size=kv_cache.shape[1],
            )
            return

        if head_dim != 128:
            raise ValueError(f"Unsupported DeepSeek V4 compressor head_dim={head_dim}")
        quant_input = normed_roped.float()
        block_max = quant_input.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
        exponent = torch.ceil(torch.log2(block_max / _DEEPSEEK_V4_FP8_MAX))
        scales = torch.exp2(exponent)
        encoded = _hpu_encode_e4m3fn_bytes((quant_input / scales).clamp(
            -_DEEPSEEK_V4_FP8_MAX,
            _DEEPSEEK_V4_FP8_MAX,
        ))
        _hpu_store_indexer_compressed_cache(
            encoded,
            scales,
            kv_cache,
            kv_slots,
        )

    def _hpu_get_compressed_slot_mapping(
        num_tokens: int,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        compress_ratio: int,
        out: torch.Tensor | None = None,
        initialize_out: bool = True,
    ) -> torch.Tensor:
        if out is None:
            slot_mapping = torch.full(
                (num_tokens, ),
                -1,
                dtype=torch.int64,
                device=query_start_loc.device,
            )
        else:
            if initialize_out:
                out.fill_(-1)
            slot_mapping = out[:num_tokens]

        num_reqs = block_table.shape[0]
        if not initialize_out and num_tokens == num_reqs:
            req_indices = torch.arange(
                num_reqs,
                dtype=torch.int64,
                device=query_start_loc.device,
            )
            positions = seq_lens.to(torch.int64) - 1
            valid = (positions + 1).remainder(compress_ratio) == 0
            compressed_positions = positions.div(compress_ratio, rounding_mode="floor")
            block_indices = compressed_positions.div(block_size,
                                                     rounding_mode="floor").clamp(min=0, max=block_table.shape[1] - 1)
            block_numbers = block_table[req_indices, block_indices].to(torch.int64)
            compressed_slots = (block_numbers * block_size + compressed_positions.remainder(block_size))
            slot_mapping.copy_(torch.where(valid, compressed_slots, -1))
            return slot_mapping

        query_lens = query_start_loc[1:num_reqs + 1] - query_start_loc[:num_reqs]
        req_indices = torch.repeat_interleave(
            torch.arange(
                num_reqs,
                dtype=torch.int64,
                device=query_start_loc.device,
            ),
            query_lens,
            output_size=num_tokens,
        )
        token_offsets = (torch.arange(
            num_tokens,
            dtype=torch.int64,
            device=query_start_loc.device,
        ) - query_start_loc[req_indices])
        positions = (seq_lens[req_indices].to(torch.int64) - query_lens[req_indices].to(torch.int64) + token_offsets)
        valid = (positions + 1).remainder(compress_ratio) == 0
        compressed_positions = positions.div(compress_ratio, rounding_mode="floor")
        block_indices = compressed_positions.div(block_size, rounding_mode="floor")
        block_indices = block_indices.clamp(min=0, max=block_table.shape[1] - 1)
        block_numbers = block_table[req_indices, block_indices].to(torch.int64)
        compressed_offsets = compressed_positions.remainder(block_size)
        compressed_slots = block_numbers * block_size + compressed_offsets
        slot_mapping.copy_(torch.where(valid, compressed_slots, -1))
        return slot_mapping

    def _hpu_build_prefill_chunk_metadata(
        start_idx: int,
        end_idx: int,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        uncompressed_seq_lens: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        compressed_seq_lens_cpu: torch.Tensor,
        block_table: torch.Tensor,
        compress_ratio: int,
        query_slice: slice | None = None,
        skip_kv_gather: bool = False,
    ):
        total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
        if total_seq_lens == 0:
            return None

        num_reqs = end_idx - start_idx
        device = block_table.device
        chunk_compressed_seq_lens = compressed_seq_lens[start_idx:end_idx]

        cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
        cu_seq_lens[:1] = 0
        torch.cumsum(chunk_compressed_seq_lens, dim=0, out=cu_seq_lens[1:])
        token_to_seq = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32, device=device),
            chunk_compressed_seq_lens,
            output_size=total_seq_lens,
        )

        chunk_query_start_loc = (query_start_loc[start_idx:end_idx + 1] - query_start_loc[start_idx])
        query_lens = (chunk_query_start_loc[1:] - chunk_query_start_loc[:-1])
        total_query_len = int((query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item())
        if query_slice is not None:
            qs_start = query_slice.start
            qs_stop = query_slice.stop
        else:
            qs_start = 0
            qs_stop = total_query_len

        request_indices = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int64, device=device),
            query_lens,
            output_size=total_query_len,
        )[qs_start:qs_stop]
        query_positions = torch.arange(qs_start, qs_stop, dtype=torch.int64, device=device)
        query_offsets = (query_positions - chunk_query_start_loc[request_indices].to(torch.int64))
        compressed_seq_starts = cu_seq_lens[:-1].to(torch.int64)
        uncompressed_seq_starts = (uncompressed_seq_lens[start_idx:end_idx].to(torch.int64) -
                                   query_lens.to(torch.int64))
        cu_seq_len_ks = compressed_seq_starts[request_indices].to(torch.int32)
        cu_seq_len_ke = (compressed_seq_starts[request_indices] +
                         (uncompressed_seq_starts[request_indices] + 1 + query_offsets).div(
                             compress_ratio, rounding_mode="floor")).to(torch.int32)

        token_start = query_start_loc_cpu[start_idx].item()
        if query_slice is not None:
            token_end = token_start + qs_stop
            token_start += qs_start
            skip_kv_gather = skip_kv_gather or qs_start > 0
        else:
            token_end = query_start_loc_cpu[end_idx].item()

        return (deepseek_v4_indexer_module.DeepseekV4IndexerPrefillChunkMetadata(
            cu_seqlen_ks=cu_seq_len_ks,
            cu_seqlen_ke=cu_seq_len_ke,
            cu_seq_lens=cu_seq_lens,
            token_to_seq=token_to_seq,
            total_seq_lens=total_seq_lens,
            block_table=block_table[start_idx:end_idx],
            token_start=token_start,
            token_end=token_end,
            num_reqs=num_reqs,
            skip_kv_gather=skip_kv_gather,
        ))

    def _hpu_build_c128a_topk_metadata(
        positions: torch.Tensor,
        compress_ratio: int,
        num_decode_tokens: int,
        token_to_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        slot_mapping: torch.Tensor,
        global_decode_buffer: torch.Tensor,
        decode_lens_buffer: torch.Tensor,
        prefill_buffer: torch.Tensor,
        max_compressed_tokens: int = 8192,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = positions.shape[0]
        num_prefill_tokens = num_tokens - num_decode_tokens

        global_decode = global_decode_buffer[:num_decode_tokens]
        decode_lens = decode_lens_buffer[:num_decode_tokens]
        prefill_local = prefill_buffer[:num_prefill_tokens]
        if num_tokens == 0:
            return global_decode, decode_lens, prefill_local

        num_compressed = ((positions.to(torch.int64) + 1).div(
            compress_ratio, rounding_mode="floor").clamp(max=max_compressed_tokens).to(torch.int32))
        offsets = torch.arange(
            max_compressed_tokens,
            dtype=torch.int32,
            device=positions.device,
        )

        if num_decode_tokens > 0:
            decode_num_compressed = num_compressed[:num_decode_tokens]
            valid_offsets = (offsets.unsqueeze(0) < decode_num_compressed.unsqueeze(1))
            block_indices = (offsets.to(torch.int64).div(block_size,
                                                         rounding_mode="floor").clamp(max=block_table.shape[1] - 1))
            decode_req_indices = token_to_req_indices[:num_decode_tokens].to(torch.int64)
            decode_block_tables = block_table[decode_req_indices]
            block_numbers = torch.gather(
                decode_block_tables,
                1,
                block_indices.unsqueeze(0).expand(num_decode_tokens, -1),
            ).to(torch.int32)
            slot_ids = (block_numbers * block_size + offsets.remainder(block_size).unsqueeze(0))
            global_decode.copy_(torch.where(valid_offsets, slot_ids, -1))
            decode_lens.copy_(torch.where(
                slot_mapping[:num_decode_tokens] >= 0,
                decode_num_compressed,
                0,
            ))

        if num_prefill_tokens > 0:
            prefill_num_compressed = num_compressed[num_decode_tokens:]
            prefill_local.copy_(
                torch.where(
                    offsets.unsqueeze(0) < prefill_num_compressed.unsqueeze(1),
                    offsets.unsqueeze(0),
                    -1,
                ))

        return global_decode, decode_lens, prefill_local

    class _HPUTorchKernel:

        def __init__(self, function):
            self.function = function

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                return self.function(*args, grid=grid, **kwargs)

            return launch

    def _hpu_next_power_of_2(value: int) -> int:
        if value <= 1:
            return 1
        return 1 << (value - 1).bit_length()

    def _hpu_compute_prefill_metadata(
        prefill_gather_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_prefills: int,
        num_decodes: int,
        window_size: int,
        BLOCK_SIZE: int,
        grid,
    ) -> None:
        del BLOCK_SIZE, grid
        prefill_seq_lens = seq_lens[num_decodes:num_decodes + num_prefills]
        prefill_query_lens = (query_start_loc[num_decodes + 1:num_decodes + num_prefills + 1] -
                              query_start_loc[num_decodes:num_decodes + num_prefills])
        prefix_lens = prefill_seq_lens - prefill_query_lens
        prefill_gather_lens.copy_(prefill_query_lens + prefix_lens.clamp(max=window_size - 1))

    def _hpu_compute_swa_indices_and_lens(
        swa_indices: torch.Tensor,
        swa_indices_stride: int,
        swa_lens: torch.Tensor,
        window_size: int,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        is_valid_token: torch.Tensor,
        block_table: torch.Tensor,
        block_table_stride: int,
        block_size: int,
        TRITON_BLOCK_SIZE: int,
        grid,
    ) -> None:
        del swa_indices_stride, block_table_stride, TRITON_BLOCK_SIZE
        num_tokens = int(grid[0])
        if num_tokens == 0:
            return

        device = swa_indices.device
        token_indices = torch.arange(num_tokens, dtype=torch.int64, device=device)
        request_indices = token_to_req_indices[:num_tokens].to(torch.int64)
        valid_tokens = is_valid_token[:num_tokens].to(torch.bool)
        query_starts = query_start_loc[request_indices].to(torch.int64)
        query_lens = (query_start_loc[request_indices + 1].to(torch.int64) - query_starts)
        prefix_lens = (seq_lens[request_indices].to(torch.int64) - query_lens)
        positions = prefix_lens + token_indices - query_starts
        positions = torch.where(valid_tokens, positions, 0)
        start_positions = (positions - window_size + 1).clamp(min=0)
        end_positions = positions + 1
        window_lens = torch.where(valid_tokens, end_positions - start_positions, 0)

        offsets = torch.arange(window_size, dtype=torch.int64, device=device)
        absolute_positions = (start_positions.unsqueeze(1) + offsets.unsqueeze(0))
        block_indices = (absolute_positions.div(block_size, rounding_mode="floor").clamp(min=0,
                                                                                         max=block_table.shape[1] - 1))
        request_block_tables = block_table[request_indices]
        block_numbers = torch.gather(request_block_tables, 1, block_indices).to(torch.int64)
        slot_ids = (block_numbers * block_size + absolute_positions.remainder(block_size)).to(torch.int32)

        valid_offsets = (offsets.unsqueeze(0) < window_lens.unsqueeze(1)) & valid_tokens.unsqueeze(1)
        swa_indices[:num_tokens].view(num_tokens, -1).copy_(torch.where(valid_offsets, slot_ids, -1))
        swa_lens[:num_tokens].copy_(torch.where(valid_tokens, window_lens.to(torch.int32), 0))

    from vllm.models.deepseek_v4.hw_agnostic.attention import (
        attention as deepseek_v4_attention_module, )
    _DEEPSEEK_V4_INV_ROPE_MODULE = import_module("vllm.models.deepseek_v4.hw_agnostic.attention.kernels."
                                                 "triton_inv_rope_einsum")
    _ORIGINAL_DEEPSEEK_V4_INV_ROPE_EINSUM = (deepseek_v4_attention_module.triton_inv_rope_einsum)
    _ORIGINAL_DEEPSEEK_V4_ATTENTION_IMPL = (
        deepseek_v4_attention_module.DeepseekV4MultiHeadLatentAttentionWrapper.attention_impl)
    _ORIGINAL_DEEPSEEK_V4_MLA_FORWARD_DECODE = (deepseek_v4_attention_module.DeepseekV4MLAAttention._forward_decode)
    _ORIGINAL_DEEPSEEK_V4_MLA_FORWARD_PREFILL = (deepseek_v4_attention_module.DeepseekV4MLAAttention._forward_prefill)
    deepseek_v4_attention_module.DeepseekV4MultiHeadLatentAttentionWrapper.attention_impl = (
        _hpu_deepseek_v4_attention_impl)
    deepseek_v4_attention_module.DeepseekV4MLAAttention._forward_decode = (_hpu_dsv4_mla_forward_decode)
    deepseek_v4_attention_module.DeepseekV4MLAAttention._forward_prefill = (_hpu_dsv4_mla_forward_prefill)
    _ORIGINAL_DEEPSEEK_V4_INDEXER_FORWARD = (deepseek_v4_attention_module.DeepseekV4Indexer.forward)
    deepseek_v4_attention_module.DeepseekV4Indexer.forward = (_hpu_deepseek_v4_indexer_forward)
    from vllm.models.deepseek_v4.hw_agnostic.attention import (
        compressor as deepseek_v4_compressor_module, )
    _ORIGINAL_DEEPSEEK_COMPRESSOR_FORWARD = (deepseek_v4_compressor_module.DeepseekCompressor.forward)
    deepseek_v4_compressor_module.DeepseekCompressor.forward = (_hpu_deepseek_compressor_forward_c4)
    from vllm.models.deepseek_v4.hw_agnostic.attention import (
        indexer as deepseek_v4_indexer_module, )
    from vllm.models.deepseek_v4.hw_agnostic.attention import (
        sparse_mla as deepseek_v4_sparse_mla_module, )
    from vllm.models.deepseek_v4.hw_agnostic.attention import (
        sparse_swa as deepseek_v4_sparse_swa_module, )
    from vllm.models.deepseek_v4.hw_agnostic.attention import (
        sparse_attn_indexer as deepseek_v4_sparse_attn_indexer_module, )
    from vllm.models.deepseek_v4.hw_agnostic.attention import (
        kernels as deepseek_v4_kernels_module, )
    from vllm.models.deepseek_v4.hw_agnostic.attention.kernels import (
        cache_utils as deepseek_v4_cache_utils_module, )
    from vllm.models.deepseek_v4.hw_agnostic.attention.kernels import (
        fused_qk_rmsnorm as deepseek_v4_fused_qk_rmsnorm_module, )
    deepseek_v4_qnorm_rope_module = import_module("vllm.models.deepseek_v4.hw_agnostic.attention.kernels."
                                                  "triton_qnorm_rope_kv_fp8_insert")
    deepseek_v4_compress_kernel_module = import_module("vllm.models.deepseek_v4.hw_agnostic.attention.kernels."
                                                       "fused_compress_quant_cache")
    deepseek_v4_save_state_module = import_module("vllm.models.deepseek_v4.hw_agnostic.attention.kernels."
                                                  "save_partial_states")
    deepseek_v4_fused_indexer_q_module = import_module("vllm.models.deepseek_v4.hw_agnostic.attention.kernels."
                                                       "fused_indexer_q")
    deepseek_v4_mla_sparse_kernel_module = import_module("vllm.models.deepseek_v4.hw_agnostic.attention.kernels."
                                                         "triton_mla_sparse")
    deepseek_v4_sparse_decode_kernel_module = import_module("vllm.models.deepseek_v4.hw_agnostic.attention.kernels."
                                                            "triton_sparse_decode_fp8")
    _ORIGINAL_DEEPSEEK_V4_SPARSE_DECODE_FP8 = (deepseek_v4_sparse_decode_kernel_module.triton_sparse_decode_fp8)
    _ORIGINAL_DEEPSEEK_V4_SWA_METADATA_BUILDER_INIT = (
        deepseek_v4_sparse_swa_module.DeepseekSparseSWAMetadataBuilder.__init__)

    def _hpu_deepseek_v4_swa_metadata_builder_init(self, *args, **kwargs) -> None:
        _ORIGINAL_DEEPSEEK_V4_SWA_METADATA_BUILDER_INIT(self, *args, **kwargs)
        self.is_valid_token = torch.ones_like(self.is_valid_token, dtype=torch.int32)

    deepseek_v4_sparse_swa_module.DeepseekSparseSWAMetadataBuilder.__init__ = (
        _hpu_deepseek_v4_swa_metadata_builder_init)

    if not hasattr(deepseek_v4_sparse_swa_module.triton, "next_power_of_2"):
        deepseek_v4_sparse_swa_module.triton.next_power_of_2 = _hpu_next_power_of_2

    deepseek_v4_fused_qk_rmsnorm_module.fused_q_kv_rmsnorm = (_hpu_fused_q_kv_rmsnorm)
    deepseek_v4_attention_module.fused_q_kv_rmsnorm = (_hpu_fused_q_kv_rmsnorm)
    deepseek_v4_attention_module.triton_inv_rope_einsum = (_hpu_dsv4_inv_rope_einsum)
    deepseek_v4_attention_module.DeepseekV4MultiHeadLatentAttentionWrapper.attn_gemm_parallel_execute = _hpu_attn_gemm_parallel_execute
    deepseek_v4_attention_module.DeepseekV4MultiHeadLatentAttentionWrapper.attention_frontend_impl = _hpu_deepseek_v4_attention_frontend_impl
    deepseek_v4_cache_utils_module.quantize_and_insert_k_cache = (_hpu_quantize_and_insert_k_cache)
    deepseek_v4_cache_utils_module.dequantize_and_gather_k_cache = (_hpu_dequantize_and_gather_k_cache)
    deepseek_v4_cache_utils_module.combine_topk_swa_indices = (_hpu_combine_topk_swa_indices)
    deepseek_v4_cache_utils_module.compute_global_topk_indices_and_lens = (_hpu_compute_global_topk_indices_and_lens)
    deepseek_v4_kernels_module.dequantize_and_gather_k_cache = (_hpu_dequantize_and_gather_k_cache)
    deepseek_v4_kernels_module.combine_topk_swa_indices = (_hpu_combine_topk_swa_indices)
    deepseek_v4_kernels_module.triton_bf16_mla_sparse_interface = (_hpu_bf16_mla_sparse_interface)
    deepseek_v4_mla_sparse_kernel_module.triton_bf16_mla_sparse_interface = (_hpu_bf16_mla_sparse_interface)
    deepseek_v4_sparse_decode_kernel_module.triton_bf16_mla_sparse_interface = (_hpu_bf16_mla_sparse_interface)
    deepseek_v4_sparse_decode_kernel_module.dequant_gather_slots = (_hpu_dequant_gather_slots)
    deepseek_v4_sparse_decode_kernel_module.triton_sparse_decode_fp8 = (_hpu_sparse_decode_fp8)
    deepseek_v4_attention_module.triton_sparse_decode_fp8 = (_hpu_sparse_decode_fp8)
    deepseek_v4_qnorm_rope_module.triton_qnorm_rope_kv_fp8_insert = (_hpu_qnorm_rope_kv_fp8_insert)
    deepseek_v4_attention_module.triton_qnorm_rope_kv_fp8_insert = (_hpu_qnorm_rope_kv_fp8_insert)
    deepseek_v4_fused_indexer_q_module.fused_indexer_q_rope_quant = (_hpu_fused_indexer_q_rope_quant)
    deepseek_v4_attention_module.fused_indexer_q_rope_quant = (_hpu_fused_indexer_q_rope_quant)
    deepseek_v4_sparse_attn_indexer_module.fp8_paged_mqa_logits_torch = (_hpu_fp8_paged_mqa_logits)
    deepseek_v4_attention_module.dequantize_and_gather_k_cache = (_hpu_dequantize_and_gather_k_cache)
    deepseek_v4_attention_module.combine_topk_swa_indices = (_hpu_combine_topk_swa_indices)
    deepseek_v4_attention_module.compute_global_topk_indices_and_lens = (_hpu_compute_global_topk_indices_and_lens)
    deepseek_v4_attention_module.triton_bf16_mla_sparse_interface = (_hpu_bf16_mla_sparse_interface)
    deepseek_v4_save_state_module.save_partial_states = (_hpu_save_partial_states)
    deepseek_v4_compressor_module.save_partial_states = (_hpu_save_partial_states)
    deepseek_v4_compress_kernel_module.compress_norm_rope_store_triton = (_hpu_compress_norm_rope_store)
    deepseek_v4_compressor_module.compress_norm_rope_store_triton = (_hpu_compress_norm_rope_store)
    deepseek_v4_sparse_mla_module._get_compressed_slot_mapping = (_hpu_get_compressed_slot_mapping)
    deepseek_v4_indexer_module._get_compressed_slot_mapping = (_hpu_get_compressed_slot_mapping)
    deepseek_v4_indexer_module.build_prefill_chunk_metadata = (_hpu_build_prefill_chunk_metadata)
    deepseek_v4_sparse_mla_module.build_c128a_topk_metadata = (_hpu_build_c128a_topk_metadata)
    deepseek_v4_sparse_swa_module._compute_prefill_metadata_kernel = (_HPUTorchKernel(_hpu_compute_prefill_metadata))
    deepseek_v4_sparse_swa_module._compute_swa_indices_and_lens_kernel = (
        _HPUTorchKernel(_hpu_compute_swa_indices_and_lens))

    hw_vocab_parallel_embedding_module = import_module(
        "vllm.model_executor.hw_agnostic.layers.vocab_parallel_embedding")
    hw_vocab_parallel_embedding_module.get_masked_input_and_mask = (_hpu_get_masked_input_and_mask)
