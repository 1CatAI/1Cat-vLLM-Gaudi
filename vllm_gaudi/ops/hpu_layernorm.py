import os
from typing import Optional, Union
import torch
from vllm.model_executor.layers.layernorm import \
    RMSNorm, GemmaRMSNorm

if os.environ.get("VLLM_HPU_TRITON_MODE", "off").strip().lower() == "off":
    _fused_add_rms_norm = None
else:
    from vllm_gaudi.ops.triton_gaudi import fused_add_rms_norm as _fused_add_rms_norm


def _tp2_allreduce_residual_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    *,
    allow_fused: bool = True,
    prefer_direct: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.forward_context import get_forward_context
    from vllm_gaudi.distributed.tp2_fused_ar_norm import tp2_allreduce_residual_rms_norm

    attn_metadata = get_forward_context().attn_metadata
    is_prompt = bool(attn_metadata is not None and getattr(attn_metadata, "is_prompt", False))
    normalized, residual = tp2_allreduce_residual_rms_norm(
        x.reshape(residual.shape),
        residual,
        weight,
        epsilon,
        is_prompt=is_prompt,
        allow_fused=allow_fused,
        prefer_direct=prefer_direct,
    )
    return normalized.reshape(x.shape), residual


@RMSNorm.register_oot
class HPURMSNorm(RMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from vllm_gaudi.extension.kernels import rms_norm
        HPUFusedRMSNorm = rms_norm()
        if residual is not None:
            orig_shape = x.shape
            if getattr(self, "_hpu_tp2_fused_ar_norm", False):
                return _tp2_allreduce_residual_norm(x, residual, self.weight, self.variance_epsilon)
            reshaped_x = x.reshape(residual.shape)
            if _fused_add_rms_norm is not None:
                fused = _fused_add_rms_norm(
                    reshaped_x,
                    residual,
                    self.weight,
                    self.variance_epsilon,
                )
                if fused is not None:
                    x, residual = fused
                    return x.reshape(orig_shape), residual
            residual = residual + reshaped_x
            # Note: HPUFusedRMSNorm requires 3D tensors as inputs
            x = HPUFusedRMSNorm.apply(residual, self.weight, self.variance_epsilon)
            return x.reshape(orig_shape), residual

        x = HPUFusedRMSNorm.apply(x, self.weight, self.variance_epsilon)
        return x


@GemmaRMSNorm.register_oot
class HPUGemmaRMSNorm(GemmaRMSNorm):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__(hidden_size, eps)
        from vllm_gaudi import envs as gaudi_envs

        self._hpu_prepared_gemma_weight = gaudi_envs.VLLM_HPU_TP2_PREPARED_GEMMA_WEIGHT and hidden_size == 5120
        if self._hpu_prepared_gemma_weight:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
            from vllm.model_executor.model_loader.weight_utils import default_weight_loader
            from vllm_gaudi.ops.prepared_gemma_weight import install_decode_gemma_weight
            from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans

            if get_tensor_model_parallel_world_size() != 2:
                raise RuntimeError("Prepared Gemma decode weights require TP2")
            install_decode_gemma_weight(self, default_weight_loader, shutdown_prepared_group_plans)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from vllm_gaudi.extension.kernels import rms_norm
        HPUFusedRMSNorm = rms_norm()
        # GemmaRMSNorm uses (1 + w) instead of w
        gemma_weight = None
        if getattr(self, "_hpu_prepared_gemma_weight", False) and x.numel() == self.weight.numel():
            from vllm.forward_context import get_forward_context

            metadata = get_forward_context().attn_metadata
            if (metadata is not None and not getattr(metadata, "is_prompt", True)
                    and getattr(metadata, "direct_gdn_state", False)):
                if not self._hpu_decode_gemma_weight_ready:
                    raise RuntimeError("Prepared Gemma decode weights must be loaded before capture")
                gemma_weight = self._hpu_decode_gemma_weight
        if gemma_weight is None:
            gemma_weight = self.weight + 1.0
        if residual is not None:
            if getattr(self, "_hpu_tp2_fused_ar_norm", False):
                # Stock HCCL remains the default. The experimental native path
                # is marked ready only after both ranks pass a startup probe.
                return _tp2_allreduce_residual_norm(x,
                                                    residual,
                                                    gemma_weight,
                                                    self.variance_epsilon,
                                                    allow_fused=getattr(self, "_hpu_tp2_gemma_native_ready", False),
                                                    prefer_direct=getattr(self, "_hpu_tp2_gemma_native_ready", False))
            orig_shape = x.shape
            residual = residual + x.reshape(residual.shape)
            # Note: HPUFusedRMSNorm requires 3D tensors as inputs
            x = HPUFusedRMSNorm.apply(residual, gemma_weight, self.variance_epsilon)
            return x.reshape(orig_shape), residual

        x = HPUFusedRMSNorm.apply(x, gemma_weight, self.variance_epsilon)
        return x
