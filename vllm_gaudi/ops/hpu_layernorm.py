from typing import Optional, Union
import torch
from vllm.model_executor.layers.layernorm import \
    RMSNorm, GemmaRMSNorm


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
                from vllm.forward_context import get_forward_context
                from vllm_gaudi.distributed.tp2_fused_ar_norm import tp2_allreduce_residual_rms_norm

                attn_metadata = get_forward_context().attn_metadata
                is_prompt = bool(attn_metadata is not None and getattr(attn_metadata, "is_prompt", False))
                normalized, residual = tp2_allreduce_residual_rms_norm(
                    x.reshape(residual.shape),
                    residual,
                    self.weight,
                    self.variance_epsilon,
                    is_prompt=is_prompt,
                )
                return normalized.reshape(orig_shape), residual
            residual = residual + x.reshape(residual.shape)
            # Note: HPUFusedRMSNorm requires 3D tensors as inputs
            x = HPUFusedRMSNorm.apply(residual, self.weight, self.variance_epsilon)
            return x.reshape(orig_shape), residual

        x = HPUFusedRMSNorm.apply(x, self.weight, self.variance_epsilon)
        return x


@GemmaRMSNorm.register_oot
class HPUGemmaRMSNorm(GemmaRMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from vllm_gaudi.extension.kernels import rms_norm
        HPUFusedRMSNorm = rms_norm()
        # GemmaRMSNorm uses (1 + w) instead of w
        gemma_weight = self.weight + 1.0
        if residual is not None:
            orig_shape = x.shape
            residual = residual + x.reshape(residual.shape)
            # Note: HPUFusedRMSNorm requires 3D tensors as inputs
            x = HPUFusedRMSNorm.apply(residual, gemma_weight, self.variance_epsilon)
            return x.reshape(orig_shape), residual

        x = HPUFusedRMSNorm.apply(x, gemma_weight, self.variance_epsilon)
        return x
