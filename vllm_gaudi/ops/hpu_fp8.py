from functools import partial
from typing import Optional

import torch
from vllm_gaudi import envs
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory as FusedMoE

from vllm.model_executor.layers.quantization import fp8
from vllm.model_executor.layers.quantization.fp8 import (Fp8LinearMethod as OrigFp8LinearMethod, Fp8MoEMethod,
                                                         Fp8Config)
from vllm.model_executor.layers.quantization.online import fp8 as online_fp8
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod as OrigFp8PerTensorOnlineLinearMethod,
    _Fp8OnlineLinearBase,
)
from vllm.model_executor.utils import replace_parameter
import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi.extension.ops import (VllmMixtureOfExpertsOpFP8PerChannel, VllmMixtureOfExpertsOpFP8)
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.ops.hpu_fused_moe import (_normalize_moe_activation, model_has_quant_config, select_experts_from_routed)
from vllm_gaudi.v1.worker.hpu_dp_utils import dispatch_hidden_states, dispatch_tensor, get_hpu_dp_metadata

from vllm.model_executor.kernels.linear import _POSSIBLE_FP8_BLOCK_KERNELS, _POSSIBLE_FP8_KERNELS
from vllm.platforms import PlatformEnum
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    PerTensorTorchFP8ScaledMMLinearKernel,
    ChannelWiseTorchFP8ScaledMMLinearKernel,
)


class HPUPerTensorTorchFP8ScaledMMLinearKernel(PerTensorTorchFP8ScaledMMLinearKernel):

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None


class HPUChannelWiseTorchFP8ScaledMMLinearKernel(ChannelWiseTorchFP8ScaledMMLinearKernel):

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None


class HPUFp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """HPU stub for block-scaled FP8 linear.

    The actual computation is handled by HPU-specific ops in
    Fp8LinearMethod.apply(), so this kernel only needs to satisfy
    the kernel selection interface.
    """

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None

    def apply_weights(self, layer, x, bias=None):
        raise NotImplementedError("HPU uses Fp8LinearMethod.apply() directly")

    def apply_block_scaled_mm(self, A, B, As, Bs):
        raise NotImplementedError("HPU uses Fp8LinearMethod.apply() directly")


if PlatformEnum.OOT not in _POSSIBLE_FP8_KERNELS:
    _POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = [
        HPUPerTensorTorchFP8ScaledMMLinearKernel,
        HPUChannelWiseTorchFP8ScaledMMLinearKernel,
    ]

if PlatformEnum.OOT not in _POSSIBLE_FP8_BLOCK_KERNELS:
    _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT] = [
        HPUFp8BlockScaledMMLinearKernel,
    ]


class Fp8LinearMethod(OrigFp8LinearMethod):

    def create_weights(self, *args, **kwargs) -> None:
        # The range conversion wrapper is only valid for E4M3FN tensors and
        # their serialized inverse scales. Applying it to a BF16 checkpoint
        # doubles source weights before online quantization.
        if hpu_ops.is_hpu_gaudi2 and self.quant_config.is_checkpoint_fp8_serialized:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.quant_config = self.quant_config
        input_scale = None
        if self.block_quant:
            layer = hpu_ops.fp8_block_linear_postprocess_weights(layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
            return
        # If checkpoint not serialized fp8, quantize the weights.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = hpu_ops.scaled_fp8_quant(layer.weight, scale=None)
            weight = qweight.t()

        # If checkpoint is fp8 per-tensor, handle that there are N scales for N
        # shards in a fused module
        else:
            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.

            weight, weight_scale, input_scale = hpu_ops.process_fp8_weight_tensor_strategy(
                weight,
                weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            if self.act_q_static:
                assert input_scale is not None
                input_scale = input_scale.max()
            weight = weight.t()

        # Update layer with new values.
        layer.weight = Parameter(weight.data, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
        layer.input_scale = (Parameter(input_scale, requires_grad=False) if input_scale is not None else None)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return hpu_ops.apply_block_fp8_linear_hpu(
                input=x,
                layer=layer,
                block_size=self.quant_config.weight_block_size,
                bias=bias,
                do_unpad=True,
                force_channel_fp8=envs.VLLM_HPU_FORCE_CHANNEL_FP8,
            )

        weight_scale = layer.weight_scale.transpose(0, 1) if layer.weight_scale.dim() > 1 else layer.weight_scale
        input_scale = getattr(layer, 'input_scale', None)
        input_2d = x.view(-1, x.shape[-1])
        output = hpu_ops.apply_fp8_linear_hpu(input=input_2d,
                                              weight=layer.weight,
                                              weight_scale=weight_scale,
                                              input_scale=input_scale,
                                              bias=bias,
                                              trans_B=False)
        return output.view(*x.shape[:-1], -1)

    def dequant_fp8_weight(self, layer) -> torch.Tensor:
        if hasattr(layer, "updated_fp8_weight") and layer.updated_fp8_weight:
            return layer.weight
        dequant_weight = hpu_ops.dequant_block_fp8_weight_naive(
            layer.weight,
            layer.weight_scale_inv.data,
            self.quant_config.weight_block_size,
            original_M=layer.orig_M,
            original_N=layer.orig_N,
            do_unpad=True,
        )
        return dequant_weight


class HPUFp8OnlineLinearMethod(OrigFp8PerTensorOnlineLinearMethod):
    """Load BF16 weights and quantize them to PTPC FP8 on HPU.

    vLLM's generic online FP8 method dispatches CUDA custom quantization ops.
    Gaudi uses one scale per output channel and a dynamic scale per activation
    token, matching the serialized ModelOpt ``FP8_PER_CHANNEL_PER_TOKEN``
    execution layout.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # Skip the generic CUDA/ROCm kernel selector. The HPU apply path below
        # calls fp8_gemm_v2 directly after dynamic per-token quantization.
        _Fp8OnlineLinearBase.create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if layer.weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"HPU online FP8 requires floating-point source weights, got {layer.weight.dtype}")

        # Gaudi2's E4M3 MME range is 240 rather than E4M3FN's software range
        # of 448. Quantizing from BF16 directly avoids any serialized-format
        # range conversion and preserves the reconstructed FastH3 weights.
        amax = layer.weight.abs().amax(dim=-1, keepdim=True).float()
        weight_scale = (amax + 1e-8) / float(hpu_ops.FP8_MAX)
        qweight = torch.ops.hpu.cast_to_fp8_v2(
            layer.weight,
            weight_scale.reciprocal(),
            False,
            False,
            torch.float8_e4m3fn,
        )[0]

        replace_parameter(layer, "weight", qweight.t().data)
        replace_parameter(layer, "weight_scale", weight_scale.squeeze(-1).data)
        layer.input_scale = None
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = x.reshape(-1, x.shape[-1])
        output_shape = (*x.shape[:-1], layer.weight.shape[1])
        output = hpu_ops.apply_fp8_linear_hpu(
            input=input_2d,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=None,
            bias=bias,
            trans_B=False,
        )
        return output.narrow(0, 0, input_2d.shape[0]).reshape(output_shape)


class HPUFp8MoEMethod(Fp8MoEMethod):

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(quant_config, layer)

        # Disable marlin
        self.use_marlin = False
        self.fp8_backend = False

        # disable DeepGemm support.
        self.allow_deep_gemm = False

        self.use_dispatch_fn = get_config().use_dispatch_fn
        # Snapshot the (static) quant-config flag while the vLLM config context
        # is set; the forward hot path reads this cached value instead.
        self.has_moe_quant_config = model_has_quant_config()

    @property
    def is_monolithic(self) -> bool:
        return True

    def create_weights(self, *args, **kwargs) -> None:
        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        kwargs['weight_loader'] = hpu_ops.synced_weight_loader(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.local_num_experts
        ep_shift = layer.moe_config.ep_rank * num_experts

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        if layer.moe_config.dp_size > 1 and self.use_dispatch_fn:
            dispatch_fn = partial(dispatch_hidden_states, is_sequence_parallel=layer.moe_config.is_sequence_parallel)
        else:
            dispatch_fn = None

        if self.block_quant and not envs.VLLM_HPU_FORCE_CHANNEL_FP8:
            layer.moe_op = VllmMixtureOfExpertsOpFP8(
                layer.global_num_experts,
                num_experts,
                experts_min,
                experts_max,
                dispatch_fn,
            )
        else:
            layer.moe_op = VllmMixtureOfExpertsOpFP8PerChannel(
                layer.global_num_experts,
                num_experts,
                experts_min,
                experts_max,
                dispatch_fn,
            )
        if self.block_quant:
            layer = hpu_ops.fp8_block_moe_prepare_weights(layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
        else:
            if self.quant_config.activation_scheme == "static":
                if (layer.w13_input_scale is None or layer.w2_input_scale is None):
                    raise ValueError("QuantConfig has static quantization, but found "
                                     "activation scales are None.")
                layer.w13_input_scale = torch.nn.Parameter(layer.w13_input_scale.max(), requires_grad=False)
            layer = hpu_ops.fp8_channel_moe_prepare_weights(layer)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        is_sequence_parallel = layer.moe_config.is_sequence_parallel
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        # The HPU mixture_of_experts kernel (including the chunked
        # weighted_sum_reduction_bf16 reduction) compiles for int64 routing
        # tables and bf16 (x.dtype) router weights. The grouped-topk /
        # custom-routing helper returns int32 ids and float32 weights; the
        # regular-topk branch above already normalized them, but the grouped
        # path was previously left unconverted -> the bf16 reduction kernel
        # received float32 router_weights and failed to compile
        # (GLUE_INCOMPATIBLE_DATA_TYPE). Normalize for every routing path so the
        # kernel graph receives dtype-consistent inputs.
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)

        if layer.moe_config.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            if not (self.has_moe_quant_config and self.use_dispatch_fn):
                hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
                x = dispatch_tensor(x, hidden_states_across_dp, is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, is_sequence_parallel)
        elif is_sequence_parallel:
            # See HPUCompressedTensorsW8A8Fp8MoEMethod.apply_monolithic: at
            # dp_size == 1 with sequence-parallel MoE (TP>1 + EP),
            # MoERunner._maybe_combine reduce-scatters the expert output over the
            # EP group but no paired dispatch all-gather runs (dispatch_fn is
            # wired only for dp_size > 1). Restore symmetry by all-gathering the
            # inputs over the EP group so the combine leaves the token count
            # unchanged for the block's post-experts reshape.
            x = dispatch_tensor(x, None, is_sequence_parallel=True)
            topk_ids = dispatch_tensor(topk_ids, None, is_sequence_parallel=True)
            topk_weights = dispatch_tensor(topk_weights, None, is_sequence_parallel=True)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        output = layer.moe_op(
            x,
            topk_ids,
            topk_weights,
            permuted_weights=True,
            activation=_normalize_moe_activation(layer.activation),
        )
        return output.view(*(output.size(0), *input_shape[1:]))


fp8.Fp8LinearMethod = Fp8LinearMethod
fp8.Fp8MoEMethod = HPUFp8MoEMethod
online_fp8.Fp8PerTensorOnlineLinearMethod = HPUFp8OnlineLinearMethod

# OnlineQuantizationConfig stores method classes in a dispatch table at import
# time. Keep that newer entry point aligned when it is present, while the
# pinned Omni path continues to use Fp8Config(is_checkpoint_fp8_serialized=False).
try:
    from vllm.model_executor.layers.quantization.online import base as online_base
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym

    online_base._ONLINE_LINEAR_METHODS[kFp8StaticTensorSym] = HPUFp8OnlineLinearMethod
except (AttributeError, ImportError):
    pass
