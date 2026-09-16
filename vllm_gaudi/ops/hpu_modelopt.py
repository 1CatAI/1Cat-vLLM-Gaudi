# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm_gaudi.extension.ops as hpu_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

try:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
except ImportError:
    try:
        from vllm.model_executor.layers.fused_moe import RoutedExperts
    except ImportError:

        class RoutedExperts:  # type: ignore[no-redef]
            """Sentinel for vLLM revisions without RoutedExperts."""


logger = init_logger(__name__)

_SUPPORTED_FP8_ALGOS = ("FP8", "FP8_PER_CHANNEL_PER_TOKEN")


def _checkpoint_weight_loader(layer: torch.nn.Module, extra_weight_attrs: dict):
    """Build a synchronized loader with Gaudi2 E4M3 range conversion."""

    weight_loader = extra_weight_attrs.get("weight_loader")
    if hasattr(layer, "weight_loader_v2"):
        weight_loader = layer.weight_loader_v2
    if weight_loader is None:
        raise ValueError("ModelOpt HPU linear requires a checkpoint weight loader")
    if hpu_ops.is_hpu_gaudi2:
        # ModelOpt stores E4M3FN values. Gaudi2 uses the OCP E4M3 range, so
        # halve FP8 values and double inverse scales while preserving W exactly.
        weight_loader = hpu_ops.gaudi_weight_wrapper(weight_loader)
    return hpu_ops.synced_weight_loader(weight_loader)


class HPUModelOptFp8Config(ModelOptFp8Config):
    """ModelOpt FP8 configuration backed by Gaudi FP8 MME kernels."""

    def __init__(
        self,
        quant_method: str,
        is_checkpoint_fp8_serialized: bool,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
    ) -> None:
        super().__init__(quant_method, is_checkpoint_fp8_serialized, kv_cache_quant_method, exclude_modules)
        if self.quant_method == "FP8":
            self.LinearMethodCls = HPUModelOptFp8LinearMethod
        elif self.quant_method == "FP8_PER_CHANNEL_PER_TOKEN":
            self.LinearMethodCls = HPUModelOptFp8PcPtLinearMethod
        else:
            raise ValueError("Unsupported ModelOpt FP8 quant_algo on Gaudi: "
                             f"{self.quant_method}. Supported: {', '.join(_SUPPORTED_FP8_ALGOS)}.")

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, (Attention, MLAAttention)):
            return self.KVCacheMethodCls(self)

        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            return None

        if "vision_tower" in prefix or "vision_model" in prefix or "vit_large_projector" in prefix:
            return UnquantizedLinearMethod()

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return self.LinearMethodCls(self)
        if isinstance(layer, RoutedExperts):
            raise ValueError("FP8 ModelOpt fused MoE quantization is not supported on Gaudi")
        return None


class HPUModelOptFp8LinearMethod(LinearMethodBase):
    """Static per-tensor ModelOpt FP8 linear using Gaudi FP8 MME."""

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config

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
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = _checkpoint_weight_loader(layer, extra_weight_attrs)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        weight_dtype = torch.float8_e4m3fn if self.quant_config.is_checkpoint_fp8_serialized else params_dtype
        weight = ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=weight_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.quant_config.is_checkpoint_fp8_serialized:
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            weight_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("weight_scale", weight_scale)

            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight
        max_w_scale = layer.weight_scale.max()
        if not (layer.weight_scale == layer.weight_scale[0]).all():
            max_w_scale, weight = hpu_ops.requantize_with_max_scale(
                layer.weight,
                layer.weight_scale,
                layer.logical_widths,
            )
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight_scale = layer.weight_scale.transpose(0, 1) if layer.weight_scale.dim() > 1 else layer.weight_scale
        input_scale = getattr(layer, "input_scale", None)
        input_2d = x.reshape(-1, x.shape[-1])
        output_shape = (*x.shape[:-1], layer.weight.shape[1])
        output = hpu_ops.apply_fp8_linear_hpu(
            input=input_2d,
            weight=layer.weight,
            weight_scale=weight_scale,
            input_scale=input_scale,
            bias=bias,
            trans_B=False,
        )
        return output.narrow(0, 0, input_2d.shape[0]).reshape(output_shape)


class HPUModelOptFp8PcPtLinearMethod(LinearMethodBase):
    """Per-channel weight/per-token activation ModelOpt FP8 linear.

    Checkpoint FP8 weights stay quantized. Each flattened BF16 token receives
    a dynamic activation scale and executes through ``hpu.fp8_gemm_v2`` with
    the checkpoint's per-output-channel inverse weight scales.
    """

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config

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
        del input_size, output_size, params_dtype
        if not self.quant_config.is_checkpoint_fp8_serialized:
            raise ValueError("FP8_PER_CHANNEL_PER_TOKEN requires an FP8-serialized checkpoint")

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = _checkpoint_weight_loader(layer, extra_weight_attrs)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(output_size_per_partition, dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise TypeError("FP8_PER_CHANNEL_PER_TOKEN checkpoint weight must remain FP8 before HPU post-processing, "
                            f"got {layer.weight.dtype}")
        if layer.weight_scale.dtype != torch.float32 or layer.weight_scale.shape != (layer.weight.shape[0], ):
            raise ValueError("FP8_PER_CHANNEL_PER_TOKEN requires one FP32 inverse scale per output channel; "
                             f"got dtype={layer.weight_scale.dtype}, shape={tuple(layer.weight_scale.shape)}")
        layer.weight = Parameter(layer.weight.data.t(), requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data.contiguous(), requires_grad=False)

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


modelopt.ModelOptFp8Config = HPUModelOptFp8Config
modelopt.ModelOptFp8LinearMethod = HPUModelOptFp8LinearMethod
modelopt.ModelOptFp8PcPtLinearMethod = HPUModelOptFp8PcPtLinearMethod
