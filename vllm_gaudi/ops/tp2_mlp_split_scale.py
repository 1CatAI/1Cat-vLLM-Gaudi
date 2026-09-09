# SPDX-License-Identifier: Apache-2.0
"""Expose independent gate/up scaling to the decoder compiler."""

import torch
import torch.nn.functional as F

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm_gaudi.extension import ops as hpu_ops


def split_scaled_gate_up(x, weight, weight_scale):
    """Keep the MME BF16 result and the effective FP32 scale multiplication."""
    quantized, scale = hpu_ops.dynamic_quant(x)
    raw = torch.ops.hpu.fp8_gemm_v2(quantized, False, weight, True, None, torch.bfloat16, None, None)
    gate_raw, up_raw = raw.chunk(2, -1)
    gate_scale, up_scale = weight_scale.reshape(1, -1).chunk(2, -1)
    gate = (gate_raw.float() * (scale * gate_scale)).bfloat16()
    up = (up_raw.float() * (scale * up_scale)).bfloat16()
    return F.silu(gate) * up


class HpuSplitScaleMLP(Qwen2MoeMLP):
    """Retain the loaded modules and their names, including the row reduction."""

    def __init__(self, original):
        torch.nn.Module.__init__(self)
        self.gate_up_proj = original.gate_up_proj
        self.down_proj = original.down_proj
        self.act_fn = original.act_fn
        self.expert_gate = original.expert_gate
        self.training = original.training

    def forward(self, x):
        metadata = get_forward_context().attn_metadata
        decode = (x.dtype == torch.bfloat16 and x.shape[-1] == 5120 and x.numel() == 5120 and metadata is not None
                  and not getattr(metadata, "is_prompt", True) and getattr(metadata, "direct_gdn_state", False)
                  and getattr(metadata, "num_accepted_tokens", None) is None
                  and not getattr(metadata, "dflash_full_query", False))
        if not decode:
            return super().forward(x)
        activation = split_scaled_gate_up(x.reshape(1, 5120), self.gate_up_proj.weight,
                                          self.gate_up_proj.weight_scale_inv)
        output, _ = self.down_proj(activation.reshape(*x.shape[:-1], -1))
        return output


def _validate_mlp(mlp):
    from vllm_gaudi.ops.hpu_fp8 import Fp8LinearMethod

    if type(mlp) not in (Qwen2MoeMLP, HpuSplitScaleMLP) or mlp.expert_gate is not None:
        raise RuntimeError("Split MLP scaling requires the supported dense SiLU MLP")
    gate, down = mlp.gate_up_proj, mlp.down_proj
    if (type(gate) is not MergedColumnParallelLinear or type(down) is not RowParallelLinear
            or not isinstance(mlp.act_fn, SiluAndMul) or gate.tp_size != 2 or down.tp_size != 2 or gate.gather_output
            or not down.input_is_parallel or not gate.return_bias or not down.return_bias or gate.bias is not None
            or down.bias is not None):
        raise RuntimeError("Split MLP scaling does not support alternate projections, bias, LoRA or sequence parallel")
    for projection, shape in ((gate, (17408, 5120)), (down, (5120, 8704))):
        scale = getattr(projection, "weight_scale_inv", None)
        method = projection.quant_method
        if (type(method) is not Fp8LinearMethod or not method.block_quant
                or projection.weight.dtype != torch.float8_e4m3fn or tuple(projection.weight.shape) != shape
                or not projection.weight.is_contiguous() or scale is None or scale.dtype != torch.float32
                or tuple(scale.shape) not in ((shape[0], ), (1, shape[0]))):
            raise RuntimeError("Split MLP scaling requires loaded channel FP8 weights and the qualified geometry")


def prepare_mlp_split_scale(layers):
    """Validate the whole group topology before replacing any loaded module."""
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm_gaudi import envs

    if (not hpu_ops.is_hpu_gaudi2 or get_tensor_model_parallel_world_size() != 2 or not envs.VLLM_HPU_FORCE_CHANNEL_FP8
            or not envs.VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT):
        raise RuntimeError("Split MLP scaling requires Gaudi2 TP2 channel FP8 and native dynamic quantization")
    if not layers:
        raise RuntimeError("Split MLP scaling requires decoder layers")
    for layer in layers:
        if getattr(layer, "use_attn_reduce_scatter_for_moe", False):
            raise RuntimeError("Split MLP scaling does not support sequence parallel")
        _validate_mlp(getattr(layer, "mlp", None))
    for layer in layers:
        if not isinstance(layer.mlp, HpuSplitScaleMLP):
            layer.mlp = HpuSplitScaleMLP(layer.mlp)
