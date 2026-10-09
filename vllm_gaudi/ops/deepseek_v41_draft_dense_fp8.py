# SPDX-License-Identifier: Apache-2.0
"""Draft input/output projections reuse C1 FP8 preparation and consumers."""
import copy
import hashlib

import torch
from torch import nn
from vllm_gaudi.ops.deepseek_v41_math import quantize_activation
from vllm_gaudi.ops.deepseek_v41_qkv import FusedQKVInput


class DraftInputProjection(FusedQKVInput, nn.Module):
    def __init__(self, query, kv, linear):
        super().__init__()
        self.weights = nn.Module()
        self.weights.add_module('wq_a', query)
        self.weights.add_module('wkv', kv)
        self.linear = linear
        self.qkv_fused_input = True
        self._fused_qkv_weight = None
        self._fused_qkv_quantized = False
        self.prepare_qkv_input_weight()

    def forward(self, value):
        # Both original BF16 projections consume the same block32-rounded
        # activation. Keep that boundary once before the common C1 FP8 body.
        return self._project_qkv_input(quantize_activation(value))


def prepare(attention):
    import numpy as np
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import shapes_for_tp
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import covering_scale, encode_gaudi2

    records = {}

    def projection(name, direct):
        original = getattr(attention.weights, name)
        weight = original.weight
        expected = shapes_for_tp(attention.tensor_parallel_size, (name,))[name]
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != expected:
            raise ValueError('Draft dense FP8 requires the checkpoint BF16 TP projection')
        source = weight.detach().cpu().float().numpy()
        scales = covering_scale(np.max(np.abs(source), axis=1, keepdims=True))
        encoded = encode_gaudi2(source / scales)
        result = copy.copy(original)
        result._buffers = original._buffers.copy()
        result._parameters = original._parameters.copy()
        result._modules = original._modules.copy()
        result.weight = torch.from_numpy(encoded).view(torch.float8_e4m3fn).to(weight.device)
        result.register_buffer('channel_scale', torch.from_numpy(scales.T.copy()).to(weight.device), False)
        result.dense_fp8 = True
        result.dense_fp8_direct_input = direct
        records[name] = dict(shape=list(source.shape), source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                             encoded_sha256=hashlib.sha256(encoded.tobytes()).hexdigest())
        return result

    attention.add_module('draft_input_projection',
                         DraftInputProjection(projection('wq_a', True), projection('wkv', True), attention.linear))
    attention.add_module('draft_output_projection', projection('wo_b', False))
    attention.draft_dense_preparation = dict(quantization='C1 channel E4M3 RNE/FTZ',
                                             original_weights_retained=True, projections=records)
