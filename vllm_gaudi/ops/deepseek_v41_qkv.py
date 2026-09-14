# SPDX-License-Identifier: Apache-2.0
"""Main 92c82b97 Q/KV input fusion, shared by bounded and paged DSpark."""

import torch
import torch.nn.functional as F
from vllm_gaudi.ops.deepseek_v41_math import quantize_activation


class FusedQKVInput:
    def prepare_qkv_input_weight(self):
        """Bind one persistent Q/KV input matrix for the attention path.

        ``wq_a`` and ``wkv`` consume the same activation and currently each
        invokes the block quantization helper before its own GEMM.  The fused
        matrix keeps the original output ordering (Q first, KV second), so
        splitting the result is algebraically identical while allowing one
        quantization and one MME launch.  The two legacy weight attributes are
        rebound as views to avoid retaining a second device allocation.
        """
        if not self.qkv_fused_input or self._fused_qkv_weight is not None:
            return
        q_module, kv_module = self.weights.wq_a, self.weights.wkv
        q_weight, kv_weight = q_module.weight, kv_module.weight
        if q_weight.ndim != 2 or kv_weight.ndim != 2 or q_weight.shape[1] != kv_weight.shape[1]:
            raise ValueError("V4.1 QKV fusion requires matching input K dimensions")
        if q_weight.dtype != torch.bfloat16 or kv_weight.dtype != torch.bfloat16:
            raise ValueError("V4.1 QKV fusion requires prepared BF16 input weights")
        q_quantized = hasattr(q_module, "scale")
        kv_quantized = hasattr(kv_module, "scale")
        if q_quantized != kv_quantized:
            raise ValueError("V4.1 QKV fusion cannot combine mismatched activation contracts")
        fused = torch.cat((q_weight, kv_weight), dim=0).contiguous()
        self.register_buffer("fused_wqa_wkv", fused, False)
        # Keep compatibility with code which introspects the individual
        # matrices, while making both views point at the single allocation.
        q_module.weight = self.fused_wqa_wkv[:q_weight.shape[0]]
        kv_module.weight = self.fused_wqa_wkv[q_weight.shape[0]:]
        self._fused_qkv_weight = self.fused_wqa_wkv
        self._fused_qkv_quantized = q_quantized

    def invalidate_qkv_input_weight(self):
        """Drop a fused view before a prepared weight reload or migration."""
        if "fused_wqa_wkv" in self._buffers:
            self._buffers.pop("fused_wqa_wkv")
        self._fused_qkv_weight = None
        self._fused_qkv_quantized = False

    def _project_qkv_input(self, value):
        if self._fused_qkv_weight is None:
            query = self.linear(value, self.weights.wq_a)
            kv = self.linear(value, self.weights.wkv)
            return query, kv
        fused_value = quantize_activation(value) if self._fused_qkv_quantized else value
        qkv = F.linear(fused_value, self._fused_qkv_weight)
        q_width = self.weights.wq_a.weight.shape[0]
        return qkv[..., :q_width], qkv[..., q_width:]
