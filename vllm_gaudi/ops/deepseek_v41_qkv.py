# SPDX-License-Identifier: Apache-2.0
"""Main 92c82b97 Q/KV input fusion, shared by bounded and paged DSpark."""

import torch
import torch.nn.functional as F
from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.ops.deepseek_v41_math import quantize_activation


def concatenate_static_weights(*weights):
    """Join immutable matrices without leaving a concat in an HPU recipe.

    These weights are prepared once while the model is loading.  On HPU the
    ordinary ``torch.cat`` is nevertheless submitted through the graph
    compiler, where the V2 segmented runtime can retain it until the first
    unrelated graph is compiled.  Materializing the small, immutable join on
    the host and uploading it once keeps that startup operation out of every
    captured decode graph and preserves the source FP32 bytes exactly.
    """
    if not weights:
        raise ValueError("At least one static weight is required")
    device = weights[0].device
    if any(weight.device != device for weight in weights):
        raise ValueError("Static projection weights must share a device")
    if device.type == "hpu":
        return torch.cat(tuple(weight.cpu() for weight in weights), dim=0).to(device)
    return torch.cat(weights, dim=0)


def direct_dense_fp8(value, weight, channel, *, prequant=None):
    q, scale = (torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(value.contiguous())
                if prequant is None else prequant)
    return torch.ops.hpu.fp8_gemm_v2(q, False, weight, True, None, torch.bfloat16, scale, channel, None, False)


class FusedQKVInput:

    def prepare_dense_bits12_weights(self):
        if not gaudi_envs.VLLM_HPU_DSV41_DSPARK_DENSE_BITS12:
            return
        from vllm_gaudi.ops.deepseek_v41_dense_bits12 import prepare

        if self._fused_qkv_weight is None or "fused_qkv_channel" in self._buffers:
            raise ValueError("Lossless dense packing requires the original fused BF16 Q/KV weights")
        prepare(self, self._fused_qkv_weight, "dense_bits12_qkv")
        for name in ("wq_b", "wo_b"):
            module = getattr(self.weights, name)
            if getattr(module, "dense_fp8", False) or getattr(module, "dense_fp8_direct_input", False):
                raise ValueError("Lossless BF16 storage cannot replace a different prepared FP8 numerical path")
            prepare(module, module.weight, "dense_bits12")

    def prepare_dense_kn_weights(self):
        """Keep the decoder's BF16 dense operands in contiguous K,N order.

        Prepare immutable operands once, without adding a transpose to replay.
        The alternate layout retains the original MME dtype, quantization and
        output rounding. Existing weights remain the prefill/C1 reference.
        """
        from vllm_gaudi import envs

        if not envs.VLLM_HPU_DSV41_DSPARK_DENSE_KN:
            return

        def transpose(weight):
            # A host materialization avoids a deferred startup transpose being
            # absorbed into an unrelated captured recipe.
            return weight.cpu().t().contiguous().to(weight.device)

        if self._fused_qkv_weight is not None and "fused_qkv_channel" not in self._buffers:
            weight = self._fused_qkv_weight
            if weight.dtype == torch.bfloat16:
                self.register_buffer("dspark_qkv_kn", transpose(weight), False)
        for name in ("wq_b", "wo_b"):
            module = getattr(self.weights, name)
            if (module.weight.dtype == torch.bfloat16 and not getattr(module, "dense_fp8", False)
                    and not getattr(module, "dense_fp8_direct_input", False)):
                module.register_buffer("dspark_dense_kn", transpose(module.weight), False)

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
        fp8 = getattr(q_module, "dense_fp8_direct_input", False)
        if fp8 != getattr(kv_module, "dense_fp8_direct_input", False):
            raise ValueError("V4.1 QKV fusion requires identical FP8 activation contracts")
        expected = torch.float8_e4m3fn if fp8 else torch.bfloat16
        if q_weight.dtype != expected or kv_weight.dtype != expected:
            raise ValueError("V4.1 QKV fusion requires matching prepared weight dtypes")
        q_quantized = hasattr(q_module, "scale")
        kv_quantized = hasattr(kv_module, "scale")
        if q_quantized != kv_quantized:
            raise ValueError("V4.1 QKV fusion cannot combine mismatched activation contracts")
        fused = concatenate_static_weights(q_weight, kv_weight).contiguous()
        self.register_buffer("fused_wqa_wkv", fused, False)
        # Keep compatibility with code which introspects the individual
        # matrices, while making both views point at the single allocation.
        q_module.weight = self.fused_wqa_wkv[:q_weight.shape[0]]
        kv_module.weight = self.fused_wqa_wkv[q_weight.shape[0]:]
        self._fused_qkv_weight = self.fused_wqa_wkv
        self._fused_qkv_quantized = q_quantized
        if fp8:
            channel = torch.cat((q_module.channel_scale.cpu(), kv_module.channel_scale.cpu()), dim=1).to(fused.device)
            self.register_buffer("fused_qkv_channel", channel, False)

    def invalidate_qkv_input_weight(self):
        """Drop a fused view before a prepared weight reload or migration."""
        if "fused_wqa_wkv" in self._buffers:
            self._buffers.pop("fused_wqa_wkv")
        self._buffers.pop("fused_qkv_channel", None)
        self._buffers.pop("dspark_qkv_kn", None)
        self._buffers.pop("dspark_hw_qkv", None)
        self._buffers.pop("dense_bits12_qkv_high", None)
        self._buffers.pop("dense_bits12_qkv_low", None)
        for name in ("wq_b", "wo_b"):
            module = getattr(self.weights, name, None)
            if module is not None:
                module._buffers.pop("dense_bits12_high", None)
                module._buffers.pop("dense_bits12_low", None)
        query = getattr(self.weights, "wq_b", None)
        if query is not None:
            query._buffers.pop("dspark_hw_weight", None)
        self._fused_qkv_weight = None
        self._fused_qkv_quantized = False

    def _project_qkv_input(self, value, *, token_group=None, prequant=None, roundtrip=None):
        if roundtrip is not None and (
                prequant is not None or self._fused_qkv_weight is None or not self._fused_qkv_quantized
                    or "fused_qkv_channel" in self._buffers or "dspark_hw_qkv" in self._buffers
                    or roundtrip.shape != value.shape or roundtrip.dtype != torch.bfloat16
                    or roundtrip.device != value.device or self._fused_qkv_weight.dtype != torch.bfloat16):
            raise ValueError("Prepared BF16 QKV roundtrip has an incompatible input/weight contract")
        if prequant is not None and (self._fused_qkv_weight is None or "fused_qkv_channel" not in self._buffers):
            raise ValueError("Prequantized QKV requires a prepared FP8 fused input weight")
        if self._fused_qkv_weight is None:
            if token_group is not None:
                raise ValueError("Token-owned QKV requires a prepared replicated fused input weight")
            query = self.linear(value, self.weights.wq_a)
            kv = self.linear(value, self.weights.wkv)
            return query, kv
        if "dspark_hw_qkv" in self._buffers and value.ndim == 2 and 2 <= value.shape[0] <= 6:
            from vllm_gaudi.ops.deepseek_v41_hw_dense import project
            scaled = (quantize_activation(value) if self._fused_qkv_quantized
                      and not gaudi_envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE_FUSED_QUANT else value)
            qkv = project(scaled, self.dspark_hw_qkv,
                          self.dspark_hw_qkv_input_scale, self.dspark_hw_qkv_weight_scale)
        elif "fused_qkv_channel" in self._buffers:
            qkv = (direct_dense_fp8(value, self._fused_qkv_weight, self.fused_qkv_channel) if prequant is None else
                   direct_dense_fp8(value, self._fused_qkv_weight, self.fused_qkv_channel, prequant=prequant))
        else:
            fused_value = (roundtrip if roundtrip is not None else
                           quantize_activation(value) if self._fused_qkv_quantized else value)
            prepared_kn = self._buffers.get("dspark_qkv_kn")
            if (gaudi_envs.VLLM_HPU_DSV41_DSPARK_DENSE_BITS12 and "dense_bits12_qkv_high" in self._buffers
                    and fused_value.ndim == 2 and 2 <= fused_value.shape[0] <= 6):
                from vllm_gaudi.ops.deepseek_v41_dense_bits12 import project
                qkv = project(fused_value, self, "dense_bits12_qkv")
            elif prepared_kn is not None and fused_value.ndim == 2 and 2 <= fused_value.shape[0] <= 6:
                qkv = torch.mm(fused_value, prepared_kn)
            else:
                qkv = F.linear(fused_value, self._fused_qkv_weight)
        if token_group is not None:
            from vllm_gaudi.ops.deepseek_v41_prefill_sequence_state import gather_tokens
            qkv = gather_tokens(qkv.contiguous(), group=token_group)
        q_width = self.weights.wq_a.weight.shape[0]
        return qkv[..., :q_width], qkv[..., q_width:]

    def _project_fp8_qkv_prologue(self, value, positions, *, prequant=None):
        """Feed the unscaled input GEMM accumulator into shared Q/KV norms.

        The C1 BF16 projection boundary is retained in TPC registers. The
        query and KV consumers receive their existing normalized/rotated
        operands, including the normalized query needed by the indexer.
        """
        query_weight = self.weights.wq_b
        if (self._fused_qkv_weight is None or "fused_qkv_channel" not in self._buffers
                or not getattr(query_weight, "dense_fp8", False)):
            raise ValueError("FP8 QKV prologue requires the prepared C1 input and query weights")
        q, scale = (torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(value.contiguous())
                    if prequant is None else prequant)
        prologue = (torch.ops.custom_op.custom_deepseek_v41_fp8_qkv_prologue_split_gaudi2
                    if self.fp8_qkv_prologue_split else
                    torch.ops.custom_op.custom_deepseek_v41_fp8_qkv_prologue_gaudi2)
        normalized, query, kv = prologue(
            q, scale, self._fused_qkv_weight, self.fused_qkv_channel,
            self.weights.q_norm.weight, self.weights.kv_norm.weight,
            query_weight.weight, query_weight.channel_scale,
            positions.to(torch.int32).contiguous(), self._rotary_native_table(), self.eps)
        return normalized, query.reshape(-1, self.heads, 512), kv


class FusedCompressorInput:
    """Load-time fusion for the ratio-2 CSA2 compressor projections."""

    def prepare_compressor_input_weight(self):
        if not self.compressor_fused_input or self._fused_compressor_weight is not None:
            return
        if not self.owns_kv or self.ratio != 2:
            return
        compressor = self.weights.compressor
        kv_weight = compressor.wkv.weight
        gate_weight = compressor.wgate.weight
        if kv_weight.ndim != 2 or gate_weight.ndim != 2 or kv_weight.shape[1] != gate_weight.shape[1]:
            raise ValueError("V4.1 Compressor fusion requires matching input K dimensions")
        if kv_weight.dtype != torch.float32 or gate_weight.dtype != torch.float32:
            raise ValueError("V4.1 ratio-2 Compressor fusion requires FP32 weights")
        if getattr(compressor.wkv, "bias", None) is not None or getattr(compressor.wgate, "bias", None) is not None:
            raise ValueError("V4.1 ratio-2 Compressor fusion does not support projection bias")
        fused = concatenate_static_weights(kv_weight, gate_weight).contiguous()
        self.register_buffer("fused_compressor_wkv_wgate", fused, False)
        self._fused_compressor_kv_width = kv_weight.shape[0]
        compressor.wkv.weight = self.fused_compressor_wkv_wgate[:kv_weight.shape[0]]
        compressor.wgate.weight = self.fused_compressor_wkv_wgate[kv_weight.shape[0]:]
        self._fused_compressor_weight = self.fused_compressor_wkv_wgate

    def invalidate_compressor_input_weight(self):
        if "fused_compressor_wkv_wgate" in self._buffers:
            self._buffers.pop("fused_compressor_wkv_wgate")
        self._fused_compressor_weight = None
        self._fused_compressor_kv_width = 0

    def _project_compressor_input(self, value):
        compressor = self.weights.compressor
        value = value.float()
        if self._fused_compressor_weight is None:
            return self.linear(value, compressor.wkv).float(), self.linear(value, compressor.wgate).float()
        projected = F.linear(value, self._fused_compressor_weight)
        width = self._fused_compressor_kv_width
        return projected[..., :width], projected[..., width:]
