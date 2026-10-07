# SPDX-License-Identifier: Apache-2.0
"""Traceable V4.1 arithmetic with explicit checkpoint encodings.

Algorithms follow DeepSeek-V4.1-Flash dba1be0 inference/model.py and kernel.py,
and vLLM #56214 e47aa780. Byte codecs avoid assuming that Gaudi2's default
FP8 conversion has the checkpoint's E4M3FN exponent range.
"""

import math

import torch
import torch.nn.functional as F

from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.ops.deepseek_v41_prefill_regions import prefill_function_region
from vllm_gaudi.ops.deepseek_v41_native_trace import prefill_span

NATIVE_KV_CODEC_TOKENS = 16384


def row_mean_square(value, *, request_batch=False):
    # Synapse uses a two-stage 24-part reduction for C1, but a different
    # accumulation tree for a generic batch. Keep each request's C1 shape in
    # the compiled graph. This loop is unrolled during compilation; replay
    # does not execute Python or copy statistics to the host.
    if request_batch and value.shape[0] > 1:
        return torch.cat(tuple(value[i : i + 1].square().mean(-1, keepdim=True) for i in range(value.shape[0])), dim=0)
    return value.square().mean(-1, keepdim=True)


def rms_norm(x, weight, eps=1e-20, *, request_batch=False):
    value = x.float()
    return (value * torch.rsqrt(row_mean_square(value, request_batch=request_batch) + eps) * weight.float()).to(x.dtype)


def final_rms_norm(x, weight, eps=1e-20):
    """Use the qualified low-launch-cost row norm for the decoder tail.

    The custom kernel is profitable for the B1/B2 production decode shapes,
    while the batch-generic graph remains faster at B32.  This dispatch is an
    internal shape choice under the same request-slot contract; it does not
    change the server batch capacity or create a C1-only execution path.
    """
    if (
        x.device.type == "hpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[-1] == 5120
        and x.shape[0] <= 2
        and hasattr(torch.ops.custom_op, "custom_deepseek_v41_attention_norm_bf16_gaudi2")
    ):
        return torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2(x.contiguous(), weight, eps)
    return rms_norm(x, weight, eps)


def final_collapse_rms_norm(residual, pre_mix, weight, eps=1e-20):
    """Collapse the four mHC streams and normalize at the PP1 tail.

    The native B1/B2 path retains the production BF16 boundary between the
    FP32 collapse and RMSNorm. Larger batches share the same state contract
    through the batch-generic graph.
    """
    if (
        residual.device.type == "hpu"
        and residual.dtype == torch.bfloat16
        and residual.ndim == 3
        and residual.shape[1:] == (4, 5120)
        and residual.shape[0] <= 2
        and pre_mix.dtype == torch.float32
        and pre_mix.shape == residual.shape[:2]
        and hasattr(torch.ops.custom_op, "custom_deepseek_v41_final_collapse_norm_bf16_gaudi2")
    ):
        return torch.ops.custom_op.custom_deepseek_v41_final_collapse_norm_bf16_gaudi2(
            residual.contiguous(), pre_mix.contiguous(), weight, eps
        )
    if (
        residual.device.type == "hpu"
        and residual.dtype == torch.bfloat16
        and residual.ndim == 3
        and residual.shape[1:] == (4, 5120)
        and 6 < residual.shape[0] <= NATIVE_KV_CODEC_TOKENS
        and pre_mix.dtype == torch.float32
        and pre_mix.shape == residual.shape[:2]
        and hasattr(torch.ops.custom_op, "custom_deepseek_v41_prefill_mhc_collapse_gaudi2")
    ):
        # Preserve the FP32 four-stream reduction and its BF16 boundary
        # without expanding the complete residual into FP32 temporaries.
        # The following norm keeps the same large-batch reduction as before.
        value = torch.ops.custom_op.custom_deepseek_v41_prefill_mhc_collapse_gaudi2(
            residual.contiguous(), pre_mix.contiguous()
        )
        return final_rms_norm(value, weight, eps)
    value = (residual.float() * pre_mix.unsqueeze(-1)).sum(1).to(residual.dtype)
    return final_rms_norm(value, weight, eps)


def e4m3_decode(code):
    bits = code.to(torch.int32)
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    normal = (1.0 + mantissa.float() / 8.0) * torch.exp2(exponent.float() - 7.0)
    value = torch.where(exponent == 0, mantissa.float() * 2**-9, normal)
    value = torch.where((bits & 127) == 127, float("nan"), value)
    return torch.where((bits & 128) != 0, -value, value)


def _negative(value):
    # HPU signbit may lower to a floating comparison, losing negative zero.
    dtype = torch.int16 if value.dtype in (torch.bfloat16, torch.float16) else torch.int32
    return value.contiguous().view(dtype) < 0


def e4m3_encode(value):
    """Finite saturating round-to-nearest-even; preserve the sign of zero."""
    magnitude = value.float().abs().clamp(max=448.0)
    exponent = torch.floor(torch.log2(magnitude.clamp_min(2**-6)))
    mantissa = torch.round(magnitude * torch.exp2(3.0 - exponent)).to(torch.int32)
    normal = ((exponent.to(torch.int32) + 7) << 3) + mantissa - 8
    subnormal = torch.round(magnitude * 512.0).to(torch.int32)
    code = torch.where(magnitude < 2**-6, subnormal, normal).clamp(0, 126)
    code = torch.where(torch.isnan(value), 127, code)
    return (code | (_negative(value).to(torch.int32) << 7)).to(torch.uint8)


def ue8m0_decode(code):
    value = torch.exp2(code.float() - 127.0)
    return torch.where(code == 255, float("nan"), value)


def fp4_encode(value):
    magnitude = value.float().abs()
    code = torch.zeros_like(magnitude, dtype=torch.int32)
    for index, boundary in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        upper = magnitude >= boundary if index % 2 else magnitude > boundary
        code = code + upper.to(torch.int32)
    return (code | (_negative(value).to(torch.int32) << 3)).to(torch.uint8)


def fp4_decode(code):
    magnitude = code.to(torch.int32) & 7
    value = torch.where(
        magnitude < 4,
        magnitude.float() * 0.5,
        torch.where(magnitude < 6, magnitude.float() - 2.0, magnitude.float() * 2.0 - 8.0),
    )
    return torch.where((code.to(torch.int32) & 8) != 0, -value, value)


def pack_swa(value):
    if (
        gaudi_envs.VLLM_HPU_DSV41_NATIVE_KV_PACK
        and value.device.type == "hpu"
        and value.dtype == torch.bfloat16
        and value.ndim >= 2
        and value.numel() // value.shape[-1] <= NATIVE_KV_CODEC_TOKENS
    ):
        shape = value.shape
        result = torch.ops.custom_op.custom_deepseek_v41_swa_pack_bf16_gaudi2(value.reshape(-1, shape[-1]).contiguous())
        return result.reshape(*shape[:-1], shape[-1] * 33 // 32)
    return _pack_swa_torch(value)


def _pack_swa_torch(value):
    groups = value.float().unflatten(-1, (-1, 32))
    maximum = groups.abs().amax(-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(maximum / 448.0))
    scale = torch.exp2(exponent)
    packed = e4m3_encode(groups / scale.unsqueeze(-1)).flatten(-2)
    packed = (packed & 127) | (_negative(value).to(torch.uint8) << 7)
    return torch.cat((packed, (exponent + 127).to(torch.uint8)), dim=-1)


def unpack_swa(packed, width=512):
    values = e4m3_decode(packed[..., :width]).unflatten(-1, (-1, 32))
    scales = ue8m0_decode(packed[..., width : width + width // 32])
    return (values * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def quantize_activation(value):
    if gaudi_envs.VLLM_HPU_DSV41_QUANT_ROUNDTRIP and value.device.type == "hpu" and value.dtype == torch.bfloat16:
        shape = value.shape
        # Retain the lower-startup codec for small request batches. The wide
        # path also serves the two B16 lanes of the B32 pipeline; count token
        # rows rather than treating independent heads as extra requests.
        operation = (
            torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_wide_bf16_gaudi2
            if gaudi_envs.VLLM_HPU_DSV41_PREFILL_VECTOR_QUANT and shape[0] >= 16
            else torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_bf16_gaudi2
        )
        result = operation(value.reshape(-1, shape[-1]).contiguous())
        return result.reshape(shape)
    return unpack_swa(pack_swa(value), value.shape[-1])


def pack_fp4(value, group=16):
    if (
        gaudi_envs.VLLM_HPU_DSV41_NATIVE_KV_PACK
        and value.device.type == "hpu"
        and value.dtype == torch.bfloat16
        and value.ndim >= 2
        and value.numel() // value.shape[-1] <= NATIVE_KV_CODEC_TOKENS
        and group in (16, 32)
    ):
        shape = value.shape
        op = (
            torch.ops.custom_op.custom_deepseek_v41_fp4_pack_g16_bf16_gaudi2
            if group == 16
            else torch.ops.custom_op.custom_deepseek_v41_fp4_pack_g32_bf16_gaudi2
        )
        result = op(value.reshape(-1, shape[-1]).contiguous())
        return result.reshape(*shape[:-1], shape[-1] // 2 + shape[-1] // group)
    return _pack_fp4_torch(value, group)


def _pack_fp4_torch(value, group=16):
    groups = value.float().unflatten(-1, (-1, group))
    maximum = groups.abs().amax(-1)
    if group == 16:
        code = e4m3_encode(maximum.clamp_min(6 * 2**-9) / 6.0)
        scale = e4m3_decode(code)
    else:
        exponent = torch.ceil(torch.log2(maximum.clamp_min(6 * 2**-126) / 6.0))
        code, scale = (exponent + 127).to(torch.uint8), torch.exp2(exponent)
    # Compare before division. HPU reciprocal-based division can move an exact
    # FP4 midpoint to either side of the tie; these binary thresholds times an
    # E4M3/power-of-two scale are exact in FP32 for BF16 activations.
    magnitude = groups.abs()
    codes = torch.zeros_like(groups, dtype=torch.int32)
    for index, boundary in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        threshold = scale.unsqueeze(-1) * boundary
        upper = magnitude >= threshold if index % 2 else magnitude > threshold
        codes = codes + upper.to(torch.int32)
    # At the minimum UE8M0 scale some thresholds are subnormal and TPC
    # flushes them to zero; exact zero must still select the zero FP4 code.
    codes = torch.where(magnitude == 0, 0, codes)
    values = codes.to(torch.uint8).flatten(-2) | (_negative(value).to(torch.uint8) << 3)
    packed = values[..., 0::2] | (values[..., 1::2] << 4)
    return torch.cat((packed, code), dim=-1)


def unpack_fp4(packed, width=512, group=16):
    values = packed[..., : width // 2]
    codes = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2)
    values = fp4_decode(codes).unflatten(-1, (-1, group))
    scale_codes = packed[..., width // 2 : width // 2 + width // group]
    scales = e4m3_decode(scale_codes) if group == 16 else ue8m0_decode(scale_codes)
    return (values * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def fp4_roundtrip(value, group=32):
    """Apply the checkpoint FP4 rounding contract without a packed HBM tensor."""
    if (
        gaudi_envs.VLLM_HPU_DSV41_NATIVE_KV_PACK
        and value.device.type == "hpu"
        and value.dtype == torch.bfloat16
        and value.ndim >= 2
        and value.numel() // value.shape[-1] <= 16384
        and group == 32
        and hasattr(torch.ops.custom_op, "custom_deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2")
    ):
        shape = value.shape
        result = torch.ops.custom_op.custom_deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2(
            value.reshape(-1, shape[-1]).contiguous()
        )
        return result.reshape(shape)
    return unpack_fp4(pack_fp4(value, group), value.shape[-1], group)


def rotary_table(width, length, base, original_length=0, factor=16, beta_fast=32, beta_slow=1):
    # CPU preparation preserves the reference's adjacent-pair RoPE convention.
    freq = 1.0 / (base ** (torch.arange(0, width, 2, dtype=torch.float32, device="cpu") / width))
    if original_length:

        def corrected(rotations):
            return width * math.log(original_length / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low, high = max(math.floor(corrected(beta_fast)), 0), min(math.ceil(corrected(beta_slow)), width - 1)
        ramp = ((torch.arange(width // 2, device="cpu") - low) / max(high - low, 1e-3)).clamp(0, 1)
        freq = freq / factor * ramp + freq * (1 - ramp)
    angles = torch.outer(torch.arange(length, device="cpu"), freq)
    return torch.stack((angles.cos(), angles.sin()), dim=-1)


def apply_rope(value, positions, table, inverse=False):
    if (
        gaudi_envs.VLLM_HPU_DSV41_NATIVE_ROPE
        and value.device.type == "hpu"
        and value.dtype == torch.bfloat16
        and value.ndim in (2, 3)
        and (value.ndim == 2 or 1 <= value.shape[1] <= 128)
        and 1 <= value.shape[0] <= 6
        and 128 <= value.shape[-1] <= 512
        and value.shape[-1] % 128 == 0
        and table.shape[-2:] == (32, 2)
    ):
        op = (
            torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2
            if inverse
            else torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2
        )
        shaped = value.reshape(value.shape[0], -1, value.shape[-1]).contiguous()
        return op(shaped, positions.to(torch.int32).contiguous(), table.reshape(-1, 64)).reshape(value.shape)
    return _apply_rope_torch(value, positions, table, inverse)


def _apply_rope_torch(value, positions, table, inverse=False):
    width = table.shape[-2] * 2
    pairs = value[..., -width:].float().unflatten(-1, (-1, 2))
    phase = table.index_select(0, positions.long())
    if value.ndim == 3:
        phase = phase.unsqueeze(1)
    real, imag = pairs[..., 0], pairs[..., 1]
    cosine, sine = phase[..., 0], phase[..., 1]
    if inverse:
        sine = -sine
    rotated = torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), -1)
    return torch.cat((value[..., :-width], rotated.flatten(-2).to(value.dtype)), dim=-1)


@prefill_function_region
def _prefill_hc_collapse(residual, previous_pre):
    """Collapse four streams without materializing the FP32 broadcast product."""
    return torch.ops.custom_op.custom_deepseek_v41_prefill_mhc_collapse_gaudi2(
        residual.contiguous(), previous_pre.contiguous()
    )


@prefill_span("mhc_pre")
def hc_pre(
    residual,
    previous_pre,
    fn,
    scale,
    base,
    eps=1e-20,
    hc_eps=1e-6,
    iterations=20,
    packed_fn=None,
    *,
    prefill=False,
    logical_tokens=None,
    request_batch=False,
    batch_control_reuse=False,
    batch_control_prefetch=False,
    collapsed_input=None,
):
    """vLLM mhc_pre_delayed_torch's previous-sublayer mixing contract."""
    copies = residual.shape[1]
    flat_bf16 = residual.flatten(1)
    gate_tokens = residual.shape[0] if logical_tokens is None else logical_tokens
    if not isinstance(gate_tokens, int) or gate_tokens < residual.shape[0]:
        raise ValueError("mHC logical token count must cover the owned rows")
    if (
        gaudi_envs.VLLM_HPU_DSV41_MHC_CONTROL_RRMS
        and packed_fn is not None
        and flat_bf16.device.type == "hpu"
        and flat_bf16.dtype == torch.bfloat16
        and 1 <= gate_tokens <= 2048
        and flat_bf16.shape[-1] == 20480
        and packed_fn.shape == (24, 20480)
    ):
        control_op = (torch.ops.custom_op.custom_deepseek_v41_control_rrms_parallel_bf16_gaudi2
                      if not prefill and gate_tokens == 1 and gaudi_envs.VLLM_HPU_DSV41_MHC_PARALLEL_CONTROL else
                      torch.ops.custom_op.custom_deepseek_v41_control_rrms_unpack_bf16_gaudi2
                      if not prefill and gaudi_envs.VLLM_HPU_DSV41_MHC_LINEAR_LOAD else
                      torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2)
        control = control_op(flat_bf16.contiguous(), packed_fn, eps)
        projection, rrms = control[:, :24], control[:, 24:]
    else:
        flat = flat_bf16.float()
        if (
            gaudi_envs.VLLM_HPU_DSV41_TPC_MHC
            and flat.device.type == "hpu"
            and flat.shape[1] == 20480
            and (flat.shape[0] == 1 or (request_batch and flat.shape[0] <= 64))
            and fn.shape == (24, 20480)
        ):
            op = (
                torch.ops.custom_op.custom_deepseek_v41_control_prefetch_f32_gaudi2
                if request_batch and batch_control_prefetch and flat.shape[0] >= 32
                else torch.ops.custom_op.custom_deepseek_v41_control_batch4_f32_gaudi2
                if request_batch and flat.shape[0] >= 4 and batch_control_reuse
                else torch.ops.custom_op.custom_deepseek_v41_control_gemv_f32_gaudi2
            )
            projection = op(flat, fn)
        else:
            projection = F.linear(flat, fn)
        rrms = torch.rsqrt(row_mean_square(flat, request_batch=request_batch) + eps)
    # The fused TPC gate is a decode/small-prefill win.  At C8192 its single
    # TPC program measures slower than the compiler's wide elementwise chain,
    # so retain the same math and native
    # Sinkhorn while letting large-M prefill use the better scheduled graph.
    if (
        gaudi_envs.VLLM_HPU_DSV41_MHC_GATES_FUSED
        and residual.device.type == "hpu"
        and iterations == 20
        and hc_eps == 1e-6
        and copies == 4
        and projection.ndim == 2
        and projection.shape[-1] == 24
        and gate_tokens <= 2048
    ):
        from vllm_gaudi.ops.deepseek_v41_mhc_gate_schedule import native_gates
        gates = native_gates(
            projection.contiguous(), rrms.contiguous(), scale.contiguous(), base.contiguous(), prefill=prefill
        )
        pre, post = gates[:, :copies], gates[:, copies : 2 * copies]
        comb = gates[:, 2 * copies :].reshape(-1, copies, copies)
    else:
        mixes = projection * rrms
        pre = torch.sigmoid(mixes[:, :copies] * scale[0] + base[:copies]) + hc_eps
        post = torch.sigmoid(mixes[:, copies : 2 * copies] * scale[1] + base[copies : 2 * copies]) * 2.0
        comb = mixes[:, 2 * copies :].reshape(-1, copies, copies) * scale[2]
        comb = torch.softmax(comb + base[2 * copies :].reshape(1, copies, copies), -1) + hc_eps
        if residual.device.type == "hpu" and iterations == 20 and hc_eps == 1e-6 and copies == 4:
            comb = torch.ops.custom_op.custom_deepseek_v4_sinkhorn4_gaudi2(comb.contiguous())
        else:
            comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
            for _ in range(iterations - 1):
                comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
                comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    if collapsed_input is not None:
        collapsed = collapsed_input
    elif (
        prefill
        and gaudi_envs.VLLM_HPU_DSV41_PREFILL_MHC_POST
        and residual.device.type == "hpu"
        and residual.dtype == torch.bfloat16
        and residual.shape[1:] == (4, 5120)
        and 1 <= residual.shape[0] <= NATIVE_KV_CODEC_TOKENS
    ):
        collapsed = _prefill_hc_collapse(residual, previous_pre)
    else:
        collapsed = (residual.float() * previous_pre.unsqueeze(-1)).sum(1).to(residual.dtype)
    return collapsed, pre, post, comb


def hc_control_and_collapse(residual, previous_pre, packed_fn, eps, *, collapsed_input=None, swizzled_fn=None,
                            bf16_fn=None):
    """Carry the prepared controller output to the post consumer.

    The packed 25-value output owns its storage across native recipe boundaries.
    Its final value is the production RRMS, so the post kernel need not repeat
    the residual reduction or change its FP32 sum order.
    """
    control_op = (torch.ops.custom_op.custom_deepseek_v41_control_rrms_parallel_bf16_gaudi2
                  if residual.shape[0] == 1 and gaudi_envs.VLLM_HPU_DSV41_MHC_PARALLEL_CONTROL else
                  torch.ops.custom_op.custom_deepseek_v41_control_rrms_unpack_bf16_gaudi2
                  if gaudi_envs.VLLM_HPU_DSV41_MHC_LINEAR_LOAD else
                  torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2)
    if residual.shape[0] == 1 and bf16_fn is not None:
        control_op = torch.ops.custom_op.custom_deepseek_v41_control_rrms_bf16_weight_gaudi2
        packed_fn = bf16_fn
    elif residual.shape[0] == 1 and swizzled_fn is not None:
        control_op = torch.ops.custom_op.custom_deepseek_v41_control_rrms_swizzled_bf16_gaudi2
        packed_fn = swizzled_fn
    control = control_op(residual.flatten(1).contiguous(), packed_fn, eps)
    collapsed = collapsed_input
    if collapsed is None:
        collapsed = (residual.float() * previous_pre.unsqueeze(-1)).sum(1).to(residual.dtype)
    return collapsed, control


def hc_post(value, residual, post, comb):
    mixed = (comb.unsqueeze(-1) * residual.float().unsqueeze(2)).sum(1)
    return (value.float().unsqueeze(1) * post.unsqueeze(-1) + mixed).to(value.dtype)


@prefill_span("mhc_input")
@prefill_function_region
def prefill_hc_input(
    residual, previous_pre, fn, scale, base, norm, eps, hc_eps, iterations, packed_fn, logical_tokens=None
):
    """Bind all mHC input-region tensors explicitly, including layer weights.

    The native collapse keeps the BF16 sublayer input boundary. FP32 control
    and norm expressions can fuse while state writes and TP collectives remain
    outside this bounded region.
    """
    value, pre, post, comb = hc_pre(
        residual,
        previous_pre,
        fn,
        scale,
        base,
        eps,
        hc_eps,
        iterations,
        packed_fn=packed_fn,
        prefill=True,
        logical_tokens=logical_tokens,
    )
    return value, pre, post, comb, rms_norm(value, norm, eps)


def bounded_prefill_hc_input(residual, previous_pre, fn, scale, base, norm, eps, hc_eps, iterations, packed_fn):
    """Research helper; complete-chain tests rejected production dispatch.

    This did not lower the peak of the already-compiled full-row region.
    Keep it isolated to the archived comparison tool, not the serving path.

    Each token's control projection/RRMS and collapse is independent. Reuse
    the existing C8192 FP32 region (never the small-token GEMV branch) for the
    C16384 bucket. The caller needs the four controls/normalized outputs, not
    the intermediate collapsed row, so no full-chunk copy of it is allocated.
    """
    tile, count = 8192, residual.shape[0]
    if count <= tile:
        return prefill_hc_input(residual, previous_pre, fn, scale, base, norm, eps, hc_eps, iterations, packed_fn)[1:]
    outputs = None
    for start in range(0, count, tile):
        stop = min(start + tile, count)
        block = prefill_hc_input(
            residual[start:stop], previous_pre[start:stop], fn, scale, base, norm, eps, hc_eps, iterations, packed_fn
        )[1:]
        if outputs is None:
            outputs = tuple(value.new_empty((count, *value.shape[1:])) for value in block)
        for destination, source in zip(outputs, block, strict=True):
            destination[start:stop].copy_(source)
        del block, source
    return outputs


def engram_update(residual, kv, q_weight, k_weight, active_mask, eps=1e-20):
    copies, width = residual.shape[1:]
    key = kv[:, : copies * width].float().reshape(-1, copies, width)
    value, h = kv[:, copies * width :].float(), residual.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * (q_weight.float() * k_weight.float()) * key).sum(-1) * rstd * width**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    gate = gate.masked_fill(~active_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * value.unsqueeze(1)).to(residual.dtype)


@prefill_function_region
def _prefill_engram_tile(residual, kv, q_weight, k_weight, active_mask, eps):
    return engram_update(residual, kv, q_weight, k_weight, active_mask, eps)


def prefill_engram_update(residual, kv, q_weight, k_weight, active_mask, eps=1e-20):
    """Bound row-independent gate/update intermediates after the full GEMM.

    Preserve every FP32 reduction and the final BF16 boundary. Token rows do
    not interact here; the scheduler and projection still own the full chunk.
    """
    tile = 512
    if residual.shape[0] <= tile:
        return _prefill_engram_tile(residual, kv, q_weight, k_weight, active_mask, eps)
    output = torch.empty_like(residual)
    for start in range(0, residual.shape[0], tile):
        stop = min(start + tile, residual.shape[0])
        output[start:stop].copy_(
            _prefill_engram_tile(residual[start:stop], kv[start:stop], q_weight, k_weight, active_mask[start:stop], eps)
        )
    return output
