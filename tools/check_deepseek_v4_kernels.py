# SPDX-License-Identifier: Apache-2.0
"""QNorm/RoPE/packed-cache source-build checks; no model or latency benchmark."""

from vllm_gaudi.entrypoints.deepseek_v4 import prepare_environment
prepare_environment()

import os
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])

HEAD_DIM = 512
FP8_DIM = 448
ROPE_DIM = 64
FP8_MAX = 448.0
QUANT_BLOCK = 64


def custom_op(
    q,
    kv,
    cache_storage,
    cache_geometry,
    slots,
    positions,
    norm_eps,
    cos_sin_cache,
):
    return (
        torch.ops.custom_op
        .custom_deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2(
            q,
            kv,
            cache_storage,
            cache_geometry,
            slots,
            positions,
            cos_sin_cache,
        )
    )


def hybrid_custom_op(
    q,
    kv,
    cache_storage,
    cache_geometry,
    slots,
    positions,
    norm_eps,
    cos_sin_cache,
):
    return (
        torch.ops.custom_op
        .custom_deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2(
            q,
            kv,
            cache_storage,
            cache_geometry,
            slots,
            positions,
            cos_sin_cache,
        )
    )


def encode_e4m3fn_bytes(values):
    abs_values = values.abs().clamp(max=FP8_MAX)
    is_subnormal = abs_values < 2.0**-6
    sub_mantissa = torch.round(abs_values * 512.0).to(torch.int32).clamp(0, 8)
    sub_code = torch.where(
        sub_mantissa == 8,
        torch.full_like(sub_mantissa, 8),
        sub_mantissa,
    )

    safe_values = abs_values.clamp_min(2.0**-6)
    exponent = torch.floor(torch.log2(safe_values))
    mantissa = torch.round(
        (safe_values / torch.exp2(exponent) - 1.0) * 8.0
    ).to(torch.int32)
    carry = mantissa == 8
    exponent = exponent.to(torch.int32) + carry.to(torch.int32)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    exponent_bits = (exponent + 7).clamp(1, 15)
    mantissa = torch.where(
        exponent_bits == 15,
        mantissa.clamp_max(6),
        mantissa,
    )
    magnitude = torch.where(
        is_subnormal,
        sub_code,
        exponent_bits * 8 + mantissa,
    )
    sign = torch.signbit(values).to(torch.int32) * 128
    return (magnitude + sign).to(torch.uint8)


def apply_pairwise_rope(values, positions, cos_sin_cache):
    num_tokens = values.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions.to(torch.int64))
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.view(num_tokens, 1, ROPE_DIM // 2)
    sin = sin.view(num_tokens, 1, ROPE_DIM // 2)
    pairs = values.float().view(
        num_tokens,
        values.shape[1],
        ROPE_DIM // 2,
        2,
    )
    real, imag = pairs.unbind(dim=-1)
    return torch.stack(
        (real * cos - imag * sin, imag * cos + real * sin),
        dim=-1,
    ).flatten(start_dim=-2)


def reference_op(q, kv, positions, norm_eps, cos_sin_cache):
    q_fp32 = q.float()
    q_normed = q_fp32 * torch.rsqrt(
        q_fp32.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    q_roped = apply_pairwise_rope(
        q_normed[..., FP8_DIM:],
        positions,
        cos_sin_cache,
    )
    q.copy_(torch.cat((q_normed[..., :FP8_DIM], q_roped), dim=-1).to(q.dtype))

    kv_roped = apply_pairwise_rope(
        kv[:, FP8_DIM:].view(kv.shape[0], 1, ROPE_DIM),
        positions,
        cos_sin_cache,
    ).view(kv.shape[0], ROPE_DIM)
    packed_kv = torch.cat((kv[:, :FP8_DIM], kv_roped.to(kv.dtype)), dim=-1)
    fp8_source = packed_kv[:, :FP8_DIM].float().view(
        kv.shape[0],
        FP8_DIM // QUANT_BLOCK,
        QUANT_BLOCK,
    )
    block_max = fp8_source.abs().amax(dim=-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(block_max / FP8_MAX))
    scale = torch.exp2(exponent).unsqueeze(-1)
    fp8_bytes = encode_e4m3fn_bytes(
        (fp8_source / scale).clamp(-FP8_MAX, FP8_MAX)
    ).view(kv.shape[0], FP8_DIM)
    bf16_bytes = packed_kv[:, FP8_DIM:].contiguous().view(torch.uint8)
    scale_bytes = torch.cat(
        (
            (exponent + 127.0).clamp(0.0, 255.0).to(torch.uint8),
            torch.zeros(
                (kv.shape[0], 1),
                dtype=torch.uint8,
                device=kv.device,
            ),
        ),
        dim=-1,
    )
    return torch.cat((fp8_bytes, bf16_bytes, scale_bytes), dim=-1)


def make_case(position, num_heads=32, num_tokens=1):
    torch.manual_seed(20260830 + position)
    q = (torch.randn(num_tokens, num_heads, HEAD_DIM) * 0.75).to(
        torch.bfloat16
    )
    qkv_storage = (
        torch.randn(num_tokens, 2 * HEAD_DIM) * 0.5
    ).to(torch.bfloat16)
    _, kv = qkv_storage.split(HEAD_DIM, dim=-1)
    positions = torch.arange(
        position,
        position + num_tokens,
        dtype=torch.int32,
    )
    norm_eps = torch.tensor([1e-6], dtype=torch.float32)
    angles = torch.randn(
        max(256, position + num_tokens),
        ROPE_DIM // 2,
        dtype=torch.float32,
    )
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    block_size = 64
    block_count = 4
    block_stride = block_size * (576 + 8) + 29
    base_offset = 37
    cache_storage = torch.randint(
        0,
        256,
        (base_offset + block_count * block_stride + 17,),
        dtype=torch.uint8,
    )
    cache_geometry = torch.tensor(
        [base_offset, block_stride, block_size, block_count],
        dtype=torch.int32,
    )
    slots = (
        torch.arange(num_tokens, dtype=torch.int32) + 65
    ).remainder(block_size * block_count)
    return (
        q,
        qkv_storage,
        cache_storage,
        cache_geometry,
        slots,
        positions,
        norm_eps,
        cos_sin_cache,
    )


def inputs_to_hpu(case):
    (
        q,
        qkv_storage,
        cache_storage,
        cache_geometry,
        slots,
        positions,
        norm_eps,
        cos_sin_cache,
    ) = case
    q_hpu = q.to("hpu")
    qkv_hpu = qkv_storage.to("hpu")
    _, kv_hpu = qkv_hpu.split(HEAD_DIM, dim=-1)
    return (
        q_hpu,
        kv_hpu,
        cache_storage.to("hpu"),
        cache_geometry.to("hpu"),
        slots.to("hpu"),
        positions.to("hpu"),
        norm_eps.to("hpu"),
        cos_sin_cache.to("hpu"),
    )


def cpu_reference(case):
    (
        q,
        qkv_storage,
        cache_storage,
        cache_geometry,
        slots,
        positions,
        norm_eps,
        cos_sin_cache,
    ) = case
    q_ref = q.clone()
    _, kv_ref = qkv_storage.split(HEAD_DIM, dim=-1)
    packed_ref = reference_op(
        q_ref,
        kv_ref,
        positions,
        norm_eps,
        cos_sin_cache,
    )
    expected_cache = cache_storage.clone()
    block_size = int(cache_geometry[2])
    for token_idx, slot_tensor in enumerate(slots):
        slot = int(slot_tensor)
        block = slot // block_size
        token = slot % block_size
        block_offset = int(cache_geometry[0]) + block * int(cache_geometry[1])
        data_offset = block_offset + token * 576
        scale_offset = block_offset + block_size * 576 + token * 8
        expected_cache[data_offset : data_offset + 576] = packed_ref[
            token_idx, :576
        ]
        expected_cache[scale_offset : scale_offset + 8] = packed_ref[
            token_idx, 576:
        ]
    return q_ref, packed_ref, expected_cache


def reference_inputs_to_hpu(case):
    inputs = inputs_to_hpu(case)
    return inputs[0], inputs[1], inputs[5], inputs[6], inputs[7]


def mismatch_details(actual, expected, limit=16):
    mismatch = actual != expected
    indices = mismatch.nonzero()[:limit]
    return [
        (
            tuple(int(component) for component in index),
            actual[tuple(index)].item(),
            expected[tuple(index)].item(),
        )
        for index in indices
    ]


def bf16_ulp_distance(actual, expected):
    def ordered_bits(values):
        raw = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        magnitude = raw & 0x7FFF
        return torch.where(
            (raw & 0x8000) != 0,
            0x8000 - magnitude,
            0x8000 + raw,
        )

    return (ordered_bits(actual) - ordered_bits(expected)).abs()


def validate(position, mode, candidate_fn, reference_fn, num_tokens=1):
    case = make_case(position, num_tokens=num_tokens)
    expected_q, expected_packed, expected_cache = cpu_reference(case)
    candidate_inputs = inputs_to_hpu(case)
    candidate_ack = candidate_fn(*candidate_inputs)
    reference_inputs = reference_inputs_to_hpu(case)
    reference_output = reference_fn(*reference_inputs)
    torch.hpu.synchronize()
    actual_q = candidate_inputs[0].cpu()
    actual_cache = candidate_inputs[2].cpu()
    actual_ack = candidate_ack.cpu()
    hpu_reference_q = reference_inputs[0].cpu()
    hpu_reference_packed = reference_output.cpu()

    cpu_q_mismatches = int((actual_q != expected_q).sum().item())
    hpu_q_mismatches = int((actual_q != hpu_reference_q).sum().item())
    cpu_q_ulp = bf16_ulp_distance(actual_q, expected_q)
    hpu_q_ulp = bf16_ulp_distance(actual_q, hpu_reference_q)
    cache_mismatches = int((actual_cache != expected_cache).sum().item())
    ack_mismatches = int((actual_ack != 1).sum().item())
    payload_hpu_mismatches = int(
        (expected_packed != hpu_reference_packed).sum().item()
    )
    print(
        "correctness",
        f"mode={mode}",
        f"position={position}",
        f"tokens={num_tokens}",
        f"q_vs_cpu_mismatch={cpu_q_mismatches}",
        f"q_vs_cpu_max_ulp={int(cpu_q_ulp.max().item())}",
        f"q_vs_hpu_mismatch={hpu_q_mismatches}",
        f"q_vs_hpu_max_ulp={int(hpu_q_ulp.max().item())}",
        f"cache_byte_mismatch={cache_mismatches}",
        f"ack_mismatch={ack_mismatches}",
        f"payload_vs_hpu_mismatch={payload_hpu_mismatches}",
    )
    if int(cpu_q_ulp.max().item()) > 1 or int(hpu_q_ulp.max().item()) > 1:
        raise AssertionError(
            "Q exceeds one BF16 ULP: "
            f"cpu={mismatch_details(actual_q, expected_q)}, "
            f"hpu={mismatch_details(actual_q, hpu_reference_q)}"
        )
    if cache_mismatches or ack_mismatches:
        raise AssertionError(
            "cache payload mismatch: "
            f"{mismatch_details(actual_cache, expected_cache)}"
        )


def validate_hybrid(position, mode, candidate_fn, num_tokens=1):
    case = make_case(position, num_tokens=num_tokens)
    baseline_inputs = inputs_to_hpu(case)
    baseline_ack = custom_op(*baseline_inputs)
    candidate_inputs = inputs_to_hpu(case)
    candidate_ack = candidate_fn(*candidate_inputs)
    torch.hpu.synchronize()
    baseline_q = baseline_inputs[0].cpu()
    baseline_cache = baseline_inputs[2].cpu()
    expected_hybrid_q = baseline_q.clone()
    expected_hybrid_q[..., :FP8_DIM] = (
        expected_hybrid_q[..., :FP8_DIM]
        .float()
        .to(torch.float8_e4m3fn)
        .to(torch.bfloat16)
    )
    actual_q = candidate_inputs[0].cpu()
    actual_cache = candidate_inputs[2].cpu()
    actual_ack = candidate_ack.cpu()
    baseline_ack = baseline_ack.cpu()
    q_mismatches = int((actual_q != expected_hybrid_q).sum().item())
    nope_mismatches = int(
        (
            actual_q[..., :FP8_DIM]
            != expected_hybrid_q[..., :FP8_DIM]
        ).sum().item()
    )
    rope_mismatches = int(
        (
            actual_q[..., FP8_DIM:]
            != expected_hybrid_q[..., FP8_DIM:]
        ).sum().item()
    )
    cache_mismatches = int((actual_cache != baseline_cache).sum().item())
    ack_mismatches = int((actual_ack != baseline_ack).sum().item())
    print(
        "hybrid_correctness",
        f"mode={mode}",
        f"position={position}",
        f"tokens={num_tokens}",
        f"q_mismatch={q_mismatches}",
        f"nope_mismatch={nope_mismatches}",
        f"rope_mismatch={rope_mismatches}",
        f"cache_byte_mismatch={cache_mismatches}",
        f"ack_mismatch={ack_mismatches}",
    )
    if q_mismatches or cache_mismatches or ack_mismatches:
        raise AssertionError(
            "hybrid qnorm mismatch: "
            f"q={mismatch_details(actual_q, expected_hybrid_q)}, "
            f"cache={mismatch_details(actual_cache, baseline_cache)}"
        )



if __name__ == "__main__":
    compiled = torch.compile(hybrid_custom_op, backend="hpu_backend", fullgraph=True)
    for position in (0, 7, 127):
        validate_hybrid(position, "direct", hybrid_custom_op)
        validate_hybrid(position, "compiled", compiled)
