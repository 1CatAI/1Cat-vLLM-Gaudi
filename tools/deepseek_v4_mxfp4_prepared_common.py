# SPDX-License-Identifier: Apache-2.0
"""Shared inputs and references for the prepared MXFP4 candidate."""
from check_deepseek_v4_mxfp4_mme import make_inputs
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut, prepare_mxfp4_q16, prepare_mxfp4_s16

import torch


def compile_prepared_moe():
    """Compile the one-boundary prepared MoE including exact FP32 routing."""
    candidate = torch.compile(
        torch.ops.custom_op.custom_deepseek_v4_mxfp4_prepared_moe_bf16_gaudi2,
        backend="hpu_backend",
        fullgraph=True,
        dynamic=False,
    )

    def moe(x, ids, router, q13, q2, s13, s2, lookup,
            token_to_chunk, token_in_chunk, normal_scales=False):
        del token_to_chunk, token_in_chunk
        return candidate(
            x, ids, router, q13, q2, s13, s2, lookup, normal_scales
        )

    return moe


def make_prepared_inputs(sample_path=None):
    """Create full TP2-shape Q16/S16 tensors and the comparable standard inputs."""
    standard, ids, router = make_inputs(sample_path)
    q13 = torch.full((256, 16, 131072), 0x1111, dtype=torch.int16, device="hpu")
    q2 = torch.full((256, 32, 32768), 0x1111, dtype=torch.int16, device="hpu")
    s13 = torch.full((256, 16, 16384), 120 << 7, dtype=torch.int16, device="hpu").view(torch.bfloat16)
    s2 = torch.full((256, 32, 4096), 120 << 7, dtype=torch.int16, device="hpu").view(torch.bfloat16)
    for expert in ids.flatten().tolist():
        q13[expert].copy_(prepare_mxfp4_q16(standard[3][expert]))
        q2[expert].copy_(prepare_mxfp4_q16(standard[4][expert]))
        s13[expert].copy_(prepare_mxfp4_s16(standard[5][expert]))
        s2[expert].copy_(prepare_mxfp4_s16(standard[6][expert]))
    token_to_chunk = torch.arange(6, dtype=torch.int32, device="hpu").view(1, 6)
    token_in_chunk = torch.zeros((1, 6), dtype=torch.int32, device="hpu")
    prepared = [
        standard[0], standard[1], standard[2], q13, q2, s13, s2,
        mxfp4_bf16_lut("hpu"), token_to_chunk, token_in_chunk,
    ]
    torch.hpu.synchronize()
    return prepared, standard, ids, router


def native_reference(x, ids, router, w13, w2, s13, s2):
    return torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights(
        x,
        ids,
        router,
        w13,
        w2,
        s13,
        s2,
        block_size=32,
        permuted_weights=True,
        activation="silu",
        experts_min=0,
        experts_max=5,
        is_fp4=True,
        chunk_size=0,
        total_experts=0,
    )


def selected_native_inputs(standard, ids):
    selected = [tuple(t[expert] for expert in ids.flatten().tolist()) for t in standard[3:]]
    local_ids = torch.arange(6, dtype=torch.int32, device="hpu").view(1, 6)
    return [standard[0], local_ids, standard[2], *selected]


def comparison(actual, reference):
    actual_float, reference_float = actual.float(), reference.float()
    return {
        "exact": torch.equal(actual, reference),
        "nonzero": int(torch.count_nonzero(actual_float - reference_float)),
        "max_abs": float((actual_float - reference_float).abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(actual_float - reference_float)
            / torch.linalg.vector_norm(reference_float).clamp_min(1e-30)
        ),
    }
