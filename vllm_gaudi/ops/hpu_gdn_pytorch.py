# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU-native PyTorch implementations for Qwen3.5 GDN ops.

These implementations intentionally avoid Triton/CUDA-only kernels and run
entirely with PyTorch tensor ops on the active device (HPU for Gaudi runs).
Phase 1 scope:
- non-mixed prefill/decode support
- no speculative decode tensor layout support yet
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import vllm_gaudi.envs as gaudi_envs
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

# Set VLLM_GDN_LEGACY_PHASE_B=1 to use the original _phase_b_step-based loop
# in hpu_chunk_gdr_phase_b.  Slower but numerically identical to the reference
# implementation — useful for debugging accuracy issues.
_USE_LEGACY_PHASE_B = os.getenv("VLLM_GDN_LEGACY_PHASE_B", "0") == "1"

# Set VLLM_GDN_COMPUTE_FP32=0 to opt the general GDN implementation into
# bfloat16 compute. The established fallback remains FP32; promoted tactics
# can request BF16 explicitly without changing unsupported model shapes.
_GDN_COMPUTE_DTYPE = torch.float32 if os.getenv("VLLM_GDN_COMPUTE_FP32", "1") == "1" else torch.bfloat16

# NVIDIA's optimized GDN kernels keep the numerically sensitive triangular
# inverse and recurrent state in FP32 while feeding the bulk Q/K/V products
# with BF16. These independent switches let the HPU path retain that boundary
# without promoting every MME operation to FP32.
_GDN_SOLVE_FP32 = os.getenv("VLLM_GDN_SOLVE_FP32", "0") == "1"
_GDN_STATE_FP32 = os.getenv("VLLM_GDN_STATE_FP32", "0") == "1"

# Set VLLM_GDN_EXACT_SOLVE=1 to use exact row-by-row forward substitution
# instead of the Neumann iterative solver.  Exact but ~2.6x slower (127
# Python-loop iterations for chunk_size=128).  Useful for isolating
# accuracy issues to the solver vs other sources.
_USE_EXACT_SOLVE = os.getenv("VLLM_GDN_EXACT_SOLVE", "0") == "1"

# FlashQLA's portable algebra removes the pairwise gate exponential from the
# triangular solve and phase-B local attention. Keep these import-time flags
# next to the other graph-shaping switches so torch.compile sees one static
# implementation per server process.
_USE_FLASHQLA_REFORMULATION = gaudi_envs.VLLM_GDN_FLASHQLA
_FLASHQLA_FACTORIZE_PHASE_B = gaudi_envs.VLLM_GDN_FLASHQLA_FACTORIZE_PHASE_B
_FLASHQLA_FP32_SCALING = gaudi_envs.VLLM_GDN_FLASHQLA_FP32_SCALING
_USE_QWEN38_NATIVE_COMPACT_KKT = (
    gaudi_envs.VLLM_GDN_QWEN38_NATIVE_COMPACT_KKT
)

_USE_BF16_BMM_F32 = gaudi_envs.VLLM_GDN_BF16_BMM_F32
_USE_SOLVE_BF16_BMM_F32 = gaudi_envs.VLLM_GDN_SOLVE_BF16_BMM_F32
_USE_NATIVE_RECURRENT_SCAN = gaudi_envs.VLLM_GDN_NATIVE_RECURRENT_SCAN
if (
    _USE_BF16_BMM_F32
    or _USE_SOLVE_BF16_BMM_F32
    or _USE_NATIVE_RECURRENT_SCAN
):
    _mixed_bmm_extension = Path(
        gaudi_envs.VLLM_GDN_BF16_BMM_F32_EXTENSION,
    ).expanduser().resolve()
    if not _mixed_bmm_extension.is_file():
        raise RuntimeError(
            "VLLM_GDN_BF16_BMM_F32_EXTENSION must point to the built "
            "flashqla_compound_probe extension."
        )
    torch.ops.load_library(str(_mixed_bmm_extension))
    try:
        torch.ops.custom_op.flashqla_bf16_bmm_f32
    except AttributeError as exc:
        raise RuntimeError(
            "The configured extension does not register "
            "custom_op::flashqla_bf16_bmm_f32."
        ) from exc
    if _USE_NATIVE_RECURRENT_SCAN:
        try:
            torch.ops.custom_op.flashqla_recurrent_scan_probe
        except AttributeError as exc:
            raise RuntimeError(
                "The configured extension does not register "
                "custom_op::flashqla_recurrent_scan_probe."
            ) from exc
        if _GDN_STATE_FP32 or _USE_BF16_BMM_F32:
            raise RuntimeError(
                "VLLM_GDN_NATIVE_RECURRENT_SCAN requires BF16 recurrent "
                "state and cannot use the FP32-output BMM path."
            )

def resolve_hpu_gdn_chunk_size(model_config) -> tuple[int, bool]:
    """Resolve the HPU GDN chunk size and whether bucketing must align to it."""
    hf_text_config = getattr(model_config, "hf_text_config", None)
    model_has_explicit_size = (
        hf_text_config is not None
        and (
            getattr(hf_text_config, "mamba_chunk_size", None) is not None
            or getattr(hf_text_config, "chunk_size", None) is not None
        )
    )

    override = gaudi_envs.VLLM_GDN_CHUNK_SIZE
    if override != 0:
        if override < 32 or override % 32 != 0:
            raise ValueError(
                "VLLM_GDN_CHUNK_SIZE must be 0 or a positive multiple of 32, "
                f"got {override}."
            )
        return override, True

    if model_has_explicit_size:
        return model_config.get_mamba_chunk_size(), True
    return 128, False


def resolve_hpu_gdn_neumann_iters() -> int:
    """Resolve the iteration budget for the approximate triangular solve."""
    neumann_iters = gaudi_envs.VLLM_GDN_NEUMANN_ITERS
    if neumann_iters <= 0:
        raise ValueError(
            "VLLM_GDN_NEUMANN_ITERS must be a positive integer, "
            f"got {neumann_iters}."
        )
    return neumann_iters


def resolve_hpu_gdn_fused_state_matmul() -> bool:
    """Resolve whether GDN phase B fuses its two state projections."""
    return gaudi_envs.VLLM_GDN_FUSED_STATE_MATMUL


def resolve_hpu_gdn_deferred_output_add() -> bool:
    """Resolve whether phase B avoids in-place writes to output views."""
    return gaudi_envs.VLLM_GDN_DEFERRED_OUTPUT_ADD


def resolve_hpu_gdn_recursive_solver_base() -> int:
    """Resolve the recursive GDN unit-lower inverse base size."""
    base = gaudi_envs.VLLM_GDN_RECURSIVE_SOLVER_BASE
    if base != 0 and (base < 2 or base & (base - 1)):
        raise ValueError(
            "VLLM_GDN_RECURSIVE_SOLVER_BASE must be 0 or a power of two "
            f"greater than one, got {base}."
        )
    return base


def resolve_hpu_gdn_compact_repeated_kkt() -> bool:
    """Resolve whether repeated GDN key heads share their KKT product."""
    return gaudi_envs.VLLM_GDN_COMPACT_REPEATED_KKT


def resolve_hpu_gdn_compact_repeated_local_attn() -> bool:
    """Resolve whether repeated GDN Q/K heads share their local product."""
    return gaudi_envs.VLLM_GDN_COMPACT_REPEATED_LOCAL_ATTN


def resolve_hpu_gdn_compiled_qk_l2norm() -> bool:
    """Resolve whether GDN Q/K L2Norm remains in the compiled graph."""
    return gaudi_envs.VLLM_GDN_COMPILED_QK_L2NORM


def resolve_hpu_gdn_fused_rmsnorm_gated() -> bool:
    """Resolve whether Qwen GDN prefill uses Habana FusedRMSNorm."""
    return gaudi_envs.VLLM_GDN_FUSED_RMSNORM_GATED


def hpu_fused_rmsnorm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    activation: str,
) -> torch.Tensor:
    """Apply HPU FusedRMSNorm followed by the Qwen GDN output gate."""
    if activation not in ("silu", "swish", "sigmoid"):
        raise ValueError(f"Unsupported GDN output gate activation: {activation}.")

    from vllm_gaudi.extension.kernels import rms_norm

    original_shape = x.shape
    fused_rms_norm = rms_norm()
    if fused_rms_norm is None:
        raise RuntimeError(
            "Habana FusedRMSNorm is unavailable on this HPU software stack."
        )
    normalized = fused_rms_norm.apply(
        x.reshape(1, -1, x.shape[-1]),
        weight,
        epsilon,
    ).reshape(original_shape)
    z_float = z.to(torch.float32)
    gate = (
        torch.sigmoid(z_float)
        if activation == "sigmoid"
        else torch.nn.functional.silu(z_float)
    )
    return (normalized.to(torch.float32) * gate).to(x.dtype)


def _preprocess_qk_l2norm_compiled(q, k):
    """Normalize Q/K inside the compiled graph for validated HPU stacks."""
    q = _l2norm_last_dim(q.to(torch.float32))
    k = _l2norm_last_dim(k.to(torch.float32))
    return q, k


@torch._dynamo.disable
def _preprocess_qk_l2norm(q, k):
    """Normalize Q/K eagerly for compatibility with older HPU compilers."""
    q = _l2norm_last_dim(q.to(torch.float32))
    k = _l2norm_last_dim(k.to(torch.float32))
    return q, k


def hpu_chunk_gdr_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    chunk_size: int,
    num_seqs: int,
    seq_len: int,
    compile_qk_l2norm: bool = False,
    compute_dtype: torch.dtype | None = None,
    preserve_compact_qk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, float, int,
           int, int]:
    """Preprocessing stage of chunk GDR: head repeat, l2norm, flatten, cumsum.

    Returns (qf, kf, vf, bf, g_cumsum, init_state,
             H, num_chunks, scale, Kdim, Vdim, S).
    """
    _, _, qk_heads, Kdim = q.shape
    _, _, HV, Vdim = v.shape
    device = q.device

    if qk_heads != HV:
        if HV % qk_heads == 0:
            repeat = HV // qk_heads
            if not preserve_compact_qk:
                q = q.repeat_interleave(repeat, dim=2)
                k = k.repeat_interleave(repeat, dim=2)
                qk_heads = HV
        else:
            raise ValueError(
                "Unsupported head mapping: "
                f"q/k heads={qk_heads}, value heads={HV}."
            )

    if use_qk_l2norm_in_kernel:
        if compile_qk_l2norm:
            q, k = _preprocess_qk_l2norm_compiled(q, k)
        else:
            q, k = _preprocess_qk_l2norm(q, k)

    if scale is None:
        scale = k.shape[-1]**-0.5

    selected_compute_dtype = _GDN_COMPUTE_DTYPE if compute_dtype is None else compute_dtype
    qf = q.reshape(-1, qk_heads, Kdim).to(selected_compute_dtype)
    kf = k.reshape(-1, qk_heads, Kdim).to(selected_compute_dtype)
    vf = v.reshape(-1, HV, Vdim).to(selected_compute_dtype)
    gf = g.reshape(-1, HV).to(torch.float32)
    bf = beta.reshape(-1, HV).to(selected_compute_dtype)

    S = num_seqs
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    total_tokens = S * seq_len

    g_active = gf[:total_tokens]
    padded_len = num_chunks * chunk_size
    g_block = g_active.reshape(S, seq_len, HV)
    if padded_len > seq_len:
        pad_block = torch.zeros(
            S,
            padded_len - seq_len,
            HV,
            dtype=gf.dtype,
            device=device,
        )
        g_block = torch.cat((g_block, pad_block), dim=1)
    g_block = g_block.reshape(S, num_chunks, chunk_size, HV)
    g_cumsum = torch.cumsum(g_block, dim=2).reshape(
        S,
        padded_len,
        HV,
    )[:, :seq_len].reshape(total_tokens, HV)

    if initial_state is None:
        init_state = torch.zeros(
            (S, HV, Vdim, Kdim),
            dtype=torch.float32,
            device=device,
        )
    else:
        init_state = initial_state.to(torch.float32)

    return (qf[:total_tokens], kf[:total_tokens], vf[:total_tokens], bf[:total_tokens], g_cumsum, init_state, HV,
            num_chunks, scale, Kdim, Vdim, S)


def hpu_chunk_gdr_phase_a(
    qf: torch.Tensor,
    kf: torch.Tensor,
    vf: torch.Tensor,
    bf: torch.Tensor,
    g_cumsum: torch.Tensor,
    seq_len: int,
    chunk_size: int,
    S: int,
    num_chunks: int,
    H: int,
    Kdim: int,
    Vdim: int,
    neumann_iters: int,
    recursive_solver_base: int = 0,
    compact_repeated_kkt: bool = False,
    qk_head_repeat: int = 1,
    solve_in_fp32: bool | None = None,
    compact_qk_inputs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase A: batched stages 2-4 for ALL chunks at once.

    Returns (u_all, w_all, q_chunks, k_chunks, g_chunks).
    All shaped [S, num_chunks, tc, H, dim].
    """
    device = qf.device
    tc = chunk_size
    if compact_qk_inputs:
        if qk_head_repeat <= 1 or H % qk_head_repeat != 0:
            raise ValueError(
                "Compact Q/K inputs require a valid repeated-head mapping."
            )
        qk_heads = H // qk_head_repeat
    else:
        qk_heads = H

    # Reshape to [S, seq_len, H, dim] then [S, C, tc, H, dim]
    q_seqs = qf.reshape(S, seq_len, qk_heads, Kdim)
    k_seqs = kf.reshape(S, seq_len, qk_heads, Kdim)
    v_seqs = vf.reshape(S, seq_len, H, Vdim)
    g_seqs = g_cumsum.reshape(S, seq_len, H)
    b_seqs = bf.reshape(S, seq_len, H)

    padded_len = num_chunks * tc
    if padded_len > seq_len:
        pad_len = padded_len - seq_len
        q_seqs = torch.cat(
            [
                q_seqs,
                torch.zeros(
                    S,
                    pad_len,
                    qk_heads,
                    Kdim,
                    dtype=qf.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        k_seqs = torch.cat(
            [
                k_seqs,
                torch.zeros(
                    S,
                    pad_len,
                    qk_heads,
                    Kdim,
                    dtype=kf.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_seqs = torch.cat([v_seqs, torch.zeros(S, pad_len, H, Vdim, dtype=vf.dtype, device=device)], dim=1)
        g_last_valid = g_seqs[:, -1:, :]
        g_seqs = torch.cat([g_seqs, g_last_valid.expand(S, pad_len, H)], dim=1)
        b_seqs = torch.cat([b_seqs, torch.zeros(S, pad_len, H, dtype=bf.dtype, device=device)], dim=1)

    q_chunks = q_seqs.reshape(S, num_chunks, tc, qk_heads, Kdim)
    k_chunks = k_seqs.reshape(S, num_chunks, tc, qk_heads, Kdim)
    v_chunks = v_seqs.reshape(S, num_chunks, tc, H, Vdim)
    g_chunks = g_seqs.reshape(S, num_chunks, tc, H)
    b_chunks = b_seqs.reshape(S, num_chunks, tc, H)

    SC = S * num_chunks
    k_head_major = k_chunks.reshape(
        SC,
        tc,
        qk_heads,
        Kdim,
    ).permute(0, 2, 1, 3)
    if compact_qk_inputs:
        k_flat = None
    else:
        k_flat = k_head_major.reshape(SC * H, tc, Kdim)
    v_flat = v_chunks.reshape(SC, tc, H, Vdim).permute(0, 2, 1, 3).reshape(SC * H, tc, Vdim)
    g_flat = g_chunks.reshape(SC, tc, H).permute(0, 2, 1).reshape(SC * H, tc)
    b_flat = b_chunks.reshape(SC, tc, H).permute(0, 2, 1).reshape(SC * H, tc)

    eye = torch.eye(tc, dtype=qf.dtype, device=device)

    # Stage 2: chunk_scaled_dot_kkt
    coeff = b_flat.unsqueeze(-1) * (torch.exp(g_flat.unsqueeze(-1) - g_flat.unsqueeze(-2))).to(b_flat.dtype)
    if compact_qk_inputs:
        compact_dot = torch.matmul(
            k_head_major,
            k_head_major.transpose(-1, -2),
        )
        grouped_coeff = coeff.reshape(
            SC,
            qk_heads,
            qk_head_repeat,
            tc,
            tc,
        )
        a_lower = torch.tril(
            compact_dot.unsqueeze(2) * grouped_coeff,
            diagonal=-1,
        ).reshape(SC * H, tc, tc)
    elif compact_repeated_kkt and qk_head_repeat > 1:
        if H % qk_head_repeat != 0:
            raise ValueError(
                "qk_head_repeat must divide the expanded GDN head count, "
                f"got qk_head_repeat={qk_head_repeat}, H={H}."
            )
        compact_heads = H // qk_head_repeat
        compact_k = k_chunks[..., ::qk_head_repeat, :].reshape(SC, tc, compact_heads,
                                                               Kdim).permute(0, 2, 1, 3)
        compact_k = compact_k.reshape(SC * compact_heads, tc, Kdim)
        compact_dot = torch.bmm(compact_k, compact_k.transpose(1, 2))
        compact_dot = compact_dot.reshape(SC, compact_heads, tc, tc)
        grouped_coeff = coeff.reshape(SC, compact_heads, qk_head_repeat, tc, tc)
        a_lower = torch.tril(
            compact_dot.unsqueeze(2) * grouped_coeff,
            diagonal=-1,
        ).reshape(SC * H, tc, tc)
    else:
        assert k_flat is not None
        dot = torch.bmm(k_flat, k_flat.transpose(1, 2))
        a_lower = torch.tril(dot * coeff, diagonal=-1)
    lmat = (eye.unsqueeze(0) + a_lower).to(qf.dtype)

    # Stage 3: solve_tril
    use_fp32_solve = _GDN_SOLVE_FP32 if solve_in_fp32 is None else solve_in_fp32
    solve_dtype = torch.float32 if use_fp32_solve else qf.dtype
    solve_eye = eye.to(solve_dtype)
    A_solve = _hpu_solve_lower_triangular_batched(
        lmat.to(solve_dtype),
        solve_eye,
        use_vectorized=True,
        neumann_iters=neumann_iters,
        recursive_base=recursive_solver_base,
    ).to(qf.dtype)

    # Stage 4: recompute u, w
    rhs_u = v_flat * b_flat.unsqueeze(-1)
    u_flat = torch.bmm(A_solve, rhs_u)
    if compact_qk_inputs:
        grouped_scale = (b_flat * torch.exp(g_flat)).reshape(
            SC,
            qk_heads,
            qk_head_repeat,
            tc,
        )
        grouped_rhs_w = (
            k_head_major.unsqueeze(2) * grouped_scale.unsqueeze(-1)
        )
        grouped_solve = A_solve.reshape(
            SC,
            qk_heads,
            qk_head_repeat,
            tc,
            tc,
        )
        w_flat = torch.matmul(grouped_solve, grouped_rhs_w).reshape(
            SC * H,
            tc,
            Kdim,
        )
    else:
        assert k_flat is not None
        rhs_w = k_flat * (
            b_flat * torch.exp(g_flat)
        ).unsqueeze(-1)
        w_flat = torch.bmm(A_solve, rhs_w)

    u_all = u_flat.reshape(SC, H, tc, Vdim).permute(
        0, 2, 1, 3
    ).reshape(S, num_chunks, tc, H, Vdim)
    w_all = w_flat.reshape(SC, H, tc, Kdim).permute(
        0, 2, 1, 3
    ).reshape(S, num_chunks, tc, H, Kdim)

    return u_all, w_all, q_chunks, k_chunks, g_chunks


def hpu_flashqla_chunk_gdr_phase_a(
    qf: torch.Tensor,
    kf: torch.Tensor,
    vf: torch.Tensor,
    bf: torch.Tensor,
    g_cumsum: torch.Tensor,
    seq_len: int,
    chunk_size: int,
    S: int,
    num_chunks: int,
    H: int,
    Kdim: int,
    Vdim: int,
    neumann_iters: int,
    recursive_solver_base: int = 0,
    compact_repeated_kkt: bool = False,
    qk_head_repeat: int = 1,
    solve_in_fp32: bool | None = None,
    masked_triangular_decay: bool = False,
    fp32_rhs_scaling: bool = False,
    fp32_output_scaling: bool = False,
    gate_center_mode: int = 0,
    return_local_decay: bool = False,
    compact_qk_inputs: bool = False,
    compact_qk_factor_gate: bool = False,
    factorized_gate_transform: bool = False,
    native_compact_kkt: bool = False,
) -> tuple[torch.Tensor, ...]:
    """FlashQLA's gate-free KKT solve using stable gate reconstruction.

    FlashQLA solves the gate-free unit-lower matrix first, then reconstructs
    ``Ag = G * L0^-1 * beta`` where every materialized gate exponent is a
    causal decay in ``[0, 1]``. This mirrors the upstream fused kernel. Do not
    materialize ``D`` and ``D^-1`` separately: their positive exponent range
    can overflow even though the final causal decay is finite.

    ``factorized_gate_transform`` is a Gaudi research path that returns U in a
    centered gate-free basis and W in an unscaled gate-free basis. Optimized
    Phase B cancels those bases without materializing ``Ag`` or pairwise gate
    decay. Its separate exponent factors require a bounded per-chunk gate span.
    """
    device = qf.device
    tc = chunk_size
    if compact_qk_inputs:
        if qk_head_repeat <= 1 or H % qk_head_repeat != 0:
            raise ValueError(
                "Compact Q/K inputs require a valid repeated-head mapping."
            )
        qk_heads = H // qk_head_repeat
    else:
        qk_heads = H
    if native_compact_kkt and (not compact_qk_inputs or tc != 64):
        raise ValueError(
            "Native compact KKT requires compact Q/K inputs and chunk_size=64."
        )

    q_seqs = qf.reshape(S, seq_len, qk_heads, Kdim)
    k_seqs = kf.reshape(S, seq_len, qk_heads, Kdim)
    v_seqs = vf.reshape(S, seq_len, H, Vdim)
    g_seqs = g_cumsum.reshape(S, seq_len, H)
    b_seqs = bf.reshape(S, seq_len, H)

    padded_len = num_chunks * tc
    if padded_len > seq_len:
        pad_len = padded_len - seq_len
        q_seqs = torch.cat(
            [
                q_seqs,
                torch.zeros(
                    S,
                    pad_len,
                    qk_heads,
                    Kdim,
                    dtype=qf.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        k_seqs = torch.cat(
            [
                k_seqs,
                torch.zeros(
                    S,
                    pad_len,
                    qk_heads,
                    Kdim,
                    dtype=kf.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_seqs = torch.cat(
            [v_seqs, torch.zeros(S, pad_len, H, Vdim, dtype=vf.dtype, device=device)],
            dim=1,
        )
        g_last_valid = g_seqs[:, -1:, :]
        g_seqs = torch.cat([g_seqs, g_last_valid.expand(S, pad_len, H)], dim=1)
        b_seqs = torch.cat(
            [b_seqs, torch.zeros(S, pad_len, H, dtype=bf.dtype, device=device)],
            dim=1,
        )

    q_chunks = q_seqs.reshape(S, num_chunks, tc, qk_heads, Kdim)
    k_chunks = k_seqs.reshape(S, num_chunks, tc, qk_heads, Kdim)
    v_chunks = v_seqs.reshape(S, num_chunks, tc, H, Vdim)
    g_chunks = g_seqs.reshape(S, num_chunks, tc, H)
    b_chunks = b_seqs.reshape(S, num_chunks, tc, H)

    SC = S * num_chunks
    k_head_major = k_chunks.reshape(
        SC,
        tc,
        qk_heads,
        Kdim,
    ).permute(0, 2, 1, 3)
    lmat = None
    if compact_qk_inputs:
        k_flat = None
    else:
        k_flat = k_head_major.reshape(SC * H, tc, Kdim)
    v_flat = v_chunks.reshape(SC, tc, H, Vdim).permute(0, 2, 1, 3).reshape(SC * H, tc, Vdim)
    g_flat = g_chunks.reshape(SC, tc, H).permute(0, 2, 1).reshape(SC * H, tc)
    b_flat = b_chunks.reshape(SC, tc, H).permute(0, 2, 1).reshape(SC * H, tc)

    strict_lower_mask = None
    if masked_triangular_decay:
        positions = torch.arange(tc, device=device)
        strict_lower_mask = positions.unsqueeze(-1) > positions.unsqueeze(0)

    if compact_qk_inputs:
        compact_dot = torch.matmul(
            k_head_major,
            k_head_major.transpose(-1, -2),
        )
        grouped_beta = b_flat.reshape(
            SC,
            qk_heads,
            qk_head_repeat,
            tc,
        )
        if native_compact_kkt:
            lmat = torch.ops.custom_op.qwen38_compact_kkt_bf16_gaudi2(
                compact_dot,
                grouped_beta,
            ).reshape(SC * H, tc, tc)
        else:
            a_product = compact_dot.unsqueeze(2) * grouped_beta.unsqueeze(-1)
            a_lower = (
                torch.tril(a_product, diagonal=-1)
                if strict_lower_mask is None
                else a_product * strict_lower_mask
            ).reshape(SC * H, tc, tc)
    elif compact_repeated_kkt and qk_head_repeat > 1:
        if H % qk_head_repeat != 0:
            raise ValueError(
                "qk_head_repeat must divide the expanded GDN head count, "
                f"got qk_head_repeat={qk_head_repeat}, H={H}."
        )
        compact_heads = H // qk_head_repeat
        compact_k = k_chunks[..., ::qk_head_repeat, :].reshape(
            SC,
            tc,
            compact_heads,
            Kdim,
        )
        compact_k = compact_k.permute(0, 2, 1, 3).reshape(SC * compact_heads, tc, Kdim)
        compact_dot = torch.bmm(
            compact_k,
            compact_k.transpose(1, 2),
        ).reshape(SC, compact_heads, tc, tc)
        grouped_beta = b_flat.reshape(
            SC,
            compact_heads,
            qk_head_repeat,
            tc,
        )
        a_product = compact_dot.unsqueeze(2) * grouped_beta.unsqueeze(-1)
        a_lower = (
            torch.tril(a_product, diagonal=-1)
            if strict_lower_mask is None
            else a_product * strict_lower_mask
        ).reshape(SC * H, tc, tc)
    else:
        assert k_flat is not None
        dot = torch.bmm(k_flat, k_flat.transpose(1, 2))
        a_product = dot * b_flat.unsqueeze(-1)
        a_lower = (
            torch.tril(a_product, diagonal=-1)
            if strict_lower_mask is None
            else a_product * strict_lower_mask
        )

    causal_decay = None

    eye = torch.eye(tc, dtype=qf.dtype, device=device)
    if lmat is None:
        lmat = eye.unsqueeze(0) + a_lower
    use_fp32_solve = _GDN_SOLVE_FP32 if solve_in_fp32 is None else solve_in_fp32
    solve_dtype = torch.float32 if use_fp32_solve else qf.dtype
    solve_eye = eye.to(solve_dtype)
    a0_inv = _hpu_solve_lower_triangular_batched(
        lmat.to(solve_dtype),
        solve_eye,
        use_vectorized=True,
        neumann_iters=neumann_iters,
        recursive_base=recursive_solver_base,
    ).to(qf.dtype)

    # These controls belonged to the rejected uncentered D/D^-1 prototype.
    # Keep them in the research API while both the stable and centered paths
    # use their fixed scaling contracts.
    del fp32_rhs_scaling, fp32_output_scaling, gate_center_mode

    if factorized_gate_transform:
        # Ag = D @ A0^-1 @ D^-1 @ diag(beta). Push both diagonal
        # transforms around the BMM so the 64x64 pairwise decay and Ag
        # tensors are never materialized. Centering bounds both exponent
        # ranges while preserving their product.
        gate_center = (g_flat[:, :1] + g_flat[:, -1:]) * 0.5
        right_scale = torch.exp(gate_center - g_flat).to(a0_inv.dtype)
        beta_scale = b_flat.to(a0_inv.dtype)
        u_rhs = (
            v_flat.to(a0_inv.dtype)
            * (right_scale * beta_scale).unsqueeze(-1)
        )
        u_flat = torch.bmm(a0_inv, u_rhs)

        if compact_qk_inputs:
            grouped_inverse = a0_inv.reshape(
                SC,
                qk_heads,
                qk_head_repeat,
                tc,
                tc,
            )
            grouped_beta = beta_scale.reshape(
                SC,
                qk_heads,
                qk_head_repeat,
                tc,
            )
            grouped_rhs_w = (
                k_head_major.to(a0_inv.dtype).unsqueeze(2)
                * grouped_beta.unsqueeze(-1)
            )
            w_flat = torch.matmul(
                grouped_inverse,
                grouped_rhs_w,
            ).reshape(SC * H, tc, Kdim)
        else:
            assert k_flat is not None
            w_flat = torch.bmm(
                a0_inv,
                k_flat.to(a0_inv.dtype) * beta_scale.unsqueeze(-1),
            )
    else:
        if causal_decay is None:
            gate_delta = g_flat.unsqueeze(-1) - g_flat.unsqueeze(-2)
            # Zero the non-causal half before exp so a large positive
            # upper-triangle delta can never overflow, then clear that half.
            if strict_lower_mask is None:
                causal_decay = torch.tril(
                    torch.exp(torch.tril(gate_delta))
                )
            else:
                causal_mask = strict_lower_mask | torch.eye(
                    tc,
                    dtype=torch.bool,
                    device=device,
                )
                causal_decay = torch.exp(
                    torch.where(causal_mask, gate_delta, float("-inf"))
                )
            causal_decay = causal_decay.to(a0_inv.dtype)
        ag = (
            a0_inv
            * causal_decay
            * b_flat.to(a0_inv.dtype).unsqueeze(-2)
        )
        u_flat = torch.bmm(ag, v_flat.to(a0_inv.dtype))
        gate_exp = torch.exp(g_flat).to(a0_inv.dtype)
        if compact_qk_inputs:
            grouped_gate = gate_exp.reshape(
                SC,
                qk_heads,
                qk_head_repeat,
                tc,
            )
            grouped_ag = ag.reshape(
                SC,
                qk_heads,
                qk_head_repeat,
                tc,
                tc,
            )
            if compact_qk_factor_gate:
                # Scaling the coefficient avoids a repeated K tensor.
                grouped_ag = grouped_ag * grouped_gate.unsqueeze(-2)
                grouped_rhs_w = k_head_major.to(a0_inv.dtype).unsqueeze(2)
            else:
                grouped_rhs_w = (
                    k_head_major.to(a0_inv.dtype).unsqueeze(2)
                    * grouped_gate.unsqueeze(-1)
                )
            w_flat = torch.matmul(grouped_ag, grouped_rhs_w).reshape(
                SC * H,
                tc,
                Kdim,
            )
        else:
            assert k_flat is not None
            w_flat = torch.bmm(
                ag,
                k_flat.to(a0_inv.dtype)
                * gate_exp.unsqueeze(-1),
            )

    if factorized_gate_transform:
        u_all = u_flat.reshape(S, num_chunks, H, tc, Vdim)
        w_all = w_flat.reshape(S, num_chunks, H, tc, Kdim)
    else:
        u_all = u_flat.reshape(SC, H, tc, Vdim).permute(
            0, 2, 1, 3
        ).reshape(S, num_chunks, tc, H, Vdim)
        w_all = w_flat.reshape(SC, H, tc, Kdim).permute(
            0, 2, 1, 3
        ).reshape(S, num_chunks, tc, H, Kdim)
    if return_local_decay:
        if causal_decay is None:
            raise ValueError(
                "factorized_gate_transform cannot return local decay"
            )
        return (
            u_all,
            w_all,
            q_chunks,
            k_chunks,
            g_chunks,
            causal_decay,
        )
    return u_all, w_all, q_chunks, k_chunks, g_chunks


def hpu_chunk_gdr_phase_b(
    u_all: torch.Tensor,
    w_all: torch.Tensor,
    q_chunks: torch.Tensor,
    k_chunks: torch.Tensor,
    g_chunks: torch.Tensor,
    init_state: torch.Tensor,
    scale: float,
    S: int,
    num_chunks: int,
    seq_len: int,
    H: int,
    Kdim: int,
    Vdim: int,
    output_final_state: bool,
    output_dtype: torch.dtype | None = None,
    fused_state_matmul: bool = False,
    deferred_output_add: bool = False,
    factorized_local_decay: bool = False,
    local_decay: torch.Tensor | None = None,
    compact_repeated_local_attn: bool = False,
    qk_head_repeat: int = 1,
    compact_qk_inputs: bool = False,
    native_recurrent_scan: bool | None = None,
    centered_gate_free_phase_a: bool = False,
    phase_a_head_major: bool = False,
    state_in_fp32: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Phase B: sequential loop — stages 5-6 (state-dependent).

    Dispatches between optimized (hoisted precompute) and legacy
    (_phase_b_step-based) paths based on VLLM_GDN_LEGACY_PHASE_B env var.
    """
    if _USE_LEGACY_PHASE_B:
        if centered_gate_free_phase_a or phase_a_head_major:
            raise ValueError(
                "Centered gate-free Phase A requires optimized Phase B."
            )
        if (
            native_recurrent_scan is True
            or native_recurrent_scan is None
            and _USE_NATIVE_RECURRENT_SCAN
        ):
            raise ValueError(
                "Native recurrent scan requires the optimized phase-B path."
            )
        if compact_qk_inputs:
            raise ValueError(
                "Compact Q/K inputs require the optimized phase-B path."
            )
        return _hpu_chunk_gdr_phase_b_legacy(
            u_all,
            w_all,
            q_chunks,
            k_chunks,
            g_chunks,
            init_state,
            scale,
            S,
            num_chunks,
            seq_len,
            H,
            Kdim,
            Vdim,
            output_final_state,
            output_dtype,
        )
    return _hpu_chunk_gdr_phase_b_optimized(
        u_all,
        w_all,
        q_chunks,
        k_chunks,
        g_chunks,
        init_state,
        scale,
        S,
        num_chunks,
        seq_len,
        H,
        Kdim,
        Vdim,
        output_final_state,
        output_dtype,
        fused_state_matmul,
        deferred_output_add,
        factorized_local_decay,
        local_decay,
        compact_repeated_local_attn,
        qk_head_repeat,
        compact_qk_inputs,
        native_recurrent_scan,
        centered_gate_free_phase_a,
        phase_a_head_major,
        state_in_fp32,
    )


def _eager_reshape_output(core_h, S, padded_len, seq_len, H, Vdim):
    """Reshape core_h to output tensor in eager mode."""
    return core_h.permute(0, 1, 3, 2, 4).reshape(S, padded_len, H, Vdim)[:, :seq_len, :, :].reshape(-1, H, Vdim)


def _bf16_bmm_f32(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Run batched BF16 MME inputs with an FP32 Synapse output."""
    batch_shape = lhs.shape[:-2]
    output_shape = (*batch_shape, lhs.shape[-2], rhs.shape[-1])
    lhs_flat = lhs.reshape(-1, lhs.shape[-2], lhs.shape[-1]).to(
        torch.bfloat16,
    )
    rhs_flat = rhs.reshape(-1, rhs.shape[-2], rhs.shape[-1]).to(
        torch.bfloat16,
    )
    output = torch.ops.custom_op.flashqla_bf16_bmm_f32(
        lhs_flat,
        rhs_flat,
    )
    return output.reshape(output_shape)


def _solve_bmm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if _USE_SOLVE_BF16_BMM_F32:
        return _bf16_bmm_f32(lhs, rhs)
    return torch.bmm(lhs, rhs)


def _hpu_chunk_gdr_phase_b_optimized(
    u_all: torch.Tensor,
    w_all: torch.Tensor,
    q_chunks: torch.Tensor,
    k_chunks: torch.Tensor,
    g_chunks: torch.Tensor,
    init_state: torch.Tensor,
    scale: float,
    S: int,
    num_chunks: int,
    seq_len: int,
    H: int,
    Kdim: int,
    Vdim: int,
    output_final_state: bool,
    output_dtype: torch.dtype | None = None,
    fused_state_matmul: bool = False,
    deferred_output_add: bool = False,
    factorized_local_decay: bool = False,
    local_decay: torch.Tensor | None = None,
    compact_repeated_local_attn: bool = False,
    qk_head_repeat: int = 1,
    compact_qk_inputs: bool = False,
    native_recurrent_scan: bool | None = None,
    centered_gate_free_phase_a: bool = False,
    phase_a_head_major: bool = False,
    state_in_fp32: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Optimized Phase B: chunk-local precompute hoisted out of the loop.

    Loop body keeps only the recurrent matmul/add:
      out_i   = core_i + C_i @ state_i
      state_{i+1} = M_i @ state_i + N_i

    Internal recurrent state is stored as [S, H, K, V].
    """
    tc = u_all.shape[3] if phase_a_head_major else u_all.shape[2]
    padded_len = num_chunks * tc
    device = u_all.device
    compute_dtype = q_chunks.dtype
    if compact_qk_inputs:
        if qk_head_repeat <= 1 or H % qk_head_repeat != 0:
            raise ValueError(
                "Compact Q/K inputs require a valid repeated-head mapping."
            )
        qk_heads = H // qk_head_repeat
    else:
        qk_heads = H

    # [S, C, H, ...]
    if phase_a_head_major:
        u_h = u_all.to(compute_dtype)
        w_h = w_all.to(compute_dtype)
    else:
        u_h = u_all.permute(0, 1, 3, 2, 4).to(compute_dtype)
        w_h = w_all.permute(0, 1, 3, 2, 4).to(compute_dtype)
    q_h = q_chunks.permute(0, 1, 3, 2, 4)
    k_h = k_chunks.permute(0, 1, 3, 2, 4).to(compute_dtype)
    g_h = g_chunks.permute(0, 1, 3, 2)

    g_last = g_h[..., -1:]  # [S,C,H,1]
    if centered_gate_free_phase_a:
        # Phase A returned Uc in a centered gate-free basis and W0 in the
        # unscaled gate-free basis. Local gate factors then cancel before
        # every large Phase-B matmul.
        gate_center = (g_h[..., :1] + g_h[..., -1:]) * 0.5
        if compact_qk_inputs:
            compact_dot = torch.matmul(
                q_h,
                k_h.transpose(-1, -2),
            )
            A = compact_dot.unsqueeze(3).expand(
                S,
                num_chunks,
                qk_heads,
                qk_head_repeat,
                tc,
                tc,
            ).reshape(S, num_chunks, H, tc, tc)
            q_base = q_h.unsqueeze(3).expand(
                S,
                num_chunks,
                qk_heads,
                qk_head_repeat,
                tc,
                Kdim,
            ).reshape(S, num_chunks, H, tc, Kdim)
        else:
            A = torch.matmul(q_h, k_h.transpose(-1, -2))
            q_base = q_h
        A = torch.tril(A)
        left_scale = torch.exp(g_h - gate_center).to(compute_dtype)
        g_exp = torch.exp(g_h).to(compute_dtype)
        core_h = (
            torch.matmul(A, u_h) * left_scale.unsqueeze(-1) * scale
        )
        C_h = (
            (q_base - torch.matmul(A, w_h))
            * g_exp.unsqueeze(-1)
            * scale
        )

        if compact_qk_inputs:
            grouped_u = u_h.reshape(
                S,
                num_chunks,
                qk_heads,
                qk_head_repeat,
                tc,
                Vdim,
            )
            grouped_w = w_h.reshape(
                S,
                num_chunks,
                qk_heads,
                qk_head_repeat,
                tc,
                Kdim,
            )
            grouped_k_t = k_h.transpose(-1, -2).unsqueeze(3)
            N_t = torch.matmul(
                grouped_k_t,
                grouped_u,
            ).reshape(S, num_chunks, H, Kdim, Vdim)
            R_t = torch.matmul(
                grouped_k_t,
                grouped_w,
            ).reshape(S, num_chunks, H, Kdim, Kdim)
        else:
            N = torch.matmul(u_h.transpose(-1, -2), k_h)
            R = torch.matmul(w_h.transpose(-1, -2), k_h)
        n_state_scale = torch.exp(g_last - gate_center).to(
            compute_dtype,
        ).unsqueeze(-1)
        r_state_scale = torch.exp(g_last).to(compute_dtype).unsqueeze(-1)
    else:
        g_exp = torch.exp(g_h).to(compute_dtype)
        pair_decay = None
        if local_decay is not None:
            pair_decay = local_decay.reshape(
                S,
                num_chunks,
                H,
                tc,
                tc,
            ).to(compute_dtype)
        if pair_decay is not None:
            delta_exp = pair_decay[..., -1, :]
        else:
            delta_exp = torch.exp(g_last - g_h).to(compute_dtype)
        # Output decomposition:
        # out = (A @ U + (Q - A @ W) @ state_t) * scale
        if factorized_local_decay:
            gate_center = (g_h[..., :1] + g_h[..., -1:]) * 0.5
            if compact_qk_inputs:
                grouped_gate = g_h.reshape(
                    S,
                    num_chunks,
                    qk_heads,
                    qk_head_repeat,
                    tc,
                )
                grouped_center = gate_center.reshape(
                    S,
                    num_chunks,
                    qk_heads,
                    qk_head_repeat,
                    1,
                )
                q_local = q_h.unsqueeze(3) * torch.exp(
                    grouped_gate - grouped_center,
                ).to(compute_dtype).unsqueeze(-1)
                k_local = k_h.unsqueeze(3) * torch.exp(
                    grouped_center - grouped_gate,
                ).to(compute_dtype).unsqueeze(-1)
                A = torch.tril(
                    torch.matmul(q_local, k_local.transpose(-1, -2))
                ).reshape(S, num_chunks, H, tc, tc)
            else:
                q_local = q_h * torch.exp(g_h - gate_center).to(
                    compute_dtype,
                ).unsqueeze(-1)
                k_local = k_h * torch.exp(gate_center - g_h).to(
                    compute_dtype,
                ).unsqueeze(-1)
                A = torch.tril(
                    torch.matmul(q_local, k_local.transpose(-1, -2))
                )
        else:
            if pair_decay is None:
                pair_decay = torch.exp(
                    g_h.unsqueeze(-1) - g_h.unsqueeze(-2)
                ).to(compute_dtype)
            if compact_qk_inputs:
                compact_dot = torch.matmul(
                    q_h,
                    k_h.transpose(-1, -2),
                )
                grouped_decay = pair_decay.reshape(
                    S,
                    num_chunks,
                    qk_heads,
                    qk_head_repeat,
                    tc,
                    tc,
                )
                A = (
                    compact_dot.unsqueeze(3) * grouped_decay
                ).reshape(S, num_chunks, H, tc, tc)
            elif compact_repeated_local_attn and qk_head_repeat > 1:
                compact_heads = H // qk_head_repeat
                compact_q = q_h[:, :, ::qk_head_repeat]
                compact_k = k_h[:, :, ::qk_head_repeat]
                compact_dot = torch.matmul(
                    compact_q,
                    compact_k.transpose(-1, -2),
                )
                grouped_decay = pair_decay.reshape(
                    S,
                    num_chunks,
                    compact_heads,
                    qk_head_repeat,
                    tc,
                    tc,
                )
                A = (
                    compact_dot.unsqueeze(3) * grouped_decay
                ).reshape(S, num_chunks, H, tc, tc)
            else:
                A = torch.matmul(q_h, k_h.transpose(-1, -2))
                A = A * pair_decay
            # Phase A's returned local decay is already exactly zero above
            # the diagonal. Avoid another full 64x64 TPC tril after QK in
            # that path; generated pair decay still needs causal masking.
            if local_decay is None:
                A = torch.tril(A)

        if compact_qk_inputs:
            grouped_g_exp = g_exp.reshape(
                S,
                num_chunks,
                qk_heads,
                qk_head_repeat,
                tc,
            )
            Q = (
                q_h.unsqueeze(3) * grouped_g_exp.unsqueeze(-1)
            ).reshape(S, num_chunks, H, tc, Kdim)
        else:
            Q = q_h * g_exp.unsqueeze(-1)
        core_h = torch.matmul(A, u_h) * scale
        C_h = (Q - torch.matmul(A, w_h)) * scale

        u_decay = u_h * delta_exp.unsqueeze(-1)
        w_decay = w_h * delta_exp.unsqueeze(-1)
        if compact_qk_inputs:
            grouped_u = u_decay.reshape(
                S,
                num_chunks,
                qk_heads,
                qk_head_repeat,
                tc,
                Vdim,
            )
            grouped_w = w_decay.reshape(
                S,
                num_chunks,
                qk_heads,
                qk_head_repeat,
                tc,
                Kdim,
            )
            grouped_k_t = k_h.transpose(-1, -2).unsqueeze(3)
            N_t = torch.matmul(
                grouped_k_t,
                grouped_u,
            ).reshape(S, num_chunks, H, Kdim, Vdim)
            R_t = torch.matmul(
                grouped_k_t,
                grouped_w,
            ).reshape(S, num_chunks, H, Kdim, Kdim)
        else:
            N = torch.matmul(u_decay.transpose(-1, -2), k_h)
            R = torch.matmul(w_decay.transpose(-1, -2), k_h)

    alpha = (
        r_state_scale
        if centered_gate_free_phase_a
        else g_exp[..., -1:].unsqueeze(-1)
    )
    if not compact_qk_inputs:
        N_t = N.transpose(-1, -2)  # [S,C,H,K,V]
        R_t = R.transpose(-1, -2)
    if centered_gate_free_phase_a:
        N_t = N_t * n_state_scale
        R_t = R_t * r_state_scale
    k_eye = torch.eye(Kdim, dtype=compute_dtype, device=device).view(
        1,
        1,
        1,
        Kdim,
        Kdim,
    )
    M_full = alpha * k_eye - R_t  # [S,C,H,K,K]

    use_fp32_state = _GDN_STATE_FP32 if state_in_fp32 is None else state_in_fp32
    state_dtype = torch.float32 if use_fp32_state else compute_dtype
    state_t = init_state.to(state_dtype).transpose(-1, -2)  # [S,H,K,V]
    state_bias = N_t.to(state_dtype)
    if native_recurrent_scan is None:
        native_recurrent_scan = _USE_NATIVE_RECURRENT_SCAN
    if native_recurrent_scan and not fused_state_matmul:
        raise ValueError(
            "Native recurrent scan requires fused_state_matmul=True."
        )
    if native_recurrent_scan and state_dtype != torch.bfloat16:
        raise ValueError("Native recurrent scan requires BF16 state.")

    output_updates = [] if deferred_output_add else None
    if fused_state_matmul:
        state_projections = torch.cat([C_h, M_full], dim=-2)
        if not _USE_BF16_BMM_F32:
            state_projections = state_projections.to(state_dtype)
        if native_recurrent_scan:
            output_updates_flat, state_flat = (
                torch.ops.custom_op.flashqla_recurrent_scan_probe(
                    state_projections.reshape(
                        S * num_chunks * H,
                        tc + Kdim,
                        Kdim,
                    ),
                    core_h.reshape(
                        S * num_chunks * H,
                        tc,
                        Vdim,
                    ),
                    state_bias.reshape(
                        S * num_chunks * H,
                        Kdim,
                        Vdim,
                    ),
                    state_t.reshape(S * H, Kdim, Vdim),
                )
            )
            core_h = core_h + output_updates_flat.reshape(
                S,
                num_chunks,
                H,
                tc,
                Vdim,
            )
            state_t = state_flat.reshape(S, H, Kdim, Vdim)
            output_updates = None
        else:
            for ci in range(num_chunks):
                if _USE_BF16_BMM_F32:
                    projected = _bf16_bmm_f32(
                        state_projections[:, ci],
                        state_t,
                    )
                else:
                    projected = torch.matmul(
                        state_projections[:, ci],
                        state_t,
                    )
                if output_updates is None:
                    core_h[:, ci].add_(
                        projected[..., :tc, :].to(core_h.dtype)
                    )
                else:
                    output_updates.append(projected[..., :tc, :])
                state_t = projected[..., tc:, :] + state_bias[:, ci]
    else:
        for ci in range(num_chunks):
            projected_output = torch.matmul(C_h[:, ci].to(state_dtype), state_t)
            if output_updates is None:
                core_h[:, ci].add_(projected_output.to(core_h.dtype))
            else:
                output_updates.append(projected_output)
            state_t = (
                torch.matmul(M_full[:, ci].to(state_dtype), state_t)
                + state_bias[:, ci]
            )

    if output_updates is not None:
        core_h = core_h + torch.stack(output_updates, dim=1)

    out = _eager_reshape_output(core_h, S, padded_len, seq_len, H, Vdim)

    final_state = None
    if output_final_state:
        # Cast to output dtype before transpose+contiguous to halve the
        # size of the contiguous copy (e.g. fp32 -> bf16).
        st = state_t if output_dtype is None else state_t.to(output_dtype)
        final_state = st.transpose(-1, -2).contiguous()

    return out, final_state


def _recurrent_timestep_body(
    q_t: torch.Tensor,  # [H, K]
    k_t: torch.Tensor,  # [H, K]
    v_t: torch.Tensor,  # [HV, V]
    g_t: torch.Tensor,  # [HV]
    b_t: torch.Tensor,  # [HV]
    h_state: torch.Tensor,  # [HV, V, K]
    scale: float,
    HV: int,
    H: int,
    Kdim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compilable per-timestep body for hpu_fused_recurrent_gated_delta_rule.

    Returns (out_t [HV, V], updated h_state [HV, V, K]).
    """
    q_t = q_t * scale
    h_state = h_state * torch.exp(g_t).view(HV, 1, 1)
    proj = torch.sum(h_state * k_t.view(H, 1, Kdim), dim=-1)
    v_new = (v_t - proj) * b_t.view(HV, 1)
    h_state = h_state + v_new.unsqueeze(-1) * k_t.view(H, 1, Kdim)
    out_t = torch.sum(h_state * q_t.view(H, 1, Kdim), dim=-1)
    return out_t, h_state


def _hpu_solve_lower_triangular_batched(
    lmat: torch.Tensor,
    eye: torch.Tensor,
    use_vectorized: bool,
    neumann_iters: int,
    recursive_base: int = 0,
) -> torch.Tensor:
    """Compute L^{-1} for L = I + strictly-lower.

    Dispatches between exact forward substitution (VLLM_GDN_EXACT_SOLVE=1)
    and approximate Neumann iteration (default).

    **Neumann path** mirrors torch_chunk_gated_delta_rule_opt:
      inv_{k+1} = inv_k - inv_k @ ((L @ inv_k) * strict_lower_mask)

    For GDN, ``lmat`` is always ``I + tril(..., -1)``. The strict lower term is
    nilpotent, so this fixed-point style update converges quickly for the chunk
    sizes used in practice. We keep a fixed iteration budget to stay compile
    friendly on HPU.

    **Exact path** uses row-by-row forward substitution (127 Python-loop
    iterations for n=128). Numerically exact in fp32 but ~2.6× slower.

    **Accuracy note (Neumann):** The residual depends on both ``neumann_iters``
    and the learned model weights (beta, g, k projections) which determine the
    magnitude of off-diagonal entries. Different model sizes or model families
    may need a different iteration budget. Benchmark with ``hpu_tril_solve.py``
    using real weight statistics when porting to a new model.
    On Qwen3.5-9B with chunk_size=128: 14 iters ≈ residual 8, baseline (exact
    forward-sub) ≈ residual 0 but 2.6× slower.

    Args:
        lmat: [..., N, N] lower-triangular matrix with unit diagonal
        eye: [N, N] identity matrix (pre-cached for efficiency)
        use_vectorized: kept for API compatibility; Neumann path is used for both
        neumann_iters: fixed iteration budget for inverse refinement. Higher
            values improve accuracy at the cost of more bmm ops (2 per iter).
            Model-dependent: weights with larger beta or slower-decaying g
            need more iterations for the same residual.
        recursive_base: recursively invert diagonal blocks down to this base
            size. Zero keeps the inverse-refinement implementation.

    Returns:
        [..., N, N] (approximate or exact) inverse of lmat
    """
    if lmat.ndim < 2 or lmat.shape[-1] != lmat.shape[-2]:
        raise ValueError(f"Expected square matrix [..., N, N], got {tuple(lmat.shape)}")

    n = lmat.shape[-1]
    if eye.shape != (n, n):
        raise ValueError(f"Expected eye shape ({n}, {n}), got {tuple(eye.shape)}")

    if _USE_EXACT_SOLVE:
        return _solve_exact_forward_sub(lmat, eye)

    if neumann_iters <= 0:
        raise ValueError(f"neumann_iters must be > 0, got {neumann_iters}.")

    lflat = lmat.reshape(-1, n, n)
    if recursive_base:
        if recursive_base < 2 or recursive_base & (recursive_base - 1):
            raise ValueError(
                "recursive_base must be a power of two greater than one, "
                f"got {recursive_base}."
            )
        recursion_ratio = n // recursive_base if recursive_base <= n else 0
        if (
            recursive_base > n
            or n % recursive_base != 0
            or recursion_ratio & (recursion_ratio - 1)
        ):
            raise ValueError(
                "matrix size must be a power-of-two multiple of "
                f"recursive_base, got recursive_base={recursive_base}, n={n}."
            )
        return _hpu_recursive_unit_lower_inverse(
            lflat,
            recursive_base,
        ).reshape(lmat.shape)

    # Same strict-lower mask behavior as torch_chunk_gated_delta_rule_opt.
    lower_mask = torch.tril(
        torch.ones((n, n), dtype=lflat.dtype, device=lflat.device),
        diagonal=-1,
    ).unsqueeze(0)

    inv_flat = eye.unsqueeze(0).expand(lflat.shape[0], -1, -1).clone()
    # Keep the iteration count as a Python int argument for compile stability
    # while allowing external tuning.
    for _ in range(neumann_iters):
        prod = torch.bmm(lflat, inv_flat)
        err = prod * lower_mask
        update = torch.bmm(inv_flat, err)
        inv_flat = inv_flat - update

    return inv_flat.reshape(lmat.shape)


def _hpu_recursive_unit_lower_inverse(
    lflat: torch.Tensor,
    base: int,
) -> torch.Tensor:
    """Invert unit lower-triangular matrices with recursive block products."""
    n = lflat.shape[-1]
    if n == base:
        eye = torch.eye(n, dtype=lflat.dtype, device=lflat.device).unsqueeze(0)
        lower_mask = torch.tril(
            torch.ones((n, n), dtype=lflat.dtype, device=lflat.device),
            diagonal=-1,
        ).unsqueeze(0)
        inv = 2 * eye - lflat
        for _ in range((n - 1).bit_length() - 1):
            prod = torch.bmm(lflat, inv)
            err = prod * lower_mask
            inv = inv - torch.bmm(inv, err)
        return inv

    half = n // 2
    batch = lflat.shape[0]
    top_left = lflat[:, :half, :half]
    bottom_right = lflat[:, half:, half:]
    bottom_left = lflat[:, half:, :half]
    diagonal_inverse = _hpu_recursive_unit_lower_inverse(
        torch.cat((top_left, bottom_right), dim=0),
        base,
    )
    top_left_inverse = diagonal_inverse[:batch]
    bottom_right_inverse = diagonal_inverse[batch:]
    bottom_left_inverse = -_solve_bmm(
        _solve_bmm(bottom_right_inverse, bottom_left),
        top_left_inverse,
    )
    zeros = torch.zeros_like(top_left)
    top = torch.cat((top_left_inverse, zeros), dim=-1)
    bottom = torch.cat((bottom_left_inverse, bottom_right_inverse), dim=-1)
    return torch.cat((top, bottom), dim=-2)


def _solve_exact_forward_sub(
    lmat: torch.Tensor,
    eye: torch.Tensor,
) -> torch.Tensor:
    """Exact row-by-row forward substitution for L^{-1}.

    Numerically exact in fp32. 127 Python-loop iterations for n=128,
    each with a variable-size bmm. Causes many graph breaks under
    torch.compile but useful as an accuracy reference.

    Enable via VLLM_GDN_EXACT_SOLVE=1.
    """
    n = lmat.shape[-1]
    orig_shape = lmat.shape
    lflat = lmat.reshape(-1, n, n)

    result = torch.zeros_like(lflat)
    result[:, 0, :] = eye[0, :].unsqueeze(0)
    for j in range(1, n):
        prev = result[:, :j, :]  # [B, j, N]
        l_row = lflat[:, j:j + 1, :j]  # [B, 1, j]
        correction = torch.bmm(l_row, prev)  # [B, 1, N]
        result[:, j:j + 1, :] = eye[j, :].unsqueeze(0).unsqueeze(0) - correction

    return result.reshape(orig_shape)


def _materialize_seq_ranges(cu_seqlens: torch.Tensor, total_tokens: int) -> list[tuple[int, int]]:
    """Convert cu_seqlens to safe [bos, eos) ranges on CPU.

    Lazy-mode HPU tensors can produce unexpected scalar values when accessed
    repeatedly via .item() on-device. Materialize once on CPU and clamp to
    token bounds to keep Python-side loops safe.
    """
    try:
        cu_cpu = cu_seqlens.to(dtype=torch.int64, device="cpu")
    except RuntimeError as exc:
        # In some lazy/graph captures, host transfer of cu_seqlens is not
        # allowed. Fall back to one contiguous sequence to keep execution
        # alive instead of crashing.
        logger.warning(
            "[GDN seq range] Failed to materialize cu_seqlens on CPU (%s). "
            "Falling back to a single contiguous range [0, %d).",
            exc,
            total_tokens,
        )
        return [(0, total_tokens)]

    cu_list = cu_cpu.tolist()

    ranges: list[tuple[int, int]] = []
    for i in range(max(0, len(cu_list) - 1)):
        bos_raw = int(cu_list[i])
        eos_raw = int(cu_list[i + 1])
        bos = min(max(bos_raw, 0), total_tokens)
        eos = min(max(eos_raw, 0), total_tokens)
        if eos < bos:
            eos = bos
        ranges.append((bos, eos))
    return ranges


def _l2norm_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


def hpu_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch replacement for fused_gdn_gating.

    Returns:
      g: [1, num_tokens, num_heads] float32
      beta_out: [1, num_tokens, num_heads] same dtype as b
    """
    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    use_softplus = (beta * x) <= threshold
    softplus_x = torch.where(use_softplus, (1.0 / beta) * torch.log1p(torch.exp(beta * x)), x)
    g = -torch.exp(A_log.to(torch.float32)) * softplus_x
    beta_out = torch.sigmoid(b.to(torch.float32)).to(b.dtype)
    return g.unsqueeze(0), beta_out.unsqueeze(0)


def _eager_read_state(state: torch.Tensor, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Eager-only state read — isolates index_select from compiled graph."""
    return state.index_select(0, idx).to(dtype)


def hpu_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch replacement for fused_recurrent_gated_delta_rule.

    This implementation supports the non-speculative paths used by current
    Gaudi Qwen3.5 integration.
    """
    if num_accepted_tokens is not None:
        raise NotImplementedError("Speculative decode path is not implemented in phase 1.")
    if ssm_state_indices is not None and ssm_state_indices.ndim > 1:
        raise NotImplementedError("2D ssm_state_indices (spec decode) is not implemented in phase 1.")

    if beta is None:
        beta = torch.ones_like(g)
    if scale is None:
        scale = k.shape[-1]**-0.5

    # Shapes: q/k [B, T, H, K], v [B, T, HV, V], g/beta [B, T, HV]
    B, T, H, Kdim = q.shape
    _, _, HV, Vdim = v.shape
    device = q.device

    # Match upstream kernel semantics: when HV > H, each q/k head is shared
    # across a group of value heads (grouped-value attention).
    if H != HV:
        if HV % H == 0:
            repeat = HV // H
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)
            H = HV
        else:
            raise ValueError(f"Unsupported head mapping in hpu_fused_recurrent_gated_delta_rule: "
                             f"q/k heads={H}, value heads={HV}. Expected HV % H == 0.")

    # --- Vectorized decode fast path (shape-only detection) ---
    # Detect all-single-token decode from shapes alone — NO device-to-host
    # sync and NO _materialize_seq_ranges call needed.
    #   (a) cu_seqlens has N+1 entries and T == N  → N seqs, 1 token each
    #   (b) cu_seqlens is None and T == 1          → B seqs, 1 token each
    _all_single_token = ((cu_seqlens is not None and B == 1 and cu_seqlens.shape[0] - 1 == T)
                         or (cu_seqlens is None and T == 1))

    if _all_single_token:
        num_seqs = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else B

        if initial_state is None:
            final_state = torch.zeros((num_seqs, HV, Vdim, Kdim), dtype=torch.float32, device=device)
        else:
            final_state = initial_state if inplace_final_state else initial_state.clone()

        # Compute state indices and read state eagerly BEFORE reshapes,
        # so the graph break from _eager_read_state comes first.
        if ssm_state_indices is not None:
            sidx_raw = ssm_state_indices.reshape(-1).to(dtype=torch.long, device=device)
            num_slots = final_state.shape[0]
            sidx = torch.remainder(sidx_raw, num_slots)
        else:
            sidx = torch.arange(num_seqs, dtype=torch.long, device=device)

        h_batch = _eager_read_state(final_state, sidx, _GDN_COMPUTE_DTYPE)

        # Flatten token axis.
        # Compute dtype controlled by VLLM_GDN_COMPUTE_FP32 env var (default: bf16)
        qf = q.reshape(-1, H, Kdim).to(_GDN_COMPUTE_DTYPE)
        kf = k.reshape(-1, H, Kdim).to(_GDN_COMPUTE_DTYPE)
        vf = v.reshape(-1, HV, Vdim).to(_GDN_COMPUTE_DTYPE)
        gf = g.reshape(-1, HV).to(torch.float32)
        bf = beta.reshape(-1, HV).to(_GDN_COMPUTE_DTYPE)

        if use_qk_l2norm_in_kernel:
            qf = _l2norm_last_dim(qf)
            kf = _l2norm_last_dim(kf)

        out_full = torch.zeros(num_seqs, HV, Vdim, dtype=v.dtype, device=device)

        # Inline compute (no separate function boundary for this test).
        q_s = qf * scale
        h_batch = h_batch * torch.exp(gf).to(h_batch.dtype).unsqueeze(-1).unsqueeze(-1)
        proj = torch.matmul(h_batch, kf.unsqueeze(-1)).squeeze(-1)
        v_new = (vf - proj) * bf.unsqueeze(-1)
        h_batch = h_batch + v_new.unsqueeze(-1) * kf.unsqueeze(2)
        out_batch = torch.matmul(h_batch, q_s.unsqueeze(-1)).squeeze(-1)

        # Direct index_copy_ (no eager wrapper for this test).
        final_state.index_copy_(0, sidx, h_batch.to(final_state.dtype))
        out_full = out_batch.to(v.dtype)

        out_result = out_full.unsqueeze(0) if cu_seqlens is not None else out_full.view(B, T, HV, Vdim)
        return out_result, final_state

    # --- General (multi-token) fallback path ---
    return _recurrent_general_path(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        use_qk_l2norm_in_kernel,
        B,
        T,
        H,
        HV,
        Kdim,
        Vdim,
        device,
    )


@torch._dynamo.disable
def _recurrent_general_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    inplace_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    ssm_state_indices: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    B: int,
    T: int,
    H: int,
    HV: int,
    Kdim: int,
    Vdim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """General multi-token recurrent path (Python loops, dynamo-disabled)."""
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("When cu_seqlens is used, expected batch size B=1.")
        seq_ranges = _materialize_seq_ranges(cu_seqlens, B * T)
        num_seqs = len(seq_ranges)
    else:
        num_seqs = B
        seq_ranges = [(i * T, (i + 1) * T) for i in range(B)]

    if initial_state is None:
        final_state = torch.zeros((num_seqs, HV, Vdim, Kdim), dtype=torch.float32, device=device)
    else:
        final_state = initial_state if inplace_final_state else initial_state.clone()

    # Always compute in fp32 for stability and cast back on writes.
    state_work = final_state.to(torch.float32)

    # Flatten token axis for varlen path (B is expected to be 1 there).
    qf = q.reshape(-1, H, Kdim).to(torch.float32)
    kf = k.reshape(-1, H, Kdim).to(torch.float32)
    vf = v.reshape(-1, HV, Vdim).to(torch.float32)
    gf = g.reshape(-1, HV).to(torch.float32)
    bf = beta.reshape(-1, HV).to(torch.float32)

    out = torch.empty((qf.shape[0], HV, Vdim), dtype=torch.float32, device=device)

    state_indices_tensor: torch.Tensor | None = None
    state_indices_valid: torch.Tensor | None = None
    if ssm_state_indices is not None:
        state_indices_tensor = ssm_state_indices.reshape(-1).to(
            dtype=torch.long,
            device=state_work.device,
        )
        state_indices_valid = ((state_indices_tensor >= 0) & (state_indices_tensor < state_work.shape[0]))

    num_state_indices = (int(state_indices_tensor.shape[0]) if state_indices_tensor is not None else 0)
    # trip count = num_seqs (batch bucket); recompile per batch bucket
    for seq_id, (bos, eos) in enumerate(seq_ranges):
        if eos <= bos:
            continue

        if state_indices_tensor is not None and state_indices_valid is not None:
            if seq_id >= num_state_indices:
                continue

            seq_id_t = torch.tensor([seq_id], dtype=torch.long, device=state_work.device)
            valid_seq = state_indices_valid.index_select(0, seq_id_t)
            raw_idx = state_indices_tensor.index_select(0, seq_id_t)
            safe_idx = torch.where(valid_seq, raw_idx, torch.zeros_like(raw_idx))
            prev_state = state_work.index_select(0, safe_idx)
            h_state = prev_state.squeeze(0)
        else:
            h_state = state_work[seq_id]

        # trip count = padded_seq_len per sequence (seq bucket); recompile per seq bucket,
        # worst one with 2k inputs, for loop 2k times
        #TODO: vectorize this loop with a custom scan or by reshaping to
        # [num_chunks, chunk_size, H] and doing a grouped cumsum with resets at chunk boundaries.
        for t in range(bos, eos):
            q_t = qf[t]
            k_t = kf[t]
            v_t = vf[t]
            g_t = gf[t]
            b_t = bf[t]

            if use_qk_l2norm_in_kernel:
                q_t = _l2norm_last_dim(q_t)
                k_t = _l2norm_last_dim(k_t)

            out_t, h_state = _recurrent_timestep_body(
                q_t,
                k_t,
                v_t,
                g_t,
                b_t,
                h_state,
                scale,
                HV,
                H,
                Kdim,
            )
            out[t] = out_t

        # Persist state back to selected cache line.
        if state_indices_tensor is not None and state_indices_valid is not None:
            # Avoid Python-side scalar branching in graph mode: for invalid
            # indices, write back the unchanged state.
            updated_state = torch.where(
                valid_seq.view(1, 1, 1),
                h_state.unsqueeze(0),
                prev_state,
            )
            state_work.index_copy_(0, safe_idx, updated_state)
        else:
            state_work[seq_id] = h_state

    final_state.copy_(state_work.to(final_state.dtype))
    out = out.to(v.dtype)

    out = out.unsqueeze(0) if cu_seqlens is not None else out.view(B, T, HV, Vdim)

    return out, final_state


def hpu_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    prefill_num_seqs: int | None = None,
    prefill_seq_len: int | None = None,
    # NOTE: neumann_iters impacts accuracy. 14 is used for Qwen3.5; other
    # models may need re-tuning. See _hpu_solve_lower_triangular_batched docs.
    neumann_iters: int = 14,
    fused_state_matmul: bool = False,
    recursive_solver_base: int = 0,
    compact_repeated_kkt: bool = False,
    compile_qk_l2norm: bool = False,
    deferred_output_add: bool | None = None,
    qk_head_repeat_override: int | None = None,
    compact_repeated_local_attn: bool = False,
    preserve_compact_qk: bool = False,
    compact_qk_factor_gate: bool = False,
    flashqla_reformulation: bool | None = None,
    compute_dtype: torch.dtype | None = None,
    solve_in_fp32: bool | None = None,
    state_in_fp32: bool | None = None,
    masked_triangular_decay: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """PyTorch replacement for chunk_gated_delta_rule.

    This path intentionally mirrors upstream prefill call semantics without
    delegating to the fused recurrent helper.

    When ``prefill_num_seqs`` and ``prefill_seq_len`` are provided (both
    Python ints), the function bypasses ``_materialize_seq_ranges`` (which
    requires a device-to-host sync) and constructs uniform seq_ranges
    directly.  This makes the function fully compilable with torch.compile.
    """
    # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py#L132
    B, T, H, Kdim = q.shape
    _, _, HV, Vdim = v.shape
    device = q.device
    qk_head_repeat = HV // H if H != HV and HV % H == 0 else 1
    if qk_head_repeat_override is not None:
        override_is_valid = qk_head_repeat_override > 0 and (
            (H == HV and H % qk_head_repeat_override == 0)
            or (H != HV and HV == H * qk_head_repeat_override)
        )
        if not override_is_valid:
            raise ValueError(
                "qk_head_repeat_override must match either an expanded or "
                "compact grouped-head layout, got "
                f"repeat={qk_head_repeat_override}, H={H}, HV={HV}."
            )
        qk_head_repeat = qk_head_repeat_override
    compact_qk_inputs = preserve_compact_qk and H != HV
    use_flashqla_reformulation = (
        _USE_FLASHQLA_REFORMULATION
        if flashqla_reformulation is None
        else flashqla_reformulation
    )
    if preserve_compact_qk and not compact_qk_inputs:
        raise ValueError(
            "preserve_compact_qk requires fewer Q/K heads than value heads."
        )
    if masked_triangular_decay and not use_flashqla_reformulation:
        raise ValueError(
            "masked_triangular_decay requires FlashQLA reformulation."
        )
    if deferred_output_add is None:
        deferred_output_add = resolve_hpu_gdn_deferred_output_add()
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    if neumann_iters <= 0:
        raise ValueError(f"neumann_iters must be > 0, got {neumann_iters}.")

    # ---- Compile-friendly 3-stage path (HPU bucketed prefill) ----
    if prefill_num_seqs is not None and prefill_seq_len is not None \
            and prefill_num_seqs > 0:
        (qf, kf, vf, bf, g_cumsum, init_state, H_c, num_chunks, scale_c, Kdim_c, Vdim_c,
         S_c) = hpu_chunk_gdr_preprocess(
             q=q,
             k=k,
             v=v,
             g=g,
             beta=beta,
             scale=scale,
             initial_state=initial_state,
             use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
             chunk_size=chunk_size,
             num_seqs=prefill_num_seqs,
             seq_len=prefill_seq_len,
             compile_qk_l2norm=compile_qk_l2norm,
             compute_dtype=compute_dtype,
             preserve_compact_qk=compact_qk_inputs,
         )
        phase_a = (
            hpu_flashqla_chunk_gdr_phase_a
            if use_flashqla_reformulation
            else hpu_chunk_gdr_phase_a
        )
        phase_a_kwargs = {}
        if use_flashqla_reformulation:
            phase_a_kwargs["compact_qk_factor_gate"] = (
                compact_qk_factor_gate
            )
            phase_a_kwargs["return_local_decay"] = True
            phase_a_kwargs["native_compact_kkt"] = (
                _USE_QWEN38_NATIVE_COMPACT_KKT
            )
            phase_a_kwargs["masked_triangular_decay"] = (
                masked_triangular_decay
            )
        if use_flashqla_reformulation and _FLASHQLA_FP32_SCALING:
            phase_a_kwargs.update(
                fp32_rhs_scaling=True,
                fp32_output_scaling=True,
            )

        phase_a_result = phase_a(
            qf,
            kf,
            vf,
            bf,
            g_cumsum,
            seq_len=prefill_seq_len,
            chunk_size=chunk_size,
            S=S_c,
            num_chunks=num_chunks,
            H=H_c,
            Kdim=Kdim_c,
            Vdim=Vdim_c,
            neumann_iters=neumann_iters,
            recursive_solver_base=recursive_solver_base,
            compact_repeated_kkt=compact_repeated_kkt,
            qk_head_repeat=qk_head_repeat,
            solve_in_fp32=solve_in_fp32,
            compact_qk_inputs=compact_qk_inputs,
            **phase_a_kwargs,
        )
        if use_flashqla_reformulation:
            (
                u_all,
                w_all,
                q_chunks,
                k_chunks,
                g_chunks,
                local_decay,
            ) = phase_a_result
        else:
            u_all, w_all, q_chunks, k_chunks, g_chunks = phase_a_result
            local_decay = None

        out, final_state = hpu_chunk_gdr_phase_b(
            u_all,
            w_all,
            q_chunks,
            k_chunks,
            g_chunks,
            init_state,
            scale_c,
            S=S_c,
            num_chunks=num_chunks,
            seq_len=prefill_seq_len,
            H=H_c,
            Kdim=Kdim_c,
            Vdim=Vdim_c,
            output_final_state=output_final_state,
            output_dtype=initial_state.dtype if initial_state is not None else None,
            fused_state_matmul=fused_state_matmul,
            deferred_output_add=deferred_output_add,
            factorized_local_decay=(
                use_flashqla_reformulation
                and _FLASHQLA_FACTORIZE_PHASE_B
            ),
            local_decay=local_decay,
            compact_repeated_local_attn=compact_repeated_local_attn,
            qk_head_repeat=qk_head_repeat,
            compact_qk_inputs=compact_qk_inputs,
            state_in_fp32=state_in_fp32,
        )

        out = out.to(q.dtype).view(B, T, H_c, Vdim)
        return out, final_state

    # ---- Legacy paths (cu_seqlens / non-bucketed) ----
    return _hpu_chunk_gated_delta_rule_legacy(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        chunk_size,
        neumann_iters,
        B,
        T,
        H,
        HV,
        Kdim,
        Vdim,
        device,
    )


# ========================================================================
# Legacy helper functions
#
# Used only by the legacy paths (_hpu_chunk_gdr_phase_b_legacy,
# _hpu_chunk_gated_delta_rule_legacy).  Kept for debugging / accuracy
# validation.  Enable via VLLM_GDN_LEGACY_PHASE_B=1.
# ========================================================================


def _hpu_chunk_gated_delta_rule_legacy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    use_qk_l2norm_in_kernel: bool,
    chunk_size: int,
    neumann_iters: int,
    B: int,
    T: int,
    H: int,
    HV: int,
    Kdim: int,
    Vdim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Legacy chunk pipeline: cu_seqlens / non-bucketed paths.

    Handles head repeat, l2norm, cumsum, and dispatches to
    _chunk_precomputed_pipeline (vectorized) or per-head reference loop.
    """
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("When cu_seqlens is used, expected batch size B=1.")
        seq_ranges = _materialize_seq_ranges(cu_seqlens, B * T)
        num_seqs = len(seq_ranges)
    else:
        num_seqs = B
        seq_ranges = [(i * T, (i + 1) * T) for i in range(B)]

    # Match upstream grouped-value semantics.
    if H != HV:
        if HV % H == 0:
            repeat = HV // H
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)
            H = HV
        else:
            raise ValueError("Unsupported head mapping in hpu_chunk_gated_delta_rule: "
                             f"q/k heads={H}, value heads={HV}. Expected HV % H == 0.")

    if scale is None:
        scale = k.shape[-1]**-0.5

    # Match upstream ChunkGatedDeltaRuleFunction behavior: normalize full
    # q/k tensors before the core chunk pipeline.
    if use_qk_l2norm_in_kernel:
        q = _l2norm_last_dim(q.to(torch.float32))
        k = _l2norm_last_dim(k.to(torch.float32))

    # Flatten token axis for shared varlen/non-varlen logic.
    qf = q.reshape(-1, H, Kdim).to(torch.float32)
    kf = k.reshape(-1, H, Kdim).to(torch.float32)
    vf = v.reshape(-1, HV, Vdim).to(torch.float32)
    gf = g.reshape(-1, HV).to(torch.float32)
    bf = beta.reshape(-1, HV).to(torch.float32)

    # Upstream match: `chunk_local_cumsum` in fla/ops/cumsum.py.
    # Stage 1 computes per-chunk cumulative g in log-space.
    g_cumsum = torch.empty_like(gf)
    if num_seqs > 0:
        seq_len = seq_ranges[0][1] - seq_ranges[0][0]
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        total_tokens = num_seqs * seq_len
        g_active = gf[:total_tokens]
        padded_len = num_chunks * chunk_size
        if padded_len > seq_len:
            pad = torch.zeros(num_seqs * (padded_len - seq_len), gf.shape[1], dtype=gf.dtype, device=gf.device)
            g_block = g_active.reshape(num_seqs, seq_len, -1)
            pad_block = pad.reshape(num_seqs, padded_len - seq_len, -1)
            g_block = torch.cat([g_block, pad_block], dim=1)
        else:
            g_block = g_active.reshape(num_seqs, seq_len, -1)
        g_block = g_block.reshape(num_seqs, num_chunks, chunk_size, -1)
        g_cumsum_block = torch.cumsum(g_block, dim=2)
        g_cumsum[:total_tokens] = g_cumsum_block.reshape(num_seqs, -1,
                                                         gf.shape[1])[:, :seq_len, :].reshape(-1, gf.shape[1])
    else:
        for bos, eos in seq_ranges:
            for cs in range(bos, eos, chunk_size):
                ce = min(cs + chunk_size, eos)
                g_cumsum[cs:ce] = torch.cumsum(gf[cs:ce], dim=0)

    # Initial state layout: [num_seqs, H, V, K].
    if initial_state is None:
        init_state = torch.zeros(
            (num_seqs, H, Vdim, Kdim),
            dtype=torch.float32,
            device=device,
        )
    else:
        if initial_state.shape[0] != num_seqs:
            raise ValueError("The number of initial states is expected to equal the number "
                             f"of input sequences ({num_seqs}), got {initial_state.shape[0]}.")
        init_state = initial_state.to(torch.float32)

    out = torch.zeros((qf.shape[0], H, Vdim), dtype=torch.float32, device=device)
    final_state = torch.empty_like(init_state) if output_final_state else None

    eye_cache: dict[int, torch.Tensor] = {}
    use_vectorized_chunk = True
    _vectorize_seq_loop = (use_vectorized_chunk and num_seqs > 0)

    if _vectorize_seq_loop:
        total_active = num_seqs * (seq_ranges[0][1] - seq_ranges[0][0])
        out_pre, fs_pre = _chunk_precomputed_pipeline(
            qf=qf[:total_active],
            kf=kf[:total_active],
            vf=vf[:total_active],
            g_cumsum=g_cumsum[:total_active],
            bf=bf[:total_active],
            init_state=init_state,
            seq_ranges=seq_ranges,
            chunk_size=chunk_size,
            scale=scale,
            H=H,
            Kdim=Kdim,
            Vdim=Vdim,
            output_final_state=(final_state is not None),
            neumann_iters=neumann_iters,
        )
        out[:total_active] = out_pre
        if final_state is not None and fs_pre is not None:
            final_state.copy_(fs_pre)

    else:
        for seq_id, (bos, eos) in enumerate(seq_ranges):
            if eos <= bos:
                if final_state is not None:
                    final_state[seq_id] = init_state[seq_id]
                continue

            state = init_state[seq_id].clone()
            for cs in range(bos, eos, chunk_size):
                ce = min(cs + chunk_size, eos)
                tc = ce - cs

                q_chunk = qf[cs:ce]
                k_chunk = kf[cs:ce]
                v_chunk = vf[cs:ce]
                g_chunk = g_cumsum[cs:ce]
                beta_chunk = bf[cs:ce]

                if tc not in eye_cache:
                    eye_cache[tc] = torch.eye(tc, dtype=qf.dtype, device=device)

                if use_vectorized_chunk:
                    out[cs:ce], state = _chunk_vectorized_body(
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        g_chunk,
                        beta_chunk,
                        state,
                        eye_cache[tc],
                        scale,
                        neumann_iters,
                    )
                else:
                    # Per-head reference path (list accumulation for HPU compat).
                    a_solve_list: list[torch.Tensor] = []
                    for h in range(H):
                        kh = k_chunk[:, h, :]
                        bh = beta_chunk[:, h]
                        gh = g_chunk[:, h]
                        dot = kh @ kh.transpose(0, 1)
                        coeff = bh[:, None] * torch.exp(gh[:, None] - gh[None, :])
                        a_lower = torch.tril(dot * coeff, diagonal=-1)
                        lmat = eye_cache[tc] + a_lower
                        a_solve_h = _hpu_solve_lower_triangular_batched(
                            lmat,
                            eye_cache[tc],
                            use_vectorized=False,
                            neumann_iters=neumann_iters,
                        )
                        a_solve_list.append(a_solve_h.unsqueeze(0))
                    A_solve = torch.cat(a_solve_list, dim=0)

                    u_list: list[torch.Tensor] = []
                    w_list: list[torch.Tensor] = []
                    for h in range(H):
                        rhs_u = v_chunk[:, h, :] * beta_chunk[:, h:h + 1]
                        rhs_w = (k_chunk[:, h, :] * (beta_chunk[:, h] * torch.exp(g_chunk[:, h]))[:, None])
                        u_list.append((A_solve[h] @ rhs_u).unsqueeze(1))
                        w_list.append((A_solve[h] @ rhs_w).unsqueeze(1))
                    u_chunk = torch.cat(u_list, dim=1)
                    w_chunk = torch.cat(w_list, dim=1)

                    v_new_list: list[torch.Tensor] = []
                    h_start = state.clone()
                    for h in range(H):
                        state_h = h_start[h]
                        proj = w_chunk[:, h, :] @ state_h.transpose(0, 1)
                        val_raw = u_chunk[:, h, :] - proj
                        v_new_list.append(val_raw.unsqueeze(1))

                        g_last = g_chunk[-1, h]
                        val_state = val_raw * torch.exp(g_last - g_chunk[:, h])[:, None]
                        state_h = state_h * torch.exp(g_last)
                        state_h = state_h + val_state.transpose(0, 1) @ k_chunk[:, h, :]
                        state[h] = state_h
                    v_new_chunk = torch.cat(v_new_list, dim=1)

                    out_list: list[torch.Tensor] = []
                    for h in range(H):
                        qh = q_chunk[:, h, :]
                        kh = k_chunk[:, h, :]
                        vh = v_new_chunk[:, h, :]
                        hs = h_start[h]
                        gh = g_chunk[:, h]

                        base = qh @ hs.transpose(0, 1)
                        base = base * torch.exp(gh)[:, None]
                        attn = qh @ kh.transpose(0, 1)
                        attn = attn * torch.exp(gh[:, None] - gh[None, :])
                        attn = torch.tril(attn)
                        out_list.append(((base + attn @ vh) * scale).unsqueeze(1))
                    out[cs:ce] = torch.cat(out_list, dim=1)

            if final_state is not None:
                final_state[seq_id] = state

    out = out.to(q.dtype)
    out = out.unsqueeze(0) if cu_seqlens is not None else out.view(B, T, H, Vdim)

    if final_state is None:
        return out, None
    if initial_state is not None:
        final_state = final_state.to(initial_state.dtype)
    return out, final_state


def _hpu_chunk_gdr_phase_b_legacy(
    u_all: torch.Tensor,
    w_all: torch.Tensor,
    q_chunks: torch.Tensor,
    k_chunks: torch.Tensor,
    g_chunks: torch.Tensor,
    init_state: torch.Tensor,
    scale: float,
    S: int,
    num_chunks: int,
    seq_len: int,
    H: int,
    Kdim: int,
    Vdim: int,
    output_final_state: bool,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Legacy Phase B: uses _phase_b_step per chunk (reference accuracy).

    Slower but numerically identical to the original implementation.
    Enable via VLLM_GDN_LEGACY_PHASE_B=1 for debugging accuracy issues.
    """
    tc = u_all.shape[2]
    padded_len = num_chunks * tc
    device = u_all.device

    states = init_state.clone()
    out_all = torch.zeros(S, num_chunks, tc, H, Vdim, dtype=torch.float32, device=device)

    for ci in range(num_chunks):
        out_all[:, ci], states = _phase_b_step(
            u_all[:, ci],
            w_all[:, ci],
            k_chunks[:, ci],
            g_chunks[:, ci],
            q_chunks[:, ci],
            states,
            scale,
            S,
            H,
            Kdim,
            Vdim,
        )

    out = out_all.reshape(S, padded_len, H, Vdim)[:, :seq_len, :, :].reshape(-1, H, Vdim)

    final_state: torch.Tensor | None = None
    if output_final_state:
        final_state = states.to(output_dtype if output_dtype is not None else init_state.dtype)

    return out, final_state


def _phase_b_step(
    u_c: torch.Tensor,  # [S, tc, H, V]
    w_c: torch.Tensor,  # [S, tc, H, K]
    k_c: torch.Tensor,  # [S, tc, H, K]
    g_c: torch.Tensor,  # [S, tc, H]
    q_c: torch.Tensor,  # [S, tc, H, K]
    states: torch.Tensor,  # [S, H, V, K]
    scale: float,
    S: int,
    H: int,
    Kdim: int,
    Vdim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single chunk step for stages 5-6 (state update + output).

    Extracted as a standalone function so dynamo compiles it once and
    the Python for-loop re-launches the same cached graph every iteration.

    Returns (out_ci [S, tc, H, V], new_states [S, H, V, K]).
    """
    tc = g_c.shape[1]

    # Stage 5: state update (batched across S)
    h_start = states.clone()  # [S, H, V, K]
    v_new_c = u_c - torch.einsum("sthk,shvk->sthv", w_c, h_start)

    g_last = g_c[:, -1:, :]  # [S, 1, H]
    decay = torch.exp(g_last - g_c)  # [S, tc, H]
    val_state = v_new_c * decay.unsqueeze(-1)  # [S, tc, H, V]
    new_states = (h_start * torch.exp(g_last.permute(0, 2, 1)).unsqueeze(-1) +
                  torch.einsum("sthv,sthk->shvk", val_state, k_c))

    # Stage 6: output computation — merge S*H for bmm
    SH = S * H
    q_sh = q_c.permute(0, 2, 1, 3).reshape(SH, tc, Kdim)
    k_sh = k_c.permute(0, 2, 1, 3).reshape(SH, tc, Kdim)
    v_new_sh = v_new_c.permute(0, 2, 1, 3).reshape(SH, tc, Vdim)
    h_start_sh = h_start.reshape(SH, Vdim, Kdim)
    g_sh = g_c.permute(0, 2, 1).reshape(SH, tc)

    recurrent_term = torch.bmm(q_sh, h_start_sh.transpose(1, 2))
    recurrent_term = recurrent_term * torch.exp(g_sh).unsqueeze(-1)
    attn = torch.bmm(q_sh, k_sh.transpose(1, 2))
    attn = attn * torch.exp(g_sh.unsqueeze(-1) - g_sh.unsqueeze(-2))
    attn = torch.tril(attn)
    out_sh = torch.bmm(attn, v_new_sh)
    # Match torch_chunk_gated_delta_rule_opt style: core chunk output plus
    # in-place add of recurrent-state contribution from the previous state.
    out_sh.add_(recurrent_term)
    out_sh = out_sh * scale

    out_ci = out_sh.reshape(S, H, tc, Vdim).permute(0, 2, 1, 3)
    return out_ci, new_states


def _chunk_precomputed_pipeline(
    qf: torch.Tensor,  # [total_tokens, H, K]
    kf: torch.Tensor,  # [total_tokens, H, K]
    vf: torch.Tensor,  # [total_tokens, H, V]
    g_cumsum: torch.Tensor,  # [total_tokens, H]
    bf: torch.Tensor,  # [total_tokens, H]
    init_state: torch.Tensor,  # [S, H, V, K]
    seq_ranges: list[tuple[int, int]],
    chunk_size: int,
    scale: float,
    H: int,
    Kdim: int,
    Vdim: int,
    output_final_state: bool,
    neumann_iters: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Optimized chunk pipeline with precomputed stages 2-4.

    Stages 2-4 (dot product, triangular solve, u/w recomputation) are
    independent across chunks and are computed for ALL chunks at once in
    a single batched call.  Only stages 5-6 (state update and output)
    require sequential processing due to inter-chunk state dependency.

    This moves ~70% of the per-chunk compute out of the sequential loop,
    critical for long sequences (e.g. 128k tokens = 1024 chunks).
    """
    num_seqs = len(seq_ranges)
    device = qf.device
    seq_len = seq_ranges[0][1] - seq_ranges[0][0]
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    S = num_seqs

    # --- Gather per-sequence data into [S, seq_len, H, dim] blocks ---
    q_seqs = qf.reshape(S, seq_len, H, Kdim)
    k_seqs = kf.reshape(S, seq_len, H, Kdim)
    v_seqs = vf.reshape(S, seq_len, H, Vdim)
    g_seqs = g_cumsum.reshape(S, seq_len, H)
    b_seqs = bf.reshape(S, seq_len, H)

    # --- Reshape to [S, num_chunks, chunk_size, H, dim] ---
    # Pad if seq_len not divisible by chunk_size.
    padded_len = num_chunks * chunk_size
    if padded_len > seq_len:
        pad_len = padded_len - seq_len
        q_seqs = torch.cat([q_seqs, torch.zeros(S, pad_len, H, Kdim, dtype=qf.dtype, device=device)], dim=1)
        k_seqs = torch.cat([k_seqs, torch.zeros(S, pad_len, H, Kdim, dtype=kf.dtype, device=device)], dim=1)
        v_seqs = torch.cat([v_seqs, torch.zeros(S, pad_len, H, Vdim, dtype=vf.dtype, device=device)], dim=1)
        # Pad g_cumsum with the LAST VALID value (not zero).  Zero-padding
        # causes exp(0 - g_valid) = exp(|g_valid|) which overflows to Inf
        # when |g_valid| > ~88 (common with real GDN weights), producing
        # NaN via 0*Inf in the coefficient matrix.  Replicating the last
        # valid cumsum value keeps all exp differences bounded.
        g_last_valid = g_seqs[:, -1:, :]  # [S, 1, H]
        g_seqs = torch.cat([g_seqs, g_last_valid.expand(S, pad_len, H)], dim=1)
        b_seqs = torch.cat([b_seqs, torch.zeros(S, pad_len, H, dtype=bf.dtype, device=device)], dim=1)

    # [S, C, tc, H, dim] where C = num_chunks, tc = chunk_size
    tc = chunk_size
    q_chunks = q_seqs.reshape(S, num_chunks, tc, H, Kdim)
    k_chunks = k_seqs.reshape(S, num_chunks, tc, H, Kdim)
    v_chunks = v_seqs.reshape(S, num_chunks, tc, H, Vdim)
    g_chunks = g_seqs.reshape(S, num_chunks, tc, H)
    b_chunks = b_seqs.reshape(S, num_chunks, tc, H)

    # ====================================================================
    # Phase A: Precompute stages 2-4 for ALL chunks at once.
    # Flatten (S, C) into batch dim → [S*C*H, tc, dim]
    # ====================================================================
    SC = S * num_chunks
    # Merge S,C dims then permute H to batch: [S*C, tc, H, K] -> [S*C*H, tc, K]
    k_flat = k_chunks.reshape(SC, tc, H, Kdim).permute(0, 2, 1, 3).reshape(SC * H, tc, Kdim)
    v_flat = v_chunks.reshape(SC, tc, H, Vdim).permute(0, 2, 1, 3).reshape(SC * H, tc, Vdim)
    g_flat = g_chunks.reshape(SC, tc, H).permute(0, 2, 1).reshape(SC * H, tc)
    b_flat = b_chunks.reshape(SC, tc, H).permute(0, 2, 1).reshape(SC * H, tc)

    eye = torch.eye(tc, dtype=qf.dtype, device=device)

    # Stage 2: chunk_scaled_dot_kkt — all S*C*H chunks at once
    dot = torch.bmm(k_flat, k_flat.transpose(1, 2))  # [SC*H, tc, tc]
    coeff = b_flat.unsqueeze(-1) * (torch.exp(g_flat.unsqueeze(-1) - g_flat.unsqueeze(-2))).to(b_flat.dtype)
    a_lower = torch.tril(dot * coeff, diagonal=-1)
    lmat = (eye.unsqueeze(0) + a_lower).to(qf.dtype)

    # Stage 3: solve_tril — all S*C*H at once (Neumann or forward-sub)
    A_solve = _hpu_solve_lower_triangular_batched(
        lmat,
        eye,
        use_vectorized=True,
        neumann_iters=neumann_iters,
    )

    # Stage 4: recompute u, w — all S*C*H at once
    rhs_u = v_flat * b_flat.unsqueeze(-1)  # [SC*H, tc, V]
    rhs_w = k_flat * (b_flat * torch.exp(g_flat)).unsqueeze(-1)  # [SC*H, tc, K]
    u_flat = torch.bmm(A_solve, rhs_u)  # [SC*H, tc, V]
    w_flat = torch.bmm(A_solve, rhs_w)  # [SC*H, tc, K]

    # Reshape precomputed results to [S, num_chunks, tc, H, dim]
    u_all = u_flat.reshape(SC, H, tc, Vdim).permute(0, 2, 1, 3).reshape(S, num_chunks, tc, H, Vdim)
    w_all = w_flat.reshape(SC, H, tc, Kdim).permute(0, 2, 1, 3).reshape(S, num_chunks, tc, H, Kdim)

    # Also precompute q reshaped for stage 6: [S, num_chunks, tc, H, K]
    q_all = q_chunks  # already [S, C, tc, H, K]

    # ====================================================================
    # Phase B: Sequential loop — stages 5-6 only (state-dependent).
    # ====================================================================
    states = init_state.clone()  # [S, H, V, K]
    out_all = torch.zeros(S, num_chunks, tc, H, Vdim, dtype=torch.float32, device=device)

    for ci in range(num_chunks):
        out_all[:, ci], states = _phase_b_step(
            u_all[:, ci],
            w_all[:, ci],
            k_chunks[:, ci],
            g_chunks[:, ci],
            q_all[:, ci],
            states,
            scale,
            S,
            H,
            Kdim,
            Vdim,
        )

    # --- Scatter output back to flat [total_tokens, H, V] ---
    out = out_all.reshape(S, padded_len, H, Vdim)[:, :seq_len, :, :].reshape(-1, H, Vdim)

    final_state: torch.Tensor | None = None
    if output_final_state:
        final_state = states.to(init_state.dtype)

    return out, final_state


def _chunk_vectorized_body(
    q_chunk: torch.Tensor,  # [Tc, H, K]
    k_chunk: torch.Tensor,  # [Tc, H, K]
    v_chunk: torch.Tensor,  # [Tc, H, V]
    g_chunk: torch.Tensor,  # [Tc, H]
    beta_chunk: torch.Tensor,  # [Tc, H]
    state: torch.Tensor,  # [H, V, K]
    eye: torch.Tensor,  # [Tc, Tc]
    scale: float,
    neumann_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compilable vectorized chunk body for hpu_chunk_gated_delta_rule.

    Returns (out_chunk [Tc, H, V], new_state [H, V, K]).
    """
    k_h = k_chunk.permute(1, 0, 2).contiguous()  # [H, Tc, K]
    g_h = g_chunk.transpose(0, 1).contiguous()  # [H, Tc]
    beta_h = beta_chunk.transpose(0, 1).contiguous()  # [H, Tc]

    dot = torch.bmm(k_h, k_h.transpose(1, 2))
    coeff = beta_h.unsqueeze(-1) * (torch.exp(g_h.unsqueeze(-1) - g_h.unsqueeze(-2))).to(beta_h.dtype)
    a_lower = torch.tril(dot * coeff, diagonal=-1)
    lmat = (eye.unsqueeze(0) + a_lower).to(q_chunk.dtype)
    A_solve = _hpu_solve_lower_triangular_batched(
        lmat,
        eye,
        use_vectorized=True,
        neumann_iters=neumann_iters,
    )

    rhs_u = v_chunk.permute(1, 0, 2).contiguous() * beta_h.unsqueeze(-1)
    rhs_w = k_h * (beta_h * torch.exp(g_h)).unsqueeze(-1)
    u_chunk = torch.bmm(A_solve, rhs_u).permute(1, 0, 2).contiguous()
    w_chunk = torch.bmm(A_solve, rhs_w).permute(1, 0, 2).contiguous()

    h_start = state.clone()
    v_new_chunk = u_chunk - torch.einsum("thk,hvk->thv", w_chunk, h_start)

    # Prefer reshape/index broadcasting over chained unsqueeze on
    # sliced tensors for HPU graph lowering stability.
    tc = k_chunk.shape[0]
    H = k_h.shape[0]
    g_last_tc_h = g_chunk[-1:, :]  # [1, H]
    decay_tc_h = torch.exp(g_last_tc_h - g_chunk)
    val_state = v_new_chunk * decay_tc_h.reshape(tc, H, 1)
    new_state = (h_start * torch.exp(g_last_tc_h[0]).reshape(H, 1, 1) +
                 torch.einsum("thv,thk->hvk", val_state, k_chunk))

    q_h = q_chunk.permute(1, 0, 2).contiguous()
    v_new_h = v_new_chunk.permute(1, 0, 2).contiguous()
    base_h = torch.einsum("htk,hvk->htv", q_h, h_start)
    base_h = base_h * torch.exp(g_h).reshape(H, tc, 1)
    attn_h = torch.bmm(q_h, k_h.transpose(1, 2))
    g_h_l = g_h.reshape(H, tc, 1)
    g_h_r = g_h.reshape(H, 1, tc)
    attn_h = attn_h * torch.exp(g_h_l - g_h_r)
    attn_h = torch.tril(attn_h)
    out_chunk = (base_h + torch.bmm(attn_h, v_new_h)).permute(1, 0, 2) * scale

    return out_chunk, new_state
