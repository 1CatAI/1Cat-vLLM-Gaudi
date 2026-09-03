# SPDX-License-Identifier: Apache-2.0
"""FlashInfer-compatible Gated Delta Rule prefill for Intel Gaudi."""

from __future__ import annotations

from typing import Literal

import torch


def _chunk_gated_delta_rule_log_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    prefill_num_seqs: int | None = None,
    prefill_seq_len: int | None = None,
    neumann_iters: int = 14,
    fused_state_matmul: bool = False,
    recursive_solver_base: int = 0,
    compact_repeated_kkt: bool = False,
    compile_qk_l2norm: bool = False,
    flashqla_reformulation: bool = False,
    deferred_output_add: bool = False,
    compute_dtype: torch.dtype | None = None,
    solve_in_fp32: bool = False,
    state_in_fp32: bool = False,
    preserve_compact_qk: bool = False,
    masked_triangular_decay: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Internal log-gate entry point used by the vLLM adapter.

    vLLM already computes the forget gate in log space. Keeping this private
    entry point avoids an otherwise redundant ``exp`` followed by ``log`` at
    every GDN layer while the public API retains FlashInfer's alpha contract.
    """
    from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_chunk_gated_delta_rule

    return hpu_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=log_decay,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        chunk_size=chunk_size,
        prefill_num_seqs=prefill_num_seqs,
        prefill_seq_len=prefill_seq_len,
        neumann_iters=neumann_iters,
        fused_state_matmul=fused_state_matmul,
        recursive_solver_base=recursive_solver_base,
        compact_repeated_kkt=compact_repeated_kkt,
        compile_qk_l2norm=compile_qk_l2norm,
        flashqla_reformulation=flashqla_reformulation,
        deferred_output_add=deferred_output_add,
        compute_dtype=compute_dtype,
        solve_in_fp32=solve_in_fp32,
        state_in_fp32=state_in_fp32,
        preserve_compact_qk=preserve_compact_qk,
        masked_triangular_decay=masked_triangular_decay,
    )


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor | None) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.ndim != 3:
            raise ValueError(f"{name} must have [total_seq_len, heads, dim] shape, got {tuple(tensor.shape)}.")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must contain the same number of tokens.")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have matching shapes, got {tuple(q.shape)} and {tuple(k.shape)}.")
    if q.shape[-1] != v.shape[-1]:
        raise ValueError("The Gaudi GDN prefill path currently requires equal key and value dimensions.")
    if v.shape[1] % q.shape[1] != 0:
        raise ValueError("The number of value heads must be divisible by the number of Q/K heads.")
    if cu_seqlens is None:
        raise AssertionError("cu_seqlens is required for varlen mode")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional tensor with at least two elements.")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}.")


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    use_cp: Literal["auto"] | bool = "auto",
    state_indices: torch.Tensor | None = None,
    _cp_chunk_len: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute FlashInfer's public GDN prefill contract on Gaudi.

    Inputs use FlashInfer's packed three-dimensional layout and ``g`` is the
    forget gate alpha, not its logarithm. The returned recurrent state uses
    the VK/K-last layout ``[num_seqs, value_heads, value_dim, key_dim]``.

    Context-parallel checkpointing and indexed state pools are not yet part
    of the promoted Gaudi2 tactic; requesting either fails explicitly.
    """
    _validate_inputs(q, k, v, cu_seqlens)
    assert cu_seqlens is not None
    if use_cp not in ("auto", True, False):
        raise ValueError(f'use_cp must be "auto", True, or False, got {use_cp!r}.')
    if use_cp is True:
        raise NotImplementedError("Context-parallel GDN prefill is not implemented for Gaudi2.")
    if state_indices is not None:
        raise NotImplementedError("Indexed GDN prefill state pools are not implemented for Gaudi2.")
    if checkpoint_every_n_tokens != 0 or state_checkpoints is not None or checkpoint_cu_starts is not None:
        raise NotImplementedError("GDN prefill state checkpointing is not implemented for Gaudi2.")
    if _cp_chunk_len is not None:
        raise NotImplementedError("_cp_chunk_len is only meaningful for context-parallel execution.")
    if output_state is not None and not output_final_state:
        raise ValueError("output_state requires output_final_state=True.")

    total_tokens = q.shape[0]
    value_heads = v.shape[1]
    if g is None:
        log_decay = torch.zeros(total_tokens, value_heads, dtype=torch.float32, device=q.device)
    else:
        if tuple(g.shape) != (total_tokens, value_heads):
            raise ValueError(f"g has shape {tuple(g.shape)}, expected {(total_tokens, value_heads)}.")
        log_decay = torch.log(g.to(torch.float32))
    if beta is None:
        beta_value = torch.ones(total_tokens, value_heads, dtype=torch.float32, device=q.device)
    else:
        if tuple(beta.shape) != (total_tokens, value_heads):
            raise ValueError(f"beta has shape {tuple(beta.shape)}, expected {(total_tokens, value_heads)}.")
        beta_value = beta

    num_seqs = cu_seqlens.numel() - 1
    effective_initial_state = initial_state
    if effective_initial_state is None:
        effective_initial_state = torch.zeros(
            num_seqs,
            value_heads,
            v.shape[-1],
            q.shape[-1],
            dtype=torch.float32,
            device=q.device,
        )
    # A single sequence is the common serving path and its length is already
    # known from the static tensor shape, so it can use the fully compiled
    # implementation without copying cu_seqlens to the host.
    uniform_num_seqs = 1 if num_seqs == 1 else None
    uniform_seq_len = total_tokens if num_seqs == 1 else None
    result, final_state = _chunk_gated_delta_rule_log_gate(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        log_decay.unsqueeze(0),
        beta_value.unsqueeze(0),
        scale=scale,
        initial_state=effective_initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        chunk_size=64,
        prefill_num_seqs=uniform_num_seqs,
        prefill_seq_len=uniform_seq_len,
        neumann_iters=14,
        fused_state_matmul=True,
        recursive_solver_base=16,
        compact_repeated_kkt=True,
        flashqla_reformulation=True,
        deferred_output_add=True,
        compute_dtype=torch.float32,
        solve_in_fp32=True,
        state_in_fp32=True,
        preserve_compact_qk=v.shape[1] > q.shape[1],
        masked_triangular_decay=True,
    )
    result = result.squeeze(0)
    if output is not None:
        if output.shape != result.shape:
            raise ValueError(f"output has shape {tuple(output.shape)}, expected {tuple(result.shape)}.")
        output.copy_(result)
        result = output

    if not output_final_state:
        return result
    assert final_state is not None
    if output_state is not None:
        if output_state.shape != final_state.shape:
            raise ValueError(
                f"output_state has shape {tuple(output_state.shape)}, expected {tuple(final_state.shape)}.")
        output_state.copy_(final_state)
        final_state = output_state
    return result, final_state


__all__ = ["chunk_gated_delta_rule"]
