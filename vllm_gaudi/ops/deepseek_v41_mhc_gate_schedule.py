# SPDX-License-Identifier: Apache-2.0
"""Expose mHC gates independently of the arriving TP partials."""
import torch

from vllm_gaudi import envs as gaudi_envs


def native_gates(mixes, rrms, scale, base, *, prefill=False):
    op = (torch.ops.custom_op.custom_deepseek_v41_mhc_gates_positive_gaudi2
          if not prefill and gaudi_envs.VLLM_HPU_DSV41_MHC_POSITIVE_GATES else
          torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2)
    return op(mixes, rrms, scale, base)


def communication_gates_post(value, residual, control, scale, base, epsilon):
    # The control producer owns both the 24 projections and the shared RRMS.
    # This branch reads neither value nor residual, so the native tensor
    # dependency plan can execute it while the exchange remains in flight.
    gates = native_gates(
        control[:, :24].contiguous(), control[:, 24:25].contiguous(), scale, base
    )
    pre = gates[:, :4].contiguous()
    if gaudi_envs.VLLM_HPU_DSV41_MHC_GATE_PACKET:
        updated, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_gates_post_gaudi2(
            value, residual, gates
        )
    else:
        updated, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
            value, residual, gates[:, 4:8].contiguous(),
            gates[:, 8:].reshape(residual.shape[0], 4, 4).contiguous(), pre
        )
    return updated, collapsed, gates


def communication_gates_post_quant(value, residual, control, scale, base, norm_weight, epsilon):
    """Keep independent gates while publishing the next FFN's two quantizers."""
    gates = native_gates(
        control[:, :24].contiguous(), control[:, 24:25].contiguous(), scale, base
    )
    updated, collapsed, normalized, quantized, activation_scale, shared_q, shared_scale = (
        torch.ops.custom_op.custom_deepseek_v41_mhc_post_norm_statistics_gaudi2(
            value, residual, gates, norm_weight, epsilon
        )
    )
    return updated, collapsed, gates, normalized, quantized, activation_scale, shared_q, shared_scale


def communication_gates_post_quant_memory_ready(value, residual, control, scale, base, norm_weight,
                                               epsilon, flags, line, enabled):
    """Experimental acquiring reader with unchanged weighted post arithmetic."""
    gates = native_gates(control[:, :24].contiguous(), control[:, 24:25].contiguous(), scale, base)
    updated, collapsed, normalized, quantized, activation_scale, shared_q, shared_scale, status = (
        torch.ops.custom_op.private_memory_ready_post(value, residual, gates, norm_weight, flags, epsilon, line, enabled)
    )
    return updated, collapsed, gates, normalized, quantized, activation_scale, shared_q, shared_scale, status
