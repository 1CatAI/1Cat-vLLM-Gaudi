# SPDX-License-Identifier: Apache-2.0
"""CPU mHC equation oracle and upstream normalized squared-error contract.

Equations: DeepSeek-V4.1-Flash dba1be0 inference/model.py hc_mixes/hc_post
and inference/kernel.py hc_split_sinkhorn_kernel. The caller supplies the
residual at the delayed-control boundary; no model scheduling is changed.
Metric/limits: DeepGEMM 057ca596 tests/test_mega_mhc.py check_correctness,
deep_gemm/testing/numeric.py calc_diff (5e-5 BF16/FP32; 2e-4 FP8).
This runs the mathematical reference on CPU, not the upstream CUDA kernel.
"""
import torch


def normalized_error(actual, reference):
    a, b = actual.double(), reference.double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float('inf')
    denominator = (a.square() + b.square()).sum()
    # Algebraically identical to 1 - 2<a,b>/(||a||²+||b||²), without cancellation.
    return float((a-b).square().sum() / denominator) if denominator else 0.0


def routing_error_by_expert(reference_ids, reference_weights, actual_ids, actual_weights):
    """Compare the same selected experts without treating a permutation as error.

    A changed expert set or duplicate selected ID remains a failed contract.
    This aligns only the diagnostic weights; it never changes model routing.
    """
    if (reference_ids.ndim != 2 or not reference_ids.numel() or reference_ids.shape != actual_ids.shape
            or reference_weights.shape != reference_ids.shape or actual_weights.shape != actual_ids.shape):
        raise ValueError('Routing IDs and weights must have matching [tokens, experts] shapes')
    if reference_ids.dtype not in (torch.int32, torch.int64) or actual_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Routing IDs must be integer tensors')
    left, right = reference_ids.cpu().tolist(), actual_ids.cpu().tolist()
    same_set = all(len(set(a)) == len(a) and len(set(b)) == len(b) and sorted(a) == sorted(b)
                   for a, b in zip(left, right, strict=True))
    if not same_set:
        return {'same_expert_set': False, 'order_equal': left == right, 'error': float('inf')}
    indices = torch.tensor([[b.index(expert) for expert in a] for a, b in zip(left, right, strict=True)],
                           dtype=torch.int64, device=actual_weights.device)
    return {'same_expert_set': True, 'order_equal': left == right,
            'error': normalized_error(actual_weights.gather(1, indices), reference_weights)}


def gate_reference(residual, weight, scale, base, eps):
    flat = residual.flatten(1).float()
    mixes = (flat @ weight.float().T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = (mixes[:, 8:].reshape(-1, 4, 4) * scale[2] + base[8:].reshape(4, 4)).softmax(-1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return torch.cat((pre, post, comb.flatten(1)), -1)


def post_reference(peers, residual, weight, scale, base, norm, eps):
    gates = gate_reference(residual, weight, scale, base, eps)
    summed = peers[0].float()
    for peer in peers[1:]:
        summed = summed + peer.float()
    value = summed.bfloat16().float()
    pre, post, comb = gates[:, :4], gates[:, 4:8], gates[:, 8:].reshape(-1, 4, 4)
    updated = (value.unsqueeze(1) * post.unsqueeze(-1)
               + (comb.unsqueeze(-1) * residual.float().unsqueeze(2)).sum(1)).bfloat16()
    collapsed = (updated.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
    x = collapsed.float()
    normalized = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm.float()).bfloat16()
    return {0: updated, 1: collapsed, 2: pre, 3: normalized, 8: gates}


def check_outputs(observed, reference):
    return {str(index): dict(error=normalized_error(observed[index], expected), limit=5e-5,
                            max_abs=float((observed[index].float()-expected.float()).abs().max()))
            for index, expected in reference.items()}
