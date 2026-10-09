# SPDX-License-Identifier: Apache-2.0
"""Immutable draft-only BF16 banks produced by the existing exact decoder."""


def prepare_draft_weight_cache(moe):
    import torch

    if moe.topk != 3 or not moe.normal_scales or moe.n256 or moe.mtp_fp8 or moe.mtp_sat:
        raise ValueError("Draft cache requires the unchanged normal N128 top-three BF16 body")
    if not hasattr(torch.ops.custom_op, "custom_deepseek_v41_mtp_cached_moe_bf16_gaudi2"):
        raise RuntimeError("Draft cache's additive native implementation is unavailable")
    experts = moe.weights.experts
    for projection, name in (("w13", "mtp_cache13"), ("w2", "mtp_cache2")):
        q, scales = getattr(experts, projection + "_q16"), getattr(experts, projection + "_s16")
        if q.ndim != 3 or not 1 <= q.shape[0] <= 128:
            raise ValueError("Only the small draft expert bank can be restored")
        bank = torch.empty((q.shape[0], q.shape[2] // 32, q.shape[1] * 128),
                           device=q.device, dtype=torch.bfloat16)
        # Cold setup only: one bounded decoder temporary, no extra copies in
        # the native hot plan. Original packed weights remain the prefill path.
        for first in range(0, q.shape[0], 8):
            stop = min(first + 8, q.shape[0])
            ids = torch.arange(first, stop, dtype=torch.int32, device=q.device).reshape(-1, 1)
            decoded = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_dequant_bf16_gaudi2(
                ids, q, scales, moe.lookup, True)
            bank[first:stop].copy_(decoded)
            torch.hpu.synchronize()
        setattr(moe, name, bank)
