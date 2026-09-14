# SPDX-License-Identifier: Apache-2.0
"""Reversible N256 weight layout with bounded, load-time FP8 exponent preparation.

Q16 stores four adjacent N codes at the same K. Each group of the I16 scale
tensor contains 256 original E8M0 bytes followed by 256 FP8 exponent offsets.
Original scales are retained for BF16 prefill on the same compressed weights.
"""

import numpy as np

from vllm_gaudi.ops.deepseek_v41_fp8 import channel_scales, read_expert
from vllm_gaudi.ops.deepseek_v41_weights import canonical_hash

LAYOUT = {
    "version": 3,
    "q16": "n256-k-major-adjacent-n-nibbles",
    "s16": "group32-original-u8-plane-relative-fp8-exponent-u8-plane",
    "decode_partition": [256, 128],
    "format": "gaudi2-e4m3-bias7-normal-results",
    "scale": "minimum-covering-channel-power-of-two",
}
FINGERPRINT = canonical_hash(LAYOUT)


def prepare_expert(q16, s16):
    """Prepare one expert; preserve its original nibble and scale encodings."""
    if np.any(s16 & 127):
        raise ValueError("Prepared scales must retain exact E8M0 encodings")
    channels, record = channel_scales(q16, s16)
    blocks, stream = q16.shape
    if blocks % 2:
        raise ValueError("N256 expert output width must be divisible by 256")
    k = stream // 32
    n = blocks * 128
    raw = q16.view(np.uint8).reshape(blocks, k // 2, 128)
    packed = np.empty((blocks // 2, k, 128), dtype=np.uint8)
    for half in range(2):
        rows = raw[half::2]
        low = rows[..., 0::2]
        high = rows[..., 1::2]
        packed[:, 0::2, half * 64:(half + 1) * 64] = (low & 15) | ((high & 15) << 4)
        packed[:, 1::2, half * 64:(half + 1) * 64] = (low >> 4) | (high & 240)
    q = packed.view("<i2").reshape(n // 256, k * 64)
    codes = (s16.reshape(blocks, k // 32, 128) >> 7).astype(np.uint8)
    original = codes.reshape(blocks // 2, 2, k // 32, 128).transpose(0, 2, 1, 3).reshape(n // 256, k // 32, 256)
    channel = channels.reshape(n // 256, 256)
    channel_code = (channel >> 7).astype(np.int16)
    offsets = ((original.astype(np.int16) - channel_code[:, None, :]) * 8).astype(np.uint8)
    planes = np.concatenate((original, offsets), axis=-1).view("<i2").reshape(n // 256, k * 8)
    # Covers source, outputs, range scan and layout temporaries simultaneously.
    record["temporary_upper_bound_bytes"] += 8 * (q16.nbytes + s16.nbytes)
    if record["temporary_upper_bound_bytes"] > 2 * 2**30:
        raise ValueError("N256 expert preparation exceeds 2 GiB temporary budget")
    return q, planes, channel, record


def restore_expert(q, planes):
    """Recover the exact N128 Q16/S16 bytes for compatibility validation."""
    if q.dtype != np.dtype("<i2") or planes.dtype != np.dtype("<i2") or q.ndim != 2:
        raise ValueError("N256 storage must use rank-two I16 expert tensors")
    blocks, stream = q.shape
    k = stream // 64
    if stream % 8192 or planes.shape != (blocks, k * 8):
        raise ValueError("Invalid N256/K128 dimensions")
    packed = q.view(np.uint8).reshape(blocks, k, 128)
    raw = np.empty((blocks * 2, k // 2, 128), dtype=np.uint8)
    for half in range(2):
        even = packed[:, 0::2, half * 64:(half + 1) * 64]
        odd = packed[:, 1::2, half * 64:(half + 1) * 64]
        raw[half::2, :, 0::2] = (even & 15) | ((odd & 15) << 4)
        raw[half::2, :, 1::2] = (even >> 4) | (odd & 240)
    original = planes.view(np.uint8).reshape(blocks, k // 32, 512)[..., :256]
    codes = original.reshape(blocks, k // 32, 2, 128).transpose(0, 2, 1, 3).reshape(blocks * 2, k // 32 * 128)
    return raw.view("<i2").reshape(blocks * 2, k * 32), (codes.astype("<u2") << 7)


def load_projection(shard, prefix, device):
    """Load directly into the sole resident Q16/scale allocation, one expert at a time."""
    import torch
    source_q = shard.catalog[prefix + "_q16"]
    source_s = shard.catalog[prefix + "_s16"]
    experts, blocks, stream = source_q.shape
    if blocks % 2:
        raise ValueError("Prepared shard cannot be represented by the N256 expert layout")
    q = torch.empty((experts, blocks // 2, stream * 2), dtype=torch.int16, device=device)
    p = torch.empty((experts, blocks // 2, source_s.shape[2] * 2), dtype=torch.int16, device=device)
    channel = torch.empty((experts, blocks // 2, 256), dtype=torch.bfloat16, device=device)
    # Batch host-to-device copies without keeping another resident weight copy.
    # The staging allocation is at most 128 MiB, plus one bounded expert scan.
    per_expert = (q[0].numel() + p[0].numel() + channel[0].numel()) * 2
    batch = min(16, max(1, 128 * 2**20 // per_expert))
    for first in range(0, experts, batch):
        last = min(first + batch, experts)
        cpu_q = np.empty((last - first, *q.shape[1:]), dtype="<i2")
        cpu_p = np.empty((last - first, *p.shape[1:]), dtype="<i2")
        cpu_c = np.empty((last - first, *channel.shape[1:]), dtype="<u2")
        for expert in range(first, last):
            shard.check_identity()
            new_q, new_p, new_c, _ = prepare_expert(read_expert(source_q, expert), read_expert(source_s, expert))
            cpu_q[expert - first] = new_q
            cpu_p[expert - first] = new_p
            cpu_c[expert - first] = new_c
        q[first:last].copy_(torch.from_numpy(cpu_q))
        p[first:last].copy_(torch.from_numpy(cpu_p))
        channel[first:last].copy_(torch.from_numpy(cpu_c.view("<i2")).view(torch.bfloat16))
    shard.check_identity()
    return q, p, channel
