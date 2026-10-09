# SPDX-License-Identifier: Apache-2.0
"""Exact bounded N128-to-N256 startup preparation on the device.

All layout and exponent decisions use integer arithmetic. The serving loader
does not select this prototype until its complete upload/consumer gate passes.
"""
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import torch


def signed_words(value):
    return (((value + 32768) & 65535) - 32768).to(torch.int16)


def pack_bytes(value):
    return signed_words(value[..., 0::2] | (value[..., 1::2] << 8))


def prepare_expert_device(q16, s16_bits, *, compact_scales, active_k):
    """Return packed weights, scale bits, channel bits and per-expert checks."""
    batch, blocks, stream = q16.shape
    k = stream // 32
    groups = k // 32
    if (q16.dtype != torch.int16 or s16_bits.dtype != torch.int16 or blocks % 2 or stream % 4096
            or s16_bits.shape != (batch, blocks, stream // 8) or not 0 < active_k <= k or active_k % 32):
        raise ValueError("Device startup preparation requires the bounded N128 source contract")
    words = q16.to(torch.int32)
    source = words.reshape(batch, blocks, groups, 16, 64)
    a = torch.maximum(source & 7, (source >> 4) & 7).amax(3)
    b = torch.maximum((source >> 8) & 7, (source >> 12) & 7).amax(3)
    maximum = torch.stack((a, b), dim=-1).reshape(batch, blocks, groups, 128)
    scales = s16_bits.to(torch.int32).reshape(batch, blocks, groups, 128)
    codes = (scales >> 7) & 255
    offset = torch.where(maximum == 1, -8, torch.where(maximum <= 3, -7, torch.where(maximum <= 5, -6, -5)))
    nonzero = maximum != 0
    candidate = torch.where(nonzero, codes - 127 + offset, -32000)
    exponent = candidate.amax(2)
    exponent = torch.where(exponent == -32000, 0, exponent)
    channel_codes = exponent + 127
    channels = (channel_codes << 7).to(torch.int16).reshape(batch, blocks // 2, 256)
    delta = codes - channel_codes.unsqueeze(2)
    valid = (((scales & 32895) == 0) & (codes >= 2) & (codes <= 254)).flatten(1).all(1)
    valid = valid & ((exponent >= -126) & (exponent <= 127)).flatten(1).all(1)
    valid = valid & ((delta >= -5) | ~nonzero).flatten(1).all(1)
    active_delta = delta[:, :, :active_k // 32]
    eligible = ((active_delta >= -5) & (active_delta <= 6)).flatten(1).all(1)
    if active_k < k:
        eligible = eligible & (q16[:, :, active_k * 32:] == 0).flatten(1).all(1)

    rows = words.reshape(batch, blocks, k // 2, 64)
    low = (rows & 15) | ((rows >> 4) & 240)
    high = ((rows >> 4) & 15) | ((rows >> 8) & 240)
    low = pack_bytes(low).reshape(batch, blocks // 2, 2, k // 2, 32).permute(0, 1, 3, 2, 4)
    high = pack_bytes(high).reshape(batch, blocks // 2, 2, k // 2, 32).permute(0, 1, 3, 2, 4)
    packed = torch.stack((low, high), dim=3).reshape(batch, blocks // 2, stream * 2)
    original = codes.reshape(batch, blocks // 2, 2, groups, 128).permute(0, 1, 3, 2, 4)
    original = original.reshape(batch, blocks // 2, groups, 256)
    final_codes = channel_codes.reshape(batch, blocks // 2, 1, 256)
    if compact_scales:
        planes = pack_bytes(torch.cat((original, final_codes), dim=2))
        planes = planes.reshape(batch, blocks // 2, k * 4 + 128)
    else:
        differences = ((original - final_codes) * 8) & 255
        planes = pack_bytes(torch.cat((original, differences), dim=-1))
        planes = planes.reshape(batch, blocks // 2, k * 8)
    checks = torch.stack((valid, eligible), dim=-1).to(torch.int32)
    return packed, planes, channels, checks


@lru_cache(maxsize=8)
def compiled_preparation(compact_scales, active_k):
    """Share the preparation graph across layers, without retaining weight data."""

    def convert(q, s):
        return prepare_expert_device(q, s, compact_scales=compact_scales, active_k=active_k)

    return torch.compile(convert, backend="hpu_backend", fullgraph=True, dynamic=False)


def read_source_batch(source, first, last):
    """Read contiguous compressed experts directly into one bounded buffer."""
    import numpy as np

    if not 0 <= first < last <= source.shape[0] or last - first > 16:
        raise ValueError("Device weight staging requires a bounded expert range")
    count = source.nbytes // source.shape[0]
    if count * (last - first) > 128 << 20:
        raise ValueError("Device weight source batch exceeds staging budget")
    result = np.empty((last - first, *source.shape[1:]), dtype="<i2" if source.dtype == "I16" else "<u2")
    with source.file.open("rb") as stream:
        stream.seek(source.offset + first * count)
        if stream.readinto(memoryview(result).cast("B")) != result.nbytes:
            raise ValueError("Truncated prepared device weight source")
    return result


def staged_source_batches(shard, source_q, source_s, batch):
    """One reader prepares the next compressed batch; it never calls HPU APIs."""
    experts = source_q.shape[0]

    def read(first):
        last = min(first + batch, experts)
        shard.check_identity()
        q = read_source_batch(source_q, first, last)
        s = read_source_batch(source_s, first, last)
        shard.check_identity()
        return first, last, q, s

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(read, 0)
        for first in range(0, experts, batch):
            current = future.result()
            if first + batch < experts:
                future = pool.submit(read, first + batch)
            yield current
            del current


def fill_projection_device(shard, source_q, source_s, q, planes, channel, *, compact_scales, active_k):
    """Fill the sole resident allocation with bounded device preparation batches.

    Read only compressed source bytes on the host. Certificates are accumulated
    on the device and read once after the complete projection, before publication.
    No expanded or converted weight copy survives this function.
    """
    experts = source_q.shape[0]
    source_bytes = (source_q.nbytes + source_s.nbytes) // experts
    # Conservative simultaneous-lifetime allowance for the integer conversion
    # graph. Source staging stays <=128 MiB; transient device storage <=2 GiB.
    batch = min(16, max(1, (64 << 20) // source_bytes), max(1, (2 << 30) // (24 * source_bytes)))
    if source_bytes > 64 << 20:
        raise ValueError("Device expert source exceeds double-buffer staging budget")
    if 24 * source_bytes > 2 << 30:
        raise ValueError("Device expert preparation exceeds its temporary budget")
    convert = compiled_preparation(compact_scales, active_k)
    certificates = torch.empty((experts, 2), dtype=torch.int32, device=q.device)
    for first, last, raw_q, raw_s in staged_source_batches(shard, source_q, source_s, batch):
        prepared = convert(torch.from_numpy(raw_q).to(q.device), torch.from_numpy(raw_s.view("<i2")).to(q.device))
        q[first:last].copy_(prepared[0])
        planes[first:last].copy_(prepared[1])
        channel[first:last].copy_(prepared[2].view(torch.bfloat16))
        certificates[first:last].copy_(prepared[3])
        # Uploads are synchronous; do not retain the caller's previous source
        # arrays when the reader advances its two-buffer window.
        del raw_q, raw_s
    valid, eligible = certificates.bool().all(0).cpu().tolist()
    if not valid:
        raise ValueError("Device preparation rejected the source FP4/scale qualification")
    shard.check_identity()
    return bool(eligible)
