# SPDX-License-Identifier: Apache-2.0
"""Reversible N256 weight layout with bounded, load-time FP8 exponent preparation.

Q16 stores four adjacent N codes at the same K. Legacy scale groups retain
original E8M0 bytes and FP8 offsets. The compact TP4 layout retains original
bytes plus one channel-exponent row, reconstructing the same offsets on chip.
Both formats retain the original BF16 prefill scales exactly.
"""

import numpy as np
from functools import lru_cache
from types import FunctionType

from vllm_gaudi.ops.deepseek_v41_fp8 import channel_scales
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
COMPACT_LAYOUT = dict(LAYOUT, version=4, s16="group32-original-u8-plus-one-channel-exponent-u8-row")
COMPACT_FINGERPRINT = canonical_hash(COMPACT_LAYOUT)


def _fused_decode(value, ids, routing, q13, q2, s13, s2, lookup, channel13, channel2, normal, direct_finalize):
    import torch

    namespace = torch.ops.custom_op
    operation = (
        namespace.custom_deepseek_v41_expert_n256_moe_direct_finalize_prefetch_w2_fp8_gaudi2
        if direct_finalize and value.shape[0] == 1
        else namespace.custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2
    )
    return operation(value, ids, routing, q13, q2, s13, s2, lookup, channel13, channel2, normal)


@lru_cache(maxsize=32)
def _compiled_fused_decode(signature):
    import torch

    entry = FunctionType(
        _fused_decode.__code__.replace(co_name=f"n256_fused_decode_{signature}"), _fused_decode.__globals__
    )
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def run_fused_decode(value, ids, routing, q13, q2, s13, s2, lookup, channel13, channel2, normal, direct_finalize=True):
    """Reuse one compiled N256 body across layer weights and TP4 requests."""
    import torch

    if not 1 <= value.shape[0] <= 6:
        raise ValueError("Fused N256 decode requires a C1-C6 bucket")
    arguments = (value, ids, routing, q13, q2, s13, s2, lookup, channel13, channel2, normal, direct_finalize)
    if torch.compiler.is_compiling():
        return _fused_decode(*arguments)
    signature = (value.shape[0], tuple(q13.shape), tuple(q2.shape), normal, direct_finalize)
    return _compiled_fused_decode(signature)(*arguments)


def saturated_decode_eligible(planes, *, active_k=None):
    """Reuse the SAT offset gate for full and compact N256 scale planes.

    A caller may exclude a padded K suffix only after proving its FP4 codes
    are zero. Runtime SAT maps both FP8 zero encodings to positive zero.
    """
    if planes.dtype != np.dtype("<i2") or planes.ndim not in (2, 3):
        raise ValueError("Expected N256 I16 scale planes")
    words = planes.shape[-1]
    if words % 512 == 128:
        groups = (words - 128) // 128
        values = planes.view(np.uint8).reshape(-1, groups + 1, 256)
        offsets = (values[:, :-1].astype(np.int16) - values[:, -1:, :].astype(np.int16)) * 8
    elif words % 1024 == 0:
        groups = words // 256
        offsets = planes.view(np.uint8).reshape(-1, groups, 512)[:, :, 256:].view(np.int8)
    else:
        raise ValueError("Expected complete K128 full or compact planes")
    if active_k is not None:
        if active_k <= 0 or active_k % 32 or active_k > groups * 32:
            raise ValueError("Invalid active K prefix")
        offsets = offsets[:, :active_k // 32]
    return bool(np.all((offsets >= -40) & (offsets <= 48) & ((offsets & 7) == 0)))



def prepare_expert(q16, s16, *, compact_scales=False, pack_q16=None):
    """Prepare one expert; preserve its original nibble and scale encodings."""
    if np.any(s16 & 127):
        raise ValueError("Prepared scales must retain exact E8M0 encodings")
    channels, record = channel_scales(q16, s16)
    blocks, stream = q16.shape
    if blocks % 2:
        raise ValueError("N256 expert output width must be divisible by 256")
    k = stream // 32
    n = blocks * 128
    if pack_q16 is None:
        raw = q16.view(np.uint8).reshape(blocks, k // 2, 128)
        packed = np.empty((blocks // 2, k, 128), dtype=np.uint8)
        for half in range(2):
            rows = raw[half::2]
            low = rows[..., 0::2]
            high = rows[..., 1::2]
            packed[:, 0::2, half * 64 : (half + 1) * 64] = (low & 15) | ((high & 15) << 4)
            packed[:, 1::2, half * 64 : (half + 1) * 64] = (low >> 4) | (high & 240)
        q = packed.view("<i2").reshape(n // 256, k * 64)
    else:
        q = pack_q16(q16)
        if q.dtype != np.dtype("<i2") or q.shape != (n // 256, k * 64) or not q.flags.c_contiguous:
            raise ValueError("CPU packing changed the resident N256 tensor contract")
    codes = (s16.reshape(blocks, k // 32, 128) >> 7).astype(np.uint8)
    original = codes.reshape(blocks // 2, 2, k // 32, 128).transpose(0, 2, 1, 3).reshape(n // 256, k // 32, 256)
    channel = channels.reshape(n // 256, 256)
    channel_code = (channel >> 7).astype(np.int16)
    if compact_scales:
        # Retain every original E8M0 byte, including zero-weight groups. The
        # decoder reconstructs the exact modulo-256 FP8 exponent difference.
        planes = np.concatenate((original, channel_code[:, None, :].astype(np.uint8)), axis=1)
        planes = planes.view("<i2").reshape(n // 256, k * 4 + 128)
    else:
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
    compact = planes.shape == (blocks, k * 4 + 128)
    if stream % 8192 or (not compact and planes.shape != (blocks, k * 8)):
        raise ValueError("Invalid N256/K128 dimensions")
    packed = q.view(np.uint8).reshape(blocks, k, 128)
    raw = np.empty((blocks * 2, k // 2, 128), dtype=np.uint8)
    for half in range(2):
        even = packed[:, 0::2, half * 64 : (half + 1) * 64]
        odd = packed[:, 1::2, half * 64 : (half + 1) * 64]
        raw[half::2, :, 0::2] = (even & 15) | ((odd & 15) << 4)
        raw[half::2, :, 1::2] = (even >> 4) | (odd & 240)
    original = (
        planes.view(np.uint8).reshape(blocks, k // 32 + 1, 256)[:, :-1]
        if compact
        else planes.view(np.uint8).reshape(blocks, k // 32, 512)[..., :256]
    )
    codes = original.reshape(blocks, k // 32, 2, 128).transpose(0, 2, 1, 3).reshape(blocks * 2, k // 32 * 128)
    return raw.view("<i2").reshape(blocks * 2, k * 32), (codes.astype("<u2") << 7)


def load_projection(shard, prefix, device):
    """Load directly into the sole resident Q16/scale allocation, one expert at a time."""
    import os
    import torch

    prepared = os.environ.get("VLLM_HPU_DSV41_N256_PREPARED_DIR")
    if prepared and prefix.startswith("layers."):
        from vllm_gaudi.ops.deepseek_v41_n256_shards import N256PreparedShard

        if not hasattr(shard, "_n256_runtime_shard"):
            shard._n256_runtime_shard = N256PreparedShard(prepared, shard)
        shard.check_identity()
        return shard._n256_runtime_shard.projection(prefix, device)
    source_q = shard.catalog[prefix + "_q16"]
    source_s = shard.catalog[prefix + "_s16"]
    experts, blocks, stream = source_q.shape
    compact_scales = getattr(shard, "manifest", {}).get("tensor_parallel_size") == 4
    if blocks % 2:
        raise ValueError("Prepared shard cannot be represented by the N256 expert layout")
    q = torch.empty((experts, blocks // 2, stream * 2), dtype=torch.int16, device=device)
    scale_words = stream // 8 + 128 if compact_scales else source_s.shape[2] * 2
    p = torch.empty((experts, blocks // 2, scale_words), dtype=torch.int16, device=device)
    channel = torch.empty((experts, blocks // 2, 256), dtype=torch.bfloat16, device=device)
    # Batch host-to-device copies without keeping another resident weight copy.
    # The staging allocation is at most 128 MiB, plus one bounded expert scan.
    per_expert = (q[0].numel() + p[0].numel() + channel[0].numel()) * 2
    batch = min(16, max(1, 128 * 2**20 // per_expert))
    import json
    config = json.loads((shard.directory / "config.json").read_text())["text_config"]
    active_k = config["moe_intermediate_size"] // shard.tensor_parallel_size if prefix.endswith(".w2") else stream // 32
    sat_eligible = True
    if os.environ.get("VLLM_HPU_DSV41_N256_DEVICE_PREPARE", "0") == "1":
        if q.device.type != "hpu":
            raise ValueError("Device expert preparation requires an HPU allocation")
        from vllm_gaudi.ops.deepseek_v41_device_prepare import fill_projection_device

        sat_eligible = fill_projection_device(
            shard, source_q, source_s, q, p, channel, compact_scales=compact_scales, active_k=active_k)
        q.dsv41_sat_eligible = sat_eligible
        if prefix.endswith(".w2") and sat_eligible:
            q.dsv41_active_k = int(active_k)
        return q, p, channel
    from vllm_gaudi.ops.deepseek_v41_expert_load import prepare_expert_batch

    load_workers = int(os.environ.get("VLLM_HPU_DSV41_N256_LOAD_WORKERS", "1"))
    pack_q16 = None
    if pack_library := os.environ.get("VLLM_HPU_DSV41_N256_PACK_LIBRARY"):
        from vllm_gaudi.ops.deepseek_v41_startup_pack import native_q16_packer

        pack_q16 = native_q16_packer(pack_library)
    for first in range(0, experts, batch):
        last = min(first + batch, experts)
        cpu_q = np.empty((last - first, *q.shape[1:]), dtype="<i2")
        cpu_p = np.empty((last - first, *p.shape[1:]), dtype="<i2")
        cpu_c = np.empty((last - first, *channel.shape[1:]), dtype="<u2")
        for expert, new_q, new_p, new_c, eligible in prepare_expert_batch(
            shard, source_q, source_s, first, last,
            compact_scales=compact_scales, active_k=active_k, workers=load_workers, pack_q16=pack_q16,
        ):
            sat_eligible &= eligible
            cpu_q[expert - first] = new_q
            cpu_p[expert - first] = new_p
            cpu_c[expert - first] = new_c
        q[first:last].copy_(torch.from_numpy(cpu_q))
        p[first:last].copy_(torch.from_numpy(cpu_p))
        channel[first:last].copy_(torch.from_numpy(cpu_c.view("<i2")).view(torch.bfloat16))
    shard.check_identity()
    q.dsv41_sat_eligible = bool(sat_eligible)
    if prefix.endswith(".w2") and sat_eligible:
        q.dsv41_active_k = int(active_k)
    return q, p, channel
