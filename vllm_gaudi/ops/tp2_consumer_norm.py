# SPDX-License-Identifier: Apache-2.0
"""The unchanged local BF16 math after a dedicated TP2 peer exchange."""


def tp2_consumer_norm(partial, peer, residual, weight, epsilon, rms_norm):
    # Both adds retain a BF16 result. Reassociating these expressions changes
    # rounding before normalization, even though addition is commutative.
    reduced = partial + peer
    residual_out = reduced + residual.reshape(partial.shape)
    flat = residual_out.reshape(-1, weight.numel()).unsqueeze(0)
    normalized = rms_norm(flat, weight, epsilon).reshape(partial.shape)
    return normalized, residual_out
