# SPDX-License-Identifier: Apache-2.0
"""Dependency bound for an experimental V4.1 decoder prefill suffix.

Each decoder layer has a causal local window. The global compressed-KV source
is still produced for every prompt row; this bound applies only after that
source layer and only when no intermediate decoder-row output is requested.
"""


def decoder_halo_rows(decoder_layers: int, window: int = 128, alignment: int = 256) -> int:
    if decoder_layers < 1 or window < 1 or alignment < 1:
        raise ValueError("Decoder halo requires positive depth, window and alignment")
    required = window + (decoder_layers - 1) * (window - 1)
    if decoder_layers == 19 and window == 128:
        # The mathematical local-window bound is 2414 rows. In the current
        # Gaudi TP2 Reindex chain, two-rank real-weight continuation checks
        # first become bitwise exact at 4096 rows. Keep the larger operational
        # bound until the Reindex row dependency is fully characterized.
        required = max(required, 4096)
    return ((required + alignment - 1) // alignment) * alignment


def decoder_halo_mode(start: int, count: int, prompt_tokens: int, *, eligible: bool, block_tokens: int = 8192) -> str:
    """Choose a request phase without consulting device state or tensor data."""
    if not eligible:
        return "full"
    if start < 0 or count < 1 or prompt_tokens < start + count:
        raise ValueError("Decoder halo requires a bounded prompt chunk")
    retained = decoder_halo_rows(19)
    # Only a complete, aligned scheduler transaction can omit decoder rows.
    # The final block may be short. If it contains fewer than the retained
    # rows, execute the preceding full block to rebuild its trailing local
    # caches; all earlier complete blocks can still publish only global KV.
    if block_tokens <= retained or prompt_tokens <= block_tokens or start % block_tokens:
        return "full"
    tail = (prompt_tokens - 1) % block_tokens + 1
    final_start = prompt_tokens - tail
    if start == final_start and count == tail:
        return "final"
    if count != block_tokens or start + count > final_start:
        return "full"
    if tail < retained and start + count == final_start:
        return "full"
    return "prefix_only"
