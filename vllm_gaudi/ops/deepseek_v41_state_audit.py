# SPDX-License-Identifier: Apache-2.0
"""Read-only snapshots of the first native decode handoff for diagnosis."""
import hashlib
import json
from pathlib import Path

import torch


def save_single_handoff(bank, slot, position, request_id, directory, *, engram_history=None):
    """Persist the bounded, reconstructed state before native B1 consumes it.

    This runs only when explicitly enabled for diagnosis. Its host copies and
    file writes are outside a steady decode timing window. Decoded mirrors are
    rebuilt from canonical packed pages by bind_single immediately beforehand.
    No additional device graph or state mutation is introduced by the audit.
    """
    if bank.single_owner != slot or bank.pending is not None:
        raise RuntimeError("State audit requires the completed native B1 handoff")
    program, shared = bank.program, bank.program.shared
    tensors = {}

    def save(name, value):
        tensors[name] = value.detach().to(device="cpu", copy=True)

    save("block_table", shared.block_table)
    for layer in program.layers:
        attention = layer.attention
        for name in ("swa", "kv_history", "score_history"):
            value = getattr(attention, name, None)
            if value is not None:
                save(f"layer{layer.layer}.{name}", value)
        if hasattr(shared, "decoded_swa"):
            offset = attention.decoded_swa_offset
            save(f"layer{layer.layer}.decoded_swa", shared.decoded_swa[offset:offset + 256])
    mirrors_complete = True
    for source, cache in shared.sources.items():
        count = position // cache.ratio
        for name in ("decoded_main", "decoded_index_hot"):
            value = getattr(cache, name, None)
            if value is None or count > value.shape[0]:
                mirrors_complete = False
                continue
            save(f"source{source}.{name}", value[:count])
    fingerprint = hashlib.sha256(request_id.encode()).hexdigest()[:20]
    path = Path(directory) / f"request-{fingerprint}-pp{program.pp_rank}-tp{program.tp_rank}"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "request_id": request_id,
        "position": position,
        "pp_rank": program.pp_rank,
        "tp_rank": program.tp_rank,
        "slot": slot.index,
        "slot_generation": slot.generation,
        "blocks": list(bank.page_versions[slot.index][1]),
        "mirrors_complete": mirrors_complete,
        "engram_history": engram_history,
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            }
            for name, value in tensors.items()
        },
    }
    torch.save(tensors, path.with_suffix(".pt"))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path
