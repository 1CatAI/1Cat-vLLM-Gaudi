# SPDX-License-Identifier: Apache-2.0
"""Build changing C1 metadata without scalar Torch indexing loops."""
import numpy as np
import torch


def build_q1_metadata(builder, position, seq_len, block_table_cpu):
    if position != seq_len - 1:
        raise ValueError(f"DeepSeek V4 q1 position/sequence mismatch: position={position}, seq_len={seq_len}")
    name = type(builder).__name__
    if name not in ("DeepseekSparseSWAMetadataBuilder", "DeepseekV4IndexerMetadataBuilder",
                    "DeepseekV4HWAgnosticMetadataBuilder"):
        return {}
    if block_table_cpu.device.type != "cpu" or block_table_cpu.dtype != torch.int32:
        raise ValueError("Native V4 metadata requires a CPU int32 block table")
    table = block_table_cpu.numpy()[0]

    def slots(logical, block_size):
        blocks, offsets = np.divmod(np.asarray(logical, dtype=np.int64), block_size)
        if np.any(blocks < 0) or np.any(blocks >= len(table)):
            raise ValueError(f"DeepSeek V4 q1 metadata exceeds block table width: builder={name}")
        physical = table[blocks]
        return np.where(physical >= 0, physical.astype(np.int64) * block_size + offsets, -1).astype(np.int32)

    if name == "DeepseekSparseSWAMetadataBuilder":
        width = int(builder.window_size)
        start = max(position - width + 1, 0)
        length = position - start + 1
        indices = np.full((1, 1, width), -1, dtype=np.int32)
        indices[0, 0, :length] = slots(np.arange(start, position + 1), int(builder.block_size))
        return {"decode_swa_indices": torch.from_numpy(indices),
                "decode_swa_lens": torch.tensor([length], dtype=torch.int32)}
    ratio = int(builder.compress_ratio)
    if ratio <= 1:
        return {}
    compressed = -1
    if (position + 1) % ratio == 0:
        compressed = int(slots(position // ratio, int(builder.kv_cache_spec.num_states)))
    result = {"compressed_slot_mapping": torch.tensor([compressed], dtype=torch.int32)}
    if name == "DeepseekV4IndexerMetadataBuilder":
        result["compressed_seq_lens"] = torch.tensor([[seq_len // ratio]], dtype=torch.int32)
    elif ratio == 128:
        width = int(builder.c128a_max_compressed)
        count = min(seq_len // ratio, width)
        indices = np.full((1, 1, width), -1, dtype=np.int32)
        indices[0, 0, :count] = slots(np.arange(count), int(builder.kv_cache_spec.block_size) // ratio)
        result["c128a_global_decode_topk_indices"] = torch.from_numpy(indices)
        result["c128a_decode_topk_lens"] = torch.tensor([count], dtype=torch.int32)
    return result
