# SPDX-License-Identifier: Apache-2.0
"""Recover actual Engram consumer rows from checkpoint and fixed C6 history.

Read only the selected checkpoint rows. This setup is outside all device
measurements; it does not replace or qualify the serving Engram producer.
"""
import functools
import json
import os


def read_rows(item, indices):
    import numpy as np

    width = item['row_bytes']
    count = item['row_stop'] - item['row_start']
    if np.any(indices < 0) or np.any(indices >= count):
        raise ValueError('Hash is outside the TP checkpoint shard')
    descriptor = os.open(item['file'], os.O_RDONLY)
    try:
        rows = []
        for index in indices.reshape(-1):
            payload = os.pread(descriptor, width, item['shard_offset'] + int(index) * width)
            if len(payload) != width:
                raise ValueError('Truncated checkpoint Engram row')
            rows.append(np.frombuffer(payload, dtype=np.uint8).copy())
        return np.stack(rows).reshape(*indices.shape, width)
    finally:
        os.close(descriptor)


@functools.lru_cache(maxsize=1)
def hash_metadata(prepared):
    from transformers import AutoTokenizer
    from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout, build_compressed_token_map

    text = json.loads((prepared / 'config.json').read_text())['text_config']
    layout = EngramHashLayout.from_config(text)
    token_map, size = build_compressed_token_map(AutoTokenizer.from_pretrained(prepared, local_files_only=True))
    if size != layout.vocab_size:
        raise ValueError('Checkpoint tokenizer compression mismatch')
    return layout, token_map


def real_engram_rows(prepared, case, rank, tp):
    import numpy as np
    import torch
    from vllm_gaudi.ops.deepseek_v41_engram import EngramTokenHistory

    if not case['history_context_qualified'] or case['tensor_parallel_size'] != tp:
        raise ValueError('Actual request history and TP geometry are required')
    layout, token_map = hash_metadata(prepared)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    tables = json.loads((prepared / manifest['engram_host_shards'][str(rank)]['file']).read_text())['tables']
    history = EngramTokenHistory(layout, token_map)
    history.reset('native-input-gate')
    history.position = int(case['positions'][0])
    history.history = case['cursor_history'].numpy()[::-1].astype(np.int64).copy()
    ids = case['ids'].tolist()
    batch = history.prepare('native-input-gate', ids, image_mask=[value in (129264, 129265) for value in ids])
    output = []
    for i, layer in enumerate(layout.layer_ids):
        shard = layout.head_shard(layer, rank, tp)
        first, stop = shard['head_start'], shard['head_stop']
        indices = batch.hash_ids[:, i, first:stop].astype(np.int64) - shard['row_start']
        weight = read_rows(tables[f'layers.{layer}.engram.embed.weight'], indices)
        scale = read_rows(tables[f'layers.{layer}.engram.embed.scale'], indices)
        values = torch.from_numpy(weight).view(torch.float8_e4m3fn).float()
        scaling = torch.exp2(torch.from_numpy(scale.astype(np.int32)) - 127).repeat_interleave(32, -1)
        output.append((values * scaling).bfloat16())
    return tuple(output)
