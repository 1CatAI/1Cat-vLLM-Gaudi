# SPDX-License-Identifier: Apache-2.0
"""Verify the visible-prefix contract used by bounded C1 attention."""

import torch
from torch import nn

from vllm_gaudi.ops.deepseek_v41_attention import CSA2Attention, CSA2SharedState


def test_full_reindex_reuse_publish_a_visible_prefix_at_compression_boundaries():
    config = {
        "index_topk": 512,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
    }
    stages = [CSA2SharedState(config, start, start + 20, "cpu") for start in (0, 20)]
    layers = []
    for layer_id in range(2, 40):
        layer = CSA2Attention.__new__(CSA2Attention)
        nn.Module.__init__(layer)
        layer.layer = layer_id
        layer.ratio = config["compress_ratios"][layer_id]
        layer.shared = stages[layer_id // 20]
        source = max(value for value in config["index_source_layer_ids"] if value <= layer_id)
        layer.selection = layer.shared.topk[str(source)]
        layer.owns_index = layer_id in config["index_source_layer_ids"]
        layer.candidate_source = 20
        layer.compressed_offsets = torch.arange(512, dtype=torch.int32)
        layer.candidate_offsets = torch.arange(2048 * 8, dtype=torch.int32)
        layers.append(layer)
    # Include incomplete groups, the sliding-window boundary, capacity and a
    # reused position. A stale row must not retain the preceding larger prefix.
    for position in (0, 1, 5, 127, 128, 255, 256, 510, 511, 0):
        for layer in layers:
            selected = layer._select(torch.tensor([position], dtype=torch.int32))
            visible = (position + 1) // layer.ratio
            assert selected[0, :visible].tolist() == list(range(visible))
            assert (selected[0, visible:] == -1).all()
