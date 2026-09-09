# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v4_native_metadata import build_q1_metadata


def builder(name, **values):
    return type(name, (), values)()


@pytest.mark.parametrize("position", [0, 1, 3, 4, 126, 127, 128, 255, 256, 510, 511])
@pytest.mark.parametrize("blocks", [[3, 5, 1, 7], [7, -1, 3, 0]])
def test_swa_preserves_slot_order_padding_and_missing_pages(position, blocks):
    value = builder("DeepseekSparseSWAMetadataBuilder", window_size=128, block_size=128)
    result = build_q1_metadata(value, position, position + 1, torch.tensor([blocks], dtype=torch.int32))
    logical = range(max(0, position - 127), position + 1)
    expected = [blocks[p // 128] * 128 + p % 128 if blocks[p // 128] >= 0 else -1 for p in logical]
    expected += [-1] * (128 - len(expected))
    assert result["decode_swa_indices"].tolist() == [[expected]]
    assert result["decode_swa_lens"].tolist() == [min(position + 1, 128)]
    assert all(t.dtype == torch.int32 and t.is_contiguous() for t in result.values())


def test_c4_completion_and_compressed_length():
    value = builder("DeepseekV4IndexerMetadataBuilder", compress_ratio=4, kv_cache_spec=SimpleNamespace(num_states=2))
    table = torch.tensor([[3, 5]], dtype=torch.int32)
    assert {k: v.tolist() for k, v in build_q1_metadata(value, 7, 8, table).items()} == {
        "compressed_slot_mapping": [7], "compressed_seq_lens": [[2]]}
    assert build_q1_metadata(value, 8, 9, table)["compressed_slot_mapping"].tolist() == [-1]


@pytest.mark.parametrize("position,expected,slot", [
    (126, [-1, -1, -1, -1], -1), (127, [6, -1, -1, -1], 6),
    (255, [6, 7, -1, -1], 7), (511, [6, 7, 10, 11], 11)])
def test_c128_boundaries(position, expected, slot):
    value = builder("DeepseekV4HWAgnosticMetadataBuilder", compress_ratio=128, c128a_max_compressed=4,
                    kv_cache_spec=SimpleNamespace(num_states=2, block_size=256))
    result = build_q1_metadata(value, position, position + 1, torch.tensor([[3, 5]], dtype=torch.int32))
    assert result["compressed_slot_mapping"].tolist() == [slot]
    assert result["c128a_global_decode_topk_indices"].tolist() == [[expected]]
    assert result["c128a_decode_topk_lens"].tolist() == [(position + 1) // 128]


def test_invalid_metadata_fails_before_transfer():
    value = builder("DeepseekSparseSWAMetadataBuilder", window_size=128, block_size=128)
    table = torch.tensor([[3]], dtype=torch.int32)
    with pytest.raises(ValueError, match="position/sequence mismatch"):
        build_q1_metadata(value, 1, 3, table)
    with pytest.raises(ValueError, match="exceeds block table width"):
        build_q1_metadata(value, 128, 129, table)
