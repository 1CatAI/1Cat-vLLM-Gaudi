# SPDX-License-Identifier: Apache-2.0
"""Independent request hash transactions retain the scalar history contract."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from vllm_gaudi.ops.deepseek_v41_batch_input import fill_request_metadata
from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout, EngramHistoryBatch, EngramTokenHistory


def layout():
    return EngramHashLayout.from_config({
        "engram_layer_ids": [1, 14],
        "engram_num_embeddings": [10000, 10000],
        "engram_max_ngram_size": 4,
        "engram_n_heads": 8,
        "engram_compressed_vocab_size": 8,
        "engram_vocab_size": 5,
        "engram_pad_token_id": 2,
        "engram_head_dim": 256,
    })


@pytest.mark.parametrize("count", [2, 3, 16, 32, 64])
def test_batch_hashes_match_independent_histories(count):
    parameters = layout()
    actual, reference = {}, {}
    for row in range(count):
        name = f"r{row}"
        token_map = np.roll(np.arange(16) % 8, row % 3)
        length = (0, 1, 2, 3, 2048)[row % 5]
        tokens = [(i * 3 + row) % 16 for i in range(length)]
        masks = [i % 17 == 0 for i in range(length)]
        for states in (actual, reference):
            owner = EngramTokenHistory(parameters, token_map)
            owner.reset(name)
            owner.restore_prefix(name, tokens, masks)
            states[name] = owner
    for step in range(9):
        names = list(np.roll(list(actual), step))
        spans = [(name, actual[name].position, [(step * 7 + row) % 16], None if step == 0 else [row % 3 == step % 3])
                 for row, name in enumerate(names)]
        packet = EngramHistoryBatch.prepare(actual, spans, 1 << (count - 1).bit_length())
        expected = [reference[name].prepare(name, tokens, dead) for name, _, tokens, dead in spans]
        assert np.array_equal(packet.hash_ids, np.concatenate([value.hash_ids for value in expected]))
        assert not packet.hash_ids.flags.writeable
        counts = [0 if (row + step) % 4 == 0 else 1 for row in range(count)]
        for (_, value), want in zip(packet.members, expected):
            assert np.array_equal(value.compressed_ids, want.compressed_ids)
            assert np.array_equal(value.active_mask, want.active_mask)
            assert not value.compressed_ids.flags.writeable and not value.hash_ids.flags.writeable
        packet.commit(counts)
        for name, want, keep in zip(names, expected, counts):
            reference[name].commit(want, keep)
            assert actual[name].position == reference[name].position
            assert np.array_equal(actual[name].history, reference[name].history)
            assert actual[name].generation == reference[name].generation
        with pytest.raises(RuntimeError, match="stale"):
            packet.commit(counts)


@pytest.mark.parametrize("bad", ["position", "token", "image", "pending", "owner"])
def test_invalid_decode_batch_publishes_no_partial_transactions(bad):
    parameters = layout()
    histories = {name: EngramTokenHistory(parameters, np.arange(16) % 8) for name in ("a", "b")}
    for name, owner in histories.items():
        owner.reset(name)
    spans = [("a", 0, [3], None), ("b", 0, [4], None)]
    pending = None
    if bad == "position":
        spans[-1] = ("b", 1, [4], None)
    elif bad == "token":
        spans[-1] = ("b", 0, [16], None)
    elif bad == "image":
        spans[-1] = ("b", 0, [4], [False, True])
    elif bad == "pending":
        pending = histories["b"].prepare("b", [1])
    else:
        histories["b"].reset("different")
    with pytest.raises((ValueError, RuntimeError)):
        EngramHistoryBatch.prepare(histories, spans, 2)
    assert histories["a"].pending is None
    assert histories["b"].pending is pending
    assert all(owner.position == 0 for owner in histories.values())


@pytest.mark.parametrize("count,capacity", [(1, 1), (2, 2), (3, 4), (16, 16), (32, 32)])
def test_pinned_frame_values_padding_and_reorder(count, capacity):
    host = torch.full((3, capacity), 123, dtype=torch.int32)
    for step in range(4):
        requests = [
            SimpleNamespace(tokens=[7, 11, 13 + i + step], num_computed_tokens=2) for i in reversed(range(count))
        ]
        owners = [SimpleNamespace(index=i) for i in reversed(range(count))]
        fill_request_metadata(host, requests, owners)
        expected = torch.full_like(host, -1)
        expected[0].zero_()
        for row, (request, owner) in enumerate(zip(requests, owners)):
            expected[:, row] = torch.tensor([request.tokens[2], 2, owner.index])
        assert torch.equal(host, expected)
