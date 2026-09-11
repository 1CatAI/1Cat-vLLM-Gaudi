# SPDX-License-Identifier: Apache-2.0
"""Hash/history and native host-gather contracts before model graph integration."""

import gc

import numpy as np
import pytest

from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout, EngramTokenHistory


def layout():
    return EngramHashLayout.from_config({"engram_layer_ids": [1, 14], "engram_num_embeddings": [72, 204],
        "engram_max_ngram_size": 4, "engram_n_heads": 2, "engram_compressed_vocab_size": 8,
        "engram_vocab_size": 5, "engram_pad_token_id": 2, "engram_head_dim": 32})


def reference_hashes(parameters, tokens, dead):
    # Scalar reference follows vLLM #56214 _hash_ids_kernel: encountering an
    # image or the sequence start pads every older lookback of this n-gram.
    result = np.empty((len(tokens), 2, 6), dtype=np.int32)
    for position in range(len(tokens)):
        for layer in range(2):
            value, blocked = 0, False
            for shift in range(4):
                source = position - shift
                blocked = blocked or source < 0 or dead[source]
                code = 2 if blocked else tokens[source] % 8
                value ^= int(parameters.multipliers[layer, shift]) * code
                if shift:
                    for head in range(2):
                        column = (shift - 1) * 2 + head
                        result[position, layer, column] = value % int(parameters.primes[layer, column]) + int(
                            parameters.offsets[layer, column])
    return result


def test_exact_hash_head_shards_and_image_boundary():
    parameters = layout()
    assert parameters.primes.tolist() == [[5, 7, 11, 13, 17, 19], [23, 29, 31, 37, 41, 43]]
    for layer, end in zip((1, 14), (72, 204)):
        first, second = [parameters.head_shard(layer, tp) for tp in (0, 1)]
        assert first["row_stop"] == second["row_start"]
        assert first["head_stop"] == second["head_start"] == 3
        assert second["row_stop"] == end
    state = EngramTokenHistory(parameters, np.arange(16) % 8)
    state.reset("image-request")
    tokens, dead = [3, 4, 9, 8, 7, 6], [False, True, True, False, False, False]
    batch = state.prepare("image-request", tokens, dead)
    assert np.array_equal(batch.hash_ids, reference_hashes(parameters, tokens, dead))
    assert batch.active_mask.tolist() == [not value for value in dead]


def test_verify_rejection_request_switch_and_chunk_history():
    parameters = layout()
    state = EngramTokenHistory(parameters, np.arange(16) % 8)
    state.reset("a")
    prefix = state.prepare("a", [4, 5])
    state.commit(prefix, 2)
    verify = state.prepare("a", [6, 7, 8, 9, 10])
    with pytest.raises(RuntimeError, match="pending"):
        state.prepare("a", [2])
    state.commit(verify, 2)
    next_batch = state.prepare("a", [11, 12])
    expected = reference_hashes(parameters, [4, 5, 6, 7, 11, 12], [False] * 6)[-2:]
    assert next_batch.start_position == 4
    assert np.array_equal(next_batch.hash_ids, expected)
    state.discard(next_batch)
    replay = state.prepare("a", [11, 12])
    assert np.array_equal(replay.hash_ids, next_batch.hash_ids)
    state.reset("b")
    with pytest.raises(RuntimeError, match="Stale"):
        state.commit(replay, 2)
    fresh = state.prepare("b", [11, 12])
    assert np.array_equal(fresh.hash_ids, reference_hashes(parameters, [11, 12], [False] * 2))


def test_native_gather_fixed_buffers_and_completion_ownership(tmp_path):
    from vllm_gaudi.lib.dsv41_host_gather import GatherSlot, HostRows
    rows = np.arange(8 * 32, dtype=np.uint8).reshape(8, 32)
    scales = np.arange(8, dtype=np.uint8).reshape(8, 1)
    path = tmp_path / "tables.bin"
    # Deliberately unaligned offsets exercise source safetensors header offsets.
    path.write_bytes(b"header" + rows.tobytes() + b"pad" + scales.tobytes())
    table = HostRows(str(path), 6 + 4 * 32, str(path), 6 + rows.size + 3 + 4, 4, 8, 32, True, False)
    slot = GatherSlot(6, 32)
    weight_view, scale_view = slot.weights, slot.scales
    pointers = weight_view.ctypes.data, scale_view.ctypes.data
    for indices in ([7, 4, 7, 5], [5, 6, 4]):
        generation = slot.submit(table, np.array(indices, dtype=np.int32))
        slot.wait(generation)
        assert np.array_equal(weight_view[:len(indices)], rows[indices])
        assert np.array_equal(scale_view[:len(indices)], scales[indices])
        assert slot.major_faults == 0
        with pytest.raises(RuntimeError, match="earlier consumer"):
            slot.submit(table, np.array(indices, dtype=np.int32))
        with pytest.raises(RuntimeError, match="stale"):
            slot.release(generation - 1)
        slot.release(generation)
        assert pointers == (slot.weights.ctypes.data, slot.scales.ctypes.data)
    # The slot owns its table until completion; caller reference removal is safe.
    generation = slot.submit(table, np.array([4], dtype=np.int32))
    del table
    gc.collect()
    slot.wait(generation)
    assert np.array_equal(weight_view[0], rows[4])
    slot.release(generation)


def test_native_gather_rejects_wrong_head_rows(tmp_path):
    from vllm_gaudi.lib.dsv41_host_gather import GatherSlot, HostRows
    path = tmp_path / "table.bin"
    path.write_bytes(bytes(4 * 33))
    table = HostRows(str(path), 0, str(path), 4 * 32, 8, 12, 32, False, False)
    slot = GatherSlot(1, 32)
    generation = slot.submit(table, np.array([7], dtype=np.int32))
    with pytest.raises(RuntimeError, match="TP head shard"):
        slot.wait(generation)
