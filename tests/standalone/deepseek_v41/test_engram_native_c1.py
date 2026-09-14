# SPDX-License-Identifier: Apache-2.0
"""Native C1 must preserve the frozen hash and final staging bytes."""
from dataclasses import replace
import gc

import numpy as np
import pytest

from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout, EngramTokenHistory
from vllm_gaudi.ops.deepseek_v41_host import host_native

native = host_native()
HostRows = native.HostRows
NativeC1Prepare = native.NativeC1Prepare


def fixture(tmp_path, tp, width=256, overflow=False):
    layout = EngramHashLayout.from_config({
        "engram_layer_ids": [1, 14], "engram_num_embeddings": [10000, 10000],
        "engram_max_ngram_size": 4, "engram_n_heads": 8, "engram_compressed_vocab_size": 8,
        "engram_vocab_size": 5, "engram_pad_token_id": 2, "engram_head_dim": width})
    if overflow:
        values = np.array([[2**63-1, -2**63, -1, 3], [-2**63+1, 2**63-3, 7, -5]], dtype=np.int64)
        layout = replace(layout, multipliers=values)
    state = EngramTokenHistory(layout, np.arange(16) % 8)
    tables, sources, first, last = [], [], [], []
    for index, layer in enumerate(layout.layer_ids):
        shard = layout.head_shard(layer, tp)
        start, stop = shard["row_start"], shard["row_stop"]
        rows = stop
        weights = np.arange(rows * width, dtype=np.uint8).reshape(rows, width)
        weights ^= np.arange(rows, dtype=np.uint8)[:, None]
        scales = np.arange(rows * (width // 32), dtype=np.uint8).reshape(rows, width // 32)
        path = tmp_path / f"{index}-{tp}-{width}.bin"
        path.write_bytes(b"header!" + weights.tobytes() + b"pad" + scales.tobytes())
        tables.append(HostRows(str(path), 7 + start * width, str(path),
                              10 + weights.size + start * (width // 32), start, stop, width, True, False))
        sources.append((weights, scales))
        first.append(shard["head_start"])
        last.append(shard["head_stop"])
    targets = [[np.full((1, 12, width + width // 32), 197, dtype=np.uint8) for _ in range(2)] for _ in range(3)]
    args = (state.token_map, layout.multipliers, layout.primes, layout.offsets,
            np.array(first, dtype=np.int64), np.array(last, dtype=np.int64), state.pad_id, tables, targets)
    return state, NativeC1Prepare(*args), targets, sources, args


@pytest.mark.parametrize("tp", [0, 1])
@pytest.mark.parametrize("width,overflow", [(32, False), (256, True)])
def test_exact_native_hash_bytes_requests_images_and_prefill(tmp_path, tp, width, overflow):
    state, native, targets, sources, args = fixture(tmp_path, tp, width, overflow)
    reference = EngramTokenHistory(state.layout, state.token_map)
    generation = 0
    # C1 and ordinary prefill share one committed history. Rejected prefill
    # suffixes and image spans must never leak into later native hashes.
    for request in ("first", "second"):
        state.reset(request)
        reference.reset(request)
        for step, count in enumerate((1, 1, 6, 1, 2, 1, 1, 1, 1)):
            tokens = [(step * 3 + i) % 16 for i in range(count)]
            image = [step in (1, 4) or i == 2 for i in range(count)]
            expected = reference.prepare(request, tokens, image)
            if count == 1:
                generation += 1
                slot = generation % 3
                before = [[value.copy() for value in row] for row in targets]
                batch, faults = state.prepare_c1(request, tokens[0], image[0], native, generation, slot)
                assert state.position == expected.start_position
                assert faults == 0
                for layer in range(2):
                    ids = expected.hash_ids[:, layer, args[4][layer]:args[5][layer]]
                    w, s = sources[layer]
                    assert np.array_equal(targets[slot][layer], np.concatenate((w[ids], s[ids]), axis=-1))
                for other in set(range(3)) - {slot}:
                    assert all(np.array_equal(a, b) for a, b in zip(targets[other], before[other]))
                with pytest.raises(RuntimeError, match="pending"):
                    state.prepare_c1(request, tokens[0], False, native, generation + 1, (slot + 1) % 3)
                with pytest.raises(RuntimeError, match="stale"):
                    native.complete(request, batch.generation, generation - 1)
                native.complete(request, batch.generation, generation)
            else:
                batch = state.prepare(request, tokens, image)
            assert np.array_equal(batch.hash_ids, expected.hash_ids)
            assert np.array_equal(batch.compressed_ids, expected.compressed_ids)
            assert np.array_equal(batch.active_mask, expected.active_mask)
            committed = count if count == 1 else count - 1
            state.commit(batch, committed)
            reference.commit(expected, committed)
            assert np.array_equal(state.history, reference.history)
    del args
    gc.collect()
    # Native owns table mappings and final targets across caller ref removal.
    state.reset("retained")
    batch, _ = state.prepare_c1("retained", 2, False, native, generation + 1, 0)
    native.complete("retained", batch.generation, generation + 1)


def test_invalid_generation_token_slot_and_destination_alias(tmp_path):
    state, native, targets, _, args = fixture(tmp_path, 0)
    with pytest.raises(RuntimeError, match="alias"):
        NativeC1Prepare(*args[:-1], [targets[0], targets[0]])
    for bad_target in (targets[0][0].astype(np.int16), targets[0][0][:, :, ::-1]):
        with pytest.raises(RuntimeError, match="destination"):
            NativeC1Prepare(*args[:-1], [[bad_target, targets[0][1]], targets[1]])
    state.reset("a")
    for token, slot, message in ((16, 0, "vocabulary"), (1, 3, "slot")):
        with pytest.raises(RuntimeError, match=message):
            native.prepare("a", 1, 1, slot, token, False, state.history)
    batch, _ = state.prepare_c1("a", 1, False, native, 1, 0)
    native.complete("a", batch.generation, 1)
    state.commit(batch, 1)
    before = targets[0][0].copy()
    with pytest.raises(RuntimeError, match="Stale"):
        native.prepare("a", 2, 1, 0, 2, False, state.history)
    assert np.array_equal(before, targets[0][0])


def test_bad_second_layer_rows_fail_before_any_write(tmp_path):
    state, _, targets, _, args = fixture(tmp_path, 0)
    bad = list(args)
    bad[3] = args[3].copy()
    bad[3][1, :] += 100000
    native = NativeC1Prepare(*bad)
    before = [[value.copy() for value in row] for row in targets]
    with pytest.raises(RuntimeError, match="TP shard"):
        native.prepare("a", 1, 1, 0, 1, False, state.history)
    assert all(np.array_equal(a, b) for row, saved in zip(targets, before) for a, b in zip(row, saved))
    with pytest.raises(RuntimeError, match="failed"):
        native.prepare("a", 1, 2, 0, 1, False, state.history)


@pytest.mark.parametrize("tp", [0, 1])
def test_late_only_c1_hashes_both_layers_but_writes_only_layer_14(tmp_path, tp):
    state, native, targets, sources, args = fixture(tmp_path, tp)
    reference = EngramTokenHistory(state.layout, state.token_map)
    state.reset("late")
    reference.reset("late")
    expected = reference.prepare("late", [7], [False])
    before_layer1 = targets[0][0].copy()

    batch, faults = state.prepare_c1("late", 7, False, native, 1, 0, late_only=True)

    first, last = args[4][1], args[5][1]
    ids = expected.hash_ids[:, 1, first:last]
    weights, scales = sources[1]
    assert faults == 0
    assert np.array_equal(batch.hash_ids, expected.hash_ids)
    assert np.array_equal(targets[0][0], before_layer1)
    assert np.array_equal(targets[0][1], np.concatenate((weights[ids], scales[ids]), axis=-1))
    native.complete("late", batch.generation, 1)
    state.commit(batch, 1)
