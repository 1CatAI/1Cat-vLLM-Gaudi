# SPDX-License-Identifier: Apache-2.0
"""Prefix reconstruction must match committed prefill without table reads."""
import numpy as np
import pytest

from test_engram import layout, reference_hashes

from vllm_gaudi.ops.deepseek_v41_engram import EngramTokenHistory


def test_checkpoint_carries_image_sentinels_without_original_multimodal_features():
    parameters = layout()
    source = EngramTokenHistory(parameters, np.arange(16) % 8)
    target = EngramTokenHistory(parameters, np.arange(16) % 8)
    source.reset("source")
    target.reset("target")
    tokens, mask = [5, 6, 7, 8, 9], [False, False, True, False, True]
    prepared = source.prepare("source", tokens, mask)
    source.commit(prepared, len(tokens))
    checkpoint = source.snapshot_prefix("source")
    assert checkpoint == (5, (-1, 0, -1))
    target.restore_checkpoint("target", checkpoint)
    expected = source.prepare("source", [3, 4]).hash_ids
    actual = target.prepare("target", [3, 4]).hash_ids
    assert np.array_equal(actual, expected)
    # A source request can advance or disappear without changing the snapshot.
    source.reset("replacement")
    assert checkpoint == (5, (-1, 0, -1))


@pytest.mark.parametrize("checkpoint", [(-1, ()), (128, (1, 2)), (128, (1, 2, 8)), (128, (1, 2, -2)),
                                        (128, (1, 2, 1.5)), (1048577, (1, 2, 3))])
def test_invalid_compressed_checkpoint_does_not_publish_history(checkpoint):
    history = EngramTokenHistory(layout(), np.arange(16) % 8)
    history.reset("new")
    generation = history.generation
    with pytest.raises(ValueError):
        history.restore_checkpoint("new", checkpoint)
    assert history.position == 0 and history.history.size == 0 and history.generation == generation


@pytest.mark.parametrize("length", [0, 1, 2, 3, 127, 128, 2048, 8193])
@pytest.mark.parametrize("image_offset", [None, 1, 2, 3, 4])
def test_restored_prefix_matches_committed_history_and_next_hashes(length, image_offset):
    parameters = layout()
    tokens = np.arange(length, dtype=np.int64) % 16
    dead = np.zeros(length, dtype=bool)
    if image_offset is not None and length >= image_offset:
        dead[-image_offset] = True
    ref = EngramTokenHistory(parameters, np.arange(16) % 8)
    restored = EngramTokenHistory(parameters, np.arange(16) % 8)
    for history in (ref, restored):
        history.reset("same")
    # Reference uses the unchanged transactional prefill path across chunks.
    for begin in range(0, length, 127):
        batch = ref.prepare("same", tokens[begin:begin + 127], dead[begin:begin + 127])
        ref.commit(batch, len(batch.compressed_ids))
    restored.restore_prefix("same", tokens, dead)
    assert restored.position == ref.position == length
    assert np.array_equal(restored.history, ref.history)
    suffix, suffix_dead = [9, 3, 7, 5, 6, 11], [False, False, True, False, False, False]
    expected = reference_hashes(parameters, tokens.tolist() + suffix, dead.tolist() + suffix_dead)[-len(suffix):]
    actual = restored.prepare("same", suffix, suffix_dead)
    assert actual.start_position == length
    assert np.array_equal(actual.hash_ids, expected)
    # Rejecting an unconsumed preparation leaves the restored prefix intact.
    restored.discard(actual)
    assert restored.position == length
    assert np.array_equal(restored.history, ref.history)
    next_token = restored.prepare("same", suffix[:1], suffix_dead[:1])
    assert np.array_equal(next_token.hash_ids, expected[:1])


def test_prefix_restore_rejects_live_owner_and_is_atomic_on_invalid_input():
    history = EngramTokenHistory(layout(), np.arange(16) % 8)
    history.reset("a")
    original_generation = history.generation
    for tokens, mask in (([16], None), ([-1], None), ([[1]], None), ([1, 2], [False])):
        with pytest.raises(ValueError):
            history.restore_prefix("a", tokens, mask)
        assert history.position == 0 and history.history.size == 0
        assert history.generation == original_generation
    with pytest.raises(RuntimeError, match="fresh"):
        history.restore_prefix("b", [1, 2])
    pending = history.prepare("a", [1, 2])
    with pytest.raises(RuntimeError, match="fresh"):
        history.restore_prefix("a", [3, 4])
    assert history.pending is pending
    history.commit(pending, 2)
    with pytest.raises(RuntimeError, match="fresh"):
        history.restore_prefix("a", [3, 4])
    history.reset("b")
    prefix = np.array([1, 2, 3, 4])
    mask = np.array([False, True, False, False])
    history.restore_prefix("b", prefix, mask)
    prefix[:] = 0
    mask[:] = True
    assert history.history.tolist() == [-1, 3, 4]
    with pytest.raises(RuntimeError, match="Stale"):
        history.commit(pending, 2)


def test_host_prefix_admission_preserves_other_requests_and_requires_idle_owner():
    from vllm_gaudi.ops.deepseek_v41_host import EngramHost

    host = EngramHost.__new__(EngramHost)
    host.layout = layout()
    host.history = EngramTokenHistory(host.layout, np.arange(16) % 8)
    host.history.reset("existing")
    batch = host.history.prepare("existing", [7, 8, 9])
    host.history.commit(batch, 3)
    host.histories = {"existing": host.history}
    host.closed = False
    host.pending = host.device_pending = None
    existing = host.history
    with pytest.raises(ValueError):
        host.restore_prefix("new", [16])
    assert "new" not in host.histories and host.history is existing
    host.pending = object()
    with pytest.raises(RuntimeError, match="live"):
        host.restore_prefix("new", [1, 2, 3])
    host.pending = None
    host.device_pending = "device-request"
    with pytest.raises(RuntimeError, match="live"):
        host.restore_prefix("new", [1, 2, 3])
    host.device_pending = None
    assert host.restore_prefix("new", [1, 2, 3, 4], [False, True, False, False]) == 4
    assert host.history is existing and existing.position == 3
    with pytest.raises(RuntimeError, match="new request"):
        host.restore_prefix("existing", [1])
    host.activate("new")
    assert host.history.history.tolist() == [-1, 3, 4]
    prepared = host.history.prepare("new", [5])
    expected = reference_hashes(host.layout, [1, 2, 3, 4, 5], [False, True, False, False, False])[-1:]
    assert prepared.start_position == 4 and np.array_equal(prepared.hash_ids, expected)
    host.history.commit(prepared, 1)
    host.release_request("new")
    assert host.restore_prefix("new", [6, 7]) == 2
    assert existing.position == 3
