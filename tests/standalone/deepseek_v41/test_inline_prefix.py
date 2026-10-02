# SPDX-License-Identifier: Apache-2.0
"""An interior checkpoint must survive later rows without rewinding live state."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_math import pack_swa
from vllm_gaudi.ops.deepseek_v41_prefix_state import BatchPrefixSnapshot, InlinePrefixCapture
from test_prefix_state import Event, make_bank, slot_values


@pytest.mark.parametrize("start,end,boundary", [(0, 16384, 16256), (16384, 32768, 32640)])
def test_full_prompt_and_decoder_halo_checkpoint_matches_causal_ring(start, end, boundary):
    bank, source, target, neighbour = make_bank()
    live, untouched = slot_values(bank, source), slot_values(bank, neighbour)
    capture = InlinePrefixCapture(bank, source, start, end, boundary)
    expected = {}
    for layer, state in bank.layers.items():
        base = end - 4096 if layer == 1 else start
        positions = torch.arange(base, end)
        values = (positions[:, None].remainder(31) + torch.arange(512)[None, :].remainder(7)).float() / 16
        for name, _ in state.named_buffers(recurse=False):
            packed = name == "swa"
            producer = values.to(torch.bfloat16) if packed else values
            capture.record(layer, name, producer, pack=pack_swa if packed else None)
            rows = 256 if packed else 8
            causal = producer[boundary - base - rows:boundary - base]
            causal = pack_swa(causal) if packed else causal
            ring = torch.empty_like(causal)
            ring.index_copy_(0, torch.arange(boundary - rows, boundary).remainder(rows), causal)
            expected[layer, name] = ring
            # A later writer destroys its projection; saved boundary rows stay owned.
            producer.zero_()
    tensors = capture.require_complete()
    assert all(torch.equal(tensors[key], value) for key, value in expected.items())
    snapshot = BatchPrefixSnapshot.capture(bank, source, boundary, b"hash", Event(), record_done=Event,
                                           state_tensors=tensors)
    assert all(torch.equal(a, b) for a, b in zip(live, slot_values(bank, source), strict=True))
    snapshot.restore(bank, target, boundary, b"hash", Event(), record_done=Event)
    for layer, state in bank.layers.items():
        for name, value in state.named_buffers(recurse=False):
            rows = 256 if name == "swa" else 8
            assert torch.equal(value[target.index * rows:(target.index + 1) * rows], expected[layer, name])
    assert all(torch.equal(a, b) for a, b in zip(untouched, slot_values(bank, neighbour), strict=True))


def test_inline_snapshot_rejects_missing_or_incompatible_producers_without_live_writes():
    bank, source, _, _ = make_bank()
    live = slot_values(bank, source)
    capture = InlinePrefixCapture(bank, source, 0, 16384, 16256)
    with pytest.raises(RuntimeError, match="complete stage"):
        capture.require_complete()
    with pytest.raises(ValueError, match="boundary ring"):
        capture.record(0, "swa", torch.ones(128, 512, dtype=torch.bfloat16), pack=pack_swa)
    with pytest.raises(ValueError, match="complete stage"):
        BatchPrefixSnapshot.capture(bank, source, 16256, b"hash", Event(), record_done=Event, state_tensors={})
    assert all(torch.equal(a, b) for a, b in zip(live, slot_values(bank, source), strict=True))
