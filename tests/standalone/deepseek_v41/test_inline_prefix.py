# SPDX-License-Identifier: Apache-2.0
"""An interior checkpoint must survive later rows without rewinding live state."""
import pytest
import torch
from types import SimpleNamespace

from vllm_gaudi.ops.deepseek_v41_math import pack_swa
from vllm_gaudi.ops.deepseek_v41_prefix_state import BatchPrefixSnapshot, InlinePrefixCapture
from test_prefix_state import Event, make_bank, slot_values
from vllm_gaudi.v1.worker.deepseek_v41_prefix import PrefixCheckpoints


@pytest.mark.parametrize("start,sizes,boundary", [(0, [1024], 896),
                                                (16384, [4096, 1024], 21376),
                                                (32768, [8192, 1024], 41856)])
def test_checkpoint_keeps_finite_tail_and_activates_only_its_producer(start, sizes, boundary):
    bank, _, _, _ = make_bank()
    bank.program.shared = SimpleNamespace(inline_prefix_capture=None)
    runner = SimpleNamespace(vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2)),
                             model=SimpleNamespace(batch_state=bank), prefill_capacity=16384)
    runtime = PrefixCheckpoints(runner)
    runtime.operations = SimpleNamespace(captures={"source": SimpleNamespace(num_tokens=boundary)})
    chunks, offset = [], 0
    for size in sizes:
        chunks.append((offset, list(range(start + offset, start + offset + size))))
        offset += size
    assert runtime.chunks("source", start, chunks, inline_eligible=True) == chunks
    capture = runtime.inline["source"]
    for offset, values in chunks:
        runtime.activate_chunk("source", start + offset, len(values))
        active = bank.program.shared.inline_prefix_capture
        assert (active is capture) == (start + offset <= boundary < start + offset + len(values))
    runtime.activate_chunk("source", start + sum(sizes), 1)
    assert bank.program.shared.inline_prefix_capture is None


@pytest.mark.parametrize("start,end,boundary", [(0, 16384, 16256), (16384, 32768, 32640),
                                              (0, 1024, 896), (20480, 21504, 21376)])
def test_full_prompt_and_decoder_halo_checkpoint_matches_causal_ring(start, end, boundary):
    bank, source, target, neighbour = make_bank()
    live, untouched = slot_values(bank, source), slot_values(bank, neighbour)
    capture = InlinePrefixCapture(bank, source, start, end, boundary)
    expected = {}
    for layer, state in bank.layers.items():
        base = max(start, end - 4096) if layer == 1 else start
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
