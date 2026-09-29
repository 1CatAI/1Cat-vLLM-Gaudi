# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_gaudi.ops.deepseek_v41_batch import FlatBatch, Phase, QuerySpan, RequestSlots


class Event:
    done = False

    def query(self):
        return self.done


def test_cancelled_slot_waits_for_consumer_and_rejects_old_generation():
    slots = RequestSlots(1)
    first = slots.acquire("first")
    event = Event()
    slots.submitted(first, event)
    slots.release(first)
    with pytest.raises(RuntimeError, match="live consumers"):
        slots.acquire("second")
    event.done = True
    second = slots.acquire("second")
    assert second.index == first.index and second.generation == first.generation + 1
    with pytest.raises(RuntimeError, match="Stale"):
        slots.submitted(first, Event())
    with pytest.raises(RuntimeError, match="Stale"):
        slots.release(first)


def test_flat_batch_preserves_request_positions_and_distinguishes_query_count():
    slots = RequestSlots(4)
    a, b, c = [slots.acquire(name) for name in ("a", "b", "c")]
    batch = FlatBatch.create((QuerySpan(c, 128, (9, ), Phase.DECODE,
                                        (4, 7)), QuerySpan(a, 3, (5, 6, 7), Phase.PREFILL,
                                                           (2, )), QuerySpan(b, 7, (11, ), Phase.DECODE, (3, ))),
                             maximum=1048576,
                             budget=8192,
                             capacity=4)
    values = batch.tensors(8)
    assert batch.decode_bucket is None
    assert batch.offsets == (0, 1, 4, 5)
    assert values["positions"].tolist() == [128, 3, 4, 5, 7, -1, -1, -1]
    assert values["request_slots"].tolist() == [2, 0, 0, 0, 1, -1, -1, -1]
    assert values["phases"].tolist() == [2, 1, 1, 1, 2, 0, 0, 0]
    assert values["sequence_lengths"].tolist() == [129, 6, 8]
    decode = FlatBatch.create((QuerySpan(c, 128, (9, ), Phase.DECODE,
                                         (4, 7)), QuerySpan(a, 3, (5, ), Phase.DECODE,
                                                            (2, )), QuerySpan(b, 7, (11, ), Phase.DECODE, (3, ))),
                              maximum=1048576,
                              budget=64,
                              capacity=4)
    assert decode.decode_bucket == 4


def test_batch_rejects_aliases_truncation_missing_pages_and_speculation():
    slot = RequestSlots(1).acquire("request")
    valid = QuerySpan(slot, 127, (1, ), Phase.DECODE, (1, ))
    for spans in ((valid, valid), (QuerySpan(slot, 128, (1, ), Phase.DECODE,
                                             (1, )), ), (QuerySpan(slot, 0, (1, 2), Phase.DECODE, (1, )), )):
        with pytest.raises(ValueError):
            FlatBatch.create(spans, maximum=1048576, budget=8192, capacity=4)
    with pytest.raises(ValueError, match="truncate"):
        FlatBatch.create((valid, ), maximum=1048576, budget=8192, capacity=1).tensors(0)
