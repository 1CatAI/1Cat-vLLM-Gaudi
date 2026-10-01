# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from types import SimpleNamespace

from vllm_gaudi.ops.deepseek_v41_inputs import PinnedDecodeInputs


def test_staging_reuses_views_across_bucket_and_slot_reordering():
    staging = PinnedDecodeInputs(6, "cpu")
    views = {key: (id(a), id(b)) for key, (a, b) in staging.views.items()}
    for index, count in enumerate((1, 6, 2, 1, 6)):
        values = list(range(index * 100, index * 100 + count))
        ids, positions = torch.empty(count, dtype=torch.int64), torch.empty(count, dtype=torch.int32)
        staging.stage(values, 16384 + index * 8, ids, positions)
        assert ids.tolist() == values
        assert positions.tolist() == list(range(16384 + index * 8, 16384 + index * 8 + count))
    assert staging.bytes == 12 * 16
    assert views == {key: (id(a), id(b)) for key, (a, b) in staging.views.items()}


def test_staging_waits_for_real_dma_completion_before_overwriting_host_slot():
    staging = PinnedDecodeInputs(6, "cpu")
    waited = []

    class Event:
        def query(self):
            return False

        def synchronize(self):
            waited.append(staging.ids[0, 0].item())

        def record(self):
            pass

    staging.events = [Event(), Event()]
    ids, positions = torch.empty(1, dtype=torch.int64), torch.empty(1, dtype=torch.int32)
    for value in (11, 22, 33):
        staging.stage([value], value, ids, positions)
    assert waited == [11] and staging.waits == 1 and ids.item() == 33


def test_invalid_staging_range_does_not_mutate_destinations():
    staging = PinnedDecodeInputs(6, "cpu")
    ids, positions = torch.tensor([77]), torch.tensor([88], dtype=torch.int32)
    with pytest.raises(ValueError, match="range"):
        staging.stage([1], 2**31, ids, positions)
    assert ids.item() == 77 and positions.item() == 88 and staging.calls == 0


def test_decode_geometry_rebinds_after_prefill_generation_and_search_changes():
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner

    searches = []
    attention = SimpleNamespace(set_search_length=searches.append)
    program = SimpleNamespace(length=1048576, generation=1, tensor_parallel_size=4,
                              layers=[SimpleNamespace(attention=attention)])

    class Model:
        native = False

        def prepare_step(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return torch.zeros(1, 8)

    runner = object.__new__(V41ModelRunner)
    runner.request_batches = None
    runner.model = Model()
    runner.model.program = program
    runner.use_dspark = runner.direct_token_ids = False
    runner.input_views = {1: torch.empty(1, dtype=torch.int64)}
    runner.position_views = {1: torch.empty(1, dtype=torch.int32)}
    runner.position_bank = runner._next_input = None
    uploads = []

    def upload(tokens, start):
        uploads.append((tokens, start))
        runner.input_views[1].copy_(torch.tensor(tokens))
        runner.position_views[1].fill_(start)

    runner.tp4_control_inputs = {1: SimpleNamespace(upload=upload)}
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_first_rank=True, is_last_rank=True))
    runner.audit = {"target_steps": 0, "target_tokens": 0}
    runner._forward("one", [42], 16384, decode=True, search_length=32768)
    runner._forward("one", [43], 16385, decode=True, search_length=32768)
    assert searches == [32768]
    program.generation += 1
    runner._forward("two", [44], 16384, decode=True, search_length=32768)
    runner._forward("two", [45], 16385, decode=False, search_length=32768)
    assert attention.prefill_token_end == 16386
    runner._forward("two", [46], 16386, decode=True, search_length=32768)
    assert attention.prefill_token_end is None
    runner._forward("two", [47], 32768, decode=True, search_length=65536)
    assert searches == [32768, 32768, 32768, 32768, 65536]
    assert len(uploads) == 5


@pytest.mark.parametrize("start,stop", [(0, 0), (0, 2), (2, 5), (4, 5), (4, 9), (9, 12)])
def test_scheduled_token_slice_matches_full_history_without_concatenating(start, stop):
    from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState
    request = object.__new__(RequestState)
    request.prompt, request.output = [11, 12, 13, 14], [21, 22, 23]
    expected = (request.prompt + request.output)[start:stop]
    assert request.token_slice(start, stop) == expected
