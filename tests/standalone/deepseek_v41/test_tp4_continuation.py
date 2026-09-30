# SPDX-License-Identifier: Apache-2.0
"""CPU ownership gates; device dependencies still require the HPU chain gate."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_host import device_engram_parameters
from vllm_gaudi.v1.worker.deepseek_v41_v2_runner import CompletionRecord, V41AsyncOutput, V41V2ModelRunner


class NotReady:
    def synchronize(self):
        raise AssertionError("The non-output worker must not wait here")


def test_image_overlay_uses_actual_bound_ids_for_text_rows():
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner
    runner = object.__new__(V41ModelRunner)
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_first_rank=True))
    runner.input_ids = torch.tensor([11, 22])
    runner.model = SimpleNamespace(embed_input_ids=lambda ids: ids.float().unsqueeze(1))
    runner._execute_mm_encoder = lambda request: None
    runner.encoder_cache = {'image': torch.tensor([[999.]])}
    feature = SimpleNamespace(identifier='image', mm_position=SimpleNamespace(offset=0, length=1, is_embed=None))
    actual = runner._image_embeddings(SimpleNamespace(mm_features=[feature]), 0, 2, torch.tensor([101, 202]))
    assert actual.tolist() == [[999.], [202.]]


def test_decode_after_image_span_keeps_device_continuation_embedding():
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner
    runner = object.__new__(V41ModelRunner)
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_first_rank=True))
    feature = SimpleNamespace(mm_position=SimpleNamespace(offset=0, length=8))
    # No model or encoder is needed once the transaction has no image rows.
    assert runner._image_embeddings(SimpleNamespace(mm_features=[feature]), 100, 1, torch.tensor([17])) is None


@pytest.mark.parametrize("change", ["none", "generation", "precision", "search", "visible"])
def test_prefix_readiness_follows_compiled_generation(change):
    from vllm_gaudi.models.deepseek_v41_program import CompiledStage
    stage = object.__new__(CompiledStage)
    stage.owner = SimpleNamespace(generation=7, precision_fingerprint="original")
    stage.prefix_groups = 3
    stage.prefix_generation = stage._prefix_key(32768)
    if change == "generation":
        stage.owner.generation += 1
    elif change == "precision":
        stage.owner.precision_fingerprint = "changed"
    elif change == "visible":
        stage.owner.decode_token_bound = 20480
    assert stage.prefix_ready(65536 if change == "search" else 32768) is (change == "none")


@pytest.mark.parametrize("rank", range(4))
def test_device_parameters_preserve_fixed_offsets_for_six_heads(rank):
    primes = np.array([[101 + 2 * i for i in range(24)]], dtype=np.int64)
    offsets = np.array([[1000 * i for i in range(24)]], dtype=np.int64)
    start = 6 * rank
    layout = SimpleNamespace(layer_ids=[1], multipliers=np.array([[1, 3, 5, 7]]),
                             primes=primes, offsets=offsets,
                             head_shard=lambda *args: dict(head_start=start, head_stop=start+6, row_start=1000*start))
    packed = device_engram_parameters(layout, 1, rank, 17, 4)
    assert len(packed) == 47 and packed[:3] == [17, start, 1000 * start]
    assert packed[11:17] == primes[0, start:start+6].tolist()
    assert packed[23:29] == offsets[0, start:start+6].tolist()
    assert packed[35:41] == [pow(2, 64, int(p)) for p in primes[0, start:start+6]]
    assert packed[17:23] == [1] * 6
    assert packed[29:35] == packed[41:47] == [0] * 6


@pytest.mark.parametrize("rank", range(4))
def test_tp4_output_publication_does_not_wait_on_worker(rank):
    runner = object.__new__(V41V2ModelRunner)
    runner.model = SimpleNamespace(tensor_parallel_size=4, tp_rank=rank)
    record = CompletionRecord("a", 7, 16384, torch.tensor([[17]]), NotReady(), torch.tensor([17]))
    runner._completion, runner.pending = record, "batch_ready"
    runner.batch_result = V41AsyncOutput(record)
    result = runner.sample_tokens()
    assert isinstance(result, V41AsyncOutput) if rank == 0 else result is None
    assert runner._completion is record and runner.batch_result is None and runner.pending is None


@pytest.mark.parametrize("change", ["none", "pages", "position", "resume", "finished", "batch", "visible"])
def test_prefix_waits_for_scheduler_page_and_request_ownership(monkeypatch, change):
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX", "1")
    runner = object.__new__(V41V2ModelRunner)
    runner.audit = {}
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_first_rank=True))
    runner.model = SimpleNamespace(tensor_parallel_size=4, decode_prefix_ready=lambda search: True,
                                    program=SimpleNamespace(length=1048576, search_length=32768,
                                                            decode_token_bound=20480))
    record = SimpleNamespace(request_id="a", start=16400)
    cached = SimpleNamespace(req_ids=["a"], resumed_req_ids=set(),
                             num_computed_tokens=[16401], new_block_ids=[None])
    scheduled = SimpleNamespace(num_scheduled_tokens={"a": 1}, finished_req_ids=set(),
                                 scheduled_spec_decode_tokens={}, scheduled_cached_reqs=cached)
    if change == "pages":
        cached.new_block_ids[0] = ([129],)
    elif change == "position":
        cached.num_computed_tokens[0] -= 1
    elif change == "resume":
        cached.resumed_req_ids.add("a")
    elif change == "finished":
        scheduled.finished_req_ids.add("a")
    elif change == "batch":
        scheduled.num_scheduled_tokens["b"] = 1
    elif change == "visible":
        record.start = 20479
        cached.num_computed_tokens[0] = 20480
    assert runner._prefix_authorized(record, scheduled) is (change == "none")


def test_tp4_sampling_keeps_device_token_without_pp_broadcast(monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_completion
    calls = []
    runner = object.__new__(V41V2ModelRunner)
    runner.model = SimpleNamespace(tensor_parallel_size=4)
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
                                drain=lambda: calls.append("drain"), generation=4)
    runner._v2_async_step, runner._completion, runner.audit = True, None, {}
    request = SimpleNamespace(req_id="a", prompt=[1], decode_start=1)
    selected = torch.tensor([[17]], dtype=torch.int32)
    runner.pending = request, 2, 1, 1, [], True, selected
    runner.tp4_token_readback = lambda token: (token.clone(), NotReady())

    def reject_runtime_reverification(size):
        raise AssertionError("TP4 sampled-token hot path must reuse the verified readback producer")

    monkeypatch.setattr(deepseek_v41_completion, "resolve_device_runtime", reject_runtime_reverification)
    result = runner._sample_single()
    assert isinstance(result, V41AsyncOutput) and calls == ["drain"]
    assert runner._completion.device_token.data_ptr() == selected.data_ptr()
    assert runner._completion.generation == 5 and runner.pending is None
