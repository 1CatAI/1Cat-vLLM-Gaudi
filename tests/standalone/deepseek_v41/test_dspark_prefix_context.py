# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.v1.worker.deepseek_v41_prefix import draft_context_views, restore_draft_context


def program():
    layers = []
    for index in range(3):
        attention = SimpleNamespace(swa=torch.full((256, 528), index + 1, dtype=torch.uint8))
        if index == 1:
            attention.swa_decoded = SimpleNamespace(swa=torch.full((256, 512), 1., dtype=torch.bfloat16))
        layers.append(SimpleNamespace(attention=attention))
    return SimpleNamespace(draft=SimpleNamespace(layers=layers))


def test_prefix_restores_draft_history_after_an_intervening_request():
    owner = program()
    saved = tuple(value.clone() for value in draft_context_views(owner))
    for value in draft_context_views(owner):
        value.zero_()
    restore_draft_context(owner, saved)
    assert all(torch.equal(value, expected) for value, expected in zip(draft_context_views(owner), saved, strict=True))
    draft_context_views(owner)[0].fill_(255)
    assert saved[0].eq(1).all()


def test_draft_layout_failure_precedes_any_restore_write():
    owner = program()
    saved = tuple(value.clone() for value in draft_context_views(owner))
    for value in draft_context_views(owner):
        value.zero_()
    saved = (*saved[:-1], saved[-1].float())
    with pytest.raises(ValueError, match="state layout"):
        restore_draft_context(owner, saved)
    assert all(not value.any() for value in draft_context_views(owner))


def test_ordinary_prefix_has_no_draft_state():
    owner = SimpleNamespace(draft=None)
    assert draft_context_views(owner) == ()
    restore_draft_context(owner, ())


@pytest.mark.parametrize("capacity", (1, 2))
def test_speculative_prefix_admission_matches_owned_draft_capacity(monkeypatch, capacity):
    pytest.importorskip("vllm.v1.core.auxiliary_prefix_cache")
    from vllm_gaudi.entrypoints import deepseek_v41 as entrypoint
    from vllm_gaudi.ops import deepseek_v41_config as contract
    from vllm_gaudi.ops import deepseek_v41_state

    monkeypatch.setattr(entrypoint, "prepare_native_libraries", lambda: None)
    monkeypatch.setattr(deepseek_v41_state, "register_state_spec", lambda config: None)
    for name in ("PREPARED_SHARDS", "GRAPH_REPLAY", "DSPARK"):
        monkeypatch.setenv("VLLM_HPU_DSV41_" + name, "1")
    for name in ("V2", "BATCH_DECODE", "RUNTIME_INDEXER"):
        monkeypatch.setenv("VLLM_HPU_DSV41_" + name, "0")
    for name in entrypoint._PIPELINE_ONLY_FASTPATHS:
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("VLLM_HPU_TP2_STATIC_GROUP_PLAN", "1")
    monkeypatch.setenv("VLLM_HPU_TP2_PREPARED_COMM", "1")
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"), max_model_len=262144),
        parallel_config=SimpleNamespace(data_parallel_size=1, tensor_parallel_size=4, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(enable_prefix_caching=True, user_specified_block_size=False, block_size=128),
        scheduler_config=SimpleNamespace(max_num_seqs=capacity, async_scheduling=False),
        load_config=SimpleNamespace(load_format="dsv41_prepared"),
        speculative_config=SimpleNamespace(method="dspark",
                                           num_speculative_tokens=5,
                                           enable_adaptive_verification=False),
        use_v2_model_runner=False,
        lora_config=None,
        kv_transfer_config=None,
    )
    if capacity == 1:
        contract.configure(config)
    else:
        with pytest.raises(ValueError, match="one active request"):
            contract.configure(config)


@pytest.mark.parametrize("checkpoint", (False, True))
def test_prefix_transaction_retires_async_commit_before_checkpoint(checkpoint):
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner

    retired = []
    acknowledgment = object()

    class Pending(AsyncModelRunnerOutput):
        def get_output(self):
            retired.append(True)
            return ModelRunnerOutput(req_ids=["r"], req_id_to_index={"r": 0}, sampled_token_ids=[[13]])

    def capture(request_id, position):
        assert request_id == "r" and position == 257
        assert bool(retired) == checkpoint

    runner = object.__new__(V41ModelRunner)
    runner.pending = None
    runner._update = lambda scheduled: None
    runner._execute_request = lambda *args: None
    runner._finish_request = lambda: Pending()
    runner.model = SimpleNamespace(batch_state=SimpleNamespace(leave_single=lambda: None))
    runner.requests = {"r": SimpleNamespace(num_computed_tokens=256)}
    runner.request_batches = None
    runner.draft_token_ids = None
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=True))
    runner.prefix_checkpoints = SimpleNamespace(begin=lambda operations: None, capture_at=capture,
                                               finish=lambda: [acknowledgment] if checkpoint else [])
    scheduled = SimpleNamespace(num_scheduled_tokens={"r": 1},
                                auxiliary_prefix_operations=object() if checkpoint else None)
    runner.execute_model(scheduled)
    if checkpoint:
        assert runner.batch_result.sampled_token_ids == [[13]]
        assert runner.batch_result.auxiliary_prefix_acknowledgments == [acknowledgment]
    else:
        assert isinstance(runner.batch_result, Pending) and not retired
