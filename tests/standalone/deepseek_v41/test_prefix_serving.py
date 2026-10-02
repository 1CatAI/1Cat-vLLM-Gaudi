# SPDX-License-Identifier: Apache-2.0
"""CPU scheduling/worker contracts; hardware output qualification is separate."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import OPTConfig

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256

pytest.importorskip("vllm.v1.core.auxiliary_prefix_cache", reason="Requires the auxiliary-prefix engine ABI")
from vllm.v1.core.auxiliary_prefix_cache import AuxiliaryPrefixAck, AuxiliaryPrefixDescriptor, AuxiliaryPrefixOperations
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm_gaudi.ops.deepseek_v41_engram import EngramTokenHistory
from vllm_gaudi.ops.deepseek_v41_host import EngramHost
from vllm_gaudi.ops.deepseek_v41_state import V41StateSpec, register_state_spec
from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler
from vllm_gaudi.v1.worker.deepseek_v41_prefix import PrefixCheckpoints, split_at_checkpoint

from test_engram import layout
from test_prefix_state import Event, make_bank, slot_values


def make_scheduler(tmp_path):
    OPTConfig(architectures=["OPTForCausalLM"],
              hidden_size=32,
              ffn_dim=64,
              num_hidden_layers=2,
              num_attention_heads=2,
              max_position_embeddings=8192).save_pretrained(tmp_path)
    config = VllmConfig(model_config=ModelConfig(model=str(tmp_path), dtype="float16", max_model_len=8192),
                        parallel_config=ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2),
                        cache_config=CacheConfig(block_size=128, enable_prefix_caching=True),
                        scheduler_config=SchedulerConfig(max_num_seqs=32,
                                                         max_num_batched_tokens=8192,
                                                         max_model_len=8192,
                                                         enable_chunked_prefill=True,
                                                         is_encoder_decoder=False,
                                                         async_scheduling=True,
                                                         watermark=0.0))
    config.cache_config.num_gpu_blocks = 8193
    register_state_spec(config)
    spec = V41StateSpec(block_size=128,
                        state_shape=(64, 288),
                        state_dtype=torch.uint8,
                        paged=True,
                        auxiliary_prefix=True)
    grouped = UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs={"layer": spec})
    assert grouped.requires_auxiliary_prefix_state and grouped.prefix_cacheable
    cache = KVCacheConfig(num_blocks=8193, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)])
    scheduler = HPUAsyncScheduler(vllm_config=config,
                                  kv_cache_config=cache,
                                  block_size=128,
                                  hash_block_size=128,
                                  log_stats=True,
                                  structured_output_manager=StructuredOutputManager(config))
    scheduler.use_v2_model_runner = True
    scheduler._completed_batch_reentry = True
    init_none_hash(sha256)
    return scheduler


def request(name):
    params = SamplingParams(temperature=0, max_tokens=1)
    params.update_from_generation_config({}, 50256)
    return Request(name, [13] * 2048, params, None, block_hasher=get_request_block_hasher(128, sha256))


def test_real_scheduler_requires_all_rank_state_then_reuses_only_1920_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    scheduler = make_scheduler(tmp_path)
    first = request("cold")
    scheduler.add_request(first)
    cold = scheduler.schedule()
    # The local executor transports scheduler/output objects through SHM pickle.
    cold = pickle.loads(pickle.dumps(cold, protocol=pickle.HIGHEST_PROTOCOL))
    operations = cold.auxiliary_prefix_operations
    assert MsgpackDecoder(AuxiliaryPrefixOperations).decode(MsgpackEncoder().encode(operations)) == operations
    descriptor = operations.captures["cold"]
    assert descriptor.num_tokens == 1920 and len(descriptor.block_ids[0]) == 15
    manager = scheduler.kv_cache_manager
    # Page hashes exist before worker completion; they alone must not be a hit.
    assert manager.get_computed_blocks(request("early"))[1] == 0
    result = ModelRunnerOutput(
        req_ids=["cold"],
        req_id_to_index={"cold": 0},
        sampled_token_ids=[[7]],
        auxiliary_prefix_acknowledgments=[AuxiliaryPrefixAck(descriptor.ticket, rank) for rank in range(4)])
    result = pickle.loads(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    scheduler.update_from_output(cold, result)
    second = request("warm")
    scheduler.add_request(second)
    warm = scheduler.schedule()
    assert warm.num_scheduled_tokens == {"warm": 128}
    assert warm.scheduled_new_reqs[0].num_computed_tokens == 1920
    assert warm.auxiliary_prefix_operations.restores == {"warm": descriptor}
    assert not warm.auxiliary_prefix_operations.captures
    assert not manager.auxiliary_prefix_cache.can_reset
    scheduler.update_from_output(
        warm, ModelRunnerOutput(req_ids=["warm"], req_id_to_index={"warm": 0}, sampled_token_ids=[[7]]))
    assert manager.auxiliary_prefix_cache.can_reset


def test_worker_checkpoint_restores_state_and_image_history_after_source_reuse(monkeypatch):
    bank, source, target, _ = make_bank()
    host = EngramHost.__new__(EngramHost)
    host.layout = layout()
    host.history = EngramTokenHistory(host.layout, np.arange(16) % 8)
    host.history.reset("source")
    tokens, images = [i % 16 for i in range(1920)], [False] * 1920
    images[-2] = True
    prepared = host.history.prepare("source", tokens, images)
    host.history.commit(prepared, 1920)
    host.histories = {"source": host.history}
    host.closed, host.pending, host.device_pending = False, None, None
    runner = SimpleNamespace(vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=3)),
                             model=SimpleNamespace(batch_state=bank, engram_host=host),
                             state=SimpleNamespace(blocks=8193),
                             pp=SimpleNamespace(drain=lambda: None),
                             audit={},
                             requests={
                                 name: SimpleNamespace(num_computed_tokens=1920, block_ids=(list(range(1, 8193)), ))
                                 for name in ("source", "target")
                             })
    bank.acquire = bank.slots.acquire
    bank.publish_pages = lambda slot, pages, total: None  # Existing CPU bank already has identical pages.
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(PrefixCheckpoints, "_done", staticmethod(Event))
    monkeypatch.setattr(Event, "synchronize", lambda self: None, raising=False)
    runtime = PrefixCheckpoints(runner)
    descriptor = AuxiliaryPrefixDescriptor(0, 1, 1920, b"prefix", (tuple(range(1, 16)), ))
    runtime.begin(AuxiliaryPrefixOperations(captures={"source": descriptor}))
    before = slot_values(bank, source)
    runtime.capture_at("source", 1920)
    # The same state backing is overwritten by later tokens; the checkpoint survives.
    for layer in bank.layers.values():
        layer.clear_slot(source.index)
        layer.clear_slot(target.index)
    runtime.operations = None  # CPU test ends before the real all-rank transport.
    runtime.begin(AuxiliaryPrefixOperations(restores={"target": descriptor}))
    assert all(torch.equal(a, b) for a, b in zip(before, slot_values(bank, target), strict=True))
    restored = host.histories["target"]
    assert restored.position == 1920 and restored.history[-2] == -1
    actual = restored.prepare("target", [3, 4]).hash_ids
    expected = host.histories["source"].prepare("source", [3, 4]).hash_ids
    assert np.array_equal(actual, expected)


def test_prefill_checkpoint_split_preserves_all_tokens_and_absolute_offsets():
    chunks = [(0, list(range(1024))), (1024, list(range(1024, 2048)))]
    result = split_at_checkpoint(chunks, 0, 1920)
    assert [(offset, len(values)) for offset, values in result] == [(0, 1024), (1024, 896), (1920, 128)]
    assert [value for _, values in result for value in values] == list(range(2048))


def test_inline_worker_keeps_full_transaction_and_restores_interior_history(monkeypatch):
    bank, source, target, _ = make_bank()
    bank.program.shared = SimpleNamespace()
    host = EngramHost.__new__(EngramHost)
    host.layout = layout()
    host.history = EngramTokenHistory(host.layout, np.arange(16) % 8)
    host.history.reset("source")
    tokens = [i % 16 for i in range(16384)]
    prepared = host.history.prepare("source", tokens)
    host.history.commit(prepared, len(tokens))
    host.histories = {"source": host.history}
    host.closed, host.pending, host.device_pending = False, None, None
    requests = {name: SimpleNamespace(num_computed_tokens=16256, block_ids=(list(range(1, 8193)),),
                                      token_slice=lambda begin, end: tokens[begin:end])
                for name in ("source", "target")}
    runner = SimpleNamespace(vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=3)),
                             model=SimpleNamespace(batch_state=bank, engram_host=host),
                             state=SimpleNamespace(blocks=8193), pp=SimpleNamespace(drain=lambda: None),
                             audit={}, requests=requests)
    bank.acquire = bank.slots.acquire
    bank.publish_pages = lambda *_: None
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(PrefixCheckpoints, "_done", staticmethod(Event))
    monkeypatch.setattr(Event, "synchronize", lambda self: None, raising=False)
    runtime = PrefixCheckpoints(runner)
    descriptor = AuxiliaryPrefixDescriptor(0, 1, 16256, b"prefix", (tuple(range(1, 128)),))
    runtime.begin(AuxiliaryPrefixOperations(captures={"source": descriptor}))
    chunks = [(0, tokens)]
    assert runtime.chunks("source", 0, chunks, inline_eligible=True) == chunks
    capture = bank.program.shared.inline_prefix_capture
    for layer, state in bank.layers.items():
        for name, value in state.named_buffers(recurse=False):
            capture.record(layer, name, torch.full((16384, value.shape[1]), layer + 7, dtype=value.dtype))
    expected = {key: value.clone() for key, value in capture.require_complete().items()}
    live = slot_values(bank, source)
    runtime.capture_at("source", 16384)
    assert all(torch.equal(a, b) for a, b in zip(live, slot_values(bank, source), strict=True))
    assert host.histories["source"].position == 16384
    runtime.operations = None
    runtime.begin(AuxiliaryPrefixOperations(restores={"target": descriptor}))
    restored = host.histories["target"]
    assert restored.position == 16256
    fresh = EngramTokenHistory(host.layout, np.arange(16) % 8)
    fresh.reset("fresh")
    fresh.restore_prefix("fresh", tokens[:16256])
    assert np.array_equal(restored.prepare("target", [3, 4]).hash_ids, fresh.prepare("fresh", [3, 4]).hash_ids)
    for (layer, name), values in expected.items():
        rows = values.shape[0]
        assert torch.equal(getattr(bank.layers[layer], name)[target.index * rows:(target.index + 1) * rows], values)


def test_fallback_checkpoint_retains_boundary_using_only_prepared_buckets(monkeypatch):
    from vllm_gaudi.v1.worker.deepseek_v41_runner import PREFILL_COMPUTE_BUCKETS
    runtime = object.__new__(PrefixCheckpoints)
    runtime.runner = SimpleNamespace(prefill_capacity=16384)
    descriptor = AuxiliaryPrefixDescriptor(0, 1, 16256, b"prefix", (tuple(range(1, 128)),))
    runtime.operations = AuxiliaryPrefixOperations(captures={"source": descriptor})
    tokens = list(range(16384))
    chunks = runtime.chunks("source", 0, [(0, tokens)])
    assert [value for _, values in chunks for value in values] == tokens
    assert all(len(values) in set(PREFILL_COMPUTE_BUCKETS) | {1} for _, values in chunks)
    assert any(offset + len(values) == 16256 for offset, values in chunks)
    assert not any(offset < 16256 < offset + len(values) for offset, values in chunks)
    assert max(len(values) for _, values in chunks) == 8192


def test_checkpoint_publication_preserves_owner_for_remaining_prompt_tail(monkeypatch):
    from test_single_batch_state import program
    from vllm_gaudi.ops.deepseek_v41_batch_state import BatchStageState

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *_a, **_k: self)
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(PrefixCheckpoints, "_done", staticmethod(Event))
    model = program()
    model.pp_rank, model.tp_rank, model.generation, model.precision_fingerprint = 0, 0, 1, "test"
    bank = BatchStageState(model, 2)
    source, other = bank.acquire("source"), bank.acquire("other")
    bank.publish_pages(source, [1, 2, 3, 4], 32)
    runner = SimpleNamespace(vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2)),
                             model=SimpleNamespace(batch_state=bank, engram_host=None),
                             state=SimpleNamespace(blocks=32), pp=SimpleNamespace(drain=lambda: None), audit={},
                             requests={"source": SimpleNamespace(block_ids=([1, 2, 3, 4],))})
    runtime = PrefixCheckpoints(runner)
    descriptor = AuxiliaryPrefixDescriptor(0, 1, 256, b"prefix", ((1, 2),))
    runtime.begin(AuxiliaryPrefixOperations(captures={"source": descriptor}))
    bank.bind_prefill(source)
    attention = model.layers[0].attention
    attention.swa.fill_(91)
    attention.kv_history.fill_(29)
    runtime.capture_at("source", 256)
    assert bank.prefill_owner == source
    checkpoint = runtime.store.entries[0]
    assert torch.all(checkpoint.tensors[2, "swa"] == 91)
    attention.swa.fill_(73)
    attention.kv_history.fill_(47)
    bank.begin([other, source])
    assert bank.prefill_owner is None
    assert torch.all(bank.layers[2].swa[:256] == 73)
    assert torch.all(bank.layers[2].kv_history[:8] == 47)
    assert torch.all(checkpoint.tensors[2, "swa"] == 91)
    assert torch.all(checkpoint.tensors[2, "kv_history"] == 29)
