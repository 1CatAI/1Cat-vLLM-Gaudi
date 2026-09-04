# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from types import SimpleNamespace
import habana_frameworks.torch  # noqa: F401
from habana_frameworks.torch.utils.internal import is_lazy
from vllm.model_executor.model_loader import get_model

from vllm.model_executor.layers.attention import Attention
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig, set_current_vllm_config)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_utils import (estimate_max_model_len, get_kv_cache_configs)
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData, SchedulerOutput)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor)
from vllm.v1.sample.metadata import SamplingMetadata
import vllm_gaudi.extension.environment as environment
import vllm_gaudi.v1.worker.hpu_model_runner as model_runner_module
from vllm_gaudi.v1.worker.hpu_model_runner import (
    HPUModelRunner,
    HpuModelAdapter,
    maybe_set_mamba_kv_cache_groups_ids,
    should_synchronize_hybrid_prefill_output,
)
from vllm_gaudi.v1.worker.hpu_input_batch import InputBatch

BLOCK_SIZE = 128
NUM_BLOCKS = 10
DEVICE = current_platform.device_type


def initialize_kv_cache(runner: HPUModelRunner):
    """
    Only perform necessary steps in HPUModelRunner.initialize_kv_cache()
    """
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size,
                          layers=["layer.0"],
                          layer_stride=tensor_size,
                          block_stride=attn_spec.page_size_bytes),
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)],
    )
    runner.kv_cache_config = kv_cache_config
    runner.input_batch = InputBatch(
        max_num_reqs=runner.max_num_seqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size],
        kernel_block_sizes=[kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size],
    )


#    runner.initialize_attn_backend(kv_cache_config)


def get_vllm_config():
    model_config = ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    # PR #51718: get_kv_cache_configs now calls get_resolved_kv_cache_layout(),
    # which raises "KV cache layout has not been resolved yet" unless the layout
    # is pre-set (normally done once by the engine core). Resolve it here so the
    # unit tests can call get_kv_cache_configs directly, mirroring upstream.
    cache_config.kv_cache_layout = "LBNHC"
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config


@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        model_config = vllm_config.model_config
        num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = model_config.get_head_size()
        # We need to update the environment before creating Attention
        environment.set_vllm_config(vllm_config)
        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(num_heads, head_size, 0.1)
        runner = HPUModelRunner(vllm_config, DEVICE)
        initialize_kv_cache(runner)
        yield runner


def _schedule_new_request(*req_ids: str) -> SchedulerOutput:
    new_reqs = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    for req_id in req_ids:
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[1, 2, 3],
                mm_features=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=([0], ),
                num_computed_tokens=0,
                lora_request=None,
            ))
        num_scheduled_tokens[req_id] = 3
        total_num_scheduled_tokens += num_scheduled_tokens[req_id]

    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _is_req_scheduled(model_runner, req_id: str) -> bool:
    return req_id in model_runner.input_batch.req_id_to_index


def _is_req_added(model_runner, req_id: str) -> bool:
    return req_id in model_runner.requests


def _is_sampling_metadata_changed(model_runner, sampling_metadata_before: SamplingMetadata):
    return model_runner.input_batch.sampling_metadata is not (sampling_metadata_before)


def _is_req_state_block_table_match(model_runner, req_id: str) -> bool:
    req_index = model_runner.input_batch.req_id_to_index[req_id]
    block_table = model_runner.input_batch.block_table[0]
    req_state = model_runner.requests[req_id]
    if block_table.num_blocks_per_row[req_index] != len(req_state.block_ids[0]):
        return False
    num_blocks = block_table.num_blocks_per_row[req_index]
    return (block_table.block_table.np[req_index, :num_blocks] == req_state.block_ids[0]).all()


def test_update_states_new_request(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_finished(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # finish req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids={req_id},
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert not _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)


def test_update_states_request_resumed(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # unschedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    # resume req
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids={req_id},
        new_token_ids=[[]],
        new_block_ids=[([0], )],
        num_computed_tokens=[0],
        num_output_tokens=[0],
        all_token_ids={},
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_get_nans_in_logits(model_runner, dist_init):
    req_ids = ("req_0", "req_1")

    scheduler_output = _schedule_new_request(*req_ids)
    model_runner._update_states(scheduler_output)

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
    ], device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor([
        [1.0, float('nan'), 3.0],
        [4.0, float('nan'), float('nan')],
    ], device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 1, "req_1": 2}

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, float('nan'), float('nan')],
    ], device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 2}

    result = model_runner._get_nans_in_logits(logits=None)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor([
        [1.0, float('nan'), 3.0],
    ], device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {'req_0': 1, 'req_1': 0}

    logits = torch.tensor([
        [float('nan'), float('nan'), 2.0],
        [1.0, 2.0, 3.0],
        [float('nan'), 2.0, 3.0],
    ],
                          device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {'req_0': 2, 'req_1': 0}


def test_update_states_no_changes(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # schedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert not _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_unscheduled(model_runner, dist_init):
    req_ids = ("req_0", "req_1")

    # new reqs
    scheduler_output = _schedule_new_request(*req_ids)

    model_runner._update_states(scheduler_output)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert _is_req_scheduled(model_runner, req_ids[1])

    # unschedule req_1
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_ids[0]: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert not _is_req_scheduled(model_runner, req_ids[1])


def test_update_config(model_runner):
    # Simple update
    model_runner.update_config({"load_config": {"load_format": "dummy"}})
    assert model_runner.load_config.load_format == "dummy"
    # Raise error on non-existing config
    with pytest.raises(AssertionError):
        model_runner.update_config({"do_not_exist_config": "dummy"})


def test_reload_weights_before_load_model(model_runner):
    with pytest.raises(AssertionError):
        model_runner.reload_weights()


def test_tp2_fused_text_only_mm_inputs_defer_embedding(monkeypatch):
    embedded = []
    runner = SimpleNamespace(
        supports_mm_inputs=True,
        uses_mrope=False,
        is_mm_embed=SimpleNamespace(copy_to_gpu=lambda _tokens: torch.tensor([], dtype=torch.bool)),
        attn_backend_name='HPUAttentionBackendV1',
        model=SimpleNamespace(embed_input_ids=lambda *args, **kwargs: embedded.append((args, kwargs))),
        _execute_mm_encoder=lambda *_args: None,
        _gather_mm_embeddings=lambda *_args, **_kwargs: ([], torch.tensor([], dtype=torch.bool)),
        _extract_mm_kwargs=lambda _scheduler_output: {},
    )
    monkeypatch.setattr(model_runner_module, 'get_config', lambda: SimpleNamespace(tp2_fused_ar_norm=True))

    inputs_embeds, model_mm_kwargs = HPUModelRunner._get_model_mm_inputs(
        runner,
        torch.tensor([[1, 2, 3]]),
        3,
        SimpleNamespace(),
        ['request'],
    )

    assert inputs_embeds is None
    assert model_mm_kwargs == {}
    assert embedded == []


def test_init_kv_cache_with_kv_sharing_invalid_target_layer_order(default_vllm_config: None):
    torch.set_default_dtype(torch.bfloat16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} must come before the current layer"
    environment.set_vllm_config(get_vllm_config())
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
                kv_sharing_target_layer_name=layer_1,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            )
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_layer_not_exist(default_vllm_config: None):
    torch.set_default_dtype(torch.bfloat16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    invalid_layer = "model.layers.0.cross_attn.attn"
    error_msg = f"{invalid_layer} is not a valid Attention layer in the model"
    environment.set_vllm_config(get_vllm_config())
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                # invalid layer: cross_attn.atn doesn't exist!
                kv_sharing_target_layer_name=invalid_layer,
            )
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_same_as_current(default_vllm_config: None):
    torch.set_default_dtype(torch.bfloat16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} cannot be the same as the current layer"
    environment.set_vllm_config(get_vllm_config())
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name=layer_1,
            )
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_without_kv_sharing(default_vllm_config: None):
    torch.set_default_dtype(torch.bfloat16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    vllm_config = get_vllm_config()
    environment.set_vllm_config(vllm_config)
    with set_current_vllm_config(vllm_config):
        fwd_context = {
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            )
        }
        # suppress var not used error
        assert fwd_context is not None
    # Set high context length to test max context length estimation
    vllm_config.model_config.max_model_len = 3_000_000
    vllm_ctx = vllm_config.compilation_config.static_forward_context
    runner = HPUModelRunner(vllm_config, DEVICE)
    kv_cache_spec = runner.get_kv_cache_spec()
    assert len(kv_cache_spec) == 2
    #assert len(runner.shared_kv_cache_layers) == 0

    available_memory = 20 * GiB_bytes
    # page size for layer 0's kv_cache_spec *should be* 256KB:
    # block_size*num_heads*head_size*dtype_size*coeff = 128*8*64*2*2 = 262144
    page_size = kv_cache_spec[layer_0].page_size_bytes
    assert page_size == 262144
    num_expected_blocks = 40960  # 20GB / 256KB (page_size) / 2 (num layers)
    assert num_expected_blocks == available_memory // page_size // 2
    kv_cache_config = get_kv_cache_configs(vllm_config, [kv_cache_spec], [available_memory])[0]
    assert kv_cache_config.num_blocks == num_expected_blocks
    # PR #51718 coalesces same-spec layers into ONE backing KVCacheTensor
    # (layers=[layer_0, layer_1]) whose size spans the full allocation.
    assert len(kv_cache_config.kv_cache_tensors) == 1
    assert kv_cache_config.kv_cache_tensors[0].size == available_memory

    max_context_len =\
        estimate_max_model_len(vllm_config, kv_cache_spec, 5 * GiB_bytes)
    # max context len with KV sharing should be 2x as large as without
    assert max_context_len == 1310720

    # important: override tensor size to prevent large mem alloc during test
    # this will only allocate one block worth of memory per layer
    kv_cache_config.num_blocks = 1
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        page_size = kv_cache_spec[kv_cache_tensor.layers[0]].page_size_bytes
        kv_cache_tensor.size = page_size * len(kv_cache_tensor.layers)
        kv_cache_tensor.layer_stride = page_size
        kv_cache_tensor.block_stride = page_size

    runner.initialize_kv_cache(kv_cache_config)

    layer_0_kv = vllm_ctx[layer_0].kv_cache[0]
    layer_1_kv = vllm_ctx[layer_1].kv_cache[0]
    # check layer 1 kv cache does NOT share memory with layer 0
    assert id(layer_1_kv) != id(layer_0_kv)

    # check layer 1 added to kv cache group's layer names
    assert len(kv_cache_config.kv_cache_groups) == 1
    assert len(kv_cache_config.kv_cache_groups[0].layer_names) == 2
    assert kv_cache_config.kv_cache_groups[0].layer_names[0] == layer_0
    assert kv_cache_config.kv_cache_groups[0].layer_names[1] == layer_1


def test_init_kv_cache_with_kv_sharing_valid(default_vllm_config: None):
    torch.set_default_dtype(torch.bfloat16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    vllm_config = get_vllm_config()
    environment.set_vllm_config(vllm_config)
    with set_current_vllm_config(vllm_config):
        fwd_context = {
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name="model.layers.0.self_attn.attn",
            )
        }
        # suppress var not used error
        assert fwd_context is not None
    # Set high context length to test max context length estimation
    vllm_config.model_config.max_model_len = 3_000_000
    vllm_ctx = vllm_config.compilation_config.static_forward_context
    runner = HPUModelRunner(vllm_config, DEVICE)
    kv_cache_spec = runner.get_kv_cache_spec()
    assert len(kv_cache_spec) == 1
    assert layer_0 in kv_cache_spec
    assert runner.shared_kv_cache_layers[layer_1] == layer_0

    available_memory = 20 * GiB_bytes
    # page size for layer 0's kv_cache_spec is 256KB
    # with KV sharing, we can allocate (available_mem//page_size//1) blocks
    # which is twice as many as without KV sharing
    page_size = 128 * 8 * 64 * 2 * 2  # 128 for block_size, 2 for K+V, 2 for bfloat16
    num_expected_blocks = available_memory / page_size  # 20GB / 256KB
    kv_cache_config = get_kv_cache_configs(vllm_config, [kv_cache_spec], [available_memory])[0]
    assert kv_cache_config.num_blocks == num_expected_blocks
    assert len(kv_cache_config.kv_cache_tensors) == 1
    # Each layer now has twice the available memory for KV cache
    # compared to no KV sharing
    assert kv_cache_config.kv_cache_tensors[0].size == available_memory

    max_context_len =\
        estimate_max_model_len(vllm_config, kv_cache_spec, 5 * GiB_bytes)
    # max context len with KV sharing should be 2x as large as without
    assert max_context_len == 2 * 1310720

    # important: override tensor size to prevent large mem alloc during test
    # this will only allocate 1 block worth of memory (32kb)
    kv_cache_config.num_blocks = 1
    kv_cache_config.kv_cache_tensors[0].size =\
        kv_cache_spec[layer_0].page_size_bytes

    runner.initialize_kv_cache(kv_cache_config)

    layer_0_kv = vllm_ctx[layer_0].kv_cache[0]
    layer_1_kv = vllm_ctx[layer_1].kv_cache[0]
    # check layer 1 kv cache shares memory with layer 0
    assert id(layer_1_kv) == id(layer_0_kv)

    # check layer 1 added to kv cache group's layer names
    assert len(kv_cache_config.kv_cache_groups) == 1
    assert len(kv_cache_config.kv_cache_groups[0].layer_names) == 2
    assert kv_cache_config.kv_cache_groups[0].layer_names[0] == layer_0
    assert kv_cache_config.kv_cache_groups[0].layer_names[1] == layer_1


@pytest.mark.skipif(is_lazy(), reason="Test skipped because lazy mode is enabled.")
def test_model_torch_regional_compilation(default_vllm_config: None, dist_init, model_runner):
    from vllm_gaudi.utils import HPUCompileConfig
    from vllm.model_executor.models.opt import OPTDecoderLayer
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding  # noqa
    from torch.nn.modules.normalization import LayerNorm
    from torch._dynamo.eval_frame import OptimizedModule

    def assert_compilation(model, layer_name, module):
        submodule = model.get_submodule(layer_name)
        assert isinstance(submodule, OptimizedModule), (
            f"Layer: '{module.__name__}' was not wrapped with OptimizedModule"  # noqa
        )
        assert isinstance(submodule._orig_mod, module), (
            f"_orig_mod is different from the original module: '{module.__name__}'"  # noqa
        )

    vllm_config = get_vllm_config()
    model = get_model(vllm_config=vllm_config)
    model_runner.compile_config = HPUCompileConfig()
    model_runner.regional_compilation_layers_list = [LayerNorm, VocabParallelEmbedding]

    model_runner._regional_compilation(model)

    for i in range(len(model.get_submodule("model.decoder.layers"))):
        assert_compilation(model, f"model.decoder.layers.{i}", OPTDecoderLayer)
    assert_compilation(model, "lm_head", VocabParallelEmbedding)
    assert_compilation(model, "model.decoder.final_layer_norm", LayerNorm)
    assert_compilation(model, "model.decoder.embed_tokens", VocabParallelEmbedding)


def test_mamba_cache_groups_handle_whole_model_compile_wrapper():
    base_model = torch.nn.Module()
    base_model.config = SimpleNamespace(architectures=[])

    adapter = object.__new__(HpuModelAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.model = base_model

    compiled_model = SimpleNamespace(_orig_mod=adapter)
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])

    maybe_set_mamba_kv_cache_groups_ids(compiled_model, kv_cache_config)


@pytest.mark.parametrize(
    ("use_async", "num_mamba_layers", "num_prefills", "expected"),
    [
        (True, 48, 1, True),
        (True, 48, 0, False),
        (True, 0, 1, False),
        (False, 48, 1, False),
    ],
)
def test_should_synchronize_hybrid_prefill_output(use_async, num_mamba_layers, num_prefills, expected):
    assert should_synchronize_hybrid_prefill_output(use_async, num_mamba_layers, num_prefills) is expected


def test_cache_block_capacity_keeps_hybrid_block_units_separate():
    runner = object.__new__(HPUModelRunner)
    runner.enable_bucketing = True
    runner.bucketing_manager = SimpleNamespace(num_hpu_blocks=None)
    runner.attn_block_size = 128

    runner._set_cache_block_capacity(
        scheduler_blocks=862,
        attention_kernel_blocks=862 * 7,
    )

    assert runner.bucketing_manager.num_hpu_blocks == 6034
    assert runner._PAD_BLOCK_ID == 6034
    assert runner._PAD_SLOT_ID == 6034 * 128
    assert runner._MAMBA_PAD_BLOCK_ID == 862
    assert runner._dummy_num_blocks == 862


def test_direct_gdn_state_accepts_contiguous_request_prefix_and_free_padding():
    runner = object.__new__(HPUModelRunner)
    runner._direct_gdn_state_enabled = True
    runner._compact_gdn_enabled = True
    runner.use_prefix_caching = False
    runner._compact_gdn_group_ids = {1, 3}
    runner._compact_gdn_group_offset = {1: 0, 3: 1}
    runner._gdn_max_reqs = 4
    runner._gdn_slot_free_list = []

    indices = torch.zeros(4, 4, dtype=torch.int32)
    indices[1] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    indices[3] = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    assert runner._can_use_direct_gdn_state(indices, num_indices=4, target_bs=4)
    assert not runner._can_use_direct_gdn_state(indices, num_indices=4, target_bs=4, tokens_per_request=2)

    padded_indices = indices.clone()
    padded_indices[1] = torch.tensor([1, 2, 3, -1], dtype=torch.int32)
    padded_indices[3] = torch.tensor([5, 6, 7, -1], dtype=torch.int32)
    runner._gdn_slot_free_list = [3]
    assert runner._can_use_direct_gdn_state(padded_indices, num_indices=3, target_bs=4)

    runner._gdn_slot_free_list = []
    assert not runner._can_use_direct_gdn_state(padded_indices, num_indices=3, target_bs=4)

    runner._gdn_slot_free_list = [3]
    padded_indices[3, 3] = 8
    assert not runner._can_use_direct_gdn_state(padded_indices, num_indices=3, target_bs=4)

    indices[3] = torch.tensor([6, 5, 7, 8], dtype=torch.int32)
    assert not runner._can_use_direct_gdn_state(indices, num_indices=4, target_bs=4)


def test_model_adapter_selects_hidden_states_inside_logits_region():
    adapter = SimpleNamespace(model=SimpleNamespace(compute_logits=lambda hidden_states: hidden_states * 3), )
    hidden_states = torch.arange(24).view(2, 3, 4)
    logits_indices = torch.tensor([1, 4])

    selected, logits = HpuModelAdapter.select_and_compute_logits(
        adapter,
        hidden_states,
        logits_indices,
    )

    expected = hidden_states.view(-1, 4)[logits_indices]
    assert torch.equal(selected, expected)
    assert torch.equal(logits, expected * 3)


def test_model_adapter_fuses_plain_greedy_with_logits_region():
    adapter = SimpleNamespace(model=SimpleNamespace(compute_logits=lambda hidden_states: hidden_states * 3), )
    hidden_states = torch.arange(24).view(2, 3, 4)
    logits_indices = torch.tensor([1, 4])

    selected, sampled_token_ids = HpuModelAdapter.select_compute_logits_and_greedy(
        adapter,
        hidden_states,
        logits_indices,
    )

    expected = hidden_states.view(-1, 4)[logits_indices]
    assert torch.equal(selected, expected)
    assert torch.equal(sampled_token_ids, torch.tensor([[3], [3]], dtype=torch.int32))


def test_plain_greedy_fusion_is_strictly_gated(monkeypatch):
    runner = object.__new__(HPUModelRunner)
    runner.speculative_config = None
    runner.use_structured_output = False
    runner.input_batch = SimpleNamespace(logitsprocs=SimpleNamespace(non_argmax_invariant=[
        object.__new__(model_runner_module.MinTokensLogitsProcessor),
        object.__new__(model_runner_module.LogitBiasLogitsProcessor),
    ]))
    runner.requests = {"req": SimpleNamespace(sampling_params=SamplingParams(temperature=0.0))}

    monkeypatch.setenv("VLLM_HPU_FUSED_GREEDY_LOGITS", "true")
    assert runner._can_fuse_plain_greedy_sampling(["req"])

    runner.requests["req"].sampling_params = SamplingParams(temperature=0.0, logprobs=1)
    assert not runner._can_fuse_plain_greedy_sampling(["req"])

    runner.requests["req"].sampling_params = SamplingParams(temperature=0.7)
    assert not runner._can_fuse_plain_greedy_sampling(["req"])

    runner.requests["req"].sampling_params = SamplingParams(temperature=0.0)
    runner.input_batch.logitsprocs.non_argmax_invariant.append(SimpleNamespace())
    assert not runner._can_fuse_plain_greedy_sampling(["req"])

    monkeypatch.setenv("VLLM_HPU_FUSED_GREEDY_LOGITS", "false")
    runner.input_batch.logitsprocs.non_argmax_invariant.pop()
    assert not runner._can_fuse_plain_greedy_sampling(["req"])


def test_max_cudagraph_capture_size_defaults_to_max_num_batched_tokens(model_runner):
    """max_cudagraph_capture_size defaults to max_num_batched_tokens when not configured."""
    assert model_runner.max_cudagraph_capture_size == model_runner.max_num_batched_tokens


def test_max_cudagraph_capture_size_uses_explicit_value():
    """max_cudagraph_capture_size uses the configured value when explicitly set."""
    vllm_config = get_vllm_config()
    vllm_config.compilation_config.max_cudagraph_capture_size = 256
    with set_current_vllm_config(vllm_config):
        environment.set_vllm_config(vllm_config)
        num_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = vllm_config.model_config.get_head_size()
        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(num_heads, head_size, 0.1)
        runner = HPUModelRunner(vllm_config, DEVICE)
        assert runner.max_cudagraph_capture_size == 256


@pytest.mark.parametrize(
    "is_prompt,batch_size,seq_len,num_blocks,block_size,max_capture,expected",
    [
        # Prefill within limits → use graphs
        (True, 1, 128, 0, 128, 512, True),
        # Prefill exceeding limits → skip graphs
        (True, 1, 256, 4, 128, 512, False),
        # Prefill at exact boundary → use graphs
        (True, 1, 256, 2, 128, 512, True),
        # Prefill just over boundary → skip graphs
        (True, 1, 256, 2, 128, 511, False),
        # Decode never skips graphs even with many tokens
        (False, 256, 1, 100, 128, 512, True),
        # Decode with many blocks → still use graphs
        (False, 64, 1, 1000, 128, 512, True),
    ])
def test_use_graphs(model_runner, is_prompt, batch_size, seq_len, num_blocks, block_size, max_capture, expected):
    model_runner.max_cudagraph_capture_size = max_capture
    attn_metadata = SimpleNamespace(is_prompt=is_prompt,
                                    block_size=block_size,
                                    seq_len=lambda: seq_len,
                                    num_blocks=lambda: num_blocks)
    result = model_runner._use_graphs(attn_metadata, batch_size)
    assert result == expected


def test_use_graphs_enforce_eager(model_runner):
    """When enforce_eager is set, never use graphs."""
    orig = model_runner.model_config.enforce_eager
    try:
        model_runner.model_config.enforce_eager = True
        attn_metadata = SimpleNamespace(is_prompt=False, block_size=128, seq_len=lambda: 1, num_blocks=lambda: 0)
        assert model_runner._use_graphs(attn_metadata, 1) is False
    finally:
        model_runner.model_config.enforce_eager = orig
