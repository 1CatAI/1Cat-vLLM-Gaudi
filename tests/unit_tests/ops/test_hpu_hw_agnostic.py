# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import vllm_gaudi.ops.hpu_hw_agnostic as hpu_hw_agnostic
import vllm_gaudi.ops.hpu_weights as hpu_weights

from vllm_gaudi.ops.hpu_hw_agnostic import (
    HPUHwAgnosticFp8LinearMethod,
    HPUHwAgnosticMxfp4MoEMethod,
    _hpu_bf16_mla_sparse_interface,
    _hpu_combine_topk_swa_indices,
    _hpu_deepseek_score_projection,
    _hpu_compiled_attention_frontend,
    _hpu_dsv4_can_direct_decode_dispatch,
    _hpu_fill_short_context_topk_indices,
    _hpu_gate_linear_forward,
    _hpu_hw_agnostic_moe_runner_forward,
    _hpu_dsv4_inv_rope_einsum,
    _hpu_dsv4_is_compile_only,
)
from vllm_gaudi.ops.hpu_weights import (
    _cache_hpu_dsv4_attention_block_fp8_weight,
    _cache_hpu_dsv4_compressor_norm_weight,
    _cache_hpu_router_weight,
    _cache_hpu_shared_expert_block_fp8_weight,
)
from vllm_gaudi.extension.ops import _gather_mxfp4_decode_weights
from vllm_gaudi.ops.deepseek_v4_flashmla import (
    HPUFlashMLABackend,
    HPUFlashMLALayerType,
    build_hpu_flashmla_decode_plan,
    flashmla_output_buffer_is_compatible,
    parse_hpu_flashmla_backend,
)


def test_dsv4_compressor_norm_weight_cache_is_exact_fp32():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(
        torch.linspace(-1, 1, 512, dtype=torch.bfloat16),
        requires_grad=False,
    )

    cached_bytes = _cache_hpu_dsv4_compressor_norm_weight(
        "model.layers.2.self_attn.attn.compressor.norm",
        module,
    )

    assert cached_bytes == 512 * 4
    assert module._hpu_dsv4_weight_fp32.dtype == torch.float32
    assert torch.equal(
        module._hpu_dsv4_weight_fp32,
        module.weight.detach().float(),
    )
    assert "_hpu_dsv4_weight_fp32" not in module.state_dict()


def test_dsv4_compressor_norm_weight_cache_ignores_other_modules():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(
        torch.ones(512, dtype=torch.bfloat16),
        requires_grad=False,
    )

    assert not _cache_hpu_dsv4_compressor_norm_weight(
        "model.layers.2.self_attn.q_norm",
        module,
    )
    assert not hasattr(module, "_hpu_dsv4_weight_fp32")


def _set_dsv4_attention_context(
    monkeypatch,
    indexer,
    *,
    num_decodes: int,
    num_prefills: int,
) -> None:
    indexer.prefix = "model.layers.0.self_attn.indexer"
    swa_metadata = SimpleNamespace(
        num_decodes=num_decodes,
        num_prefills=num_prefills,
    )
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                "model.layers.0.self_attn.swa_cache": swa_metadata,
            }
        ),
    )


def test_dsv4_compile_only_mode_tracks_bridge_state(monkeypatch):
    from habana_frameworks.torch.internal import bridge_config

    monkeypatch.setattr(
        bridge_config,
        "get_pt_compile_only_mode",
        lambda: True,
    )

    assert _hpu_dsv4_is_compile_only()


def test_dsv4_compile_only_process_guard_overrides_thread_state(monkeypatch):
    from habana_frameworks.torch.internal import bridge_config

    monkeypatch.setattr(
        bridge_config,
        "get_pt_compile_only_mode",
        lambda: False,
    )
    previous = hpu_hw_agnostic._set_hpu_dsv4_compile_only(True)
    try:
        assert _hpu_dsv4_is_compile_only()
    finally:
        hpu_hw_agnostic._set_hpu_dsv4_compile_only(previous)


def test_dsv4_qnorm_tpc_prewarm_state(monkeypatch):
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_DSV4_QNORM_TPC_PREWARMED",
        False,
    )

    previous = hpu_hw_agnostic._set_hpu_dsv4_qnorm_tpc_prewarmed(True)

    assert not previous
    assert hpu_hw_agnostic._DSV4_QNORM_TPC_PREWARMED


def test_flashmla_backend_prefers_direct_packed_fp8_path():
    plan = build_hpu_flashmla_decode_plan(
        requested_backend="auto",
        swa_only=False,
        has_local_topk=True,
        uses_sequential_topk=True,
        tpc_enabled=True,
        mme_enabled=True,
        tpc_eligible=True,
        mme_eligible=True,
    )

    assert plan.backend == HPUFlashMLABackend.FLASHMLA
    assert plan.layer_type == HPUFlashMLALayerType.C4
    assert plan.uses_local_topk
    assert plan.uses_sequential_topk


def test_flashmla_backend_falls_back_to_mme_for_swa_only():
    plan = build_hpu_flashmla_decode_plan(
        requested_backend="flashmla_hpu",
        swa_only=True,
        has_local_topk=False,
        uses_sequential_topk=False,
        tpc_enabled=True,
        mme_enabled=True,
        tpc_eligible=False,
        mme_eligible=True,
    )

    assert plan.backend == HPUFlashMLABackend.MME
    assert plan.layer_type == HPUFlashMLALayerType.SWA_ONLY


def test_flashmla_accepts_unpadded_query_in_padded_workspace():
    q = torch.empty(1, 32, 512, dtype=torch.bfloat16)
    out = torch.empty(1, 64, 512, dtype=torch.bfloat16)

    assert flashmla_output_buffer_is_compatible(q, out)
    assert not flashmla_output_buffer_is_compatible(q, out[:, :31])


def test_flashmla_backend_parser_validates_values():
    assert parse_hpu_flashmla_backend("tpc") == HPUFlashMLABackend.FLASHMLA
    with pytest.raises(ValueError, match="VLLM_HPU_DSV4_ATTENTION_BACKEND"):
        parse_hpu_flashmla_backend("unknown")


def test_flashmla_direct_decode_dispatch_gate(monkeypatch):
    q = torch.empty(1, 32, 512, dtype=torch.bfloat16)
    out = torch.empty(1, 64, 512, dtype=torch.bfloat16)
    monkeypatch.setenv("VLLM_HPU_DSV4_DIRECT_DECODE_DISPATCH", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_ATTENTION_BACKEND", "auto")

    assert _hpu_dsv4_can_direct_decode_dispatch(
        q, out, swa_only=False
    )
    assert _hpu_dsv4_can_direct_decode_dispatch(
        q, out, swa_only=True
    )

    monkeypatch.setenv("VLLM_HPU_DSV4_ATTENTION_BACKEND", "mme")
    assert not _hpu_dsv4_can_direct_decode_dispatch(
        q, out, swa_only=False
    )


def test_ordered_c128_compressor_requires_dedicated_gate(monkeypatch):
    q = torch.empty(1, 32, 512, dtype=torch.bfloat16)
    out = torch.empty(1, 64, 512, dtype=torch.bfloat16)
    mla_attn = SimpleNamespace(
        compress_ratio=128,
        swa_cache_layer=SimpleNamespace(prefix="swa"),
        prefix="mla",
    )
    metadata = {
        "swa": SimpleNamespace(
            num_prefills=0,
            num_decodes=1,
            num_decode_tokens=1,
            decode_swa_indices=torch.zeros(1, 1, dtype=torch.int32),
            decode_swa_lens=torch.ones(1, dtype=torch.int32),
        ),
        "mla": SimpleNamespace(
            c128a_global_decode_topk_indices=torch.zeros(
                1, 1, dtype=torch.int32
            ),
            c128a_decode_topk_lens=torch.ones(1, dtype=torch.int32),
        ),
    }
    monkeypatch.setenv("VLLM_HPU_DSV4_TPC_ORDERED_COMPRESSOR", "1")
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_hpu_dsv4_can_direct_decode_dispatch",
        lambda *args, **kwargs: True,
    )

    assert not hpu_hw_agnostic._hpu_dsv4_can_order_compressor_decode(
        mla_attn, q, out, metadata
    )
    monkeypatch.setenv(
        "VLLM_HPU_DSV4_TPC_ORDERED_C128_COMPRESSOR", "1"
    )
    assert hpu_hw_agnostic._hpu_dsv4_can_order_compressor_decode(
        mla_attn, q, out, metadata
    )


def test_dsv4_swa_metadata_uses_native_int32_validity(monkeypatch):
    builder = SimpleNamespace()

    def original_init(instance):
        instance.is_valid_token = torch.zeros(8, dtype=torch.bool)

    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_ORIGINAL_DEEPSEEK_V4_SWA_METADATA_BUILDER_INIT",
        original_init,
    )

    hpu_hw_agnostic._hpu_deepseek_v4_swa_metadata_builder_init(builder)

    assert builder.is_valid_token.dtype == torch.int32
    assert builder.is_valid_token.shape == (8,)


def test_short_indexer_can_skip_unread_cache_write(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_SKIP", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN", "0")
    monkeypatch.setenv("VLLM_HPU_DSV4_MME_PAGED_SPARSE_ATTN", "0")

    def unexpected_compressor(*args):
        raise AssertionError("unused indexer cache write was executed")

    indexer = SimpleNamespace(
        max_model_len=8,
        topk_tokens=8,
        compress_ratio=4,
        topk_indices_buffer=torch.empty((1, 8), dtype=torch.int32),
        compressor=unexpected_compressor,
    )
    _set_dsv4_attention_context(
        monkeypatch, indexer, num_decodes=1, num_prefills=0
    )
    positions = torch.tensor([7], dtype=torch.int32)

    result = hpu_hw_agnostic._hpu_deepseek_v4_indexer_forward(
        indexer,
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        None,
        positions,
        None,
    )

    torch.testing.assert_close(
        result,
        torch.tensor([[0, 1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
    )


def test_attention_core_materializes_topk_without_running_indexer(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_SKIP", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP", "1")
    materialized = []
    attention_outputs = []

    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_hpu_materialize_short_indexer_topk",
        lambda indexer, positions: materialized.append((indexer, positions)),
    )

    indexer = SimpleNamespace(max_model_len=128, topk_tokens=512)
    _set_dsv4_attention_context(
        monkeypatch, indexer, num_decodes=1, num_prefills=0
    )
    layer = SimpleNamespace(
        n_local_heads=1,
        head_dim=1,
        compressor=lambda *args: None,
        rotary_emb=None,
        indexer=indexer,
        indexer_rotary_emb=None,
        _fused_qnorm_rope_kv_insert=lambda q, kv, positions, metadata: q,
        mla_attn=lambda q, kv, positions, output: attention_outputs.append(
            output
        ),
    )
    hidden_states = torch.ones((1, 1))
    positions = torch.tensor([7], dtype=torch.int32)
    output = torch.empty((1, 1, 1))
    frontend = (
        torch.ones((1, 1)),
        torch.ones((1, 1)),
        torch.ones((1, 1)),
        torch.ones((1, 1)),
        None,
    )

    hpu_hw_agnostic._hpu_dsv4_execute_attention_core(
        layer, hidden_states, positions, output, frontend
    )

    assert materialized == [(indexer, positions)]
    assert attention_outputs == [output]


def test_attention_core_skips_topk_fill_for_sequential_kernel(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_SKIP", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_TPC_PAGED_SPARSE_ATTN", "1")
    materialized = []
    attention_outputs = []
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_hpu_materialize_short_indexer_topk",
        lambda indexer, positions: materialized.append((indexer, positions)),
    )
    indexer = SimpleNamespace(
        max_model_len=128,
        topk_tokens=512,
        prefix="model.layers.0.self_attn.indexer",
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=1),
        ),
    )
    _set_dsv4_attention_context(
        monkeypatch, indexer, num_decodes=1, num_prefills=0
    )
    layer = SimpleNamespace(
        n_local_heads=1,
        head_dim=1,
        compressor=lambda *args: None,
        rotary_emb=None,
        indexer=indexer,
        indexer_rotary_emb=None,
        _fused_qnorm_rope_kv_insert=lambda q, kv, positions, metadata: q,
        mla_attn=lambda q, kv, positions, output: attention_outputs.append(
            output
        ),
    )
    output = torch.empty((1, 1, 1))
    frontend = tuple(torch.ones((1, 1)) for _ in range(4)) + (None,)

    hpu_hw_agnostic._hpu_dsv4_execute_attention_core(
        layer,
        torch.ones((1, 1)),
        torch.tensor([7], dtype=torch.int32),
        output,
        frontend,
    )

    assert materialized == []
    assert attention_outputs == [output]


def test_attention_core_fused_compressor_flashmla_skips_separate_ops(
    monkeypatch,
):
    fused_calls = []
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={}),
    )
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_hpu_dsv4_can_fuse_compressor_flashmla",
        lambda *args: True,
    )
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_hpu_dsv4_fused_compressor_flashmla_decode",
        lambda *args: fused_calls.append(args),
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("separate Compressor/MLA path was executed")

    layer = SimpleNamespace(
        n_local_heads=1,
        head_dim=1,
        compressor=unexpected,
        rotary_emb=None,
        indexer=None,
        mla_attn=unexpected,
        _fused_qnorm_rope_kv_insert=lambda q, kv, positions, metadata: q,
    )
    output = torch.empty((1, 1, 1))
    positions = torch.tensor([7], dtype=torch.int32)
    kv_score = torch.ones((1, 2))

    hpu_hw_agnostic._hpu_dsv4_execute_attention_core(
        layer,
        torch.ones((1, 1)),
        positions,
        output,
        (
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            kv_score,
            None,
        ),
    )

    assert len(fused_calls) == 1
    assert fused_calls[0][2] is kv_score
    assert fused_calls[0][3] is positions
    assert fused_calls[0][4] is output


def test_attention_core_fused_qnorm_compressor_keeps_mla_downstream(
    monkeypatch,
):
    fused_calls = []
    direct_calls = []
    dependency = torch.tensor([1], dtype=torch.int32)
    swa_metadata = SimpleNamespace()
    mla_metadata = SimpleNamespace()
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"swa": swa_metadata, "mla": mla_metadata}
        ),
    )
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_hpu_dsv4_can_fuse_qnorm_compressor",
        lambda *args: True,
    )

    def fused(*args):
        fused_calls.append(args)
        return args[1], dependency

    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_hpu_dsv4_fused_qnorm_compressor_decode",
        fused,
    )

    def direct(*args, **kwargs):
        direct_calls.append((args, kwargs))
        return True

    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_hpu_dsv4_try_direct_decode_dispatch",
        direct,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("separate QNorm/Compressor/MLA path was executed")

    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_hpu_dsv4_can_fuse_compressor_flashmla",
        unexpected,
    )
    mla_attn = SimpleNamespace(
        prefix="mla",
        swa_cache_layer=SimpleNamespace(prefix="swa"),
        kv_cache_storage=torch.empty(1),
        kv_cache_geometry=torch.empty(1),
    )
    layer = SimpleNamespace(
        n_local_heads=1,
        head_dim=1,
        compressor=unexpected,
        rotary_emb=None,
        indexer=None,
        mla_attn=mla_attn,
        _fused_qnorm_rope_kv_insert=unexpected,
    )
    output = torch.empty((1, 1, 1))
    positions = torch.tensor([7], dtype=torch.int32)
    kv_score = torch.ones((1, 2))

    hpu_hw_agnostic._hpu_dsv4_execute_attention_core(
        layer,
        torch.ones((1, 1)),
        positions,
        output,
        (
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            kv_score,
            None,
        ),
    )

    assert len(fused_calls) == 1
    assert len(direct_calls) == 1
    assert direct_calls[0][0][5] is mla_metadata
    assert direct_calls[0][0][7] is output
    assert direct_calls[0][1]["compressor_dependency"] is dependency


def test_deepseek_score_projection_bf16_mme_keeps_fp32_output(monkeypatch):
    hidden_states = torch.tensor(
        [[0.5, -0.25, 0.125, 0.75]], dtype=torch.bfloat16
    )
    weight = torch.tensor(
        [[0.25, 0.5, -0.75, 0.125], [-0.5, 0.25, 0.5, 0.75]],
        dtype=torch.bfloat16,
    )

    monkeypatch.setenv("VLLM_HPU_DSV4_BF16_SCORE_PROJECTION", "0")
    reference = _hpu_deepseek_score_projection(hidden_states, weight)
    monkeypatch.setenv("VLLM_HPU_DSV4_BF16_SCORE_PROJECTION", "1")
    candidate = _hpu_deepseek_score_projection(hidden_states, weight)

    assert candidate.dtype == torch.float32
    torch.testing.assert_close(candidate, reference, atol=2e-3, rtol=2e-3)


def test_dsv4_frontend_selects_bf16_compressor_scores(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV4_COMPILED_ATTN_FRONTEND", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_NATIVE_FP8_ATTN_FRONTEND", "0")
    monkeypatch.setenv("VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS", "1")

    cached = torch.ones((4, 4), dtype=torch.bfloat16)
    layer = SimpleNamespace(
        eps=1e-6,
        fused_wqa_wkv=SimpleNamespace(
            _hpu_block_fp8_weight_dequant_transposed=cached,
        ),
        wq_b=SimpleNamespace(
            _hpu_block_fp8_weight_dequant_transposed=cached,
        ),
        q_norm=SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16)),
        kv_norm=SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16)),
        compressor=SimpleNamespace(
            fused_wkv_wgate=SimpleNamespace(
                weight=torch.ones((8, 4), dtype=torch.bfloat16),
            ),
        ),
        indexer=None,
    )
    expected = tuple(
        torch.full((1, 1), value, dtype=torch.bfloat16)
        for value in range(4)
    )

    def bf16_frontend(*args):
        return expected

    def unexpected_f32_frontend(*args):
        raise AssertionError("F32 compressor frontend was selected")

    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_HPU_DSV4_FRONTEND_COMPRESSOR_BF16",
        bf16_frontend,
    )
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_HPU_DSV4_FRONTEND_COMPRESSOR",
        unexpected_f32_frontend,
    )

    result = _hpu_compiled_attention_frontend(
        layer,
        torch.ones((1, 4), dtype=torch.bfloat16),
    )

    assert result == (*expected, None)


def test_dsv4_frontend_skips_safe_short_indexer_projection(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV4_COMPILED_ATTN_FRONTEND", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_NATIVE_FP8_ATTN_FRONTEND", "0")
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_SKIP", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP", "1")

    cached = torch.ones((4, 4), dtype=torch.bfloat16)
    layer = SimpleNamespace(
        eps=1e-6,
        fused_wqa_wkv=SimpleNamespace(
            _hpu_block_fp8_weight_dequant_transposed=cached,
        ),
        wq_b=SimpleNamespace(
            _hpu_block_fp8_weight_dequant_transposed=cached,
        ),
        q_norm=SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16)),
        kv_norm=SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16)),
        compressor=SimpleNamespace(
            fused_wkv_wgate=SimpleNamespace(weight=cached),
        ),
        indexer=SimpleNamespace(
            max_model_len=128,
            topk_tokens=512,
            compressor=SimpleNamespace(
                fused_wkv_wgate=SimpleNamespace(weight=cached),
            ),
        ),
    )
    _set_dsv4_attention_context(
        monkeypatch, layer.indexer, num_decodes=1, num_prefills=0
    )
    expected = tuple(torch.full((1, 1), value) for value in range(4))

    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_HPU_DSV4_FRONTEND_COMPRESSOR",
        lambda *args: expected,
    )
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_HPU_DSV4_FRONTEND_COMPRESSOR_INDEXER",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("safe short indexer projection was executed")
        ),
    )

    result = _hpu_compiled_attention_frontend(
        layer,
        torch.ones((1, 4), dtype=torch.bfloat16),
    )

    assert result == (*expected, None)


@pytest.mark.parametrize(
    ("num_decodes", "num_prefills", "expected"),
    [
        (1, 0, True),
        (0, 1, False),
        (1, 1, False),
    ],
)
def test_short_indexer_cache_skip_is_decode_only(
    monkeypatch, num_decodes, num_prefills, expected
):
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_SKIP", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP", "1")
    indexer = SimpleNamespace(max_model_len=128, topk_tokens=512)
    _set_dsv4_attention_context(
        monkeypatch,
        indexer,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
    )

    assert (
        hpu_hw_agnostic._hpu_can_skip_short_indexer_cache(indexer)
        is expected
    )


def test_dsv4_frontend_falls_back_when_context_exceeds_topk(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV4_COMPILED_ATTN_FRONTEND", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_NATIVE_FP8_ATTN_FRONTEND", "0")
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_SKIP", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_CACHE_SKIP", "1")

    cached = torch.ones((4, 4), dtype=torch.bfloat16)
    layer = SimpleNamespace(
        eps=1e-6,
        fused_wqa_wkv=SimpleNamespace(
            _hpu_block_fp8_weight_dequant_transposed=cached,
        ),
        wq_b=SimpleNamespace(
            _hpu_block_fp8_weight_dequant_transposed=cached,
        ),
        q_norm=SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16)),
        kv_norm=SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16)),
        compressor=SimpleNamespace(
            fused_wkv_wgate=SimpleNamespace(weight=cached),
        ),
        indexer=SimpleNamespace(
            max_model_len=513,
            topk_tokens=512,
            compressor=SimpleNamespace(
                fused_wkv_wgate=SimpleNamespace(weight=cached),
            ),
        ),
    )
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_HPU_DSV4_FRONTEND_COMPRESSOR",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("unsafe indexer projection was skipped")
        ),
    )

    result = _hpu_compiled_attention_frontend(
        layer,
        torch.ones((1, 4), dtype=torch.bfloat16),
    )

    assert result is None


def test_dsv4_inline_frontend_selects_uncompiled_c4_path(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS", "0")
    monkeypatch.setenv("VLLM_HPU_DSV4_SHORT_INDEXER_SKIP", "1")
    cached = torch.ones((4, 4), dtype=torch.bfloat16)
    layer = SimpleNamespace(
        fused_wqa_wkv=SimpleNamespace(
            _hpu_block_fp8_weight_dequant_transposed=cached,
        ),
        wq_b=SimpleNamespace(
            _hpu_block_fp8_weight_dequant_transposed=cached,
        ),
        q_norm=SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16)),
        kv_norm=SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16)),
        compressor=SimpleNamespace(
            fused_wkv_wgate=SimpleNamespace(weight=cached),
        ),
        indexer=SimpleNamespace(
            max_model_len=128,
            topk_tokens=512,
            compressor=SimpleNamespace(
                fused_wkv_wgate=SimpleNamespace(weight=cached),
            ),
        ),
    )
    expected = tuple(torch.full((1, 1), value) for value in range(5))

    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_hpu_dsv4_frontend_compressor_indexer",
        lambda *args: expected,
    )

    result = hpu_hw_agnostic._hpu_inline_attention_frontend(
        layer, torch.ones((1, 4), dtype=torch.bfloat16)
    )

    assert result == expected


def test_dsv4_mixed_compressor_preserves_fp32_frontend(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV4_COMPILED_ATTN_FRONTEND", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_NATIVE_FP8_ATTN_FRONTEND", "0")
    monkeypatch.setenv("VLLM_HPU_DSV4_TPC_BF16_COMPRESS_INPUTS", "0")
    monkeypatch.setenv("VLLM_HPU_DSV4_TPC_MIXED_COMPRESS_INPUTS", "1")

    cached = torch.ones((4, 4), dtype=torch.bfloat16)
    layer = SimpleNamespace(
        eps=1e-6,
        fused_wqa_wkv=SimpleNamespace(
            _hpu_block_fp8_weight_dequant_transposed=cached,
        ),
        wq_b=SimpleNamespace(
            _hpu_block_fp8_weight_dequant_transposed=cached,
        ),
        q_norm=SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16)),
        kv_norm=SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16)),
        compressor=SimpleNamespace(
            fused_wkv_wgate=SimpleNamespace(
                weight=torch.ones((8, 4), dtype=torch.bfloat16),
            ),
        ),
        indexer=None,
    )
    expected = (
        torch.ones((1, 1), dtype=torch.bfloat16),
        torch.ones((1, 1), dtype=torch.bfloat16),
        torch.ones((1, 1), dtype=torch.bfloat16),
        torch.ones((1, 1), dtype=torch.float32),
    )

    def f32_frontend(*args):
        return expected

    def unexpected_bf16_frontend(*args):
        raise AssertionError("BF16 compressor frontend was selected")

    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_HPU_DSV4_FRONTEND_COMPRESSOR",
        f32_frontend,
    )
    monkeypatch.setattr(
        hpu_hw_agnostic,
        "_HPU_DSV4_FRONTEND_COMPRESSOR_BF16",
        unexpected_bf16_frontend,
    )

    result = _hpu_compiled_attention_frontend(
        layer,
        torch.ones((1, 4), dtype=torch.bfloat16),
    )

    assert result == (*expected, None)


def test_mxfp4_decode_gather_preserves_topk_order_and_remaps_experts():
    expert_ids = torch.tensor([[3, 0, 2]], dtype=torch.int32)
    router_weights = torch.tensor([[0.2, 0.5, 0.3]])
    stacked = torch.arange(4).view(4, 1)

    local_ids, sorted_weights, w13, w2, s13, s2 = (
        _gather_mxfp4_decode_weights(
            expert_ids,
            router_weights,
            stacked,
            stacked + 10,
            stacked + 20,
            stacked + 30,
        )
    )

    torch.testing.assert_close(local_ids, torch.tensor([[0, 1, 2]], dtype=torch.int32))
    torch.testing.assert_close(sorted_weights, router_weights)
    torch.testing.assert_close(w13.flatten(), torch.tensor([3, 0, 2]))
    torch.testing.assert_close(w2.flatten(), torch.tensor([13, 10, 12]))
    torch.testing.assert_close(s13.flatten(), torch.tensor([23, 20, 22]))
    torch.testing.assert_close(s2.flatten(), torch.tensor([33, 30, 32]))


def test_short_context_topk_selects_every_compressed_candidate():
    output = torch.empty((4, 8), dtype=torch.int32)
    positions = torch.tensor([0, 3, 7, 31], dtype=torch.int64)

    result = _hpu_fill_short_context_topk_indices(
        output,
        positions,
        topk_tokens=8,
        compress_ratio=4,
    )

    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1, -1],
                [0, 1, -1, -1, -1, -1, -1, -1],
                [0, 1, 2, 3, 4, 5, 6, 7],
            ],
            dtype=torch.int32,
        ),
    )


def test_hw_agnostic_moe_runner_bypasses_opaque_forward_entry(monkeypatch):
    class FakeRoutedExperts:
        def __init__(self):
            self.initialized = False

        def _ensure_moe_quant_config_init(self):
            self.initialized = True

        def __call__(self, **kwargs):
            return kwargs["x"] * 3

    class FakeRunner:
        def __init__(self):
            self.moe_config = SimpleNamespace(dp_size=1, pcp_size=1)
            self.routed_experts = FakeRoutedExperts()
            self.gate = None
            self.forward_entry_called = False
            self._shared_experts = SimpleNamespace(output=None)
            self._quant_method = SimpleNamespace(topk_indices_dtype=torch.int32)
            self.router = SimpleNamespace(select_experts=lambda **kwargs: (None, None))

        def apply_routed_input_transform(self, hidden_states):
            return hidden_states + 1, hidden_states

        def _maybe_pad_hidden_states(
            self,
            shared_experts_input,
            hidden_states,
        ):
            assert shared_experts_input is not None
            return hidden_states, None, None

        def _maybe_sync_shared_experts_stream(
            self,
            shared_experts_input,
        ):
            assert shared_experts_input is not None

        def _maybe_apply_shared_experts(self, shared_experts_input, order):
            self._shared_experts.output = (shared_experts_input + 1) * 2

        def _maybe_combine(self, shared_output, fused_output):
            return shared_output, fused_output

        def _maybe_apply_routed_scale_to_output(
            self,
            shared_output,
            fused_output,
        ):
            return shared_output, fused_output

        def apply_routed_output_transform(self, fused_output):
            return fused_output

        def _maybe_reduce_final_output(self, output, trunc_size):
            assert trunc_size is None
            return output

        def _forward_entry(self, *args, **kwargs):
            self.forward_entry_called = True
            raise AssertionError("opaque custom op must not be called")

    monkeypatch.setattr(hpu_hw_agnostic, "_HPU_MOE_QUALITY_DEBUG", False)
    monkeypatch.setattr(hpu_hw_agnostic, "_HPU_MOE_QUALITY_CALL_INDEX", 17)

    runner = FakeRunner()
    hidden_states = torch.ones(2, 4)
    output = _hpu_hw_agnostic_moe_runner_forward(
        runner,
        hidden_states,
        hidden_states,
    )

    assert runner.routed_experts.initialized
    assert not runner.forward_entry_called
    assert hpu_hw_agnostic._HPU_MOE_QUALITY_CALL_INDEX == 17
    torch.testing.assert_close(
        output,
        (hidden_states + 1) * 5,
    )


def test_shared_expert_block_fp8_weight_cache():
    layer = torch.nn.Module()
    layer.quant_method = SimpleNamespace(
        block_quant=True,
        quant_config=SimpleNamespace(
            weight_block_size=[2, 2]
        ),
    )
    layer.weight = torch.nn.Parameter(
        torch.arange(16, dtype=torch.bfloat16).view(4, 4),
        requires_grad=False,
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )
    layer._hpu_orig_M = 4
    layer._hpu_orig_N = 4

    cached_bytes = _cache_hpu_shared_expert_block_fp8_weight(
        "model.layers.0.ffn.shared_experts.gate_up_proj",
        layer,
        torch.bfloat16,
    )

    expected = torch.tensor(
        [
            [0.0, 1.0, 4.0, 6.0],
            [4.0, 5.0, 12.0, 14.0],
            [24.0, 27.0, 40.0, 44.0],
            [36.0, 39.0, 56.0, 60.0],
        ],
        dtype=torch.bfloat16,
    )
    assert cached_bytes == expected.numel() * expected.element_size()
    assert (
        "_hpu_block_fp8_weight_dequant_transposed"
        not in dict(layer.named_parameters())
    )
    torch.testing.assert_close(
        layer._hpu_block_fp8_weight_dequant_transposed,
        expected.T.contiguous(),
    )


def test_dsv4_attention_block_fp8_weight_cache(monkeypatch):
    layer = torch.nn.Module()
    layer.quant_method = SimpleNamespace(
        block_quant=True,
        quant_config=SimpleNamespace(weight_block_size=[2, 2]),
    )
    layer.weight = torch.nn.Parameter(
        torch.arange(16, dtype=torch.bfloat16).view(4, 4),
        requires_grad=False,
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )
    layer._hpu_orig_M = 4
    layer._hpu_orig_N = 4
    monkeypatch.setenv("VLLM_HPU_DSV4_BF16_ATTN_WEIGHT_CACHE", "1")

    cached_bytes = _cache_hpu_dsv4_attention_block_fp8_weight(
        "model.layers.1.attn.fused_wqa_wkv",
        layer,
        torch.bfloat16,
    )

    assert cached_bytes == 32
    assert layer._hpu_dsv4_bf16_attention_cache
    assert layer._hpu_block_fp8_weight_dequant_transposed.shape == (4, 4)

    wo_b_layer = torch.nn.Module()
    wo_b_layer.quant_method = layer.quant_method
    wo_b_layer.weight = layer.weight
    wo_b_layer.weight_scale_inv = layer.weight_scale_inv
    wo_b_layer._hpu_orig_M = 4
    wo_b_layer._hpu_orig_N = 4
    wo_b_cached_bytes = _cache_hpu_dsv4_attention_block_fp8_weight(
        "model.layers.1.attn.wo_b",
        wo_b_layer,
        torch.bfloat16,
    )
    assert wo_b_cached_bytes == 32
    assert wo_b_layer._hpu_block_fp8_weight_dequant_transposed.shape == (4, 4)


def test_dsv4_wo_a_block_fp8_weight_cache(monkeypatch):
    layer = torch.nn.Module()
    layer.quant_method = SimpleNamespace(
        block_quant=True,
        quant_config=SimpleNamespace(weight_block_size=[2, 2]),
    )
    layer.weight = torch.nn.Parameter(
        torch.arange(32, dtype=torch.bfloat16).view(8, 4),
        requires_grad=False,
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.ones(4, 2, dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer._hpu_orig_M = 8
    layer._hpu_orig_N = 4
    layer.bmm_batch_size = 2
    monkeypatch.setenv("VLLM_HPU_DSV4_BF16_ATTN_WEIGHT_CACHE", "1")

    cached_bytes = _cache_hpu_dsv4_attention_block_fp8_weight(
        "model.layers.1.attn.wo_a",
        layer,
        torch.bfloat16,
    )

    assert cached_bytes == 64
    assert layer._hpu_dsv4_wo_a_bmm_weight.shape == (2, 4, 4)
    assert not hasattr(layer, "_hpu_block_fp8_weight_dequant_transposed")


def test_dsv4_cached_wo_a_matches_reference():
    rotary_emb = SimpleNamespace(
        cos_sin_cache=torch.tensor(
            [[1.0, 0.5, 0.25, -0.75]], dtype=torch.bfloat16
        )
    )
    output = torch.arange(16, dtype=torch.bfloat16).view(1, 4, 4) / 8
    positions = torch.tensor([0], dtype=torch.int32)
    weight = torch.arange(48, dtype=torch.bfloat16).view(2, 3, 8) / 16
    wo_a = torch.nn.Module()
    wo_a.weight = torch.nn.Parameter(
        weight.view(6, 8), requires_grad=False
    )
    wo_a.register_buffer(
        "_hpu_dsv4_wo_a_bmm_weight", weight, persistent=False
    )

    reference = hpu_hw_agnostic._ORIGINAL_DEEPSEEK_V4_INV_ROPE_EINSUM(
        rotary_emb, output, positions, 4, 2, 3, wo_a
    )
    candidate = _hpu_dsv4_inv_rope_einsum(
        rotary_emb, output, positions, 4, 2, 3, wo_a
    )

    torch.testing.assert_close(candidate, reference, atol=0, rtol=0)


def test_dsv4_attention_native_fp8_weight_cache(monkeypatch):
    layer = torch.nn.Module()
    layer.quant_method = SimpleNamespace(
        block_quant=True,
        quant_config=SimpleNamespace(weight_block_size=[2, 2]),
    )
    layer.weight = torch.nn.Parameter(
        torch.arange(16, dtype=torch.bfloat16).view(4, 4),
        requires_grad=False,
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.ones(2, 2, dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer._hpu_orig_M = 4
    layer._hpu_orig_N = 4
    monkeypatch.setenv("VLLM_HPU_DSV4_BF16_ATTN_WEIGHT_CACHE", "0")
    monkeypatch.setenv("VLLM_HPU_DSV4_NATIVE_FP8_ATTN_FRONTEND", "1")

    def fake_dynamic_quant(value):
        return (
            value.to(torch.float8_e4m3fn),
            torch.ones(value.shape[0], 1, dtype=torch.float32),
        )

    monkeypatch.setattr(hpu_weights.hpu_ops, "dynamic_quant", fake_dynamic_quant)
    cached_bytes = _cache_hpu_dsv4_attention_block_fp8_weight(
        "model.layers.1.attn.fused_wqa_wkv",
        layer,
        torch.bfloat16,
    )

    assert cached_bytes == 32
    assert layer._hpu_dsv4_native_fp8_weight.dtype == torch.float8_e4m3fn
    assert layer._hpu_dsv4_native_fp8_weight.shape == (4, 4)
    assert layer._hpu_dsv4_native_fp8_weight_scale.shape == (4,)
    assert not hasattr(layer, "_hpu_dsv4_bf16_attention_cache")


def test_hw_agnostic_fp8_linear_uses_cached_weight():
    method = HPUHwAgnosticFp8LinearMethod.__new__(
        HPUHwAgnosticFp8LinearMethod
    )
    method.block_quant = True
    method.quant_config = SimpleNamespace(
        weight_block_size=[2, 2]
    )

    layer = torch.nn.Module()
    layer.register_buffer(
        "_hpu_block_fp8_weight_dequant_transposed",
        torch.tensor(
            [[1.0, 3.0], [2.0, 4.0]],
            dtype=torch.bfloat16,
        ),
        persistent=False,
    )
    x = torch.tensor(
        [[2.0, 1.0]], dtype=torch.bfloat16
    )

    output = method.apply(layer, x)

    torch.testing.assert_close(
        output,
        torch.tensor(
            [[4.0, 10.0]], dtype=torch.bfloat16
        ),
    )


def test_dsv4_attention_cache_is_decode_only(monkeypatch):
    method = HPUHwAgnosticFp8LinearMethod.__new__(
        HPUHwAgnosticFp8LinearMethod
    )
    method.block_quant = True
    method.quant_config = SimpleNamespace(weight_block_size=[2, 2])
    layer = torch.nn.Module()
    layer._hpu_dsv4_bf16_attention_cache = True
    layer.register_buffer(
        "_hpu_block_fp8_weight_dequant_transposed",
        torch.eye(2, dtype=torch.bfloat16),
        persistent=False,
    )
    layer.weight = torch.nn.Parameter(
        torch.ones((2, 2), dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.ones((1, 1), dtype=torch.float32),
        requires_grad=False,
    )
    layer._hpu_orig_M = 2
    layer._hpu_orig_N = 2
    expected = torch.full((2, 2), 9.0, dtype=torch.bfloat16)
    monkeypatch.setattr(
        hpu_hw_agnostic.hpu_ops,
        "apply_block_fp8_linear_hpu",
        lambda **kwargs: expected,
    )

    output = method.apply(
        layer,
        torch.ones((2, 2), dtype=torch.bfloat16),
    )

    assert output is expected


def test_combine_topk_swa_indices_matches_sparse_prefill_layout():
    device = torch.device("hpu")
    topk_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [6, 7, 0, 1],
        ],
        dtype=torch.int32,
        device=device,
    )
    query_start_loc = torch.tensor(
        [7, 9, 12], dtype=torch.int64, device=device
    )
    seq_lens = torch.tensor(
        [4, 5], dtype=torch.int32, device=device
    )
    gather_lens = torch.tensor(
        [3, 4], dtype=torch.int32, device=device
    )

    combined_indices, combined_lens = (
        _hpu_combine_topk_swa_indices(
            topk_indices,
            query_start_loc,
            seq_lens,
            gather_lens,
            window_size=3,
            compress_ratio=2,
            topk=4,
            M=20,
            N=8,
        )
    )
    combined_indices = combined_indices.cpu()
    combined_lens = combined_lens.cpu()

    expected_prefixes = [
        [0, 7, 8, 9],
        [1, 2, 8, 9, 10],
        [23, 27, 28, 29],
        [24, 25, 28, 29, 30],
        [26, 27, 29, 30, 31],
    ]
    assert combined_indices.shape == (5, 128)
    assert combined_lens.tolist() == [4, 5, 4, 5, 5]
    for row, expected in zip(
        combined_indices, expected_prefixes, strict=True
    ):
        assert row[:len(expected)].tolist() == expected
        assert torch.all(row[len(expected):] == -1)


def test_bf16_mla_sparse_interface_matches_dense_reference(monkeypatch):
    device = torch.device("hpu")
    q = torch.tensor(
        [
            [[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5]],
            [[0.5, 0.5, 1.0, 0.0], [1.0, -0.5, 0.0, 0.5]],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.5]],
            [[0.0, 1.0, 0.5, 0.0]],
            [[0.5, 0.5, 1.0, -0.5]],
            [[-0.5, 1.0, 0.0, 1.0]],
            [[1.0, -0.5, 0.5, 0.0]],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    indices = torch.tensor(
        [
            [[0, 2, -1, -1]],
            [[1, 3, 4, -1]],
        ],
        dtype=torch.int32,
        device=device,
    )

    out, max_logits, softmax_lse = (
        _hpu_bf16_mla_sparse_interface(
            q,
            kv,
            indices,
            sm_scale=0.5,
            d_v=3,
            block_dpe=1,
        )
    )

    q_cpu = q.float().cpu()
    kv_cpu = kv.float().cpu()[:, 0]
    indices_cpu = indices.cpu()[:, 0]
    safe_indices = indices_cpu.clamp_min(0)
    selected = kv_cpu[safe_indices]
    valid = indices_cpu >= 0
    logits = torch.matmul(
        q_cpu, selected.transpose(1, 2)
    ) * 0.5
    logits = logits.masked_fill(
        ~valid.unsqueeze(1), -float("inf")
    )
    expected_max = logits.amax(dim=-1)
    expected_lse = torch.logsumexp(logits, dim=-1)
    expected_out = torch.matmul(
        torch.softmax(logits, dim=-1),
        selected[..., :3],
    )

    torch.testing.assert_close(
        out.float().cpu(), expected_out, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        max_logits.cpu(), expected_max, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        softmax_lse.cpu(), expected_lse, atol=1e-5, rtol=1e-5
    )

    attn_sink = torch.tensor(
        [-0.25, 0.25], dtype=torch.float32, device=device
    )
    sink_out, sink_max, sink_lse = _hpu_bf16_mla_sparse_interface(
        q,
        kv,
        indices,
        sm_scale=0.5,
        d_v=3,
        block_dpe=1,
        attn_sink=attn_sink,
    )
    logits_with_sink = torch.cat(
        (
            logits,
            attn_sink.cpu().view(1, 2, 1).expand(2, -1, -1),
        ),
        dim=-1,
    )
    expected_sink_out = torch.matmul(
        torch.softmax(logits_with_sink, dim=-1)[..., :-1],
        selected[..., :3],
    )

    torch.testing.assert_close(
        sink_out.float().cpu(),
        expected_sink_out,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        sink_max.cpu(),
        logits_with_sink.amax(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        sink_lse.cpu(),
        torch.logsumexp(logits_with_sink, dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )

    monkeypatch.setenv("VLLM_HPU_DSV4_FUSED_SDPA", "1")
    fused_out, fused_max, fused_lse = _hpu_bf16_mla_sparse_interface(
        q,
        kv,
        indices,
        sm_scale=0.5,
        d_v=3,
        block_dpe=1,
        attn_sink=attn_sink,
    )
    torch.testing.assert_close(
        fused_out.float().cpu(),
        expected_sink_out,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        fused_max.cpu(),
        logits_with_sink.amax(dim=-1),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        fused_lse.cpu(),
        torch.logsumexp(logits_with_sink, dim=-1),
        atol=2e-2,
        rtol=2e-2,
    )

    invalid_indices = torch.full_like(indices[:1], -1)
    invalid_out, invalid_max, invalid_lse = (
        _hpu_bf16_mla_sparse_interface(
            q[:1],
            kv,
            invalid_indices,
            sm_scale=0.5,
            d_v=3,
            block_dpe=1,
            attn_sink=attn_sink,
        )
    )
    torch.testing.assert_close(
        invalid_out.cpu(), torch.zeros_like(invalid_out).cpu()
    )
    expected_invalid_stat = attn_sink.cpu().view(1, -1)
    torch.testing.assert_close(invalid_max.cpu(), expected_invalid_stat)
    torch.testing.assert_close(invalid_lse.cpu(), expected_invalid_stat)


def test_mxfp4_moe_apply_preserves_2d_strided_input():
    device = torch.device("hpu")
    x = torch.arange(
        16, dtype=torch.bfloat16, device=device
    ).view(2, 8)[:, :4]
    topk_ids = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.int32, device=device
    )
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.6, 0.4]],
        dtype=torch.bfloat16,
        device=device,
    )
    assert not x.is_contiguous()

    captured = {}

    class FakeMoeOp:
        def __call__(
            self,
            hidden_states,
            expert_routing_table,
            router_weights,
            **kwargs,
        ):
            captured["hidden_states"] = hidden_states
            captured["expert_routing_table"] = (
                expert_routing_table
            )
            captured["router_weights"] = router_weights
            captured["kwargs"] = kwargs
            return hidden_states

    layer = type(
        "FakeLayer",
        (),
        {"moe_op": FakeMoeOp(), "activation": "silu"},
    )()
    method = HPUHwAgnosticMxfp4MoEMethod.__new__(
        HPUHwAgnosticMxfp4MoEMethod
    )

    output = method.apply(
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert captured["hidden_states"] is x
    assert captured["expert_routing_table"] is topk_ids
    assert captured["router_weights"] is topk_weights
    assert captured["kwargs"]["permuted_weights"] is True
    assert output is x


def test_router_gate_uses_cached_fp32_weight():
    class FakeGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.allow_router_gemm = True
            self.weight = torch.nn.Parameter(
                torch.linspace(
                    -0.5,
                    0.5,
                    8 * 64,
                    dtype=torch.float32,
                    device="hpu",
                ).view(8, 64).to(torch.bfloat16),
                requires_grad=False,
            )
            _cache_hpu_router_weight(self)

        def forward(self, x):
            return _hpu_gate_linear_forward(self, x)[0]

    gate = FakeGate()
    assert gate._hpu_router_weight_fp32.dtype == torch.float32
    assert gate._hpu_router_weight_fp32.device.type == "hpu"
    compiled_gate = torch.compile(gate, backend="hpu_backend")
    x = torch.linspace(
        -1,
        1,
        4 * 64,
        dtype=torch.float32,
        device="hpu",
    ).view(4, 64).to(torch.bfloat16)
    output = compiled_gate(x)

    expected = torch.nn.functional.linear(
        x.float().cpu(),
        gate.weight.float().cpu(),
    )
    torch.testing.assert_close(
        output.cpu(), expected, atol=1e-5, rtol=1e-5
    )
