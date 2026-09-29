# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the four-way tensor / single-stage serving contract."""

from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention
from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import _weighted_index_scores
from vllm_gaudi.ops.deepseek_v41_sampling import local_greedy_candidate, select_greedy_candidate


@pytest.mark.parametrize("tp_size", (2, 4))
@pytest.mark.parametrize("token_rows", (False, True))
def test_index_scores_cover_all_heads_and_keep_source_width(tp_size, token_rows):
    generator = torch.Generator().manual_seed(4104)
    tokens, heads, columns = 3, 32, 19
    local_heads = heads // tp_size
    keys = torch.randn(columns, 128, generator=generator).bfloat16()
    packed = pack_fp4(keys, 32)
    restored = unpack_fp4(packed, 128, 32)
    query = torch.randn(tokens, heads, 128, generator=generator).bfloat16()
    weights = torch.randn(tokens, heads, generator=generator).bfloat16()
    positions = torch.tensor([9, 18, 30], dtype=torch.int32)
    rows = torch.arange(columns, dtype=torch.int32)
    if token_rows:
        rows = rows.expand(tokens, -1).clone()
        rows[0, -1] = -1
    attention = SimpleNamespace(
        shared=SimpleNamespace(physical_rows=lambda ids, ratio: ids),
        cache=SimpleNamespace(index=packed),
        ratio=1,
        index_heads=local_heads,
        tensor_parallel_size=tp_size,
    )
    actual = PagedCSA2Attention._scores(attention, positions, rows, query, weights)
    # Each rank sums its own contiguous head interval before a BF16 TP sum.
    partials = []
    for rank in range(tp_size):
        sl = slice(rank * local_heads, (rank + 1) * local_heads)
        scores = torch.matmul(query[:, sl], restored.T).relu() * weights[:, sl, None]
        partials.append(scores.sum(1))
    expected = torch.stack(partials, 1).sum(1).float()
    valid = (rows >= 0) & (rows < (positions + 1).unsqueeze(-1))
    expected = expected.masked_fill(~valid, -torch.inf)
    assert actual.shape == (tokens, columns)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    helper = _weighted_index_scores(query, weights, restored, local_heads, False)
    torch.testing.assert_close(helper.masked_fill(~valid, -torch.inf), expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", (2, 4))
def test_index_query_scale_uses_global_head_count(monkeypatch, tp_size):
    from vllm_gaudi.ops import deepseek_v41_paged_attention as attention_module
    local_heads = 32 // tp_size
    projection, weight_projection = object(), object()
    attention = SimpleNamespace(
        index_heads=local_heads,
        tensor_parallel_size=tp_size,
        weights=SimpleNamespace(indexer=SimpleNamespace(wq_b=projection, weights_proj=weight_projection)),
        linear=lambda value, weight: torch.ones(2, local_heads * 128 if weight is projection else local_heads),
        _rope=lambda value, positions: value,
        gather=lambda value, dim: torch.cat([value] * tp_size, dim),
    )
    monkeypatch.setattr(attention_module, "fp4_roundtrip", lambda value, group: value)
    query, weights = PagedCSA2Attention._prepare_index_queries(attention, torch.ones(2, 5), torch.ones(2, 5),
                                                               torch.arange(2))
    assert query.shape == (2, 32, 128)
    torch.testing.assert_close(weights, torch.full((2, 32), 1 / 64), rtol=0, atol=0)


def test_pp1_forward_preserves_final_output_and_completion_never_uses_peer(monkeypatch):
    from vllm_gaudi.v1.worker import deepseek_v41_runner as runner_module
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True, ranks=[0])
    monkeypatch.setattr(runner_module, "get_pp_group", lambda: group)
    for name in ("PACKED_PP", "DEVICE_COMMIT", "PREFILL_PP_WAVEFRONT"):
        monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_" + name, False)

    def no_peer(*args, **kwargs):
        raise AssertionError("PP1 must not issue a pipeline send or broadcast")

    monkeypatch.setattr(runner_module.dist, "isend", no_peer)
    monkeypatch.setattr(runner_module.dist, "broadcast", no_peer)
    buffers = runner_module.PPBuffers("cpu", capacity=7, dspark=False, device_commit=False)
    assert buffers.prefill_hidden.numel() == buffers.prefill_pre.numel() == 0
    buffers.exchange = no_peer
    output = torch.ones(1, 5120, dtype=torch.bfloat16)

    class Model:
        native = False

        def prepare_step(self, *args, **kwargs):
            pass

        def __call__(self, ids, positions, **kwargs):
            return output

    runner = object.__new__(runner_module.V41ModelRunner)
    runner.model, runner.use_dspark, runner.direct_token_ids = Model(), False, False
    runner.input_views = {1: torch.empty(1, dtype=torch.int64)}
    runner.position_views = {1: torch.empty(1, dtype=torch.int32)}
    runner.position_bank = runner._next_input = None
    runner.pp = buffers
    runner.audit = {"target_steps": 0, "target_tokens": 0}
    assert runner._forward("r", [42], 0, decode=False) is output
    assert runner._forward("r", [42], 1, decode=True) is output
    assert buffers.finish_single(7, None) == (7, [])
    assert buffers.finish_single(1, 42) == (1, [42])
    assert buffers.sends == buffers.receives == 0
    assert buffers.commits == 2


def test_pp1_final_collapse_and_four_way_greedy_ties():
    from vllm_gaudi.models.deepseek_v41_program import PreparedStage
    residual = torch.arange(32).reshape(2, 4, 4).float().bfloat16()
    pre = torch.ones(2, 4) / 4
    stage = SimpleNamespace(shared=SimpleNamespace(),
                            layers=[],
                            is_last_stage=True,
                            weights=SimpleNamespace(norm=SimpleNamespace(weight=torch.ones(4))),
                            config={"text_config": {
                                "rms_norm_eps": 1e-6
                            }})
    value, _, aux = PreparedStage._forward_impl(stage, residual, pre, torch.arange(2), torch.ones(2), ())
    assert value.shape == (2, 4) and aux is None
    # TP2 and TP3 have the same maximum; the smaller global token ID wins.
    logits = torch.tensor([[0., 1., 0., 3., 7., 1., 7., 2.]])
    candidates = [local_greedy_candidate(chunk, rank) for rank, chunk in enumerate(logits.chunk(4, -1))]
    assert select_greedy_candidate(torch.cat(candidates, -1)).item() == 4


def test_trace_worker_coordinates_use_recorded_topology():
    from tools.collect_deepseek_v41_trace import worker_coordinates
    tp4 = {"topology": {"tensor_parallel_size": 4, "pipeline_parallel_size": 1}}
    assert [worker_coordinates(rank, tp4) for rank in range(4)] == [(0, rank) for rank in range(4)]
    assert [worker_coordinates(rank, {}) for rank in range(4)] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    with pytest.raises(ValueError, match="Trace topology"):
        worker_coordinates(4, tp4)


@pytest.mark.parametrize("tokens,expected", [(1, "compiled"), (6, "compiled"), (128, "prefill"), (8192, "prefill"),
                                             (16384, "prefill")])
def test_tp4_model_forward_reaches_compiled_decode_without_tp2_replay(monkeypatch, tokens, expected):
    from vllm_gaudi.models.deepseek_v41 import HpuDeepseekV41ForCausalLM
    from vllm_gaudi.models.deepseek_v41_program import PrefillInput
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED", "1")
    calls = []

    class Stage:
        loaded, length = True, 1048576

        def __call__(self, residual, pre, positions, ids, engram):
            calls.append("prefill")
            assert isinstance(residual, PrefillInput)
            residual, pre = residual.take()
            return residual[:, 0], pre, None

    def compiled(residual, pre, positions, ids, engram):
        calls.append("compiled")
        return residual[:, 0], pre, None

    model = SimpleNamespace(program=Stage(),
                            tensor_parallel_size=4,
                            pp_rank=0,
                            native=False,
                            step_use_replay=False,
                            step_ticket=object(),
                            _decode_prefix=None,
                            is_first_stage=True,
                            is_last_stage=True,
                            ordinary=compiled,
                            engram_host=SimpleNamespace(wait=lambda ticket: ()),
                            compiled_input_calls=0,
                            compiled_input=lambda ids:
                            (torch.ones(ids.numel(), 4, 5120).bfloat16(), torch.ones(ids.numel(), 4)),
                            embed_input_ids=lambda ids: torch.ones(ids.numel(), 5120, dtype=torch.bfloat16))
    ids = torch.arange(tokens)
    output = HpuDeepseekV41ForCausalLM.forward(model, ids, ids.int())
    assert calls == [expected]
    assert model.compiled_input_calls == int(tokens <= 6)
    assert output.shape == (tokens, 5120)


def test_owned_prefill_releases_initial_activation_after_first_layer(monkeypatch):
    import weakref
    from types import MethodType
    from vllm_gaudi.models.deepseek_v41_program import PrefillInput, PreparedStage
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED", "1")
    owner = PrefillInput(torch.ones(128, 4, 64, dtype=torch.bfloat16), torch.ones(128, 4))
    original = weakref.ref(owner.residual)

    class Layer:

        def __init__(self, layer):
            self.layer = layer

        def __call__(self, residual, pre, positions, mask, rows):
            if isinstance(residual, PrefillInput):
                residual, pre = residual.take()
            if self.layer == 2:
                assert original() is None, "Outer call frames retain the consumed initial residual"
            return residual + 1, pre, None

    stage = SimpleNamespace(tensor_parallel_size=4,
                            is_last_stage=False,
                            shared=SimpleNamespace(prefill_main_workspace=None),
                            layers=[Layer(0), Layer(2)])
    stage._forward_impl = MethodType(PreparedStage._forward_impl, stage)
    ids = torch.arange(128)
    output, _, _ = PreparedStage.forward(stage, owner, None, ids, ids, ())
    assert torch.equal(output, torch.full_like(output, 3))
    assert owner.residual is None and owner.pre_mix is None
    with pytest.raises(RuntimeError, match="already been consumed"):
        owner.take()
