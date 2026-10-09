# SPDX-License-Identifier: Apache-2.0
"""Native shape gates and unchanged control math with a precomputed collapse."""
import os

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_math import hc_post, hc_pre


@pytest.fixture(scope='module', params=['bf16', 'f32'])
def native_meta(request):
    if os.getenv('DSV41_TEST_NATIVE_META') != '1':
        pytest.skip('Requires built library; no HPU allocation')
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    suffix = '_f32' if request.param == 'f32' else ''
    return (getattr(torch.ops.custom_op, f'custom_deepseek_v41_mhc_post_collapse{suffix}_gaudi2'),
            torch.float32 if request.param == 'f32' else torch.bfloat16)


def operands(tokens, device='meta'):
    shapes = ((tokens, 5120), (tokens, 4, 5120), (tokens, 4), (tokens, 4, 4), (tokens, 4))
    return [torch.empty(s, dtype=torch.bfloat16 if i < 2 else torch.float32, device=device)
            for i, s in enumerate(shapes)]


@pytest.mark.parametrize('tokens', [1, 2, 6])
def test_native_shapes(native_meta, tokens):
    operation, collapsed_dtype = native_meta
    updated, collapsed = operation(*operands(tokens))
    assert updated.shape == (tokens, 4, 5120) and updated.dtype == torch.bfloat16
    assert collapsed.shape == (tokens, 5120) and collapsed.dtype == collapsed_dtype


@pytest.mark.parametrize('ranks', [2, 4, 8])
@pytest.mark.parametrize('tokens', [1, 2, 6])
def test_peer_rows_keep_residual_and_collapsed_shapes(native_meta, ranks, tokens):
    operation, collapsed_dtype = native_meta
    if collapsed_dtype != torch.bfloat16:
        pytest.skip('FP32 collapse does not implement deferred peer summation')
    args = operands(tokens)
    args[0] = torch.empty((ranks, tokens, 5120), dtype=torch.bfloat16, device='meta')
    updated, collapsed = operation(*args)
    assert updated.shape == (tokens, 4, 5120)
    assert collapsed.shape == (tokens, 5120)


@pytest.mark.parametrize('ranks', [0, 1, 9])
def test_peer_rows_reject_invalid_rank_count(native_meta, ranks):
    if native_meta[1] != torch.bfloat16:
        pytest.skip('Only the BF16 peer consumer is extended')
    args = operands(1)
    args[0] = torch.empty((ranks, 1, 5120), dtype=torch.bfloat16, device='meta')
    with pytest.raises(RuntimeError):
        native_meta[0](*args)


@pytest.mark.parametrize('tokens', [0, 7])
def test_native_rejects_unsupported_bucket(native_meta, tokens):
    with pytest.raises(RuntimeError):
        native_meta[0](*operands(tokens))


@pytest.mark.parametrize('index', range(5))
def test_native_rejects_wrong_dtype(native_meta, index):
    args = operands(1)
    args[index] = args[index].to(torch.float64)
    with pytest.raises(RuntimeError):
        native_meta[0](*args)


@pytest.mark.parametrize('tokens', [1, 2, 6])
def test_precomputed_collapse_preserves_all_control_outputs(tokens):
    g = torch.Generator().manual_seed(41100 + tokens)
    value = torch.randn(tokens, 5120, generator=g).bfloat16()
    residual = torch.randn(tokens, 4, 5120, generator=g).bfloat16()
    post = torch.rand(tokens, 4, generator=g)
    comb = torch.rand(tokens, 4, 4, generator=g)
    pre = torch.rand(tokens, 4, generator=g)
    updated = hc_post(value, residual, post, comb)
    collapsed = (updated.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
    control = torch.randn(24, 20480, generator=g) / 100
    args = (updated, pre, control, torch.ones(3), torch.zeros(24))
    expected = hc_pre(*args)
    observed = hc_pre(*args, collapsed_input=collapsed)
    for a, b in zip(observed, expected, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("tokens", [1, 2, 6, 7, 256])
@pytest.mark.parametrize("decode", [False, True])
def test_speculative_control_scope_keeps_prompt_arithmetic(monkeypatch, tokens, decode):
    from vllm_gaudi import envs
    from vllm_gaudi.ops.deepseek_v41_math import mhc_control_scope

    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_DSPARK", True)
    assert mhc_control_scope(decode=decode, tokens=tokens) == (decode and tokens <= 6)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_DSPARK", False)
    assert mhc_control_scope(decode=decode, tokens=tokens)
