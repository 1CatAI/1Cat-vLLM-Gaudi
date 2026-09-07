# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_direct_state_matches_indexed_decode(batch: int, state_dtype: torch.dtype):
    torch.manual_seed(7)
    dim = 64
    width = 4
    state_start = 2

    x = torch.randn(batch, dim, dtype=torch.bfloat16)
    weight = torch.randn(dim, width, dtype=torch.bfloat16)
    bias = torch.randn(dim, dtype=torch.bfloat16)
    indexed_pool = torch.randn(batch + state_start + 1, width - 1, dim, dtype=state_dtype)
    direct_pool = indexed_pool[state_start:state_start + batch].clone()
    indices = torch.arange(state_start, state_start + batch, dtype=torch.int32)
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32)

    indexed_out = hpu_causal_conv1d_update(
        x,
        indexed_pool,
        weight,
        bias,
        activation="silu",
        conv_state_indices=indices,
        query_start_loc=query_start_loc,
    )
    direct_out = hpu_causal_conv1d_update(
        x,
        direct_pool,
        weight,
        bias,
        activation="silu",
        conv_state_indices=None,
        query_start_loc=query_start_loc,
        direct_state_layout=True,
    )
    torch.testing.assert_close(direct_out, indexed_out, rtol=0, atol=0)
    torch.testing.assert_close(
        direct_pool,
        indexed_pool[state_start:state_start + batch],
        rtol=0,
        atol=0,
    )


def test_direct_state_rejects_index_tensor():
    with pytest.raises(ValueError, match="cache_indices"):
        hpu_causal_conv1d_update(
            torch.zeros(1, 8),
            torch.zeros(1, 3, 8),
            torch.zeros(8, 4),
            conv_state_indices=torch.zeros(1, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            direct_state_layout=True,
        )


def _spec_conv_reference(x, state_rows, weight, bias, accepted):
    batch, tokens, dim = x.shape
    width = weight.shape[1]
    history_len = width - 1
    outputs = torch.empty_like(x)
    updated = torch.empty_like(state_rows)
    for row in range(batch):
        offset = int(accepted[row]) - 1
        history = state_rows[row, offset:offset + history_len]
        sequence = torch.cat((history, x[row]), dim=0)
        value = torch.zeros(tokens, dim, dtype=x.dtype)
        for tap in range(width):
            value += sequence[tap:tap + tokens] * weight[:, tap]
        if bias is not None:
            value += bias
        outputs[row] = value
        new_state = torch.cat((history[1:], x[row]), dim=0)
        if new_state.shape[0] < state_rows.shape[1]:
            new_state = torch.nn.functional.pad(new_state, (0, 0, 0, state_rows.shape[1] - new_state.shape[0]))
        updated[row] = new_state[:state_rows.shape[1]]
    return outputs, updated


def test_speculative_conv_uses_accepted_checkpoint_and_rewrites_lattice():
    generator = torch.Generator().manual_seed(23)
    batch, tokens, dim, width = 2, 4, 7, 3
    capacity = width - 2 + tokens
    pool = torch.randn(6, capacity, dim, generator=generator)
    original = pool.clone()
    state_indices = torch.tensor([1, 4], dtype=torch.int32)
    accepted = torch.tensor([1, 3], dtype=torch.int32)
    x = torch.randn(batch, tokens, dim, generator=generator)
    weight = torch.randn(dim, width, generator=generator)
    bias = torch.randn(dim, generator=generator)
    query_start_loc = torch.arange(0, (batch + 1) * tokens, tokens, dtype=torch.int32)

    expected_output, expected_state = _spec_conv_reference(
        x,
        original.index_select(0, state_indices.long()),
        weight,
        bias,
        accepted,
    )
    actual = hpu_causal_conv1d_update(
        x.reshape(batch * tokens, dim),
        pool,
        weight,
        bias,
        activation=None,
        conv_state_indices=state_indices,
        num_accepted_tokens=accepted,
        query_start_loc=query_start_loc,
        max_query_len=tokens,
    )

    torch.testing.assert_close(actual.reshape_as(expected_output), expected_output)
    torch.testing.assert_close(pool.index_select(0, state_indices.long()), expected_state)
    torch.testing.assert_close(pool[0], original[0])


def test_speculative_conv_rollback_matches_two_committed_tokens():
    generator = torch.Generator().manual_seed(29)
    batch, tokens, dim, width = 1, 4, 5, 4
    capacity = width - 2 + tokens
    pool = torch.zeros(3, capacity, dim)
    state_index = torch.tensor([1], dtype=torch.int32)
    pool[1, :width - 1] = torch.randn(width - 1, dim, generator=generator)
    weight = torch.randn(dim, width, generator=generator)
    first = torch.randn(batch, tokens, dim, generator=generator)
    second = torch.randn(batch, tokens, dim, generator=generator)
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32)

    hpu_causal_conv1d_update(
        first.reshape(tokens, dim),
        pool,
        weight,
        activation=None,
        conv_state_indices=state_index,
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
        query_start_loc=query_start_loc,
        max_query_len=tokens,
    )
    before_second = pool[1].clone()
    expected, expected_state = _spec_conv_reference(
        second,
        before_second.unsqueeze(0),
        weight,
        None,
        torch.tensor([2], dtype=torch.int32),
    )
    actual = hpu_causal_conv1d_update(
        second.reshape(tokens, dim),
        pool,
        weight,
        activation=None,
        conv_state_indices=state_index,
        num_accepted_tokens=torch.tensor([2], dtype=torch.int32),
        query_start_loc=query_start_loc,
        max_query_len=tokens,
    )

    torch.testing.assert_close(actual.reshape_as(expected), expected)
    torch.testing.assert_close(pool[1], expected_state[0])


def test_single_token_transition_loads_accepted_history_and_canonicalizes_state():
    generator = torch.Generator().manual_seed(31)
    batch, draft_tokens, dim, width = 2, 8, 5, 4
    capacity = width - 2 + draft_tokens
    pool = torch.randn(5, capacity, dim, generator=generator)
    state_indices = torch.tensor([1, 3], dtype=torch.int32)
    accepted = torch.tensor([2, 8], dtype=torch.int32)
    transition = torch.randn(batch, 1, dim, generator=generator)
    weight = torch.randn(dim, width, generator=generator)
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32)

    before_transition = pool.index_select(0, state_indices.long()).clone()
    expected, expected_state = _spec_conv_reference(
        transition,
        before_transition,
        weight,
        None,
        accepted,
    )
    actual = hpu_causal_conv1d_update(
        transition.reshape(batch, dim),
        pool,
        weight,
        activation=None,
        conv_state_indices=state_indices,
        num_accepted_tokens=accepted,
        query_start_loc=query_start_loc,
        max_query_len=1,
    )

    torch.testing.assert_close(actual.reshape_as(expected), expected)
    torch.testing.assert_close(pool.index_select(0, state_indices.long()), expected_state)

    # A following eight-token block now starts from checkpoint zero. This
    # catches the one-token -> speculative transition that previously reused
    # stale history after a scheduler boundary step.
    next_block = torch.randn(batch, draft_tokens, dim, generator=generator)
    before_next = pool.index_select(0, state_indices.long()).clone()
    expected_next, expected_next_state = _spec_conv_reference(
        next_block,
        before_next,
        weight,
        None,
        torch.ones(batch, dtype=torch.int32),
    )
    actual_next = hpu_causal_conv1d_update(
        next_block.reshape(batch * draft_tokens, dim),
        pool,
        weight,
        activation=None,
        conv_state_indices=state_indices,
        num_accepted_tokens=torch.ones(batch, dtype=torch.int32),
        query_start_loc=torch.arange(0, (batch + 1) * draft_tokens, draft_tokens, dtype=torch.int32),
        max_query_len=draft_tokens,
    )

    torch.testing.assert_close(actual_next.reshape_as(expected_next), expected_next)
    torch.testing.assert_close(pool.index_select(0, state_indices.long()), expected_next_state)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("tokens", [1, 8])
@pytest.mark.parametrize("accepted_offset", [1, 4, 8])
def test_speculative_conv_optional_rounding_matches_ordinary_decode(dtype, batch, tokens, accepted_offset):
    rng = torch.Generator().manual_seed(433)
    dim = 128
    x = torch.randn(batch, tokens, dim, dtype=dtype, generator=rng) * .1
    weight = torch.randn(dim, 4, dtype=dtype, generator=rng) * .1
    bias = torch.randn(dim, dtype=dtype, generator=rng) * .01
    pool = torch.randn(5, 10, dim, dtype=dtype, generator=rng) * .1
    indices = torch.tensor([3, 1][:batch], dtype=torch.int32)
    selected = pool.index_select(0, indices.long())
    ordinary_state = selected[:, accepted_offset - 1:accepted_offset + 2].clone()
    expected = []
    for token in range(tokens):
        expected.append(
            hpu_causal_conv1d_update(x[:, token],
                                     ordinary_state,
                                     weight,
                                     bias=bias,
                                     activation="silu",
                                     query_start_loc=torch.arange(batch + 1, dtype=torch.int32),
                                     direct_state_layout=True))
    expected = torch.stack(expected, dim=1)
    actual = hpu_causal_conv1d_update(x.reshape(-1, dim),
                                      pool,
                                      weight,
                                      bias=bias,
                                      activation="silu",
                                      conv_state_indices=indices,
                                      num_accepted_tokens=torch.full((batch, ), accepted_offset, dtype=torch.int32),
                                      query_start_loc=torch.arange(batch + 1, dtype=torch.int32) * tokens,
                                      max_query_len=tokens,
                                      round_before_activation=True).reshape_as(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    updated = pool.index_select(0, indices.long())
    committed = updated[:, :3] if tokens == 1 else updated[:, -3:]
    torch.testing.assert_close(committed, ordinary_state, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("tokens", [1, 8])
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("transposed_state", [False, True])
@pytest.mark.parametrize("round_before", [False, True])
def test_full_query_conv_preserves_output_and_all_state(dtype, batch, tokens, width, transposed_state, round_before):
    rng = torch.Generator().manual_seed(782)
    dim = 64
    original = torch.randn(6, width - 2 + 8, dim, dtype=dtype, generator=rng) * .1
    if transposed_state:
        original = original.transpose(1, 2).contiguous()
    pools = [original.clone(), original.clone()]
    weight = torch.randn(dim, width, dtype=dtype, generator=rng) * .1
    bias = torch.randn(dim, dtype=dtype, generator=rng) * .01
    indices = torch.tensor([4, 1][:batch], dtype=torch.int32)
    qsl = torch.arange(batch + 1, dtype=torch.int32) * tokens
    # Exercise repeated mutation, every accepted prefix, and the existing
    # clamp contract, rather than checking only a fresh zero state.
    for accepted in (0, 1, 2, 3, 4, 5, 6, 7, 8, 99):
        x = torch.randn(batch * tokens, dim, dtype=dtype, generator=rng) * .1
        outputs = [
            hpu_causal_conv1d_update(
                x,
                pool,
                weight,
                bias=bias,
                activation="silu",
                conv_state_indices=indices,
                num_accepted_tokens=torch.full((batch, ), accepted, dtype=torch.int32),
                query_start_loc=qsl,
                max_query_len=tokens,
                round_before_activation=round_before,
                full_query_valid_state=fast,
            ) for fast, pool in zip((False, True), pools, strict=True)
        ]
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
        torch.testing.assert_close(pools[1], pools[0], rtol=0, atol=0)
    # An owned-row specialization must not touch any unrelated request.
    untouched = [index for index in range(original.shape[0]) if index not in indices.tolist()]
    torch.testing.assert_close(pools[1][untouched], original[untouched], rtol=0, atol=0)


def test_general_spec_conv_keeps_ragged_and_invalid_rows_safe():
    rng = torch.Generator().manual_seed(119)
    batch, tokens, dim = 3, 8, 64
    pool = torch.randn(6, 10, dim, generator=rng)
    before = pool.clone()
    x = torch.randn(batch * tokens, dim, generator=rng)
    output = hpu_causal_conv1d_update(
        x,
        pool,
        torch.randn(dim, 4, generator=rng),
        activation="silu",
        conv_state_indices=torch.tensor([2, 4, -1], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1, 4, 8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8, 11, 11], dtype=torch.int32),
        max_query_len=tokens,
    ).reshape(batch, tokens, dim)
    assert torch.count_nonzero(output[1, 3:]) == 0
    assert torch.count_nonzero(output[2]) == 0
    torch.testing.assert_close(pool[[0, 1, 3, 5]], before[[0, 1, 3, 5]], rtol=0, atol=0)
    assert torch.count_nonzero(pool[4, 5:]) == 0
