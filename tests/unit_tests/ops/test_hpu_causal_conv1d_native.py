from unittest import mock

import pytest
import torch

import vllm_gaudi.ops.causal_conv1d_pytorch as causal_conv


def test_native_causal_conv_transposes_weight_and_forwards_metadata():
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    weight = torch.randn(16, 4, dtype=torch.bfloat16)
    bias = torch.randn(16, dtype=torch.bfloat16)
    state = torch.zeros(2, 3, 16, dtype=torch.bfloat16)
    query_start_loc = torch.tensor([0, 8], dtype=torch.int32)
    cache_indices = torch.tensor([0], dtype=torch.int64)
    has_initial_state = torch.tensor([False])

    expected_output = torch.empty_like(x)
    expected_state = torch.empty_like(state)
    operation = mock.Mock(return_value=(expected_output, expected_state))

    with mock.patch.object(
        causal_conv,
        "_resolve_hpu_causal_conv1d_fwd",
        return_value=operation,
    ):
        output, updated_state = causal_conv.hpu_causal_conv1d_fwd_native(
            x,
            weight,
            bias,
            state,
            query_start_loc,
            cache_indices,
            has_initial_state,
        )

    assert output is expected_output
    assert updated_state is expected_state
    args = operation.call_args.args
    assert args[0] is x
    assert args[1] is state
    torch.testing.assert_close(args[2], weight.transpose(0, 1))
    assert args[2].is_contiguous()
    assert args[3] is bias
    assert args[4] is has_initial_state
    assert args[5] is query_start_loc
    assert args[6] is cache_indices
    assert operation.call_args.kwargs == {
        "activation": True,
        "pad_slot_id": -1,
    }


def test_native_causal_conv_builds_false_initial_state_mask():
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    weight = torch.randn(16, 4, dtype=torch.bfloat16)
    state = torch.zeros(2, 3, 16, dtype=torch.bfloat16)
    operation = mock.Mock(return_value=(x, state))

    with mock.patch.object(
        causal_conv,
        "_resolve_hpu_causal_conv1d_fwd",
        return_value=operation,
    ):
        causal_conv.hpu_causal_conv1d_fwd_native(
            x,
            weight,
            None,
            state,
            torch.tensor([0, 8], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int64),
            None,
            activation=None,
        )

    mask = operation.call_args.args[4]
    assert mask.dtype == torch.bool
    assert mask.tolist() == [False]
    assert operation.call_args.kwargs["activation"] is False


@pytest.mark.parametrize(
    ("x_shape", "weight_shape", "message"),
    [
        ((1, 8, 16), (16, 4), "token-major 2-D"),
        ((8, 16), (15, 4), "shape \\[dim, width\\]"),
    ],
)
def test_native_causal_conv_rejects_invalid_shapes(x_shape, weight_shape, message):
    with pytest.raises(ValueError, match=message):
        causal_conv.hpu_causal_conv1d_fwd_native(
            torch.empty(x_shape),
            torch.empty(weight_shape),
            None,
            torch.empty(2, 3, 16),
            torch.tensor([0, 8]),
            torch.tensor([0]),
            torch.tensor([False]),
        )
