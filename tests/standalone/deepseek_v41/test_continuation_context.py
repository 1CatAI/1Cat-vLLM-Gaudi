# SPDX-License-Identifier: Apache-2.0
"""A long-context fixture must continue from its actual committed prompt."""
import pytest

from tools.check_deepseek_v41_tp4_continuation import continuation_position


@pytest.mark.parametrize('context', [16384, 41984, 62464, 262144])
def test_continuation_positions_follow_context_through_warm_and_measured_steps(context):
    positions = [continuation_position(context, index) for index in range(206)]
    assert positions[:6] == list(range(context, context + 6))
    assert positions[6:] == list(range(context + 6, context + 206))


@pytest.mark.parametrize('context,index', [(-1, 0), (0, -1)])
def test_continuation_rejects_negative_positions(context, index):
    with pytest.raises(ValueError, match='nonnegative'):
        continuation_position(context, index)


def test_forced_feedback_uses_native_i32_sampler_contract():
    import torch
    from tools.check_deepseek_v41_tp4_continuation import forced_feedback_token

    sample = torch.zeros((1, 1), dtype=torch.int32)
    forced = forced_feedback_token(129276, sample)
    assert forced.dtype == torch.int32 and forced.shape == (1, 1)
    assert forced.item() == 129276 and sample.item() == 0
    with pytest.raises(ValueError, match='I32'):
        forced_feedback_token(129276, sample.long())
