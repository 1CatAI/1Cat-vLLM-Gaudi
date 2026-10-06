# SPDX-License-Identifier: Apache-2.0
"""Shared sampling packet contracts for TP shapes and concurrent row buckets."""
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def t(shape, dtype=torch.float32):
    return torch.empty(shape, device='meta', dtype=dtype)


@pytest.mark.parametrize('batch', [1, 2, 6])
@pytest.mark.parametrize('ranks', [2, 4, 8])
def test_tp_geometry_and_concurrent_row_contracts(batch, ranks):
    width = 128
    values, ids, norms, cuts = torch.ops.custom_op.custom_deepseek_v41_sampling_unpack_gaudi2(
        t((batch, ranks * (3 + 2 * width))), ranks, width)
    assert [tuple(v.shape) for v in (values, ids, norms, cuts)] == [(batch, ranks * width), (batch, ranks * width),
                                                                    (batch, ranks * 3), (batch, ranks)]
    assert ids.dtype == torch.int32
    probabilities, coverage = torch.ops.custom_op.custom_deepseek_v41_sampling_mask_gaudi2(
        values, t(values.shape), t(values.shape), cuts, t((batch, 4)), t((batch, 1)), ranks)
    selected = torch.ops.custom_op.custom_deepseek_v41_sampling_select_gaudi2(probabilities, ids, t((batch, 4)),
                                                                              t((batch, 1), torch.int32))
    assert coverage.dtype == selected.dtype == torch.int32
    assert coverage.shape == selected.shape == (batch, 1)


@pytest.mark.parametrize('ranks,width', [(1, 128), (9, 128), (4, 120), (4, 512)])
def test_invalid_wire_geometry_rejected(ranks, width):
    with pytest.raises(RuntimeError, match='packet layout'):
        torch.ops.custom_op.custom_deepseek_v41_sampling_unpack_gaudi2(t((1, ranks * (3 + 2 * width))), ranks, width)


def test_controls_cannot_broadcast_silently_inside_native_kernel():
    with pytest.raises(RuntimeError, match='operand shape'):
        torch.ops.custom_op.custom_deepseek_v41_sampling_mask_gaudi2(t((2, 512)), t((2, 512)), t((2, 512)), t((2, 4)),
                                                                     t((1, 4)), t((2, 1)), 4)
