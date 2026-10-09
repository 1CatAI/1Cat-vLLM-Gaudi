# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.getenv("DSV41_MHC_POST_STATS_LIBRARY")
    if not path:
        pytest.skip("Set the additive C1 post statistics registration")
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(path)


def arguments(rows, peers=False):
    def t(shape, dtype):
        return torch.empty(shape, device="meta", dtype=dtype)
    return (t((4, rows, 5120) if peers else (rows, 5120), torch.bfloat16),
            t((rows, 4, 5120), torch.bfloat16), t((rows, 24), torch.float32),
            t((5120,), torch.bfloat16), 1e-6)


@pytest.mark.parametrize("rows", [2, 5, 6])
def test_distinct_routed_and_shared_quantizers(rows):
    result = torch.ops.custom_op.custom_deepseek_v41_dspark_mhc_post_norm_statistics_gaudi2(*arguments(rows))
    assert [x.shape for x in result] == [(rows, 4, 5120), (rows, 5120), (rows, 5120),
                                       (rows, 5120), (rows, 1), (rows, 5120), (rows, 1)]
    assert [x.dtype for x in result] == [torch.bfloat16, torch.bfloat16, torch.bfloat16,
                                       torch.float8_e4m3fn, torch.float32, torch.float8_e4m3fn, torch.float32]


def test_rank_major_peer_geometry():
    result = torch.ops.custom_op.custom_deepseek_v41_dspark_mhc_post_norm_statistics_gaudi2(*arguments(6, True))
    assert result[0].shape == (6, 4, 5120)


def test_wrong_gate_packet_rejected():
    args = list(arguments(6))
    args[2] = torch.empty((6, 28), device="meta", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_dspark_mhc_post_norm_statistics_gaudi2(*args)
