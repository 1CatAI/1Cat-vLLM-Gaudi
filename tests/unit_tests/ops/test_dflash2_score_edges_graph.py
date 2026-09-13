# SPDX-License-Identifier: Apache-2.0
"""DFlash2 score contraction parity across compiled request-batch changes."""
import os
from unittest import mock

import pytest
import torch

from flashinfer_gaudi import dflash2


def _einsum_reference(predecessor, successor, candidates, unary, hidden, anchors):
    top_k = candidates.shape[-1]
    previous = torch.cat((anchors[:, None, None].expand(-1, 1, top_k), candidates[:, :-1]), dim=1)
    weighted = predecessor[previous] * hidden[:, :, None]
    return unary[:, :, None] + torch.einsum("blpr,blcr->blpc", weighted, successor[candidates])


def _inputs(batch, steps, top_k, rank, dtype, seed, noncontiguous=False, device="cpu"):
    rng = torch.Generator().manual_seed(seed)
    vocab = 257
    predecessor = torch.randn(vocab, rank, dtype=dtype, generator=rng).to(device)
    successor = torch.randn(vocab, rank, dtype=dtype, generator=rng).to(device)
    candidates = torch.randint(vocab, (batch, steps, top_k), dtype=torch.int32, generator=rng).to(device)
    unary = torch.randn(batch, steps, top_k, generator=rng).to(device)
    hidden = torch.randn(batch, steps + int(noncontiguous), rank, dtype=dtype, generator=rng).to(device)
    if noncontiguous:
        hidden = hidden[:, 1:]
    anchors = torch.randint(vocab, (batch, ), dtype=torch.int32, generator=rng).to(device)
    return predecessor, successor, candidates, unary, hidden, anchors


@pytest.mark.parametrize("batch,steps,top_k,rank", [
    (0, 7, 16, 256),
    (1, 0, 16, 256),
    (1, 1, 1, 5),
    (2, 4, 3, 5),
    (1, 7, 16, 256),
    (8, 7, 16, 256),
    (16, 7, 16, 256),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_score_bmm_preserves_einsum_contract(batch, steps, top_k, rank, dtype, noncontiguous):
    inputs = _inputs(batch, steps, top_k, rank, dtype, 419, noncontiguous)
    expected = _einsum_reference(*inputs)
    with mock.patch.object(dflash2, "_is_hpu", return_value=True):
        actual = dflash2.score_edges(*inputs)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(dflash2.select_path_reference(inputs[2], actual),
                               dflash2.select_path_reference(inputs[2], expected),
                               rtol=0,
                               atol=0)


@pytest.mark.skipif(os.environ.get("FLASHINFER_GAUDI_TEST_HPU") != "1",
                    reason="Set FLASHINFER_GAUDI_TEST_HPU=1 only with an available test card")
def test_score_bmm_hpu_force_static_batch_transitions():
    from habana_frameworks.torch.dynamo.compile_backend import config as hpu_config

    torch._dynamo.reset()
    compiled = torch.compile(dflash2.score_edges,
                             backend="hpu_backend",
                             fullgraph=True,
                             options={"force_static_compile": True})
    try:
        with mock.patch.object(hpu_config, "use_eager_fallback", False):
            for seed in (419, 523, 917):
                for batch in (1, 8, 1, 2, 16, 4, 8, 1):
                    inputs = _inputs(batch, 7, 16, 256, torch.bfloat16, seed, True, "hpu")
                    expected = _einsum_reference(*inputs).cpu()
                    actual = compiled(*inputs).cpu()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    candidates = inputs[2].cpu()
                    torch.testing.assert_close(dflash2.select_path_reference(candidates, actual),
                                               dflash2.select_path_reference(candidates, expected),
                                               rtol=0,
                                               atol=0)
    finally:
        torch._dynamo.reset()
