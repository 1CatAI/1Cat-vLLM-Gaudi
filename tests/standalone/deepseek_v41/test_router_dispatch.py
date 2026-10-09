# SPDX-License-Identifier: Apache-2.0
"""The target router feature must retain the draft's checkpoint contract."""
from types import SimpleNamespace

import torch


def test_target_native_router_flag_does_not_select_mtp_top3(monkeypatch):
    from vllm_gaudi.models.deepseek_v41_program import PreparedMoE

    monkeypatch.setenv('VLLM_HPU_DSV41_ROUTER_TOP6', '1')
    monkeypatch.setenv('VLLM_HPU_DSV41_PREFILL_MXFP4', '0')
    monkeypatch.setenv('VLLM_HPU_DSV41_CONCURRENT_MOE_ROWS', '0')
    weights = SimpleNamespace(gate=SimpleNamespace(weight=torch.empty(128, 5120, device='meta')))
    draft = PreparedMoE(weights, 3, True, torch.ones(1), lambda value: value,
                        tensor_parallel_size=2, draft=True)
    assert not draft.router_top6
