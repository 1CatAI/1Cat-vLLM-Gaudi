# SPDX-License-Identifier: Apache-2.0
"""Allocation and replay guards for the built native preflight adapter."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

if os.environ.get("DSV41_TEST_NATIVE_PREFLIGHT") != "1":
    pytest.skip("Requires an explicitly selected native adapter", allow_module_level=True)

import habana_frameworks.torch.core  # noqa: E402,F401
from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge, _verify_prepared_runtime  # noqa: E402
from vllm_gaudi.ops.tp2_graph_inputs import FixedDecodeInputs  # noqa: E402


@pytest.fixture(scope="module")
def bridge():
    path = Path(os.environ["VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE"]).resolve()
    value = _load_bridge(path)
    _verify_prepared_runtime(path)
    assert value.fixed_input_preflight_api_version == 1
    return value


@pytest.fixture(params=["cpu"] + (["hpu"] if os.environ.get("DSV41_TEST_HPU") == "1" else []))
def device(request):
    return request.param


def setup(device, bridge, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_NATIVE_INPUT_PREFLIGHT", "1")
    pool = torch.arange(16, dtype=torch.float32, device=device)
    first = dict(hidden_states=pool[:8].view(2, 4),
                 positions=torch.tensor([3], device=device),
                 state_tensors=(pool[8:], ),
                 state_generation=4,
                 metadata=SimpleNamespace(is_prompt=False),
                 adapter=SimpleNamespace(name="deepseek_v41_pp1"))
    if device == "hpu":
        torch.hpu.synchronize()
    bindings = FixedDecodeInputs(torch.nn.Identity(),
                                 first, [[first["hidden_states"], first["positions"]]],
                                 native_bridge=bridge)
    return pool, first, bindings


def test_changed_inputs_and_inplace_state_updates(bridge, device, monkeypatch):
    _, first, bindings = setup(device, bridge, monkeypatch)
    assert bindings.updates(first) == []
    first["state_tensors"][0].fill_(19)
    changed = dict(first, hidden_states=torch.full((2, 4), 7., device=device))
    if device == "hpu":
        torch.hpu.synchronize()
    updates = bindings.updates(changed)
    assert len(updates) == 1 and updates[0][0] is first["hidden_states"]
    bindings.apply(updates)
    assert torch.equal(first["hidden_states"].cpu(), torch.full((2, 4), 7.))
    assert torch.equal(first["state_tensors"][0].cpu(), torch.full((8, ), 19.))


@pytest.mark.parametrize("kind", [
    "shape", "stride", "dtype", "not_tensor", "state_replaced", "state_offset", "state_count", "state_not_tensor",
    "alias_offset", "generation", "prefill"
])
def test_invalid_contract_rejected_before_copy(bridge, device, monkeypatch, kind):
    pool, first, bindings = setup(device, bridge, monkeypatch)
    changed = dict(first, positions=torch.tensor([9], device=device))
    if kind == "shape":
        changed["hidden_states"] = first["hidden_states"].view(1, 8)
    elif kind == "stride":
        changed["hidden_states"] = torch.zeros(4, 2, device=device).t()
    elif kind == "dtype":
        changed["hidden_states"] = first["hidden_states"].to(torch.bfloat16)
    elif kind == "not_tensor":
        changed["hidden_states"] = None
    elif kind == "state_replaced":
        changed["state_tensors"] = (first["state_tensors"][0].clone(), )
    elif kind == "state_offset":
        changed["state_tensors"] = (pool[:8], )
    elif kind == "state_count":
        changed["state_tensors"] = ()
    elif kind == "state_not_tensor":
        changed["state_tensors"] = (None, )
    elif kind == "alias_offset":
        changed["hidden_states"] = pool[8:].view(2, 4)
    elif kind == "generation":
        changed["state_generation"] += 1
    elif kind == "prefill":
        changed["metadata"] = SimpleNamespace(is_prompt=True)
    if device == "hpu":
        torch.hpu.synchronize()
    assert bindings.updates(changed) is None
    assert first["positions"].cpu().item() == 3
    assert torch.equal(pool.cpu(), torch.arange(16).float())


@pytest.mark.parametrize("target", ["state", "destination"])
def test_inplace_storage_replacement_invalidates_held_tensor(bridge, monkeypatch, target):
    _, first, bindings = setup("cpu", bridge, monkeypatch)
    tensor = first["state_tensors"][0] if target == "state" else first["hidden_states"]
    tensor.set_(torch.zeros_like(tensor))
    assert bindings.updates(first) is None


def test_pool_pointer_facade_cannot_hide_alias_offset(bridge, monkeypatch):
    pool, first, bindings = setup("cpu", bridge, monkeypatch)
    monkeypatch.setattr(torch.Tensor, "data_ptr", lambda _: 0)
    assert bindings.updates(dict(first, hidden_states=pool[8:].view(2, 4))) is None
    updates = bindings.updates(dict(first, hidden_states=torch.ones(2, 4)))
    assert len(updates) == 1


def test_api_mismatch_fails_before_preparation(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_NATIVE_INPUT_PREFLIGHT", "1")
    first = dict(metadata={}, adapter=SimpleNamespace(name="deepseek_v41_pp0"))
    with pytest.raises(RuntimeError, match="version 1"):
        FixedDecodeInputs(torch.nn.Identity(), first, [], native_bridge=SimpleNamespace())
