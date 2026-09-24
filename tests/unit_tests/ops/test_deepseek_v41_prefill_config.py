# SPDX-License-Identifier: Apache-2.0
"""A selected native prompt plan must reach its device-grouped implementation."""
import pytest

from vllm_gaudi import envs
from vllm_gaudi.ops.deepseek_v41_prefill_plan import validate_prefill_plan_config

DEPENDENCIES = (
    "VLLM_HPU_DSV41_PREFILL_GROUPED",
    "VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES",
    "VLLM_HPU_DSV41_PREFILL_ROUTE_OUTPUT",
)


def test_native_plan_selects_complete_expert_route(monkeypatch):
    for name in DEPENDENCIES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN", "1")
    assert all(getattr(envs, name) for name in DEPENDENCIES)
    validate_prefill_plan_config(n256=True)


@pytest.mark.parametrize("disabled", DEPENDENCIES)
def test_partial_plan_configuration_cannot_silently_use_legacy_path(monkeypatch, disabled):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN", "1")
    monkeypatch.setenv(disabled, "0")
    with pytest.raises(ValueError, match="Native prefill plan cannot execute"):
        validate_prefill_plan_config(n256=True)


def test_native_plan_requires_matching_weight_layout(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN", "1")
    with pytest.raises(ValueError, match="N256 prepared weights"):
        validate_prefill_plan_config(n256=False)


def test_unselected_plan_keeps_existing_dispatch(monkeypatch):
    for name in (*DEPENDENCIES, "VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN"):
        monkeypatch.delenv(name, raising=False)
    assert not any(getattr(envs, name) for name in DEPENDENCIES)
    validate_prefill_plan_config(n256=False)
