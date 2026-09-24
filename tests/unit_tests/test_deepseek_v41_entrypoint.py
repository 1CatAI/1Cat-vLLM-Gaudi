# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from vllm_gaudi.entrypoints.deepseek_v41 import (
    _C1_FASTPATH_DEFAULTS,
    _NUMERIC_FASTPATH_DEFAULTS,
    _PREFILL_MOE_DEFAULTS,
    prepare_default_fastpaths,
)

_PROFILE_KEYS = set(_C1_FASTPATH_DEFAULTS) | set(_NUMERIC_FASTPATH_DEFAULTS) | set(_PREFILL_MOE_DEFAULTS) | {
    "VLLM_HPU_DSV41_DEFAULT_FASTPATHS",
    "VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS",
    "VLLM_HPU_DSV41_DSPARK",
    "VLLM_HPU_DSV41_WO_A_FP8_SIDECAR",
    "VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR",
    "VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR",
    "VLLM_HPU_DSV41_PREFILL_GROUPED_FP8",
    "VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP",
    "VLLM_HPU_DSV41_EXPERT_N256",
}


@pytest.fixture(autouse=True)
def restore_profile_environment():
    original = {key: os.environ[key] for key in _PROFILE_KEYS if key in os.environ}
    yield
    for key in _PROFILE_KEYS:
        os.environ.pop(key, None)
    os.environ.update(original)


def _clear_profile(monkeypatch):
    for key in _PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_default_profile_enables_qualified_numeric_bundle(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    for sidecar in ("wo_a_fp8", "attention_dense_fp8", "engram_fp8"):
        (tmp_path / "sidecars" / sidecar).mkdir(parents=True)

    prepare_default_fastpaths(tmp_path)

    assert os.environ["VLLM_HPU_DSV41_DSPARK"] == "0"
    assert all(os.environ[key] == value for key, value in _C1_FASTPATH_DEFAULTS.items())
    assert all(os.environ[key] == value for key, value in _NUMERIC_FASTPATH_DEFAULTS.items())
    assert all(os.environ[key] == value for key, value in _PREFILL_MOE_DEFAULTS.items())
    assert os.environ["VLLM_HPU_DSV41_WO_A_FP8_SIDECAR"] == str((tmp_path / "sidecars" / "wo_a_fp8").resolve())
    assert os.environ["VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR"] == str(
        (tmp_path / "sidecars" / "attention_dense_fp8").resolve())
    assert os.environ["VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR"] == str((tmp_path / "sidecars" / "engram_fp8").resolve())
    assert os.environ["VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS"] == "8192"
    assert os.environ["VLLM_HPU_DSV41_PREFILL_KV_REUSE"] == "1"
    assert os.environ["VLLM_HPU_DSV41_PREFILL_PP_WAVEFRONT"] == "1"
    assert "VLLM_HPU_DSV41_PREFILL_GROUPED_FP8" not in os.environ
    assert os.environ["VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP"] == "1"


def test_numeric_profile_can_be_disabled_without_sidecars(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    monkeypatch.setenv("VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS", "0")

    prepare_default_fastpaths(tmp_path)

    assert os.environ["VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS"] == "0"
    assert not any(key in os.environ for key in _NUMERIC_FASTPATH_DEFAULTS)
    assert "VLLM_HPU_DSV41_WO_A_FP8_SIDECAR" not in os.environ
    assert "VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR" not in os.environ
    assert "VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR" not in os.environ


def test_experimental_numeric_bundle_discovers_sidecars(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    monkeypatch.setenv("VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS", "1")
    model = tmp_path / "prepared"
    for sidecar in ("wo_a_fp8", "attention_dense_fp8", "engram_fp8"):
        (model / "sidecars" / sidecar).mkdir(parents=True)

    prepare_default_fastpaths(model)

    assert all(os.environ[key] == value for key, value in _NUMERIC_FASTPATH_DEFAULTS.items())
    assert os.environ["VLLM_HPU_DSV41_WO_A_FP8_SIDECAR"] == str((model / "sidecars" / "wo_a_fp8").resolve())
    assert os.environ["VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR"] == str(
        (model / "sidecars" / "attention_dense_fp8").resolve())
    assert os.environ["VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR"] == str((model / "sidecars" / "engram_fp8").resolve())


@pytest.mark.parametrize("n256", (False, True))
def test_prefill_moe_defaults_require_n256_storage(monkeypatch, tmp_path, n256):
    _clear_profile(monkeypatch)
    monkeypatch.setenv("VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS", "0")
    if n256:
        monkeypatch.setenv("VLLM_HPU_DSV41_EXPERT_N256", "1")

    prepare_default_fastpaths(tmp_path)

    if n256:
        assert all(os.environ[key] == value for key, value in _PREFILL_MOE_DEFAULTS.items())
    else:
        assert not any(key in os.environ for key in _PREFILL_MOE_DEFAULTS)


def test_default_profile_respects_individual_override(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    monkeypatch.setenv("VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_WO_A_FP8", "0")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS", "0")
    (tmp_path / "sidecars" / "attention_dense_fp8").mkdir(parents=True)
    (tmp_path / "sidecars" / "engram_fp8").mkdir(parents=True)

    prepare_default_fastpaths(tmp_path)

    assert os.environ["VLLM_HPU_DSV41_WO_A_FP8"] == "0"
    assert "VLLM_HPU_DSV41_WO_A_FP8_SIDECAR" not in os.environ
    assert os.environ["VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS"] == "0"


@pytest.mark.parametrize(("runner", "adapter"), (("0", None), (None, "0")))
def test_default_profile_respects_v2_opt_out(monkeypatch, tmp_path, runner, adapter):
    _clear_profile(monkeypatch)
    monkeypatch.setenv("VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS", "0")
    if runner is not None:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner)
    if adapter is not None:
        monkeypatch.setenv("VLLM_HPU_DSV41_V2", adapter)

    prepare_default_fastpaths(tmp_path)

    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert os.environ["VLLM_HPU_DSV41_V2"] == "0"


def test_aggregate_opt_out_leaves_environment_unchanged(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    monkeypatch.setenv("VLLM_HPU_DSV41_DEFAULT_FASTPATHS", "0")

    prepare_default_fastpaths(tmp_path)

    assert "VLLM_HPU_DSV41_DSPARK" not in os.environ
    assert not any(key in os.environ for key in _C1_FASTPATH_DEFAULTS)
