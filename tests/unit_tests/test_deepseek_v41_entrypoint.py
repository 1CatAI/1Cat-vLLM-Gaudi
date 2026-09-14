# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from vllm_gaudi.entrypoints.deepseek_v41 import (
    _C1_FASTPATH_DEFAULTS,
    prepare_default_fastpaths,
)

_PROFILE_KEYS = set(_C1_FASTPATH_DEFAULTS) | {
    "VLLM_HPU_DSV41_DEFAULT_FASTPATHS",
    "VLLM_HPU_DSV41_DSPARK",
    "VLLM_HPU_DSV41_WO_A_FP8_SIDECAR",
    "VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR",
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


def test_default_c1_fastpaths_and_sidecar_discovery(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    model = tmp_path / "prepared"
    for sidecar in ("wo_a_fp8", "attention_dense_fp8"):
        (model / "sidecars" / sidecar).mkdir(parents=True)

    prepare_default_fastpaths(model)

    assert os.environ["VLLM_HPU_DSV41_DSPARK"] == "0"
    assert all(os.environ[key] == value for key, value in _C1_FASTPATH_DEFAULTS.items())
    assert os.environ["VLLM_HPU_DSV41_WO_A_FP8_SIDECAR"] == str((model / "sidecars" / "wo_a_fp8").resolve())
    assert os.environ["VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR"] == str(
        (model / "sidecars" / "attention_dense_fp8").resolve())


def test_default_profile_respects_individual_override(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    monkeypatch.setenv("VLLM_HPU_DSV41_WO_A_FP8", "0")
    (tmp_path / "sidecars" / "attention_dense_fp8").mkdir(parents=True)

    prepare_default_fastpaths(tmp_path)

    assert os.environ["VLLM_HPU_DSV41_WO_A_FP8"] == "0"
    assert "VLLM_HPU_DSV41_WO_A_FP8_SIDECAR" not in os.environ


def test_aggregate_opt_out_leaves_environment_unchanged(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    monkeypatch.setenv("VLLM_HPU_DSV41_DEFAULT_FASTPATHS", "0")

    prepare_default_fastpaths(tmp_path)

    assert "VLLM_HPU_DSV41_DSPARK" not in os.environ
    assert not any(key in os.environ for key in _C1_FASTPATH_DEFAULTS)
