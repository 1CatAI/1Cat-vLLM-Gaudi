# SPDX-License-Identifier: Apache-2.0
import os
import json
from pathlib import Path
import sys

import pytest

from vllm_gaudi.entrypoints.deepseek_v41 import (
    _C1_FASTPATH_DEFAULTS,
    _NUMERIC_FASTPATH_DEFAULTS,
    _PREFILL_MOE_DEFAULTS,
    _PIPELINE_ONLY_FASTPATHS,
    _TP4_FASTPATH_DEFAULTS,
    _SINGLE_STAGE_NATIVE_DEFAULTS,
    prepare_default_fastpaths,
)

_PROFILE_KEYS = set().union(
    _PIPELINE_ONLY_FASTPATHS,
    _TP4_FASTPATH_DEFAULTS,
    _SINGLE_STAGE_NATIVE_DEFAULTS,
    _C1_FASTPATH_DEFAULTS,
    _NUMERIC_FASTPATH_DEFAULTS,
    _PREFILL_MOE_DEFAULTS,
    _SINGLE_STAGE_NATIVE_DEFAULTS,
) | {
    "VLLM_HPU_DSV41_DEFAULT_FASTPATHS",
    "VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS",
    "VLLM_HPU_DSV41_DSPARK",
    "VLLM_HPU_DSV41_WO_A_FP8_SIDECAR",
    "VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR",
    "VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG",
    "VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR",
    "VLLM_HPU_DSV41_PREFILL_GROUPED_FP8",
    "VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP",
    "VLLM_HPU_DSV41_EXPERT_N256",
    "VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS",
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

    prepare_default_fastpaths(tmp_path, tensor_parallel_size=2, pipeline_parallel_size=2)

    assert os.environ["VLLM_HPU_DSV41_DSPARK"] == "0"
    assert all(os.environ[key] == value for key, value in _C1_FASTPATH_DEFAULTS.items())
    assert all(os.environ[key] == value for key, value in _NUMERIC_FASTPATH_DEFAULTS.items())
    assert all(os.environ[key] == value for key, value in _PREFILL_MOE_DEFAULTS.items())
    assert os.environ["VLLM_HPU_DSV41_WO_A_FP8_SIDECAR"] == str((tmp_path / "sidecars" / "wo_a_fp8").resolve())
    assert os.environ["VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR"] == str(
        (tmp_path / "sidecars" / "attention_dense_fp8").resolve()
    )
    assert os.environ["VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR"] == str((tmp_path / "sidecars" / "engram_fp8").resolve())
    assert "VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS" not in os.environ
    assert os.environ["VLLM_HPU_DSV41_PREFILL_KV_REUSE"] == "1"
    assert os.environ["VLLM_HPU_DSV41_PREFILL_PP_WAVEFRONT"] == "1"
    assert "VLLM_HPU_DSV41_PREFILL_GROUPED_FP8" not in os.environ
    assert os.environ["VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP"] == "1"


def test_numeric_profile_can_be_disabled_without_sidecars(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    monkeypatch.setenv("VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS", "0")

    prepare_default_fastpaths(tmp_path, tensor_parallel_size=2, pipeline_parallel_size=2)

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

    prepare_default_fastpaths(model, tensor_parallel_size=2, pipeline_parallel_size=2)

    assert all(os.environ[key] == value for key, value in _NUMERIC_FASTPATH_DEFAULTS.items())
    assert os.environ["VLLM_HPU_DSV41_WO_A_FP8_SIDECAR"] == str((model / "sidecars" / "wo_a_fp8").resolve())
    assert os.environ["VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR"] == str(
        (model / "sidecars" / "attention_dense_fp8").resolve()
    )
    assert os.environ["VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR"] == str((model / "sidecars" / "engram_fp8").resolve())


@pytest.mark.parametrize("n256", (False, True))
def test_prefill_moe_defaults_require_n256_storage(monkeypatch, tmp_path, n256):
    _clear_profile(monkeypatch)
    monkeypatch.setenv("VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS", "0")
    if n256:
        monkeypatch.setenv("VLLM_HPU_DSV41_EXPERT_N256", "1")

    prepare_default_fastpaths(tmp_path, tensor_parallel_size=2, pipeline_parallel_size=2)

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

    prepare_default_fastpaths(tmp_path, tensor_parallel_size=2, pipeline_parallel_size=2)

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

    prepare_default_fastpaths(tmp_path, tensor_parallel_size=2, pipeline_parallel_size=2)

    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert os.environ["VLLM_HPU_DSV41_V2"] == "0"


def test_aggregate_opt_out_leaves_environment_unchanged(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    monkeypatch.setenv("VLLM_HPU_DSV41_DEFAULT_FASTPATHS", "0")

    prepare_default_fastpaths(tmp_path, tensor_parallel_size=2, pipeline_parallel_size=2)

    assert "VLLM_HPU_DSV41_DSPARK" not in os.environ
    assert not any(key in os.environ for key in _C1_FASTPATH_DEFAULTS)


def test_tp_only_stage_inherits_c1_except_pipeline_transport(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    for profile in (_C1_FASTPATH_DEFAULTS, _NUMERIC_FASTPATH_DEFAULTS, _PREFILL_MOE_DEFAULTS):
        for key, value in profile.items():
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED_FP8", "w13_single_bucket")
    for sidecar in ("wo_a_fp8", "attention_dense_fp8", "engram_fp8"):
        (tmp_path / "sidecars" / sidecar).mkdir(parents=True)
    prepare_default_fastpaths(tmp_path)
    assert all(os.environ[key] == "0" for key in _PIPELINE_ONLY_FASTPATHS)
    assert os.environ["VLLM_HPU_DSV41_DSPARK"] == "0"
    assert os.environ["VLLM_HPU_DSV41_PREFILL_GROUPED_FP8"] == "w13_single_bucket"
    assert os.environ["VLLM_HPU_DSV41_VISION"] == "1"
    assert os.environ.get("VLLM_HPU_DSV41_PREFILL_MXFP4", "0") == "0"
    for key in ("GRAPH_REPLAY", "NATIVE_INPUT_GRAPH", "RUNTIME_INDEXER", "ENGRAM_DIRECT_INPUT"):
        assert os.environ["VLLM_HPU_DSV41_" + key] == "1"


@pytest.mark.parametrize("tp,pp", ((4, 2), (2, 1)))
def test_rejects_unsupported_topology_before_changing_environment(monkeypatch, tmp_path, tp, pp):
    _clear_profile(monkeypatch)
    before = dict(os.environ)
    with pytest.raises(ValueError, match="TP4 x PP1"):
        prepare_default_fastpaths(tmp_path, tensor_parallel_size=tp, pipeline_parallel_size=pp)
    assert dict(os.environ) == before


def test_tp4_defaults_select_optimized_moe_without_opt_in(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    for sidecar in ("wo_a_fp8", "attention_dense_fp8", "engram_fp8"):
        (tmp_path / "sidecars" / sidecar).mkdir(parents=True)
    prepare_default_fastpaths(tmp_path)
    assert all(
        os.environ[key] == value for key, value in _TP4_FASTPATH_DEFAULTS.items() if key not in _PIPELINE_ONLY_FASTPATHS
    )
    for key in (
        "EXPERT_N256_FP8",
        "EXPERT_FUSED_QUANT",
        "EXPERT_FUSED_REDUCE",
        "PREFILL_GROUPED",
        "PREFILL_DEVICE_ROUTES",
        "PREFILL_ROUTE_OUTPUT",
        "PREFILL_ACTIVE_PLAN",
    ):
        assert os.environ["VLLM_HPU_DSV41_" + key] == "1"
    assert os.environ["VLLM_HPU_DSV41_PREFILL_GROUPED_FP8"] == "w13_single_bucket"
    assert os.environ["VLLM_HPU_DSV41_PREFILL_DECODER_HALO"] == "1"
    assert os.environ["VLLM_HPU_DSV41_PREFILL_MLA_SEQUENCE"] == "1"


def test_tp4_prefill_diagnostic_disable_controls(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    for sidecar in ("wo_a_fp8", "attention_dense_fp8", "engram_fp8"):
        (tmp_path / "sidecars" / sidecar).mkdir(parents=True)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED_FP8", "")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_DECODER_HALO", "0")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_MLA_SEQUENCE", "0")
    prepare_default_fastpaths(tmp_path)
    assert os.environ["VLLM_HPU_DSV41_PREFILL_GROUPED_FP8"] == ""
    assert os.environ["VLLM_HPU_DSV41_PREFILL_DECODER_HALO"] == "0"
    assert os.environ["VLLM_HPU_DSV41_PREFILL_MLA_SEQUENCE"] == "0"


@pytest.mark.parametrize("tp,pp,expected_tile", ((2, 2, 8192), (4, 1, 16384)))
def test_c1_defaults_preserve_scheduler_prefill_tile_and_halo(monkeypatch, tmp_path, tp, pp, expected_tile):
    from vllm_gaudi.ops.deepseek_v41_decoder_halo import decoder_halo_mode
    from vllm_gaudi.ops.deepseek_v41_prefill_capacity import prefill_capacity, prefill_compute_buckets

    _clear_profile(monkeypatch)
    for sidecar in ("wo_a_fp8", "attention_dense_fp8", "engram_fp8"):
        (tmp_path / "sidecars" / sidecar).mkdir(parents=True)
    prepare_default_fastpaths(tmp_path, tensor_parallel_size=tp, pipeline_parallel_size=pp)
    capacity = prefill_capacity(16384, tp)
    assert capacity == expected_tile
    assert max(prefill_compute_buckets(capacity)) == expected_tile
    mode = decoder_halo_mode(0, 16384, 16384, eligible=True, block_tokens=capacity, allow_single_block=pp == 1)
    assert mode == ("final" if pp == 1 else "full")

    # A deliberate smaller diagnostic tile still disables the full-transaction
    # halo in the runner; inheritance must never supply that cap implicitly.
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS", "8192")
    prepare_default_fastpaths(tmp_path, tensor_parallel_size=tp, pipeline_parallel_size=pp)
    assert max(prefill_compute_buckets(capacity)) == 8192


def test_tp4_rejects_generic_moe_instead_of_silent_fallback(monkeypatch, tmp_path):
    import torch
    from vllm_gaudi.models.deepseek_v41_program import PreparedMoE

    _clear_profile(monkeypatch)
    for sidecar in ("wo_a_fp8", "attention_dense_fp8", "engram_fp8"):
        (tmp_path / "sidecars" / sidecar).mkdir(parents=True)
    prepare_default_fastpaths(tmp_path)
    monkeypatch.setenv("VLLM_HPU_DSV41_EXPERT_N256_FP8", "0")
    with pytest.raises(ValueError, match="generic MoE is not a serving fallback"):
        PreparedMoE(None, 6, True, torch.empty(128), lambda x: x, tensor_parallel_size=4)


def test_runtime_reexec_preserves_complete_launcher_capture_contract(monkeypatch, tmp_path):
    from vllm_gaudi.entrypoints.deepseek_v41 import main

    profile = tmp_path / "runtime.json"
    expected = dict(
        HABANA_PROFILE="1",
        HABANA_PROF_CONFIG=str(tmp_path / "capture.json"),
        VLLM_HPU_DSV41_RAW_TRACE="1",
        VLLM_HPU_DSV41_RAW_SCOPE_ONLY="1",
        ENABLE_PROFILER="true",
        GRAPH_VISUALIZATION="1",
        GRAPH_VISUALIZATION_DIR=str(tmp_path / "graphs"),
    )
    profile.write_text(json.dumps({"environment": {name: "0" for name in expected}}))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["deepseek_v41", "prepared", "--runtime-profile", str(profile)])
    monkeypatch.setenv("DSV41_RUN_EVIDENCE", str(tmp_path))
    monkeypatch.delenv("DSV41_SERVING_RUNTIME", raising=False)
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    captured = {}

    def execve(executable, argv, environment):
        captured.update(environment)
        raise SystemExit(0)

    monkeypatch.setattr(os, "execve", execve)
    with pytest.raises(SystemExit):
        main()
    assert {name: captured[name] for name in expected} == expected


def test_graph_reservation_follows_tp4_capture_policy(monkeypatch):
    import vllm_gaudi.entrypoints.deepseek_v41 as entry

    monkeypatch.setattr(entry, "prepare_native_libraries", lambda: None)
    monkeypatch.delenv("VLLM_GRAPH_RESERVED_MEM", raising=False)
    entry.prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    assert os.environ["VLLM_GRAPH_RESERVED_MEM"] == "0.1"
    monkeypatch.delenv("VLLM_GRAPH_RESERVED_MEM")
    entry.prepare_environment(tensor_parallel_size=2, pipeline_parallel_size=2)
    assert os.environ["VLLM_GRAPH_RESERVED_MEM"] == "0.1"
    monkeypatch.setenv("VLLM_GRAPH_RESERVED_MEM", "0.07")
    entry.prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    assert os.environ["VLLM_GRAPH_RESERVED_MEM"] == "0.07"


def test_installed_dense_precision_is_discovered_without_opt_in(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    for name in ("wo_a_fp8", "attention_dense_fp8", "engram_fp8"):
        (tmp_path / "sidecars" / name).mkdir(parents=True)
    precision = tmp_path / "sidecars/attention_dense_fp8/precision.json"
    precision.write_text('{}')
    prepare_default_fastpaths(tmp_path)
    assert os.environ["VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG"] == str(precision)
    monkeypatch.setenv("VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG", "diagnostic.json")
    prepare_default_fastpaths(tmp_path)
    assert os.environ["VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG"] == "diagnostic.json"


def test_single_stage_native_defaults_preserve_explicit_disable(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    for sidecar in ("wo_a_fp8", "attention_dense_fp8", "engram_fp8"):
        (tmp_path / "sidecars" / sidecar).mkdir(parents=True)
    monkeypatch.setenv("VLLM_HPU_DSV41_DEVICE_CLOSED_LOOP", "0")
    prepare_default_fastpaths(tmp_path, tensor_parallel_size=4, pipeline_parallel_size=1)
    assert os.environ["VLLM_HPU_DSV41_DEVICE_CLOSED_LOOP"] == "0"
    for key, value in _SINGLE_STAGE_NATIVE_DEFAULTS.items():
        if key != "VLLM_HPU_DSV41_DEVICE_CLOSED_LOOP":
            assert os.environ[key] == value


def test_pipeline_stages_keep_existing_token_ownership(monkeypatch, tmp_path):
    _clear_profile(monkeypatch)
    for sidecar in ("wo_a_fp8", "attention_dense_fp8", "engram_fp8"):
        (tmp_path / "sidecars" / sidecar).mkdir(parents=True)
    prepare_default_fastpaths(tmp_path, tensor_parallel_size=2, pipeline_parallel_size=2)
    assert not any(key in os.environ for key in _SINGLE_STAGE_NATIVE_DEFAULTS)
