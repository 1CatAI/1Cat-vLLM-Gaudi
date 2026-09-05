# SPDX-License-Identifier: Apache-2.0
"""CPU-only fail-closed ABI and static graph contract tests."""

import json
from pathlib import Path

import pytest

from flashinfer_gaudi import _bridge
from flashinfer_gaudi._bridge import BRIDGE_VERSION, sha256, validate_manifest
from flashinfer_gaudi._dispatch import BackendUnavailableError
from flashinfer_gaudi.gemm import GemmSiluArtifactV1, GemmSiluPlan


@pytest.fixture
def artifact(tmp_path):
    identity = {
        "bridge_version": BRIDGE_VERSION,
        "synapse_version": "1.24.1.test",
        "torch_version": "test",
        "cxx11_abi": True
    }
    files = {}
    for name in ("adapter", "kernel_database", "bridge", "synapse"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    return ({
        "schema_version": 1,
        "target": "gaudi2",
        "runtime": identity,
        "sha256": {
            name: sha256(path)
            for name, path in files.items()
        }
    }, identity, files)


def test_manifest_identity_is_stable(artifact):
    manifest, identity, files = artifact
    assert validate_manifest(manifest, identity, files) == validate_manifest(json.loads(json.dumps(manifest)), identity,
                                                                             files)


@pytest.mark.parametrize("name", ["adapter", "kernel_database", "bridge", "synapse"])
def test_changed_or_missing_elf_rejected_before_load(artifact, name):
    manifest, identity, files = artifact
    files[name].write_bytes(b"changed")
    with pytest.raises(BackendUnavailableError, match="checksum"):
        validate_manifest(manifest, identity, files)
    files[name].unlink()
    with pytest.raises(BackendUnavailableError, match="checksum"):
        validate_manifest(manifest, identity, files)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("schema_version", True), ("target", "gaudi3"),
                                         ("sha256", {})])
def test_manifest_contract_mismatch(artifact, field, value):
    manifest, identity, files = artifact
    manifest[field] = value
    with pytest.raises(BackendUnavailableError):
        validate_manifest(manifest, identity, files)


@pytest.mark.parametrize("field,value", [("bridge_version", "1.25.0"), ("torch_version", "different"),
                                         ("cxx11_abi", False)])
def test_runtime_mismatch_rejected(artifact, field, value):
    manifest, identity, files = artifact
    with pytest.raises(BackendUnavailableError, match="ABI mismatch"):
        validate_manifest(manifest, {**identity, field: value}, files)


def test_graph_artifact_roundtrip():
    artifact = GemmSiluArtifactV1(8, 5120, 17408, "a" * 64)
    assert GemmSiluArtifactV1.from_json(artifact.to_json()) == artifact


@pytest.mark.parametrize("dims", [(0, 128, 128), (1, 0, 128), (1, 128, 127), (True, 128, 128), (1, 128, 2**31),
                                  (2**31, 128, 128), (1, 128, 2**30)])
def test_bad_plan_shapes_reject_before_loader(monkeypatch, dims):

    def forbidden():
        pytest.fail("Invalid plan touched native loader")

    monkeypatch.setattr("flashinfer_gaudi.gemm.load_bridge_adapter", forbidden)
    with pytest.raises(ValueError):
        GemmSiluPlan(*dims)


@pytest.mark.parametrize("value", ["", "0" * 63, "z" * 64, None])
def test_graph_requires_valid_adapter_id(value):
    with pytest.raises(ValueError):
        GemmSiluArtifactV1(1, 128, 128, value)


@pytest.mark.parametrize("failure", ["version", "checksum", "schema"])
def test_loader_never_executes_mismatched_adapter(monkeypatch, artifact, failure):
    manifest, identity, files = artifact
    if failure == "version":
        identity = {**identity, "bridge_version": "1.25.0"}
    elif failure == "checksum":
        files["adapter"].write_bytes(b"changed")
    else:
        manifest["schema_version"] = 2

    def forbidden(_):
        pytest.fail("Invalid adapter reached dlopen")

    monkeypatch.setattr(_bridge, "_LOADED", None)
    monkeypatch.setattr(_bridge, "runtime_identity", lambda: identity)
    monkeypatch.setattr(_bridge, "runtime_files", lambda _: files)
    monkeypatch.setattr(Path, "read_text", lambda _: json.dumps(manifest))
    monkeypatch.setattr(_bridge.torch.ops, "load_library", forbidden)
    with pytest.raises(BackendUnavailableError):
        _bridge.load_bridge_adapter()


@pytest.mark.parametrize("gemm", ["gemm", "GEMM"])
def test_trace_audit_accepts_engine_name_variants(tmp_path, gemm):
    from tools.benchmark_flashinfer_native_gemm import audit_trace
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps({
            "traceEvents": [
                {
                    "cat": "kernel",
                    "name": gemm
                },
                {
                    "cat": "kernel",
                    "name": "flashinfer_gaudi_silu_and_mul_bf16_gaudi2"
                },
                {
                    "cat": "privateuse1_runtime",
                    "name": "Launch"
                },
            ]
        }))
    assert audit_trace(path, "out")["launch_events"] == 1


@pytest.mark.parametrize("operation", ["aten::copy_", "aten::mm", "aten::silu", "aten::mul", "aten::empty"])
def test_out_trace_rejects_tensor_decomposition_or_allocation(tmp_path, operation):
    from tools.benchmark_flashinfer_native_gemm import audit_trace
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps({
            "traceEvents": [
                {
                    "cat": "kernel",
                    "name": "gemm"
                },
                {
                    "cat": "kernel",
                    "name": "flashinfer_gaudi_silu_and_mul_bf16_gaudi2"
                },
                {
                    "cat": "cpu_op",
                    "name": operation
                },
            ]
        }))
    with pytest.raises(RuntimeError, match="audit failed"):
        audit_trace(path, "out")
