# SPDX-License-Identifier: Apache-2.0
"""Relocation and unrelated source edits must not flush computation caches."""
import hashlib
import json
from types import SimpleNamespace

from vllm_gaudi.compilation.deepseek_v41_cache_identity import (
    computation_dependencies,
    runtime_content_identity,
    semantic_environment,
)


def forward(owner, value):
    return value * 2


def changed(owner, value):
    return value * 3


def test_computation_identity_includes_selected_callable():
    first = computation_dependencies(forward, SimpleNamespace())
    second = computation_dependencies(changed, SimpleNamespace())
    assert first != second
    assert computation_dependencies(forward, SimpleNamespace(unrelated="diagnostic")) == first


def test_relocated_binary_and_cache_paths_keep_semantic_environment(tmp_path):
    before = tmp_path / "a.so"
    after = tmp_path / "b.so"
    before.write_bytes(b"same runtime")
    after.write_bytes(before.read_bytes())
    assert semantic_environment({
        "VLLM_HPU_KERNEL": str(before),
        "PT_HPU_RECIPE_CACHE_CONFIG": "old"
    }) == (semantic_environment({
        "VLLM_HPU_KERNEL": str(after),
        "PT_HPU_RECIPE_CACHE_CONFIG": "new"
    }))
    assert semantic_environment({"VLLM_HPU_DSV41_ATTN_DENSE_FP8":
                                 "0"}) != (semantic_environment({"VLLM_HPU_DSV41_ATTN_DENSE_FP8": "1"}))


def test_runtime_relocation_preserves_binary_certificate(tmp_path):
    profiles = []
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        binary = directory / "runtime.so"
        binary.write_bytes(b"identical runtime")
        metadata = directory / "abi.json"
        metadata.write_text(json.dumps(dict(path=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest())))
        profiles.append(
            dict(environment={
                "PT_HPU_LAZY_MODE": "0",
                "TMPDIR": str(directory)
            },
                 additional_libraries=[dict(path=str(binary), sha256="verified by launcher")],
                 configuration_files=[dict(path=str(metadata), sha256="verified by launcher")]))
    assert runtime_content_identity(profiles[0]) == runtime_content_identity(profiles[1])
    (tmp_path / "second/runtime.so").write_bytes(b"changed runtime")
    assert runtime_content_identity(profiles[0]) != runtime_content_identity(profiles[1])
