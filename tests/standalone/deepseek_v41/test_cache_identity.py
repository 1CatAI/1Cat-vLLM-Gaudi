# SPDX-License-Identifier: Apache-2.0
"""Relocation and unrelated source edits must not flush computation caches."""
import hashlib
import json
from types import SimpleNamespace

from vllm_gaudi.compilation.deepseek_v41_cache_identity import (
    computation_dependencies,
    runtime_content_identity,
    semantic_environment,
    lowered_keys,
)


def forward(owner, value):
    return value * 2


def changed(owner, value):
    return value * 3


IDENTITY_TEST_SCALE = 2


def mutable_global(owner, value):
    return value * IDENTITY_TEST_SCALE


def test_syntax_reuse_still_resolves_current_global_values(monkeypatch):
    first = computation_dependencies(mutable_global, SimpleNamespace())
    assert computation_dependencies(mutable_global, SimpleNamespace()) == first
    monkeypatch.setitem(mutable_global.__globals__, "IDENTITY_TEST_SCALE", 3)
    assert computation_dependencies(mutable_global, SimpleNamespace()) != first


def test_computation_identity_includes_selected_callable():
    first = computation_dependencies(forward, SimpleNamespace())
    second = computation_dependencies(changed, SimpleNamespace())
    assert first != second
    assert computation_dependencies(forward, SimpleNamespace(unrelated="diagnostic")) == first


def test_lowered_compatibility_requires_identical_graph_and_compiler(monkeypatch):
    from vllm_gaudi.compilation import deepseek_v41_cache_identity as identity

    monkeypatch.setattr(identity, "lowering_dependencies", lambda: {"compiler": "one"})
    first = lowered_keys(b"graph")
    assert first != lowered_keys(b"different graph")
    monkeypatch.setattr(identity, "lowering_dependencies", lambda: {"compiler": "two"})
    second = lowered_keys(b"graph")
    assert first[0] != second[0]
    assert first[1][0] != second[1][0]


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


def test_nested_parent_and_link_paths_are_bound_by_binary_content(tmp_path):
    profiles = []
    for name in ("first", "relocated"):
        directory = tmp_path / name
        directory.mkdir()
        binary = directory / "kernels.so"
        binary.write_bytes(b"identical compiled operators")
        sha = hashlib.sha256(binary.read_bytes()).hexdigest()
        metadata = directory / "build.json"
        metadata.write_text(
            json.dumps(
                dict(binaries={"kernels.so": sha},
                     link_command=["c++", "-shared",
                                   str(directory / "kernel.o"), "-Wl,-rpath," + str(directory)],
                     addon=dict(parent_gc_path=str(binary), parent_gc_sha256=sha),
                     softmax=dict(parent=str(binary), parent_sha256=sha))))
        profiles.append(
            dict(environment={},
                 additional_libraries=[dict(path=str(binary))],
                 configuration_files=[dict(path=str(metadata))]))
    assert runtime_content_identity(profiles[0]) == runtime_content_identity(profiles[1])
    record = json.loads(metadata.read_text())
    record["addon"]["parent_gc_sha256"] = "different content"
    metadata.write_text(json.dumps(record))
    assert runtime_content_identity(profiles[0]) != runtime_content_identity(profiles[1])


def test_uncertified_configuration_paths_remain_semantic(tmp_path):
    from vllm_gaudi.compilation.deepseek_v41_cache_identity import relocated_content

    assert relocated_content(dict(file="/first/data")) != relocated_content(dict(file="/second/data"))


def test_lambda_on_the_final_line_of_a_call_keeps_its_dependency():
    from vllm_gaudi.compilation.deepseek_v41_cache_identity import normalized_source

    def build(*, generation):
        return generation

    callback = build(generation=lambda: 17)
    assert "Constant(value=17)" in normalized_source(callback)
    assert computation_dependencies(callback, SimpleNamespace())


def test_frontend_lookup_policy_keeps_math_identity_and_bounded_legacy_lookup():
    from vllm_gaudi.compilation.deepseek_v41_cache_identity import frontend_contract_keys

    transport = "module:vllm_gaudi.compilation.deepseek_v41_frontend_cache"
    contract = dict(sources={"actual_forward": "math-one", transport: "new-lookup-policy"}, model="weights")
    first = frontend_contract_keys(contract)
    assert first == frontend_contract_keys(dict(contract, sources={**contract["sources"], transport: "another-policy"}))
    assert first != frontend_contract_keys(dict(contract, sources={
        **contract["sources"], "actual_forward": "math-two"
    }))
    assert first != frontend_contract_keys(dict(contract, model="different-weights"))
    assert len(first[1]) == 1


def test_python_source_digest_keeps_semantics_and_detects_restored_mtime(tmp_path):
    import ast
    import os
    from vllm_gaudi.compilation.deepseek_v41_cache_identity import python_source_digest, _python_source_digest

    path = tmp_path / "operator.py"
    path.write_text("def operation(x):\n    return x * 2\n")
    expected = hashlib.sha256(ast.dump(ast.parse(path.read_text()), include_attributes=False).encode()).hexdigest()
    assert python_source_digest(path) == expected
    hits = _python_source_digest.cache_info().hits
    assert python_source_digest(path) == expected
    assert _python_source_digest.cache_info().hits == hits + 1
    previous = path.stat()
    path.write_text("def operation(x):\n    return x * 3\n")
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert python_source_digest(path) != expected
