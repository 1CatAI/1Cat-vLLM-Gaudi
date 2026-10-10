# SPDX-License-Identifier: Apache-2.0
"""SDK cache aliases require equivalent model, runtime and serving certificates."""
import json

import pytest

from vllm_gaudi.compilation.deepseek_v41_cache_identity import runtime_content_identity
from vllm_gaudi.entrypoints.serving_resources import reuse_recipe_bundle


def bundle(tmp_path):
    binary = tmp_path / "runtime.so"
    binary.write_bytes(b"runtime contents")
    profile = dict(environment={}, additional_libraries=[dict(path=str(binary))], configuration_files=[])
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(profile))
    old, new = tmp_path / "old", tmp_path / "canonical"
    old.mkdir()
    new.mkdir()
    identity = dict(schema=2, model="checkpoint", arguments=["shape"], runtime=runtime_content_identity(profile))
    (old / "identity.json").write_text(json.dumps(identity))
    for rank in range(4):
        directory = old / f"rank{rank}"
        directory.mkdir()
        (directory / "recipe").write_bytes(b"SDK serialized recipe")
    return old, new, identity, profile, dict(directory=str(old), runtime_profile=str(path))


def test_recipe_aliases_retain_existing_files_and_do_not_duplicate(tmp_path):
    old, new, identity, profile, compatible = bundle(tmp_path)
    assert reuse_recipe_bundle(new, identity, profile, compatible) == 4
    assert reuse_recipe_bundle(new, identity, profile, compatible) == 0
    for rank in range(4):
        assert (new / f"rank{rank}" / "recipe").stat().st_ino == (old / f"rank{rank}" / "recipe").stat().st_ino


@pytest.mark.parametrize("changed", ("model", "arguments", "runtime"))
def test_incompatible_recipe_certificate_is_rejected(tmp_path, changed):
    old, new, identity, profile, compatible = bundle(tmp_path)
    record = json.loads((old / "identity.json").read_text())
    record[changed] = "different"
    (old / "identity.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="differs"):
        reuse_recipe_bundle(new, identity, profile, compatible)
    assert not list(new.iterdir())


def test_changed_library_does_not_adopt_old_recipes(tmp_path):
    old, new, identity, profile, compatible = bundle(tmp_path)
    (tmp_path / "runtime.so").write_bytes(b"changed runtime contents")
    with pytest.raises(ValueError, match="differs"):
        reuse_recipe_bundle(new, identity, profile, compatible)
