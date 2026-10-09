# SPDX-License-Identifier: Apache-2.0
"""Draft registration rebuilds preserve immutable parent installations."""
import json

import pytest

from tools.build_deepseek_v41_mtp_k128 import digest, install_registration


@pytest.mark.parametrize("existing", [False, True])
def test_registration_rebuild_preserves_parent(tmp_path, existing):
    parent, installed = tmp_path / "parent", tmp_path / "installed"
    parent.mkdir()
    addon = tmp_path / "draft.so"
    addon.write_bytes(b"new draft registration")
    base = parent / "base.so"
    base.write_bytes(b"immutable common decoder")
    if existing:
        (parent / addon.name).write_bytes(b"immutable old draft registration")
    manifest = {"binaries": {base.name: digest(base)}, "extra_registrations": [addon.name] if existing else []}
    (parent / "deepseek_v41_unique_build.json").write_text(json.dumps(manifest))
    before = {path.name: path.read_bytes() for path in parent.iterdir()}

    manifest_path, result = install_registration(parent, installed, addon)

    assert {path.name: path.read_bytes() for path in parent.iterdir()} == before
    assert not (installed / addon.name).is_symlink()
    assert (installed / addon.name).read_bytes() == addon.read_bytes()
    assert (installed / base.name).is_symlink()
    assert result["binaries"][addon.name] == digest(addon)
    assert result["extra_registrations"].count(addon.name) == 1
    assert manifest_path.read_bytes() == before[manifest_path.name]


def test_registration_rebuild_rejects_an_existing_output(tmp_path):
    parent, installed = tmp_path / "parent", tmp_path / "installed"
    parent.mkdir()
    installed.mkdir()
    addon = tmp_path / "draft.so"
    addon.write_bytes(b"candidate")
    with pytest.raises(FileExistsError):
        install_registration(parent, installed, addon)
