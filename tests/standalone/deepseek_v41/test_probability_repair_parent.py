# SPDX-License-Identifier: Apache-2.0
"""An installed repair database selects its immutable parent automatically."""
import hashlib
import os

import pytest

from vllm_gaudi.entrypoints.deepseek_v41 import configure_probability_repair_parent


def test_parent_is_derived_from_manifest_and_corruption_rejected(tmp_path, monkeypatch):
    variable = 'VLLM_HPU_DSV41_FULL_CDF_PARENT_KERNEL'
    monkeypatch.delenv(variable, raising=False)
    parent = tmp_path / 'parent.so'
    parent.write_bytes(b'locked parent')
    manifest = dict(repair_build=dict(parent_gc_path=str(parent),
                                     parent_gc_sha256=hashlib.sha256(parent.read_bytes()).hexdigest()))
    configure_probability_repair_parent(manifest, tmp_path / 'child.so')
    assert os.environ[variable] == str(parent)
    parent.write_bytes(b'changed parent')
    with pytest.raises(RuntimeError, match='locked parent'):
        configure_probability_repair_parent(manifest, tmp_path / 'child.so')


def test_other_database_and_parent_override(tmp_path, monkeypatch):
    variable = 'VLLM_HPU_DSV41_FULL_CDF_PARENT_KERNEL'
    monkeypatch.delenv(variable, raising=False)
    configure_probability_repair_parent({}, tmp_path / 'child.so')
    assert variable not in os.environ
    parent = tmp_path / 'parent.so'
    parent.write_bytes(b'locked parent')
    manifest = dict(repair_build=dict(parent_gc_path=str(parent),
                                     parent_gc_sha256=hashlib.sha256(parent.read_bytes()).hexdigest()))
    monkeypatch.setenv(variable, str(tmp_path / 'unrelated.so'))
    with pytest.raises(RuntimeError, match='override'):
        configure_probability_repair_parent(manifest, tmp_path / 'child.so')
