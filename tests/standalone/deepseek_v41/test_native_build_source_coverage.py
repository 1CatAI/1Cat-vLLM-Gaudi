# SPDX-License-Identifier: Apache-2.0
"""A new backend must be compiled before a native artifact is published."""
import importlib.util
from pathlib import Path

import pytest


def checker():
    path = Path(__file__).resolve().parents[3] / "tools/build_deepseek_v4.py"
    spec = importlib.util.spec_from_file_location("dsv41_native_build_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_pytorch_source_coverage


def test_registered_sources_pass_without_importing_setup(tmp_path):
    (tmp_path / "setup.py").write_text("sources = ['hpu_valid.cpp']\n")
    (tmp_path / "hpu_valid.cpp").write_text("")
    checker()(tmp_path)


def test_unregistered_backend_fails_before_build(tmp_path):
    (tmp_path / "setup.py").write_text("sources = ['hpu_valid.cpp']\n")
    (tmp_path / "hpu_valid.cpp").write_text("")
    (tmp_path / "hpu_missing.cpp").write_text("")
    with pytest.raises(RuntimeError, match="hpu_missing.cpp"):
        checker()(tmp_path)


def test_repository_source_list_is_complete():
    root = Path(__file__).resolve().parents[3]
    checker()(root / "csrc/deepseek_v4/pytorch")
