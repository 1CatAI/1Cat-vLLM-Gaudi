# SPDX-License-Identifier: Apache-2.0
"""Normal installations select guarded reuse while retaining explicit disable."""
import json
import os
from types import SimpleNamespace

import pytest

from vllm_gaudi.entrypoints import serving_resources


@pytest.mark.parametrize("disabled", (False, True))
def test_recipe_installation_selects_frontend_namespace(tmp_path, monkeypatch, disabled):
    package = tmp_path / "plugin"
    engine = tmp_path / "engine"
    model = tmp_path / "model"
    (package / "entrypoints").mkdir(parents=True)
    (engine / "vllm").mkdir(parents=True)
    model.mkdir()
    (package / "__init__.py").write_text("")
    (engine / "vllm" / "__init__.py").write_text("")
    (model / "manifest.json").write_text(json.dumps({"tensor_parallel_size": 4}))
    monkeypatch.setattr(serving_resources, "__file__", str(package / "entrypoints" / "serving_resources.py"))
    monkeypatch.setattr(serving_resources.importlib.util, "find_spec",
                        lambda _: SimpleNamespace(origin=str(engine / "vllm" / "__init__.py")))
    monkeypatch.setattr(serving_resources.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    monkeypatch.setenv("DSV41_SERVING_RUNTIME", "runtime-proof")
    monkeypatch.delenv("PT_HPU_RECIPE_CACHE_CONFIG", raising=False)
    monkeypatch.delenv("DSV41_SERVING_COMPILE_IDENTITY", raising=False)
    monkeypatch.delenv("VLLM_HPU_DSV41_FRONTEND_CACHE_DIR", raising=False)
    if disabled:
        monkeypatch.setenv("VLLM_HPU_DSV41_FRONTEND_CACHE_DIR", "")
    serving_resources.prepare_serving_resources({"recipe_cache_dir": str(tmp_path / "recipes")}, model,
                                                ["--enable-prefix-caching"])
    namespace = tmp_path / "recipes" / os.environ["DSV41_SERVING_COMPILE_IDENTITY"]
    assert (namespace / "identity.json").is_file()
    assert os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].startswith(str(namespace / "rank{rank}"))
    assert os.environ["VLLM_HPU_DSV41_FRONTEND_CACHE_DIR"] == ("" if disabled else str(namespace / "frontend"))
