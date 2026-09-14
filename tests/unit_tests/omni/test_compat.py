# SPDX-License-Identifier: Apache-2.0

import sys

import torch

from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata

from vllm_gaudi.omni.compat import install_vllm_omni_compat
from vllm_gaudi.omni import register_omni_platform
from vllm_gaudi.platform import HpuPlatform


def test_request_metadata_import_bridge(monkeypatch):
    module_names = (
        "vllm.entrypoints.generate.base.protocol",
        "vllm.entrypoints.serve.engine.protocol",
    )
    for module_name in module_names:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    install_vllm_omni_compat()

    for module_name in module_names:
        module = sys.modules[module_name]
        assert module.RequestResponseMetadata is RequestResponseMetadata


def test_hpu_omni_platform_exposes_cache_eviction(monkeypatch):
    # Platform class paths are normally resolved after vllm_omni finishes its
    # package-level compatibility patching. Reproduce that entrypoint order in
    # this isolated test module as well.
    import vllm_omni  # noqa: F401

    from vllm_gaudi.omni.platform import HPUOmniPlatform

    calls = []
    monkeypatch.setattr(HpuPlatform, "empty_cache", classmethod(lambda cls: calls.append(cls)))

    HPUOmniPlatform.empty_cache()

    assert calls == [HpuPlatform]


def test_omni_platform_entrypoint_is_inactive_without_hpu(monkeypatch):
    monkeypatch.setattr(torch.hpu, "is_available", lambda: False)

    assert register_omni_platform() is None


def test_omni_platform_entrypoint_defers_compat_until_class_resolution(monkeypatch):
    monkeypatch.setattr(torch.hpu, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(
        "vllm_gaudi.omni.compat.install_vllm_omni_compat",
        lambda: calls.append("install"),
    )

    assert register_omni_platform() == "vllm_gaudi.omni.platform.HPUOmniPlatform"
    assert calls == []
