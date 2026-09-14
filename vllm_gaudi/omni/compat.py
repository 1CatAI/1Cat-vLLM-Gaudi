# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility bridges for the vLLM and vLLM-Omni pinned revisions."""

from __future__ import annotations

import sys


def _install_protocol_modules() -> None:
    """Provide protocol module paths introduced after pinned vLLM."""

    from vllm.entrypoints.openai.engine import protocol

    for module_name in (
            "vllm.entrypoints.generate.base.protocol",
            "vllm.entrypoints.serve.engine.protocol",
    ):
        sys.modules.setdefault(module_name, protocol)


def install_vllm_omni_compat() -> None:
    """Install idempotent import bridges required by pinned vLLM-Omni."""

    _install_protocol_modules()


__all__ = ["install_vllm_omni_compat"]
