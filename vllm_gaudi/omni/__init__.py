# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-Omni integration for Intel Gaudi."""


def register_omni_platform() -> str | None:
    """Activate the Omni HPU platform only when a Gaudi device is usable."""

    try:
        import torch
    except ImportError:
        return None
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        return None

    # Keep detection side-effect free. Omni suppresses errors raised while it
    # probes platform plugins; compatibility setup therefore happens when it
    # resolves the selected class, where an import failure is fatal instead of
    # silently selecting the unspecified platform.
    return "vllm_gaudi.omni.platform.HPUOmniPlatform"


def register_omni() -> None:
    """Install Gaudi-specific vLLM-Omni compatibility hooks."""

    from vllm_gaudi.omni.compat import install_vllm_omni_compat
    from vllm_gaudi.omni.minimax_h3 import install_minimax_h3_patches

    install_vllm_omni_compat()
    install_minimax_h3_patches()


__all__ = ["register_omni", "register_omni_platform"]
