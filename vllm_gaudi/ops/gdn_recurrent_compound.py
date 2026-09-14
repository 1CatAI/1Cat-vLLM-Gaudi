# SPDX-License-Identifier: Apache-2.0
"""Opt-in loader for the mixed MME/TPC Qwen3.8 recurrent-state scope."""

from pathlib import Path

import torch

import vllm_gaudi.envs as gaudi_envs


_loaded_extension: Path | None = None
_COMPOUND_VARIANTS = {
    "plain": "hpu_gdn_recurrent_compound",
    "recompute_decay": "hpu_gdn_recurrent_superbundle_recompute_decay",
    "recompute_decay_native": (
        "hpu_gdn_recurrent_superbundle_recompute_decay_native"
    ),
    "recompute_decay_graph": (
        "hpu_gdn_recurrent_compound_recompute_decay_graph"
    ),
    "native_decay_graph": (
        "hpu_gdn_recurrent_compound_native_decay_graph"
    ),
    "native_decay_bundled": (
        "hpu_gdn_recurrent_compound_native_decay_bundled"
    ),
    "decay_copy_graph": (
        "hpu_gdn_recurrent_compound_decay_copy_graph"
    ),
    "decay_copy_round_graph": (
        "hpu_gdn_recurrent_compound_decay_copy_round_graph"
    ),
}


def _selected_op_name() -> str:
    variant = gaudi_envs.VLLM_HPU_GDN_RECURRENT_COMPOUND_VARIANT
    try:
        return _COMPOUND_VARIANTS[variant]
    except KeyError as exc:
        supported = ", ".join(sorted(_COMPOUND_VARIANTS))
        raise RuntimeError(
            "VLLM_HPU_GDN_RECURRENT_COMPOUND_VARIANT must be one of "
            f"{supported}; got {variant!r}"
        ) from exc


def _resolve_compound_op():
    return getattr(torch.ops.custom_op, _selected_op_name())


def load_gdn_recurrent_compound() -> bool:
    """Load the research backend before graph capture or state mutation."""
    global _loaded_extension
    if not gaudi_envs.VLLM_HPU_GDN_RECURRENT_COMPOUND:
        return False
    configured_path = gaudi_envs.VLLM_HPU_GDN_RECURRENT_COMPOUND_EXTENSION
    if not configured_path:
        raise RuntimeError(
            "VLLM_HPU_GDN_RECURRENT_COMPOUND_EXTENSION must identify the "
            "matching native GDN extension"
        )
    extension = Path(configured_path).expanduser().resolve()
    if not extension.is_file():
        raise RuntimeError(f"GDN recurrent compound extension does not exist: {extension}")
    if _loaded_extension is not None:
        if extension != _loaded_extension:
            raise RuntimeError(
                f"GDN recurrent compound was already loaded from {_loaded_extension}, "
                f"cannot reload {extension}"
            )
        return True
    torch.ops.load_library(str(extension))
    try:
        _resolve_compound_op()
    except AttributeError as exc:
        raise RuntimeError(
            "The configured extension does not register "
            f"custom_op::{_selected_op_name()}"
        ) from exc
    _loaded_extension = extension
    return True


def gdn_recurrent_compound(
    state: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact production-node recurrence inside one compiler scope."""
    if not gaudi_envs.VLLM_HPU_GDN_RECURRENT_COMPOUND:
        raise RuntimeError("GDN recurrent compound was called while disabled")
    if _loaded_extension is None:
        raise RuntimeError("GDN recurrent compound must be loaded before model execution")
    return _resolve_compound_op()(
        state,
        decay,
        key,
        query,
        value,
        beta,
    )
