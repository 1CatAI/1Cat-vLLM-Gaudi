# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gaudi2 backend contract for the DeepSeek V4 FlashMLA decode path.

The orchestration and layer taxonomy intentionally mirror
``vllm.models.deepseek_v4.nvidia.flashmla``.  Hardware-specific CUDA
primitives stay behind this module so the Python path and metadata contract
can continue to track upstream vLLM.
"""

from dataclasses import dataclass
from enum import Enum

import torch


class HPUFlashMLABackend(str, Enum):
    AUTO = "auto"
    FLASHMLA = "flashmla_hpu"
    MME = "mme"
    LEGACY = "legacy"


class HPUFlashMLALayerType(str, Enum):
    SWA_ONLY = "swa_only"
    C4 = "c4"
    C128 = "c128"


@dataclass(frozen=True)
class HPUFlashMLADecodePlan:
    backend: HPUFlashMLABackend
    layer_type: HPUFlashMLALayerType
    uses_local_topk: bool
    uses_sequential_topk: bool


def parse_hpu_flashmla_backend(value: str) -> HPUFlashMLABackend:
    normalized = value.strip().lower()
    aliases = {
        "flashmla": HPUFlashMLABackend.FLASHMLA,
        "tpc": HPUFlashMLABackend.FLASHMLA,
        "fsdpa": HPUFlashMLABackend.MME,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return HPUFlashMLABackend(normalized)
    except ValueError as exc:
        choices = ", ".join(backend.value for backend in HPUFlashMLABackend)
        raise ValueError(
            "VLLM_HPU_DSV4_ATTENTION_BACKEND must be one of "
            f"{choices}; got {value!r}"
        ) from exc


def classify_hpu_flashmla_layer(
    *,
    swa_only: bool,
    has_local_topk: bool,
) -> HPUFlashMLALayerType:
    if swa_only:
        return HPUFlashMLALayerType.SWA_ONLY
    if has_local_topk:
        return HPUFlashMLALayerType.C4
    return HPUFlashMLALayerType.C128


def flashmla_output_buffer_is_compatible(
    q: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Return whether ``out`` can hold the active, possibly unpadded Q heads.

    H200 FlashMLA pads its workspace to 64/128 heads.  The Gaudi2 kernel does
    not need to execute those inactive heads, but it writes into the same
    caller-owned workspace contract.
    """

    return (
        q.ndim == 3
        and out.ndim == 3
        and out.shape[0] == q.shape[0]
        and out.shape[1] >= q.shape[1]
        and out.shape[2] == q.shape[2]
        and out.dtype == q.dtype
    )


def build_hpu_flashmla_decode_plan(
    *,
    requested_backend: str,
    swa_only: bool,
    has_local_topk: bool,
    uses_sequential_topk: bool,
    tpc_enabled: bool,
    mme_enabled: bool,
    tpc_eligible: bool,
    mme_eligible: bool,
) -> HPUFlashMLADecodePlan:
    requested = parse_hpu_flashmla_backend(requested_backend)
    layer_type = classify_hpu_flashmla_layer(
        swa_only=swa_only,
        has_local_topk=has_local_topk,
    )

    if requested == HPUFlashMLABackend.LEGACY:
        selected = HPUFlashMLABackend.LEGACY
    elif requested == HPUFlashMLABackend.MME:
        selected = (
            HPUFlashMLABackend.MME
            if mme_enabled and mme_eligible
            else HPUFlashMLABackend.LEGACY
        )
    elif requested == HPUFlashMLABackend.FLASHMLA:
        if tpc_enabled and tpc_eligible:
            selected = HPUFlashMLABackend.FLASHMLA
        elif mme_enabled and mme_eligible:
            selected = HPUFlashMLABackend.MME
        else:
            selected = HPUFlashMLABackend.LEGACY
    elif tpc_enabled and tpc_eligible:
        selected = HPUFlashMLABackend.FLASHMLA
    elif mme_enabled and mme_eligible:
        selected = HPUFlashMLABackend.MME
    else:
        selected = HPUFlashMLABackend.LEGACY

    return HPUFlashMLADecodePlan(
        backend=selected,
        layer_type=layer_type,
        uses_local_topk=has_local_topk,
        uses_sequential_topk=uses_sequential_topk,
    )
