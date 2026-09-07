# SPDX-License-Identifier: Apache-2.0
"""Registry compatibility for the optional row-parallel chunking path."""

import pytest

from vllm_gaudi.lora.layers import hpu_row_parallel_linear as row_lora


@pytest.mark.parametrize("container", [tuple, list, set])
def test_registration_preserves_other_wrappers_and_is_idempotent(monkeypatch, container):
    sentinel = object()
    registry = container([
        sentinel,
        row_lora.RowParallelLinearWithLoRA,
        row_lora.RowParallelLinearWithShardedLoRA,
    ])
    monkeypatch.setattr(row_lora.lora_utils, "_all_lora_classes", registry)
    expected = container([
        sentinel,
        row_lora.HPURowParallelLinearWithLoRA,
        row_lora.HPURowParallelLinearWithShardedLoRA,
    ])
    for _ in range(2):
        row_lora.register_hpu_lora_layers()
        actual = row_lora.lora_utils._all_lora_classes
        assert type(actual) is container
        assert actual == expected


def test_registration_deduplicates_existing_hpu_wrapper(monkeypatch):
    monkeypatch.setattr(row_lora.lora_utils, "_all_lora_classes", (
        row_lora.RowParallelLinearWithLoRA,
        row_lora.HPURowParallelLinearWithLoRA,
    ))
    row_lora.register_hpu_lora_layers()
    assert row_lora.lora_utils._all_lora_classes == (
        row_lora.HPURowParallelLinearWithLoRA,
        row_lora.HPURowParallelLinearWithShardedLoRA,
    )
