# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "native_graph_validation",
    Path(__file__).resolve().parents[3] / "tools/communication/native_graph_validation.py",
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def test_rejects_a_single_bf16_rounding_difference():
    reference = torch.tensor([1.0], dtype=torch.bfloat16)
    changed = torch.tensor([1.0078125], dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="Step 7.*exact mismatch"):
        validation.assert_exact((changed, ), (reference, ), step=7)


def test_checks_earlier_snapshots_even_when_last_is_correct():
    reference = torch.tensor([1.0], dtype=torch.bfloat16)
    wrong = torch.tensor([2.0], dtype=torch.bfloat16)
    references = {0: (reference, ), 1: (reference, )}
    with pytest.raises(RuntimeError, match="Step 0.*exact mismatch"):
        validation.validate_snapshots([(0, (wrong, )), (1, (reference, ))], references)


def test_dtype_and_output_count_are_part_of_exact_contract():
    reference = torch.tensor([1.0], dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        validation.assert_exact((reference.float(), ), (reference, ), step=0)
    with pytest.raises(RuntimeError, match="output count"):
        validation.assert_exact((), (reference, ), step=0)


def test_every_snapshot_is_counted():
    references = {step: (torch.tensor([step], dtype=torch.float32), ) for step in range(32)}
    assert validation.validate_snapshots(references.items(), references) == 32
