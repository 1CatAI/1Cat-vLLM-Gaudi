# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tools.communication.tp2_validation import validate_outputs


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 100.0])
def test_invalid_values_cannot_enter_timing(value):
    expected = (torch.ones(8), torch.ones(8))
    actual = [item.clone() for item in expected]
    actual[0][0] = value
    valid, _ = validate_outputs(actual, expected, atol=0.01, rtol=0.01)
    assert not valid


@pytest.mark.parametrize("launches", [0, 1, 2])
def test_collective_execution_count_is_part_of_validation(launches):
    expected = (torch.ones(8), torch.ones(8))
    valid, errors = validate_outputs(expected, expected, atol=0, rtol=0, launches=launches, expected_launches=1)
    assert valid is (launches == 1)
    assert errors == [0, 0]


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1])
def test_reject_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        validate_outputs((), (), atol=tolerance, rtol=0)
