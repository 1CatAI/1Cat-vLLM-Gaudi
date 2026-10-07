# SPDX-License-Identifier: Apache-2.0
import pytest

from tools.archive_deepseek_v41_capture import diagnostic_capture


def test_formal_and_diagnostic_remain_distinct():
    assert not diagnostic_capture(dict(status="passed", profile="decode"))
    diagnostic = dict(status="diagnostic_passed", profile="decode", finish_reason="length")
    with pytest.raises(ValueError):
        diagnostic_capture(diagnostic)
    assert diagnostic_capture(diagnostic, allow_diagnostic=True)
    assert diagnostic["status"] == "diagnostic_passed"


@pytest.mark.parametrize("result", [
    dict(status="failed", profile="decode", finish_reason="error"),
    dict(status="failed", profile="decode", finish_reason="length"),
    dict(status="diagnostic_passed", profile="prefill"),
])
def test_allow_diagnostic_does_not_approve_failed_or_wrong_phase_capture(result):
    with pytest.raises(ValueError):
        diagnostic_capture(result, allow_diagnostic=True)


def test_legacy_truncated_capture_requires_its_explicit_option():
    result = dict(status="failed", profile="decode", finish_reason="length")
    assert diagnostic_capture(result, allow_truncated_diagnostic=True)
