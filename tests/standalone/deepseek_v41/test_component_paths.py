# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from tools.deepseek_v41_component_paths import component_scratch


def test_nested_evidence_reuses_profile_short_ipc_root():
    a = component_scratch(Path("/ssd/archive/deeply/nested/experiment/arm-a"), "/ssd/ipc/old-hash")
    b = component_scratch(Path("/ssd/archive/deeply/nested/experiment/arm-b"), "/ssd/ipc/old-hash")
    assert a.parent == b.parent == Path("/ssd/ipc")
    assert a != b


def test_long_unix_socket_path_fails_before_device_open():
    with pytest.raises(ValueError, match="sockaddr_un"):
        component_scratch(Path("/ssd/experiment"), "/" + "a" * 80 + "/old-hash")


def test_explicit_short_root_handles_long_multibyte_case():
    case = Path("/ssd/实验" + "长" * 80)
    with pytest.raises(ValueError, match="sockaddr_un"):
        component_scratch(case / "nested", None)
    assert component_scratch(case / "nested", None, Path("/ssd/ipc")).parent == Path("/ssd/ipc")
