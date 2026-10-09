# SPDX-License-Identifier: Apache-2.0
"""Tensor-ready signals must preserve producer ownership and value identity."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_tensor_ready_graph_contract():
    workspace = Path(__file__).resolve().parents[3]
    environment = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    checked = subprocess.run(
        [sys.executable, str(workspace / 'tools/check_deepseek_v41_tensor_ready_graph_contract.py')],
        cwd=workspace, env=environment, capture_output=True, text=True, check=True,
    )
    result = json.loads(checked.stdout.splitlines()[-1])
    assert result['status'] == 'passed'
    assert result['arithmetic_rejected'] and result['ambiguous_signal_rejected']
    assert result['producer_control_recipe_count'] == 1 and result['unchanged_gemms'] == 2
