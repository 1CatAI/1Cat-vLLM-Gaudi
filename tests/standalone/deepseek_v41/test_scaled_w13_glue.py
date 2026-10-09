# SPDX-License-Identifier: Apache-2.0
"""CPU regression for borrowed glue descriptor ownership, without HPU init."""
import os
from pathlib import Path
import subprocess

import pytest


def test_repeat_instantiation_preserves_descriptors(tmp_path):
    library, base = os.getenv('DSV41_SCALED_W13_GC_LIBRARY'), os.getenv('DSV41_SCALED_W13_BASE_LIBRARY')
    if not library or not base:
        pytest.skip('Set the private GC library and its matching base')
    source = Path(__file__).with_name('check_scaled_w13_glue.cpp')
    executable = tmp_path / 'check_glue'
    subprocess.run(['g++', '-O2', '-I/usr/include/habanalabs', str(source), '-ldl', '-o', str(executable)], check=True)
    environment = dict(os.environ, VLLM_HPU_DSV41_UNIQUE_BASE_KERNEL=base)
    subprocess.run([str(executable), library], env=environment, check=True)
