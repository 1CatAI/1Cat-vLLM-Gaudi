# SPDX-License-Identifier: Apache-2.0
"""Check full/sliced K maps and descriptor ownership through the installed glue."""
import os
from pathlib import Path
import subprocess

import pytest


def test_affine_maps_and_descriptor_ownership(tmp_path):
    library = os.getenv("DSV41_SPLIT_SCALE_PLANES_GC_LIBRARY")
    base = os.getenv("DSV41_SPLIT_SCALE_PLANES_BASE_LIBRARY")
    if not library or not base:
        pytest.skip("Set the private GC library and its matching base")
    source = Path(__file__).with_name("check_split_scale_planes_glue.cpp")
    executable = tmp_path / "check_glue"
    subprocess.run(["g++", "-O2", "-I/usr/include/habanalabs", str(source), "-ldl", "-o", str(executable)],
                   check=True)
    environment = dict(os.environ, VLLM_HPU_DSV41_UNIQUE_BASE_KERNEL=base)
    subprocess.run([str(executable), library, os.getenv("DSV41_EXPERT_K_TILE", "128")],
                   env=environment, check=True)
