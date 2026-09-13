# SPDX-License-Identifier: Apache-2.0
"""Build the CPU Engram gather with pybind headers from the locked torch wheel."""

from pathlib import Path
from setuptools import Extension, setup
import importlib.util

torch = importlib.util.find_spec("torch")
if torch is None or not torch.submodule_search_locations:
    raise RuntimeError("Use the candidate's locked PyTorch environment to build the host gather")
include = str(Path(next(iter(torch.submodule_search_locations))) / "include")
setup(name="dsv41_host_gather", ext_modules=[Extension(
    "dsv41_host_gather", ["host_gather.cpp"], include_dirs=[include], language="c++",
    extra_compile_args=["-O2", "-std=c++17", "-pthread"], extra_link_args=["-pthread"],
)])
