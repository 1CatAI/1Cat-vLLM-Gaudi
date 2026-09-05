# SPDX-License-Identifier: Apache-2.0
"""Build only; tools/build_flashinfer_gaudi_bridge.py seals the ABI manifest."""

import os
from pathlib import Path

from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

source = Path(os.environ["GAUDI_PYTORCH_BRIDGE_SOURCE"])
generated = Path(os.environ["GAUDI_BRIDGE_BUILD"])
deps = generated / "_deps"
includes = [
    get_include_dir(), "/usr/include/habanalabs",
    str(source),
    str(generated), os.environ["GAUDI_BRIDGE_DEPENDENCY_LAYOUT"],
    str(deps / "abseil-cpp-src"),
    str(deps / "exprtk-src/include"),
    str(deps / "fmt-src/include"),
    str(deps / "magic_enum-src/include"),
    str(source / "pytorch_helpers"),
    str(source / "python_packages/habana_frameworks/torch/jit/csrc")
]

setup(
    name="flashinfer_gaudi_bridge_ops",
    ext_modules=[
        CppExtension(
            "flashinfer_gaudi_bridge_ops",
            ["gemm_silu.cpp", "silu_quant.cpp"],
            include_dirs=includes,
            library_dirs=[get_lib_dir()],
            libraries=["habana_pytorch2_plugin.upstream", "habana_pytorch_backend.upstream"],
            extra_compile_args=["-O2", "-std=c++17", "-DFMT_HEADER_ONLY"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
