# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir

setup(
    name="flashinfer_gaudi_ops",
    ext_modules=[
        CppExtension(
            "flashinfer_gaudi_ops",
            ["hpu_flashinfer_gaudi.cpp"],
            include_dirs=[get_include_dir(), "/usr/include/habanalabs"],
            library_dirs=[get_lib_dir()],
            libraries=["habana_pytorch2_plugin.upstream"],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

