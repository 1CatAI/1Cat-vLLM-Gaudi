# SPDX-License-Identifier: Apache-2.0
"""Optional diagnostic build, using the exact serving Bridge generated headers."""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir

bridge = Path(os.environ["GAUDI_PYTORCH_BRIDGE_ROOT"]).resolve(strict=True)
build = Path(os.environ["GAUDI_PYTORCH_BRIDGE_BUILD_ROOT"]).resolve(strict=True)
includes = [
    get_include_dir(), "/usr/include/habanalabs", "/usr/include/habanalabs/hl_logger",
    str(bridge),
    str(bridge / "pytorch_helpers"),
    str(bridge / "python_packages/habana_frameworks/torch/jit/csrc"),
    str(bridge / "pytorch_helpers/habana_helpers/habana_serialization/include"),
    str(build),
    str(build / "_deps")
]
includes += [
    str(build / "_deps" / path) for path in ("abseil-cpp-src", "magic_enum-src/include", "fmt-src/include",
                                             "exprtk-src/include", "nlohmann_json-src/include")
]
setup(name="dsv41_micro_compute",
      ext_modules=[
          CppExtension("dsv41_micro_compute", [str(Path(__file__).with_name("dsv41_micro_compute.cpp"))],
                       include_dirs=includes,
                       library_dirs=[get_lib_dir()],
                       libraries=["habana_pytorch2_plugin.upstream", "habana_pytorch_backend.upstream"],
                       extra_compile_args=["-O2", "-std=c++17", "-DFMT_HEADER_ONLY=1", "-DGENERIC_HELPERS"])
      ],
      cmdclass={"build_ext": BuildExtension})
