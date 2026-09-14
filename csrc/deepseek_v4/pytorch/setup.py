# SPDX-License-Identifier: Apache-2.0
"""Build against the source/generated headers of the installed Gaudi Bridge."""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir


def directory(name):
    value = os.environ.get(name)
    if not value or not Path(value).is_dir():
        raise RuntimeError(f"{name} must point to the matching Gaudi Bridge headers")
    return Path(value).resolve()


bridge = directory("GAUDI_PYTORCH_BRIDGE_ROOT")
build = directory("GAUDI_PYTORCH_BRIDGE_BUILD_ROOT")
includes = [
    get_include_dir(), "/usr/include/habanalabs", "/usr/include/habanalabs/hl_logger", str(bridge),
    str(bridge / "pytorch_helpers"),
    str(bridge / "python_packages/habana_frameworks/torch/jit/csrc"),
    str(bridge / "pytorch_helpers/habana_helpers/habana_serialization/include"),
    str(build), str(build / "_deps"),
    str(build / "_deps/abseil-cpp-src"),
    str(build / "_deps/magic_enum-src/include"),
    str(build / "_deps/fmt-src/include"),
    str(build / "_deps/exprtk-src/include"),
    str(build / "_deps/nlohmann_json-src/include"),
]
if os.environ.get("GAUDI_HL_LOGGER_INCLUDE"):
    includes.append(str(directory("GAUDI_HL_LOGGER_INCLUDE")))

setup(
    name="hpu_dsv4_sparse_attn_pt2",
    ext_modules=[CppExtension(
        "hpu_dsv4_sparse_attn_pt2",
        ["hpu_dsv4_sparse_attn_pt2.cpp", "hpu_dsv4_mhc_pt2.cpp", "hpu_dsv4_router_pt2.cpp",
         "hpu_dsv4_mxfp4_mme_pt2.cpp", "hpu_dsv4_native_attention_pt2.cpp", "hpu_dsv4_sinkhorn_pt2.cpp",
         "hpu_dsv41_mxfp4_mme_pt2.cpp", "hpu_dsv41_expert_n256_pt2.cpp", "hpu_dsv41_indexed_moe_pt2.cpp",
         "hpu_dsv41_quant_roundtrip_pt2.cpp", "hpu_dsv41_selected_kv_pt2.cpp", "hpu_dsv41_selected_mla_pt2.cpp", "hpu_dsv41_kv_pack_pt2.cpp",
         "hpu_dsv41_rope_pt2.cpp", "hpu_dsv41_prefix_layout_pt2.cpp"],
        include_dirs=includes,
        library_dirs=[get_lib_dir()],
        libraries=["habana_pytorch2_plugin.upstream", "habana_pytorch_backend.upstream"],
        extra_compile_args=["-O2", "-std=c++17", "-DFMT_HEADER_ONLY=1", "-DGENERIC_HELPERS"],
    )],
    cmdclass={"build_ext": BuildExtension},
)
