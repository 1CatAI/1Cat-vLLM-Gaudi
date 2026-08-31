import os
from pathlib import Path

from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


bridge_source = Path(
    os.environ.get(
        "GAUDI_PYTORCH_BRIDGE_SOURCE",
        "/tmp/gaudi-pytorch-bridge-review-codex-20260830",
    )
).resolve()
if not (bridge_source / "hpu_ops" / "op_backend.h").is_file():
    raise RuntimeError(
        "GAUDI_PYTORCH_BRIDGE_SOURCE must point to the source matching the "
        "installed habana-torch-plugin build"
    )

abseil_source = Path(
    os.environ.get("ABSEIL_CPP_SOURCE", "/tmp/abseil-cpp-20250512.1")
).resolve()
if not (abseil_source / "absl" / "container" / "flat_hash_map.h").is_file():
    raise RuntimeError(
        "ABSEIL_CPP_SOURCE must point to abseil-cpp tag 20250512.1, which "
        "matches the bridge dependency"
    )

bridge_generated = Path(
    os.environ.get("GAUDI_BRIDGE_GENERATED_SOURCE", "/tmp/gaudi-bridge-generated")
).resolve()
if not (
    bridge_generated / "generated" / "env_flags" / "env_flags_generated.h"
).is_file():
    raise RuntimeError(
        "GAUDI_BRIDGE_GENERATED_SOURCE must contain generated bridge headers"
    )

bridge_deps_root = Path(
    os.environ.get("GAUDI_BRIDGE_DEPS_ROOT", "/tmp")
).resolve()
for dependency in ("magic_enum-0.9.7", "fmt-9.1.0"):
    if not (bridge_deps_root / dependency / "include").is_dir():
        raise RuntimeError(
            "GAUDI_BRIDGE_DEPS_ROOT must contain magic_enum-0.9.7 and "
            "fmt-9.1.0"
        )

exprtk_source = Path(
    os.environ.get("EXPRTK_SOURCE", "/tmp/exprtk-0.0.3-cmake")
).resolve()
if not (exprtk_source / "include" / "exprtk.hpp").is_file():
    raise RuntimeError("EXPRTK_SOURCE must point to exprtk tag 0.0.3-cmake")


setup(
    name="flashqla_compound_probe",
    ext_modules=[
        CppExtension(
            "flashqla_compound_probe",
            ["flashqla_compound_probe.cpp"],
            include_dirs=[
                get_include_dir(),
                "/usr/include/habanalabs",
                str(bridge_source),
                str(abseil_source),
                str(bridge_generated),
                str(bridge_deps_root),
                str(exprtk_source / "include"),
                str(bridge_source / "pytorch_helpers"),
                str(
                    bridge_source
                    / "python_packages"
                    / "habana_frameworks"
                    / "torch"
                    / "jit"
                    / "csrc"
                ),
            ],
            library_dirs=[get_lib_dir()],
            libraries=[
                "habana_pytorch2_plugin.upstream",
                "habana_pytorch_backend.upstream",
            ],
            extra_compile_args=["-O2", "-std=c++17", "-DFMT_HEADER_ONLY"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
