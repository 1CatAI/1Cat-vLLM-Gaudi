import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.dist import Distribution
from setuptools_scm import get_version

try:
    VERSION = get_version(write_to="vllm_gaudi/_version.py")
except LookupError:
    # The checkout action in github action CI does not checkout the tag. It
    # only checks out the commit. In this case, we set a dummy version.
    VERSION = "0.0.0"

ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)
ext_modules = []


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> list[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    try:
        requirements = _read_requirements("requirements.txt")
    except ValueError:
        print("Failed to read requirements.txt in vllm_gaudi.")
    return requirements


def _native_flashinfer_build_enabled() -> bool:
    mode = os.environ.get("VLLM_GAUDI_BUILD_FLASHINFER", "auto").strip().lower()
    if mode in ("0", "false", "off", "no"):
        return False
    if mode in ("1", "true", "on", "yes"):
        return True
    if mode != "auto":
        raise ValueError("VLLM_GAUDI_BUILD_FLASHINFER must be auto, 0, or 1.")
    return shutil.which("tpc-clang") is not None and importlib.util.find_spec("habana_frameworks") is not None


class FlashInferBuildPy(_build_py):
    """Optionally build version-pinned native kernels into the package."""

    def run(self):
        super().run()
        output_dir = os.path.join(self.build_lib, "flashinfer_gaudi", "lib")
        # Private-ABI research adapters are source-build-only, including when
        # the reusable wheel staging directory contains an earlier build.
        for name in ("flashinfer_gaudi_bridge_ops.so", "bridge_artifact_v1.json"):
            staged = Path(output_dir) / name
            if staged.is_file():
                staged.unlink()
        if not _native_flashinfer_build_enabled():
            # A previous native wheel build may have left binaries in the
            # reusable setuptools build tree. Never leak those artifacts into
            # a later explicitly pure build.
            for library in Path(output_dir).glob("*.so"):
                library.unlink()
            return
        subprocess.run(
            [
                sys.executable,
                get_path("tools", "build_flashinfer_gaudi.py"),
                "--output-dir",
                output_dir,
            ],
            check=True,
        )


class FlashInferDistribution(Distribution):
    """Mark native builds as platform wheels even though build_py creates them."""

    def has_ext_modules(self) -> bool:
        return _native_flashinfer_build_enabled() or super().has_ext_modules()


setup(
    name="vllm_gaudi",
    version=VERSION,
    author="Intel",
    long_description="Intel Gaudi plugin package for vLLM.",
    long_description_content_type="text/markdown",
    url="https://github.com/vllm-project/vllm-gaudi",
    project_urls={
        "Homepage": "https://github.com/vllm-project/vllm-gaudi",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=("docs", "examples", "tests*", "csrc")),
    package_data={
        "flashinfer_gaudi": [
            "tactics/*.json",
            *(["lib/*.so"] if _native_flashinfer_build_enabled() else []),
        ]
    },
    exclude_package_data={
        "flashinfer_gaudi": [
            "lib/flashinfer_gaudi_bridge_ops.so",
            "lib/bridge_artifact_v1.json",
            *([] if _native_flashinfer_build_enabled() else ["lib/*.so"]),
        ]
    },
    py_modules=["pytest_compat"],
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_py": FlashInferBuildPy},
    distclass=FlashInferDistribution,
    extras_require={
        # Keep the multimodal stack opt-in so text-only vLLM deployments do
        # not install video/audio dependencies. This commit is the H3 pipeline
        # baseline qualified with this plugin.
        "omni": [
            "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@767cc7977e04dc1f1ae7307e630429e9169af622",
            "transformers==5.14.1",
            "diffusers==0.40.0",
            "modelscope==1.40.0",
            "imageio-ffmpeg==0.6.0",
        ],
    },
    entry_points={
        "vllm.platform_plugins": ["hpu = vllm_gaudi:register"],
        "vllm_omni.platform_plugins": ["hpu = vllm_gaudi.omni:register_omni_platform"],
        "vllm_omni.general_plugins": ["hpu = vllm_gaudi.omni:register_omni"],
        "vllm.general_plugins": [
            "01.hpu_custom_utils = vllm_gaudi:register_utils",
            "02.hpu_custom_ops = vllm_gaudi:register_ops",
            "03.hpu_custom_models = vllm_gaudi:register_models",
            "04.hpu_tool_parsers = vllm_gaudi:register_tool_parsers",
        ],
        "pytest11": ["vllm_gaudi_compat = pytest_compat"],
    },
)
