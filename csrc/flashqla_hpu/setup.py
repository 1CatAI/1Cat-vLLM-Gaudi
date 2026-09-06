from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir


setup(
    name="flashqla_pair_transform_pt2",
    ext_modules=[
        CppExtension(
            "flashqla_pair_transform_pt2",
            ["flashqla_pair_transform_pt2.cpp"],
            include_dirs=[get_include_dir(), "/usr/include/habanalabs"],
            library_dirs=[get_lib_dir()],
            libraries=["habana_pytorch2_plugin.upstream"],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
