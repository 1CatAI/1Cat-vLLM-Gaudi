# SPDX-License-Identifier: Apache-2.0
"""Build full-CDF repair and correct the certified draw's vector scan."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build_root.resolve()
    build.mkdir(exist_ok=False, parents=True)
    kind = "probability_full_draw"
    sources=[root / f"csrc/deepseek_v41_unique/{kind}_host.cpp"]
    objects=[]
    names=["deepseek_v41_probability_full_select_gaudi2", "deepseek_v41_probability_draw_gaudi2"]
    for name in names:
        source = root / f"csrc/deepseek_v41_unique/kernels/{name}.c"
        sources.append(source)
        subprocess.run(["tpc-clang", "-Wall", "-Werror", "-O2", "-march=gaudi2",
                        "-I/usr/lib/habanatools/include", str(source), "-c", "-o", str(build / f"{name}.o")],
                       check=True)
        subprocess.run(["objcopy", "-I", "binary", "-O", "elf64-x86-64", "-B", "i386:x86-64",
                        f"./{name}.o", f"{name}_embedded.o"], cwd=build, check=True)
        objects.append(str(build / f"{name}_embedded.o"))
        with (build / f"{name}.isa").open("w") as output:
            subprocess.run(["tpc-llvm-objdump", "--triple=tpc", "--mcpu=gaudi2", "-d",
                            "--no-show-raw-insn", str(build / f"{name}.o")], stdout=output, check=True)
    library = build / "libdeepseek_v41_unique_kernels.so"
    subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    "-I/usr/include/habanalabs", str(sources[0]), *objects, "-ldl",
                    "-Wl,-z,noexecstack", "-Wl,--no-undefined", "-o", str(library)], check=True)
    (build / "build.json").write_text(json.dumps(dict(
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        binary_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        default_enabled=False, rebuilt_parent_guids=["custom_deepseek_v41_probability_draw_gaudi2"],
        remaining_parent_kernels_rebuilt=False,
        required_parent_env="VLLM_HPU_DSV41_FULL_CDF_PARENT_KERNEL",
        compiler=shutil.which("tpc-clang")), indent=2) + "\n")
    print(library)


if __name__ == "__main__":
    main()
