# SPDX-License-Identifier: Apache-2.0
"""Build only the new QK consumer kernels; existing GUIDs use the locked binary."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--stream-exp", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build_root.resolve()
    build.mkdir(exist_ok=False, parents=True)
    host = "mla_stream_exp_host.cpp" if args.stream_exp else "qk_flat_direct_host.cpp"
    sources = [root / ("csrc/deepseek_v41_unique/" + host),
               root / "csrc/deepseek_v4/kernels/deepseek_v41_selected_mla_softmax_gaudi2.c"]
    objects = []
    names = ([f"deepseek_v41_mla_stream_{kind}_{mode}_gaudi2"
              for kind in ("exp", "finish") for mode in ("publish", "reuse")]
             if args.stream_exp else [f"deepseek_v41_qk_flat_{mode}_softmax_gaudi2"
                                      for mode in ("publish", "reuse")])
    if args.stream_exp:
        sources += [root / "csrc/deepseek_v41_unique/include/mla_stream_exp_softmax.h",
                    root / "csrc/deepseek_v41_unique/include/mla_stream_exp_finish.h",
                    root / "csrc/deepseek_v4/include/deepseek_v4_qnorm_rope_kv_pack_bf16.h"]
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
        default_enabled=False, inherited_kernels_rebuilt=False,
        required_parent_env=("VLLM_HPU_DSV41_MLA_STREAM_PARENT_KERNEL" if args.stream_exp
                             else "VLLM_HPU_DSV41_QK_PARENT_KERNEL"),
        compiler=shutil.which("tpc-clang")), indent=2) + "\n")
    print(library)


if __name__ == "__main__":
    main()
