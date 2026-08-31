# FlashQLA HPU research kernels

This directory contains Gaudi2 research prototypes used to determine which
parts of FlashQLA can profitably be moved to public HPU custom operators.
They are not loaded by the vLLM Gaudi runtime. FlashQLA's TileLang kernels
target NVIDIA SMs and cannot be compiled for HPU; public Gaudi custom ops run
on TPC and cannot issue MME matrix operations from the same kernel.

The current prototypes are numerically correct but slower than the compiled
HPU graph at Qwen3.8-27B's 16K shape. See `RESULTS.md` before attempting to
wire one into the model path.

`DESIGN.md` describes the mixed MME/TPC implementation required for a useful
port. `native_backend/` contains a research-only compound-backend probe; it is
not loaded by vLLM.

Build the TPC performance library and PyTorch registration module:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python setup.py build_ext --inplace
```

Before starting Python, prepend the resulting performance library to
`GC_KERNEL_PATH` and load `flashqla_pair_transform_pt2*.so` with
`torch.ops.load_library`.
