# FlashQLA HPU kernels

This directory contains the optional Gaudi2 kernels used by the Qwen3.8 GDN
prefill path. FlashQLA's TileLang kernels target NVIDIA SMs and cannot be
compiled for HPU, so the port combines public TPC custom operators with the
compiled PyTorch MME graph. The runtime loads these kernels only when their
corresponding feature flags and extension paths are configured.

`native_backend/` contains the optional compound-backend bridge used by the
mixed-precision triangular solve experiment. It is not required by the
default path.

Build the TPC performance library and PyTorch registration module:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python setup.py build_ext --inplace
```

Before starting Python, prepend the resulting performance library to
`GC_KERNEL_PATH` and load `flashqla_pair_transform_pt2*.so` with
`torch.ops.load_library`.
