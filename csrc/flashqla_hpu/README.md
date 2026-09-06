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

The compact post-convolution Q/K operator accepts the Qwen3.8 TP1 packed
width 10240 and TP2 packed width 5120. The build preserves the 16-head TPC
variant and adds an eight-head variant for TP2. Real and fake output metadata
both retain the local compact head count. The compact-KKT operator uses the
head and repeat dimensions supplied by its tensors and supports either layout.

To validate both native layouts, set `GC_KERNEL_PATH` and
`VLLM_GDN_QWEN38_NATIVE_QK_PREP_EXTENSION` to the newly built libraries, select
an available HPU, and run `pytest tests/unit_tests/ops/test_qwen38_native_qk.py`
from the repository root.
The hardware checks are skipped when the extension path is absent.
