# Native FlashInfer-Gaudi kernels

This directory contains the version-pinned native implementation used by the
in-tree `flashinfer_gaudi` package.

- `kernels/` contains TPC-C device code.
- `host/` contains the Gaudi kernel-database glue.
- `pytorch/` registers the public op through the PyTorch CustomOp API.

Build both libraries into `flashinfer_gaudi/lib` with:

```bash
python3 tools/build_flashinfer_gaudi.py
```

The build requires the Gaudi TPC compiler, the PyTorch Gaudi bridge headers,
and the same Gaudi software version used at runtime. The optional bridge
backend is intentionally separate from this stable public-CustomOp path.

