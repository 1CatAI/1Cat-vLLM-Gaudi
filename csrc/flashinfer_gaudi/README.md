# Native FlashInfer-Gaudi kernels

This directory contains the version-pinned native implementation used by the
in-tree `flashinfer_gaudi` package.

- `kernels/` contains TPC-C device code.
- `host/` contains the Gaudi kernel-database glue.
- `pytorch/` registers the public op through the PyTorch CustomOp API.

The kernel database currently contains eight exact-shape Gaudi2 kernels:

- packed one-token Qwen3.8 GDN decode;
- packed eight-token Qwen3.8 DFlash2 GDN verification with rollback
  checkpoints;
- a seven-step, top-16 Qwen3.8 DFlash2 greedy lattice walk;
- fused selected-row DFlash2 edge scoring and path selection;
- an experimental functional, full-query MTP core returning output and all
  checkpoints, with graph-compatible CustomOp registration;
- fused BF16 SiLU-and-multiply;
- fused BF16 SiLU-and-multiply plus dynamic FP8 quantization; and
- block-scaled FP8 dequantization.

Build both libraries into `flashinfer_gaudi/lib` with:

```bash
python3 tools/build_flashinfer_gaudi.py
```

The build requires the Gaudi TPC compiler, the PyTorch Gaudi bridge headers,
and the same Gaudi software version used at runtime. The optional bridge
backend is intentionally separate from this stable public-CustomOp path.
Building a kernel only makes it available; `auto` promotion remains controlled
by the offline tactic manifest after hardware correctness and performance
qualification.
