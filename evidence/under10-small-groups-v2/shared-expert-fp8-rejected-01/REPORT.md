# V4.1 shared-expert FP8 — rejected at complete-chain microbenchmark

- Scope: four real PP0 layers, complete gate/up → SwiGLU → down → routed add → BF16 consumer boundary.
- Reference BF16 device sweep: **0.179304208 ms**.
- Candidate FP8 device sweep: **0.189978240 ms**.
- Regression: **+0.010674031 ms (5.953%)**.
- Weight traffic/storage fell from 135.039 MiB to 67.652 MiB per four-layer sweep, but two dynamic quantizers and output-scale kernels outweighed the C1 MME/memory saving.
- Cross-arm output relative L2 was 1.99e-4–3.53e-4 after the routed add.
- Decision: do not integrate and do not spend an end-to-end model load on this candidate.
