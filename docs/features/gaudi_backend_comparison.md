# Comparing Gaudi operator implementations

The checkout contains the default HPU graph, the FlashInfer-Gaudi graph tactics,
and the experimental Triton-Gaudi2 kernels. They share PyTorch HPU tensors and
the vLLM execution engine; selecting a tactic does not replace the entire model
backend.

| Selection | Configuration | Behavior |
| --- | --- | --- |
| Default HPU | `VLLM_HPU_TRITON_MODE=off`, `VLLM_HPU_FLASHINFER_GDN=0` | Existing HPU operators and graph compiler |
| FlashInfer-Gaudi | Triton `off`, `VLLM_HPU_FLASHINFER_GDN=1`, `FLASHINFER_GAUDI_BACKEND=auto` | Qualified GDN graph tactics, with existing HPU paths for other shapes |
| Triton diagnostic | `VLLM_HPU_TRITON_MODE=strict` | Explicit supported Triton candidates; rejected candidates raise |
| Combined rollout | Triton `hybrid`, FlashInfer GDN enabled | Only qualified Triton paths; GDN retains the FlashInfer/vendor selection |

Triton strict takes precedence for recurrent GDN decode and row-wise dynamic
quantization. The stateful decode adapter requires the same tensor for load and
store indices. Split convolution/GDN kernels remain standalone diagnostics due
to their recipe-reentry state limitation. TP2 fused all-reduce/RMSNorm retains
its collective boundary before a local Triton RMSNorm is considered.

## Operator benchmark

Install matching Gaudi2 Triton and SynapseAI 1.24.1 Bridge builds, and prepare
their perf-library and artifact-cache environment as for the existing Triton
operator benchmarks. Run on an otherwise unused HPU:

```bash
python tools/benchmark_gaudi_three_way.py \
  --batches 1,8,32 --warmup 20 --iterations 100 --rounds 7 \
  --isolate-cases --output comparison.json
```

The benchmark uses Qwen-sized residual RMSNorm, SiLU/multiply, per-row FP8
quantization, gated recurrent decode, and convolution plus gated recurrence.
Each shape runs all three implementations on identical seeded inputs and the
same owned cache allocation. All use static `hpu_backend` fullgraph compilation.
Correctness checks include multiple state updates and exact convolution-cache
checks. Timed rounds rotate backend order and report device-event and wall time.

FlashInfer-Gaudi uses its promoted direct-state compiled graph for GDN, not the
unpromoted native TPC tactic. For RMSNorm and SiLU it shares the ordinary HPU
implementation; its quantization entry measures the associated HPU CGUID path.
These actual implementations are recorded in JSON. The Triton convolution case
uses the existing HPU convolution plus Triton recurrent kernel, matching the
model integration.

FP8 correctness uses the existing dynamic-quant scale and dequantized-value
tolerances, and separately records FP8 code equality. Passing that numerical
gate does not imply identical model tokens. Failed checks produce no speedup.

`--isolate-cases` starts a fresh process for each shape, so its measurements do
not validate cross-shape recipe reuse. Omit it to exercise shared-process
compilation. The experimental stateful Triton path has exhibited GDN compilation
failures after changing batch specialization; it therefore remains disabled in
hybrid model execution. Preserve that diagnostic separately from isolated
operator timings.

These measurements do not include model prefill, scheduling, first-token latency,
or complete request throughput. They cannot establish an end-to-end speedup.
