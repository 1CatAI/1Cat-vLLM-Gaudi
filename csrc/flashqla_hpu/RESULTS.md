# FlashQLA Gaudi2 feasibility results

Date: 2026-08-30

Model shape: Qwen3.8-27B-FP8, single Gaudi2, 16,384 input tokens,
48 value heads, 16 Q/K heads, head dimension 128, chunk size 64.
All HPU measurements used `PT_HPU_LAZY_MODE=0` and `torch.compile`; none used
the eager model path.

## Baseline

- End-to-end prefill: 6,322 tokens/s, 2.591 seconds TTFT.
- One GDN core: 11.545 ms.
- One Phase A: 4.942 ms.
- 49,152 16x16 recursive-inverse base blocks: 1.442 ms.
- 12,288 complete 64x64 recursive inverses: 2.50 ms.

## Experiments

| Experiment | Reference | Candidate | Result |
| --- | ---: | ---: | --- |
| Gate-free FlashQLA algebra, Phase A | 4.942 ms | 4.827 ms | 2.3% faster |
| Gate-free algebra, complete GDN core | 11.638 ms | 11.625 ms | 0.1% faster |
| Fused pair transform, 12,288 matrices | 1.239 ms | 2.248 ms | 1.81x slower |
| Persistent recurrent TPC, one 16K layer | 11.545 ms | 589.863 ms | 51.1x slower |
| TPC 16x16 exact inverse | 1.442 ms | 4.337 ms | 3.01x slower |
| Eight-level affine parallel scan | 11.545 ms | 20.543 ms | 1.78x slower |
| Two-chunk blocked scan | 11.953 ms | 12.355 ms | 3.4% slower |
| Four-chunk blocked scan | 11.819 ms | 12.339 ms | 4.4% slower |

## Optimized graph result

The completed HPU graph reformulation now applies the FlashQLA similarity
transform without rematerializing the pairwise gate matrix. It also factors
the phase-B local decay into centered, row-scaled Q/K inputs and defers output
additions outside the recurrent state dependency chain.

True-shape, compiled 16K measurements on HVM0:

| Region | Control | Optimized | Change |
| --- | ---: | ---: | ---: |
| Phase A | 4.806 ms | 4.024 ms | 16.3% faster |
| Complete GDN core | 12.487 ms | 9.437 ms | 24.4% faster |
| Recurrent scan, isolated | 4.773 ms | 3.426 ms | 28.2% faster |

The complete-core output differs from the accepted BF16 operation order by
0.752% relative L2; final state differs by 0.640%. The phase-B trace moved
from 89.48% any-engine activity with 627 us idle to 99.57% activity with
20 us idle. No eager model path was used.

### Optimized-core TPC breakdown

The optimized 16K core spans 8.876 ms on device. TPC is active for 4.445 ms,
but 2.886 ms overlaps MME or DMA; TPC-only timeline is 1.559 ms. Kernel
resource time and wall coverage are different metrics because one launch can
occupy multiple TPCs and engines execute concurrently.

| TPC kernel class | TPC resource share | Wall coverage | Main source |
| --- | ---: | ---: | --- |
| BF16 multiply | 40.51% | 1.755 ms | Gate/decay scaling and Phase-B operands |
| BF16 add | 16.39% | 0.859 ms | Recurrent state update and deferred output add |
| Triangular band extraction | 13.30% | 0.550 ms | Phase-A KKT and Phase-B causal masks |
| Three largest fused TPC kernels | 23.35% | 0.991 ms | Phase-A/B elementwise subgraphs |
| Remaining TPC kernels | 6.45% | n/a | Subtract, repeat, casts, and small setup ops |

Stage traces put 1.624 ms of TPC-only time in Phase A and 1.612 ms in Phase B.
The isolated recurrent scan has only 0.295 ms of TPC-only time: its 2.085 ms
span is primarily the dependency-ordered combination of BMM, state traffic,
and additions, not a standalone TPC arithmetic bottleneck. Therefore the
useful optimization target is fusion that removes both TPC passes and
intermediate DMA, plus a shorter recurrent dependency chain. Merely replacing
individual elementwise GUIDs has a substantially smaller wall-time ceiling.

The earlier end-to-end Qwen trace corroborates the operation mix before the
BF16 graph reformulation: 77.96% of linear-attention TPC resource time was
FP32, and FP32 add plus multiply accounted for 66.63% of all TPC resource
time. The optimized graph has removed that FP32-heavy operation order from
the isolated core, so the complete optimized trace above is the forward
optimization baseline.

Use all of the following for the measured fast path:

```bash
VLLM_GDN_FLASHQLA=1
VLLM_GDN_FUSED_STATE_MATMUL=1
VLLM_GDN_DEFERRED_OUTPUT_ADD=1
VLLM_GDN_RECURSIVE_SOLVER_BASE=16
VLLM_GDN_COMPACT_REPEATED_KKT=1
```

The feature remains disabled by default pending model-level quality validation.

### Long-context scaling

The optimized kernel-only core scales without execution-throughput decay:

| Prompt tokens | Chunks | Core time | Core throughput |
| ---: | ---: | ---: | ---: |
| 16,384 | 256 | 9.974 ms | 1.643M token/s |
| 65,536 | 1,024 | 35.366 ms | 1.853M token/s |
| 131,072 | 2,048 | 69.947 ms | 1.874M token/s |

The 128K graph took about four minutes to compile and used about 5.9 GB of
host RSS while the Python recurrent loop was expanded. A 256K steady-state
core is projected near 140 ms, but was not compiled with the same unrolled
method. A native scan or loop IR is needed to keep long-context recipe
generation practical.

### Rejected follow-ups

- A custom TPC KKT-form kernel was numerically accurate but took 2.443 ms
  versus 2.087 ms for the compiled graph.
- FP8 KKT BMM had no prequantized speed advantage at the 64x128x64 shape and
  introduced 2.21% relative-L2 error; including the cast made it slower.
- Combining two independent BMM right-hand sides was 1.76x to 2.52x slower
  because it replaced dual-MME parallel work with concatenation and one GEMM.
- Chunk sizes 32 and 128 were respectively 13.0% and 1.7% slower than 64.
- A 32-chunk deferred-output pipeline improved the isolated scan but increased
  the complete-core device span from 8.827 ms to 9.017 ms.
- A head-major intermediate layout was bitwise correct, but its roughly 1%
  timing difference changed direction across repeated same-process windows.
- Habana FusedRMSNorm's residual input was slower than the compiled ordinary
  residual-plus-RMSNorm graph at the true shape.

The fused pair transform was accurate to about 1e-7 relative L2. The direct
recurrent kernel was also accurate to about 1e-7 on a nonzero-state reference.
The gate-free full-core path differed from the accepted BF16 operation order
by 0.84% relative L2, below FlashQLA's usual 2% forward tolerance.

## Conclusion

Plain PyTorch is not running eagerly here: HPU `torch.compile` already fuses
the elementwise graph and sends matrix products to MME. A TPC-only custom op
loses as soon as it replaces MME work or materializes FP32 matrices.

The CUDA FlashQLA speedup comes from keeping recurrent state in registers
while issuing tensor-core matrix operations inside the same persistent kernel.
The public Gaudi CustomOp API permits one TPC kernel per descriptor and TPC
cannot issue MME operations. A competitive port therefore needs one of:

1. An Intel-provided fused GDN/QLA graph kernel using both MME and TPC.
2. A public custom-kernel API that can schedule MME and retain SRAM state.
3. A native HPU lowering for the 64x64 KKT solve plus a persistent MME state
   scan primitive.

An out-of-tree native-backend probe compiles against bridge tag
`v1.24.1-482`, registers a mixed `batch_gemm -> add_fwd_bf16` Synapse subgraph,
and executes on Gaudi2. User bundling and strict scheduling produced the same
hardware schedule as the ordinary compiled graph. This interface therefore
does not guarantee that recurrent state stays in SRAM. See `DESIGN.md` for the
implementation and performance gates.

`torch.linalg.solve_triangular` is not a substitute in this stack. Under
`torch.compile` it falls back to `triangular_solve_cpu`; BF16 is unsupported,
and even a 16-matrix FP32 smoke test took about 302 ms on first invocation.

Applying the measured 3.050 ms/layer complete-core gain across 48 GDN layers
projects the accepted 2.591-second endpoint to about 2.445 seconds, or 6.70K
input token/s. The 9K target requires 1.820 seconds, leaving roughly another
625 ms to remove. Deleting all Q/K preprocessing, Phase A, Phase B precompute,
and scan would leave about 1.85 seconds, or 8.9K token/s. Reaching 9K therefore
also requires MLP, quantization, or full-attention work outside FlashQLA.

## Full-prefill follow-up: compact Q/K normalization and cast

The HPU prefill path used to expand Q/K from 16 heads to the 48 value-head
slots while they were still FP32, then cast the expanded result to BF16.
Normalization is head-local and cast commutes with `repeat_interleave`, so the
compact Q/K tensors can be normalized and cast before expansion. The important
saving is avoiding the 3x enlarged FP32 intermediate; an earlier experiment
that moved normalization alone measured 2.843 ms versus 2.819 ms and had no
material gain.

True-shape compiled measurements on HVM0:

| Q/K preparation order | Median |
| --- | ---: |
| Expand normalized FP32 Q/K, then cast to BF16 | 3.062 ms |
| Normalize and cast compact Q/K, then expand | 1.788 ms |

The new order is 1.71x faster, saves 1.274 ms per GDN layer, and projects to
61.2 ms across 48 layers. Both Q and K have `max_abs=0` and `relative_l2=0`
against the old order. The benchmark is
`native_backend/benchmark_qk_preprocess_order.py`.

Combining this projection with the measured 146.4 ms FlashQLA-style graph gain
would move the 2.591-second accepted endpoint to about 2.384 seconds, or 6.87K
input tokens/s. This is a projection, not a replacement for the pending
model-level quality gate on the FlashQLA-style path.

A separate diagnostic merged gating, packed-QKV splitting, normalization,
expansion, and output layout into one compiled helper. It measured 2.478 ms
versus 3.019 ms for intentionally separate compiled helpers, with exact
Q/K/V/g/beta outputs. This 0.541 ms difference is not an endpoint projection:
production HPU serving already regionally compiles a whole decoder layer, so a
Python helper boundary does not prove that the device graph gained a fused
kernel. The benchmark is `native_backend/benchmark_gdn_post_conv_prep.py` and
is retained only as a native-kernel design reference.
