# Experimental TP2 fused boundaries

The graph-native extension depends on the Bridge patch in
`patches/tp2-collective-output-metadata.patch`. Apply it to an independent,
compatible Bridge source tree, build the backend, and build the extension
against those same headers. Keep the backend, Python bindings, and extension
as one versioned runtime installation. Do not overwrite another active runtime.

This patch changes the `CollectiveOperator` virtual interface. A successful
import or an available `torch.ops.hccl` schema is not proof of ABI compatibility.
Using a different backend can omit native execution and return uninitialized
outputs. A backend built for this patch also cannot be combined arbitrarily
with bindings requiring symbols added by a separate Triton integration.

The patch includes per-node output metadata and disables shape-agnostic caching
for the custom collective, while retaining static recipe caching. A generic
public ABI handshake remains future integration work; the guarded Gemma path
currently checks compiled execution and values at startup.

`VLLM_HPU_TP2_GEMMA_FUSED_AR_NORM` is default-off and requires
`VLLM_HPU_TP2_FUSED_AR_NORM`. Its startup validation uses changing signed inputs,
fresh allocations, a three-dimensional activation, and graph-produced Gemma
effective weights. Both ranks verify a native launch, finite outputs, an exact
residual reduction/add, and bounded normalization error. Validation finishes
before projection or embedding reduction flags are mutated. It introduces no
per-token validation synchronization.

The probe is deliberately small. Before promotion, validate chained boundaries,
model quality, current-stream ordering, and matched end-to-end throughput over
multiple fresh processes. Keep unsupported shapes and prefill on stock HCCL.

`bench_tp2_fused_ar_norm.py` now refuses to time failed validation, including
nonfinite outputs and missing native calls. Its CPU reference may diverge from
the vendor normalization after long recurrent chains; compare a matched stock
HPU chain when investigating such failures instead of treating reported error
numbers or NaN-masked maxima as success. Tolerances are explicit and recorded.
