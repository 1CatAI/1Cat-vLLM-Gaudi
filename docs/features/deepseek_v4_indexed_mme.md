# Indexed MXFP4 BF16 MME candidate

This opt-in path removes selected packed-weight copies for DeepSeek V4
Gaudi2 TP2 single-token decode. It uses the router's six expert IDs in their
original order. TPC decodes E2M1/E8M0 directly to BF16; two logical batched
MME operations perform W13 and W2, with the existing BF16 activation and
router-weighting boundaries. Other shapes use the existing MoE implementation.

This remains an experimental implementation. Isolated arithmetic and SRAM
producer/consumer checks passed, but the single-layer candidate has not
demonstrated a performance improvement. Full-model token, quality and
end-to-end performance qualification remain pending; keep the default off.

The design reuses runtime expert addressing, route order, input broadcast,
and bit-oriented decoding from 1cat-vllm's SM70 QPN M1 implementation
(`mxfp4_qpn_m1_sm70.cu`, revision `22a25e877fff7db150543f020a46c9f74ad634f6`).
The CUDA MMA instructions, reordered CUDA weight format, FP16 exponent
rebias and Split-K are not part of this BF16 implementation.

Build with `tools/build_deepseek_v4.py`, then select
`VLLM_HPU_DSV4_MXFP4_INDEXED_MME=1`. The flag defaults off and takes
precedence over the old indexed TPC experiment for matching shapes.
Model loading registers the native library before the first call and passes
the layer's tensor-parallel size into dispatch; TP sizes other than two
retain the existing path even if their local tensor shapes happen to match.

The compound operator accepts the existing seven tensors and returns only
the final BF16 activation. Diagnostic dequant and linear registrations expose
smaller boundaries for contract tests. Production uses neither their returned
decoded weights nor a persistent high-precision expert cache.

When the candidate is enabled, `set_stacked_weights` checks the complete
E8M0 scale tensors once. Codes in `[2,254]` permit an exact, shorter decoder;
otherwise the complete decoder handles subnormals and NaNs. The native ops
have an optional `normal_scales=False` argument. Passing `True` requires
this range validation and immutable scales. The model handles it at load
time; there are no per-token range scans or scalar synchronizations.
Weight reload and module conversion invalidate the cached check. Call
`set_stacked_weights` again after replacing or modifying scales to requalify
the specialization. Neither decoder converts through FP8.

All decoded weights in the compound operator are recipe-internal tensors.
This is a prerequisite for compiler slicing, not proof of SRAM residency.
Qualification must check producer/consumer placement and transfers in the
compiled graph before promoting performance. A two-node logical BMM graph
can still generate multiple physical MME/TPC tasks.

Validation covers all E2M1 codes and E8M0 exponents (including subnormals,
overflow, signed zero and NaNs), dynamic/nonconsecutive/repeated expert IDs,
real checkpoint samples and graph replay. Invalid IDs produce zero decoded
weights without reading out of bounds; model routing is expected to supply
valid IDs. No inputs are mutated. Full-model token and quality checks are
required after placement and single-layer performance gates pass.
