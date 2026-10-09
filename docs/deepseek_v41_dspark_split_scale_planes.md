# DSpark split scale-plane dependency experiment

The compact prepared scale tensor stores K-dependent group scales followed by
fixed channel codes. Its existing decoder access map therefore requests the
whole scale K dimension at each workpoint. This experiment separates those
dependencies, allowing the compiler to consider K slicing without changing
the FP4 encoding, SAT arithmetic, route packing or MME consumers.

`VLLM_HPU_DSV41_DSPARK_SPLIT_SCALE_PLANES=0` is the default. The candidate
applies only to checkpoint-qualified Target C2–C6/top6 operands. C1, draft
top3 and prefill continue through their existing dispatch.

Compact group and channel tensors are views of the original prepared tensor.
They retain its outer strides and share its storage; the channel view starts
at the fixed tail. No new expert-weight copy or per-round contiguous conversion
is permitted. Legacy scale planes retain their original layout and supply an
unused channel placeholder. Selection depends on the prepared layout, rather
than tensor-parallel size.

The separate decoder GUIDs reuse the baseline glue's validation and index
axes. Group scales have an affine K128 access map; channel codes retain a
fixed full-channel map. Borrowed descriptors are copied before adapting the
validation shape. Inherited ELF queries use an independent capacity, so an
old binary larger than the new one cannot corrupt the allocation contract.

The static gate checks both layouts, shared storage, offset and stride
ownership, full and sliced K geometry, and repeated glue instantiation.
Complete-chain qualification uses the production native replay, three actual
request inputs, real consumers and communication, and three interleaved A/B
device timing pairs. Default TPC binaries must remain unchanged.

A changed access map does not prove a different SRAM pipeline or a speedup.
The saved recipes must show whether slicing or extra copies changed. Any
numerical change must pass the accepted C1 routed-expert tolerance and
teacher-forced acceptance before batch serving qualification. The candidate
remains disabled until these checks and the official request gate pass.
