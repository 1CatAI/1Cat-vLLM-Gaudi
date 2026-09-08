# Experimental native TP2 runtime

These patches supply the native APIs required by the opt-in TP2 decoder graph.
They include the matching compute program ownership, stream handoff, HCL command
relocation and collective output metadata changes. Source is pinned in
[manifest.json](manifest.json); this directory contains no runtime binaries.

This is a research backend. The Synapse source is from an older public source
snapshot with compatibility backports for the tested SDK. It is **not** a source
release of the installed SDK, and successful ABI checks do not prove compiler
or numerical equivalence. Frozen production logprobs still differ in the full
model, with a difference isolated to an FP8 MME accumulation. Full quality and
performance promotion remain incomplete. Keep all candidates disabled in normal
service profiles.

## Source and build

Use independent clean checkouts at the exact commits in the manifest. The
`bridge.patch` here includes the older collective-output-metadata patch; do not
apply both. From the plugin checkout, validate and apply each patch:

```bash
python tools/communication/apply_native_runtime.py bridge --source "$TP2_BRIDGE_SOURCE" --apply
python tools/communication/apply_native_runtime.py synapse --source "$TP2_SYNAPSE_SOURCE" --apply
python tools/communication/apply_native_runtime.py hcl --source "$TP2_HCL_SOURCE" --apply
```

Omit `--apply` to check only. The tool rejects a different base commit, patch
fingerprint or dirty source checkout. The patch application does not build,
install or replace any library.

Build HCL with its bundled dependencies and SDK development libraries. Set
`HCL_SRC_PKG_DIR` to its checkout and `HCL_LIB_DIR` to the SDK library directory;
configure `hcl/src` into a separate CMake build directory. The patch makes
`HCL_OUTPUT_DIR` default to the build directory's `lib` subdirectory. Use a
private output directory rather than installing into the SDK.

Build Synapse using the source package's build environment and the patched HCL
headers. Its `SYNAPSE_ROOT`, `SPECS_ROOT`, `SPECS_ROOT_H9`, `SPECS_EXT_ROOT`,
`THIRD_PARTIES_ROOT`, `TPC_KERNELS_ROOT`, `HCL_ROOT` and `BUILD_ROOT_LATEST`
variables must point to the corresponding source directories and private
SDK dependency directory. Configure `synapse` with CMake, `Release`,
`TESTS_ENABLED=OFF` and `AUTO_GENERATED_TESTS=OFF`, then build the `Synapse` target.
The bundled CMake rules regenerate protobuf sources with protobuf 3.9.0; generated
protobuf files and build probes are deliberately not included in the patch.
Retain the associated MME and rotator build products. Compatibility backports
provide implemented APIs and layout corrections, not successful no-op stubs.

Build Bridge using its upstream build instructions for the same PyTorch and
Python installation. Keep its backend and HCCL Python binding together. With
those private libraries selected for the new process, build the plugin extension:

```bash
python tools/communication/build_tp2_fused_ar_norm_bridge.py \
  --bridge-source "$TP2_BRIDGE_SOURCE" \
  --cmake-build-directory "$TP2_BRIDGE_BUILD" \
  --build-directory "$TP2_EXTENSION_BUILD" \
  --backend-library "$TP2_BACKEND_LIBRARY" \
  --native-hccl-library "$TP2_HCCL_BINDING" \
  --synapse-library "$TP2_SYNAPSE_LIBRARY" \
  --hcl-library "$TP2_HCL_LIBRARY"
```

The build writes an ABI sidecar containing binary and loaded dependency hashes.
Use that sidecar with the extension; the loader rejects a different runtime.
The TP2 dynamic quantization kernel is a separate target,
`tp2_dynamic_quant_kernels`, in `csrc/flashinfer_gaudi/CMakeLists.txt`. Add its
kernel database to `GC_KERNEL_PATH` only for the candidate process. No path here
requires replacing shared production libraries.

## Execution contract

The native path requires the static group plan, prepared communication, active
GDN state views and direct state update. TP2 GDN local-head shapes additionally
require `VLLM_HPU_FLASHINFER_GDN_TP2=1`; enabling the parent GDN switch alone does
not opt into them. See the [environment variable reference](../../../../docs/configuration/env_variables.md)
for the full set of prerequisite flags.

Preparation owns fixed inputs, state views, recipe metadata, compute program
storage and communication resources. The joint plan publishes prepared command
pages with typed counter and queue relocations. It retains every decoder
reduction and keeps embedding reduction outside the decoder. A partial graph or
the retained per-segment diagnostic path is not equivalent to full decoder joint
replay. Report actual queue publications and device completion separately from
the single model entry.

Invalidate on allocation, slot binding or communicator changes; drain consumers
before releasing resources. An error after state mutation terminates that
execution without retrying another implementation. Counter rollover, queue
wraparound and producer/consumer visibility are part of correctness.

## Validation

Run the plugin's TP2/GDN unit tests and the C++ tests in
`tests/unit_tests/ops/tp2_native_graph_topology_test.cpp` and
`tools/communication/tests/`. The HCL patch also includes
`hcl/tests/native_hcl_relocation_test.cpp`. The standalone command program tests
compile against the patched Synapse runtime and HCL include directories.

`probe_tp2_native_graph.py` verifies every changing output exactly against the
same-math HPU reference, retaining independent CPU diagnostics. Reuse an archived
reference with its fingerprint when available. A successful small chain or
same-runtime model comparison does not qualify equivalence to the frozen
production model. Full-model quality and end-to-end qualification are separate
gates, and remain required before enabling defaults.
