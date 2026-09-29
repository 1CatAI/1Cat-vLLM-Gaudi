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
performance promotion remain incomplete. Generic service profiles remain opt-in.
The dedicated [DeepSeek V4.1 entrypoint](../../../../docs/features/deepseek_v41.md)
enables its experimental C1 bundle by default; this does not promote that bundle
to a production-qualified runtime. DSpark remains disabled by default.

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

The native path requires the static group plan and prepared communication.
The Qwen adapter also requires active GDN state views and direct state update.
TP2 GDN local-head shapes additionally require
`VLLM_HPU_FLASHINFER_GDN_TP2=1`; enabling the parent GDN switch alone does not
opt into them. The DeepSeek V4 adapter has its own attention and mHC state
contract and does not require GDN. See the
[V4 native decoder guide](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/docs/features/deepseek_v4_native_decode.md)
and [environment variable reference](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/docs/configuration/env_variables.md).

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

Ordinary V4 AllReduce nodes use a peer transfer followed by the original BF16
addition in the compiled consumer. They do not use the Qwen residual/norm
formula. Multiple independent transfers may share a compute consumer; every
transfer retains its own completion wait and counter relocation. Contiguous
reshape aliases retain their source storage range and are refreshed when
external inputs are rebound.

## Profiler compatibility

The public Synapse snapshot uses the older profiler virtual interface. The
patch routes SDK APIs added after that interface directly to their internal
implementations; dispatching those methods through an older profiler object
would call unrelated virtual functions. Use a matching profiler shim, SDK,
plugins, parser and control executable together in a private process environment.

When that SDK needs the older one-argument logger registration overload, build
`tools/communication/tp2_profiler_logger_compat.cpp` as a shared library against
the snapshot's `hl_logger` headers and the selected Bridge logger library.
Preload this adapter only in the isolated profiling process. Select the Bridge
logger before the profiler SDK directory in the library search path. Record
all profiler and runtime fingerprints with the trace; a CPU-only trace does
not validate device acquisition.

The pinned legacy profiler removes its recipe registration after the wrapped
destroy call has released the handle. A concurrent compile or deserialization
can reuse that address and lose its new registration when the older destroy
finishes. The Synapse API patch holds an exclusive lifetime guard for destruction
and shared guards for compilation/deserialization when a wrapper is installed.
The guard extends through profiler return, permits concurrent creation, and does
not change steady native replay or skip any profiling records. CPU forced
retirement/reuse checks and a fresh full-model acquisition are required when
changing this contract.

`tools/communication/check_profiler_recipe_lifetime.py` exercises this boundary
without acquiring a device, using one archived serialized recipe and explicitly
provided Synapse/profiler hashes. In `--mode reproduce` it forces address reuse
inside the old wrapper's retirement window. In `--mode guard` it verifies that a
concurrent registration waits and remains present afterward. This tool changes
one virtual call only inside its diagnostic process; it is never loaded by the
model, benchmark or serving entrypoint.

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

## Concurrent graph fusion

### Native program address windows

After the bounded-reindex patch, apply the native program-window correction
to the isolated Synapse source:

```bash
python tools/communication/apply_native_program_window.py --source "$TP2_SYNAPSE_SOURCE" --apply
```

Native graph program storage retains the normal compute arena's 8 KiB
alignment and single 4 GiB program-counter window. If the first allocation
cannot satisfy both, it is released and one bounded padded allocation is
attempted. Address selection uses the replacement's actual address, including
when it moves. Graph-owned workspace and arithmetic are unchanged. Rebuild
Synapse and the Bridge extension with the matching dependency fingerprint;
do not bypass the runtime ABI check. The standalone
`tests/unit_tests/ops/native_program_window_test.cpp` exercises address limits,
alignment, exact-end boundaries and relocation without acquiring a device.

### Compiler state

TPC fusion owns pending replacement nodes per cluster invocation. Compilations
may run concurrently, so pending node lists and original-node sets must not be
process-static or survive failed cluster replacement. Diagnostic cluster IDs
use an atomic counter; strict graph-trait ownership checks remain enabled.
The CPU-only `tests/tpc_fuser_compile_concurrency.cpp` diagnostic compiles and
destroys independent BF16 Gaudi2 graphs from four threads through public Synapse
APIs. Link it against the isolated patched Synapse runtime and retain the loaded
library fingerprints with its result. No device acquisition is required.

## V4.1 output-window producer bundles

The experimental N512 MXFP4 decoder keeps each output window in an independent
TPC-to-MME SRAM bundle. Its small shared activation must not cause the MantaRay
multi-MME bundler to leave a decoded weight producer in DRAM. The exception
matches the new decoder GUID only; all other operators retain normal bundling.
K is not split, and the external BF16 boundary remains unchanged. Inspect the
actual compiled graph and test complete MoE output before measuring the model.
