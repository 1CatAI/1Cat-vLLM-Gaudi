# Native compound backend probe

This research-only extension verifies that an out-of-tree module can register
an ordinary Gaudi `OpBackend` and lower one PyTorch operation into a mixed
Synapse subgraph. The probe emits an MME `batch_gemm` followed by a TPC BF16
add. It is not loaded by vLLM and is not a FlashQLA implementation.

The extension intentionally uses private bridge headers and ABI. Build it only
against the exact source tag matching the installed `habana-torch-plugin`:

```bash
export GAUDI_PYTORCH_BRIDGE_SOURCE=/path/to/gaudi-pytorch-bridge
export ABSEIL_CPP_SOURCE=/path/to/abseil-cpp-20250512.1
export GAUDI_BRIDGE_GENERATED_SOURCE=/path/to/bridge-build-root
export GAUDI_BRIDGE_DEPS_ROOT=/path/containing/fmt-and-magic-enum
export EXPRTK_SOURCE=/path/to/exprtk-0.0.3-cmake
python setup.py build_ext --inplace
PT_HPU_LAZY_MODE=0 python smoke_test.py
```

After the smoke test, compare the ordinary compiled graph with the mixed-engine
lowering at the true KKT batch shape:

```bash
PT_HPU_LAZY_MODE=0 python benchmark_probe.py \
  --trace-dir /tmp/flashqla-compound-traces
```

On bridge/plugin `1.24.1.482`, the ordinary, bundled, and strict-scheduled
variants emitted the same MME/TPC event sequence and device span. Compound
registration works, but wrapping an existing graph is not itself a scheduling
or SRAM-residency optimization.

Profile the exact single-sequence recurrent scan shape without loading the
model:

```bash
PT_HPU_LAZY_MODE=0 python benchmark_recurrent.py \
  --trace-dir /tmp/flashqla-recurrent-trace
```

Benchmark the complete production phase-B implementation, including its
chunk-local precompute:

```bash
PT_HPU_LAZY_MODE=0 python benchmark_phase_b.py \
  --trace-dir /tmp/flashqla-phase-b-traces
```

The current machine pairs `habana-torch-plugin==1.24.1.482` with bridge tag
`v1.24.1-482`. A production build must reject any other bridge/plugin pair.

The probe uses a public custom-op descriptor only for frontend allocation. A
regular `KernelRegistry` entry with the same schema takes precedence during
lowering and emits the mixed-engine graph. This bypasses the public custom-op
restriction of one TPC GUID without requiring a complete bridge rebuild.
