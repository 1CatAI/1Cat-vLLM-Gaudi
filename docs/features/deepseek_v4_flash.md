# DeepSeek V4 Flash on Gaudi2

The source-integrated profile supports ordinary tensor-parallel decode on two
Gaudi2 devices with the original mixed MXFP4/FP8 checkpoint. It uses the normal
HPU worker and regional `torch.compile(hpu_backend)` execution. This is not a
claim of whole-model HPU Graph replay or speculative decoding support.

## Build From Source

Use Gaudi Software 1.24.1 with its matching PyTorch 2.11 environment. The native
Bridge adapter requires the matching Bridge source and generated build headers;
the public custom-op headers alone are insufficient for its private backend ABI.
No Docker image, precompiled research artifact, or external working tree is
required by the installed execution path.

The engine integration is carried as reviewable, version-pinned source patches.
They are applied **before installation**, never when the model starts:

```bash
git clone https://github.com/vllm-project/vllm.git ../vllm-dsv4
git -C ../vllm-dsv4 checkout fe755c88995ad468882517b6c4bdd60138d46a3a
python tools/prepare_deepseek_v4_engine.py ../vllm-dsv4
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e ../vllm-dsv4
python -m pip install --no-build-isolation -e .

export GAUDI_PYTORCH_BRIDGE_ROOT=/path/to/matching/gaudi-pytorch-bridge
export GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=/path/to/matching/bridge/build
python tools/build_deepseek_v4.py --jobs "$(nproc)"
ctest --test-dir build/deepseek_v4/kernels --output-on-failure
```

`csrc/deepseek_v4` contains TPC source, host glue, and the C++ Bridge operators.
The generated kernel database forwards stock GUIDs to the installed Gaudi
kernel library, so `GC_KERNEL_PATH` is one regular file. It does not require
`sitecustomize`, delayed environment caching, or a custom Python import hook.
The build writes a source/binary hash manifest next to its libraries.

## Run

Select two available logical modules, then run the installed entrypoint:

```bash
HABANA_VISIBLE_MODULES=0,1 python -m vllm_gaudi.entrypoints.deepseek_v4 \
    /path/to/DeepSeek-V4-Flash --port 8000
```

The entrypoint only selects ordinary configuration and invokes the standard
vLLM CLI. It defaults to TP2, one request, a maximum context of 512 tokens,
packed FP8 KV, asynchronous scheduling, and a small warmup bucket set. Use
`--worker-cpus CPU0,CPU1` to select two distinct CPUs within the process affinity;
omitting it leaves affinity unchanged. CPU placement and competing host load
affect latency and must be recorded when comparing results.

For this bounded configuration, platform setup enables the accepted projection
caches, split-KV attention, fused cache producers, packed metadata and early
output lowering. The compiler pass is maintained in `vllm_gaudi/compilation`
and runs before HPU partitioning. No post-startup control request or experiment
worker is involved. Explicit feature settings override the profile defaults.
The same bounded workload passed to the ordinary `vllm serve` command also
selects these platform defaults; the convenience entrypoint additionally fixes
the workload and warmup settings shown above.

The combined QNorm/Compressor and Compressor/MLA recipes, ordered C128 variant,
precision-changing Compressor inputs, and process-wide eager recipe cache are
not enabled by this profile. Their isolated results did not qualify them for
default model execution. Attention projection caches use BF16 computation;
loading the original checkpoint does not imply every operation uses FP8 MME.

## Qualification Scope

Use the source-build kernel checker and standalone mutation tests before model
qualification. Compare the integrated candidate with preserved baseline token
IDs using the same generation settings; do not rerun an unchanged baseline.
The client qualifier records raw SSE arrival intervals and tests exact output
equality against saved references. Performance summaries belong in the local
experiment ledger, not in the source or public PR.

The short-context profile does not establish long-context, concurrent-request,
DSpark, or general model-accuracy parity. Existing diagnostic and quality
counterexamples remain relevant; a token-exact small cohort is not a full
accuracy evaluation.

### Known Cold-Start Limitation

The source candidate reproduces the previously recorded difference between
initial and subsequent greedy completions for the same prompt. Stable requests
and the preserved quality cohort match the old source reference token for token,
but the request-history dependence has not been explained or fixed. Do not
interpret stable-token equality as a guarantee of cold-start determinism or full
model accuracy. Preserve cold failures separately; do not warm repeatedly until
an arbitrary mismatch disappears or report those attempts as a clean pass.

For an isolated compiler contract check, select a free device and run
`tools/check_deepseek_v4_compiler.py` in its own process with both
`HABANA_VISIBLE_MODULES` and `HLS_MODULE_ID` set. It checks that prefill retains
its original boundaries, decode lowering preserves producer dependencies, and
repeated compiled calls remain exact without recompilation. It does not load a
checkpoint, collect a latency baseline, or mutate any serving worker.
