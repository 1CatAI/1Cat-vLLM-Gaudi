# Engine Source Dependency

Base: `vllm-project/vllm` commit `fe755c88995ad468882517b6c4bdd60138d46a3a`.

The hardware-agnostic DeepSeek V4 foundation was imported from the upstream
hardware-pluggable development work and extended for Gaudi cache bindings,
metadata, packed-weight loading and regional execution. Existing upstream
SPDX/copyright notices are retained in every patch. This directory does not
claim that these changes are merged into official vLLM main.

Apply the ordered patch series with `tools/prepare_deepseek_v4_engine.py` in
a clean checkout of the pinned commit, then build/install the engine normally.
The script refuses a different revision or unrelated dirty worktree and is
idempotent for the already-applied series. No runtime import overlays are used.
