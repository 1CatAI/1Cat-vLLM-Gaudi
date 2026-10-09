# SPDX-License-Identifier: Apache-2.0
"""Check grouped-prefill MTP context tails and their real allocation cost."""

import json
import os
from pathlib import Path


def main():
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.models.deepseek_v41_program import draft_context_state
    torch.hpu.set_device(0)
    inputs = [
        torch.randn(8192, 4, 5120, generator=torch.Generator().manual_seed(37 + index), dtype=torch.bfloat16).to("hpu")
        for index in range(3)
    ]
    torch.hpu.synchronize()
    baseline = torch.hpu.memory_allocated()
    torch.hpu.reset_peak_memory_stats()
    full = torch.cat([draft_context_state(value, 256, grouped_prefill=False) for value in inputs], -1)
    torch.hpu.synchronize()
    full_peak = torch.hpu.max_memory_allocated() - baseline
    expected = full[-256:].cpu()
    del full
    torch.hpu.synchronize()
    baseline = torch.hpu.memory_allocated()
    torch.hpu.reset_peak_memory_stats()
    tail = torch.cat([draft_context_state(value, 256, grouped_prefill=True) for value in inputs], -1)
    torch.hpu.synchronize()
    tail_peak = torch.hpu.max_memory_allocated() - baseline
    torch.testing.assert_close(tail.cpu(), expected, rtol=0, atol=0)
    result = dict(status="passed_exact_grouped_prefill_aux_tail",
                  rows=8192,
                  retained_rows=256,
                  target_layers=[37, 38, 39],
                  context_width=15360,
                  full_peak_increment_bytes=full_peak,
                  tail_peak_increment_bytes=tail_peak,
                  measured_peak_reduction_bytes=full_peak - tail_peak,
                  scope="Three auxiliary means and concat only; full-serving peak must be remeasured",
                  decode_math_changed=False,
                  formal_qualification=False,
                  timing_claim=False)
    (Path(os.environ["DSV41_RUN_EVIDENCE"]) / "AUX_CONTEXT.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
