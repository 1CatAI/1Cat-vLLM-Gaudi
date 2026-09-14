# SPDX-License-Identifier: Apache-2.0
"""Freeze the current device Top-6 order before replacing its selection algorithm.

This is a semantic diagnostic, with no baseline performance measurement.
Run only through the normal device lease launcher.
"""

import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401


def cases():
    lane = torch.arange(384, dtype=torch.float32)
    yield "ascending", lane
    yield "descending", -lane
    yield "all_zero", torch.zeros(384)
    yield "all_one", torch.ones(384)
    yield "four_tied_groups", lane.remainder(4)
    yield "eight_tied_groups", lane.remainder(8)
    value = torch.zeros(384)
    value[[0, 63, 64, 127, 128, 191, 192, 255, 256, 319, 320, 383]] = 9
    yield "tied_winners_across_vectors", value
    value = torch.zeros(384)
    value[::2] = -0.
    yield "signed_zero", value
    value = lane.clone()
    value[[63, 64, 127, 128, 255, 256, 383]] = float("inf")
    yield "positive_infinities", value
    yield "all_negative_infinity", torch.full((384, ), -float("inf"))
    value = lane.clone()
    value[[0, 127, 128, 255, 256, 383]] = float("nan")
    yield "nan_winners", value
    yield "subnormal_bits", torch.arange(384, dtype=torch.int32).view(torch.float32)
    for seed in range(4):
        torch.manual_seed(seed)
        yield f"random_{seed}", torch.randn(384)


@torch.inference_mode()
def main():
    if os.environ.get("DSV41_TEST_HPU") != "1":
        raise RuntimeError("An explicit HPU diagnostic lease is required")
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    compiled = torch.compile(lambda x: torch.topk(x, 6, dim=-1, sorted=True).indices,
                             backend="hpu_backend",
                             fullgraph=True,
                             dynamic=False)
    device = torch.empty(1, 384, device="hpu", dtype=torch.float32)
    rows = []
    for name, value in cases():
        device.copy_(value.view(1, 384))
        actual = compiled(device).cpu().flatten().tolist()
        # Compare one proposed finite-score tie rule. Nonfinite and denormal
        # cases are preserved for interpretation, not claimed supported here.
        stable = torch.argsort(value, descending=True, stable=True)[:6].tolist()
        record = dict(name=name,
                      ids=actual,
                      stable_cpu_ids=stable,
                      matches_stable_cpu=actual == stable,
                      input_f32_bits=value.view(torch.int32).tolist())
        rows.append(record)
        print(json.dumps({key: val for key, val in record.items() if key != "input_f32_bits"}), flush=True)
    (root / "top6-device-order.json").write_text(
        json.dumps(
            {
                "purpose": "Freeze installed device sorting semantics; no speed measurement or candidate qualification",
                "torch_version": torch.__version__,
                "cases": rows
            },
            indent=2) + "\n")


if __name__ == "__main__":
    main()
