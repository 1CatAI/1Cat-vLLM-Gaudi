# SPDX-License-Identifier: Apache-2.0
"""Functional cross-process check for the decoder's four-element literal."""
import argparse
import json
import os
from pathlib import Path

import torch

from vllm_gaudi.compilation.deepseek_v41_frontend_cache import GuardedFrontendEntry


class LiteralMix(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros((6, 4), device="hpu"))

    def forward(self, inputs):
        mix = torch.tensor([1., 0., 0., 0.], device=inputs.device).expand(inputs.shape[0], 4)
        value = inputs * 2 + mix
        self.state.copy_(value)
        return value, self.state


def main():
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators, prepare_environment

    prepare_environment()
    import habana_frameworks.torch.core  # noqa: F401 - install the HPU device and backend.

    torch.hpu.set_device(0)
    load_native_operators()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from torch._dynamo.backends.registry import lookup_backend

    owner = LiteralMix()
    entry = GuardedFrontendEntry(LiteralMix.forward,
                                 owner,
                                 lookup_backend("hpu_backend"),
                                 args.cache,
                                 identity="a" * 64)
    with torch.inference_mode():
        for scalar in (1., 2., 3.):
            inputs = torch.full((6, 4), scalar, device="hpu")
            output, state = entry(inputs)
            expected = torch.full((6, 4), scalar * 2)
            expected[:, 0] += 1
            if not torch.equal(output.cpu(), expected) or not torch.equal(state.cpu(), expected):
                raise AssertionError("Literal restoration changed output or state")
            if state.data_ptr() != owner.state.data_ptr():
                raise AssertionError("Restoration did not bind current state")
    args.output.write_text(json.dumps(entry.stats, indent=2) + "\n")
    print(json.dumps(entry.stats))


if __name__ == "__main__":
    main()
