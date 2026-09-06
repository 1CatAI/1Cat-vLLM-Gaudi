# SPDX-License-Identifier: Apache-2.0

"""HPU activation overrides with opt-in Gaudi2 Triton fast paths."""

import os

import torch
from vllm.model_executor.layers.activation import SiluAndMul


if os.environ.get("VLLM_HPU_TRITON_MODE", "off").strip().lower() == "off":
    _triton_silu_and_mul = None
else:
    from vllm_gaudi.ops.triton_gaudi import silu_and_mul as _triton_silu_and_mul


@SiluAndMul.register_oot
class HPUSiluAndMul(SiluAndMul):

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if _triton_silu_and_mul is not None:
            output = _triton_silu_and_mul(x)
            if output is not None:
                return output
        return self.forward_native(x)
