# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.activation import SiluAndMul

from vllm_gaudi import envs as gaudi_envs


@SiluAndMul.register_oot
class HPUSiluAndMul(SiluAndMul):

    @staticmethod
    def forward_oot(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        token_count = x.numel() // x.shape[-1]
        if (
            gaudi_envs.VLLM_HPU_EXPLICIT_SIGMOID_SILU
            and token_count
            >= gaudi_envs.VLLM_HPU_EXPLICIT_SIGMOID_SILU_MIN_TOKENS
        ):
            gate = x[..., :d]
            return (gate * torch.sigmoid(gate)) * x[..., d:]
        return F.silu(x[..., :d]) * x[..., d:]
