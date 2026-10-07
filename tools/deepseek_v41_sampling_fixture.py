# SPDX-License-Identifier: Apache-2.0
"""Explicit immutable sampling scope for resident component fixtures."""


def full_official_sampler(gather, sample=None):
    """Capture the collective callable, not its enclosing model or host tables."""
    if sample is None:
        from vllm_gaudi.ops.deepseek_v41_sampling import sample_probabilities
        sample = sample_probabilities

    def full(local, draw):
        return sample(gather(local, dim=-1), draw, filtered=True)

    return full


class OfficialSamplerCache:
    """Reuse the initial sampler's code for each unchanged projection contract.

    Weights, control parameters and RNG state are explicit tensor operands.
    Program ownership is intentionally absent from both keys and closures.
    Dynamo still guards their shapes/dtypes/layouts; this does not reuse a
    recorded native plan with different addresses.
    """

    def __init__(self, full, compile_fn=None):
        self._full = full
        self._compile = compile_fn
        self._compiled = {}

    def get(self, use_bf16):
        use_bf16 = bool(use_bf16)
        if use_bf16 not in self._compiled:
            import torch
            from vllm_gaudi.ops.deepseek_v41_sampling import device_sampling_draw
            full = self._full

            def official(value, weight, params, seed, counter):
                draw = device_sampling_draw(params, seed, counter)
                local = (torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(value, weight)
                         if use_bf16 else torch.nn.functional.linear(value.float(), weight))
                return full(local, draw)

            compile_fn = self._compile
            self._compiled[use_bf16] = (torch.compile(
                official, backend='hpu_backend', fullgraph=True, dynamic=False)
                if compile_fn is None else compile_fn(official))
        return self._compiled[use_bf16]
