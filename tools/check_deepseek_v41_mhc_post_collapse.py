# SPDX-License-Identifier: Apache-2.0
"""Untimed native residual/collapse plus actual norm/quant consumer contract."""
import torch
import torch.nn.functional as F

from vllm_gaudi.ops.deepseek_v41_math import hc_post, quantize_activation, rms_norm


def check_mhc_post_collapse_contract(diagnostic_path=None):
    op = torch.ops.custom_op

    def parent(value, residual, post, comb, pre, weight):
        updated = hc_post(value, residual, post, comb)
        collapsed = (updated.float() * pre.unsqueeze(-1)).sum(1).to(torch.bfloat16)
        norm, quant, scale = op.custom_deepseek_v41_ffn_norm_quant_gaudi2(collapsed, weight, 1e-6)
        return updated, collapsed, norm, quant.view(torch.uint8), scale

    def candidate(value, residual, post, comb, pre, weight):
        updated, collapsed = op.custom_deepseek_v41_mhc_post_collapse_gaudi2(value, residual, post, comb, pre)
        norm, quant, scale = op.custom_deepseek_v41_ffn_norm_quant_gaudi2(collapsed, weight, 1e-6)
        return updated, collapsed, norm, quant.view(torch.uint8), scale

    functions = [torch.compile(f, backend='hpu_backend', fullgraph=True, dynamic=False) for f in (parent, candidate)]
    generator = torch.Generator().manual_seed(41129)
    checks = []
    for generation in range(8):
        for tokens in (1, 2, 6):
            for magnitude in (0., .03125, 1., 256.):
                value = (torch.randn(tokens, 5120, generator=generator) * magnitude).bfloat16()
                residual = (torch.randn(tokens, 4, 5120, generator=generator) * magnitude).bfloat16()
                packed = torch.rand(tokens, 24, generator=generator).to('hpu')
                post = packed[:, 4:8].contiguous()
                comb = packed[:, 8:].reshape(tokens, 4, 4).contiguous()
                pre = packed[:, :4].contiguous()
                weight = (torch.rand(5120, generator=generator) + .5).bfloat16()
                args = (value.to('hpu'), residual.to('hpu'), post, comb, pre, weight.to('hpu'))
                expected = [x.cpu() for x in functions[0](*args)]
                observed = [x.cpu() for x in functions[1](*args)]
                for i, (actual, wanted) in enumerate(zip(observed, expected, strict=True)):
                    if diagnostic_path is not None and not torch.equal(actual, wanted):
                        torch.save(dict(tokens=tokens, magnitude=magnitude, output_index=i,
                                        args=[x.cpu() for x in args], expected=expected, observed=observed,
                                        passed=checks), diagnostic_path)
                    assert torch.equal(actual, wanted), (generation, tokens, magnitude, i,
                                                         (actual != wanted).sum().item())
                checks.append(dict(generation=generation, tokens=tokens, magnitude=magnitude, all_outputs_exact=True))
    return checks


def check_mhc_attention_handoff(diagnostic_path=None):
    """Keep the actual generic attention norm/block-quant/QKV consumer."""
    op = torch.ops.custom_op

    def parent(value, residual, post, comb, pre, weight, projection):
        updated = hc_post(value, residual, post, comb)
        updated = op.custom_deepseek_v41_bf16_identity_gaudi2(updated.reshape(1, -1)).reshape_as(updated)
        collapsed = (updated.float() * pre.unsqueeze(-1)).sum(1).to(torch.bfloat16)
        projected = F.linear(quantize_activation(rms_norm(collapsed, weight, 1e-6).bfloat16()), projection)
        return updated, projected

    def candidate(value, residual, post, comb, pre, weight, projection):
        updated, collapsed = op.custom_deepseek_v41_mhc_post_collapse_f32_gaudi2(value, residual, post, comb, pre)
        projected = F.linear(quantize_activation(rms_norm(collapsed, weight, 1e-6).bfloat16()), projection)
        return updated, projected

    functions = [torch.compile(f, backend='hpu_backend', fullgraph=True, dynamic=False) for f in (parent, candidate)]
    generator = torch.Generator().manual_seed(41131)
    projection = (torch.randn(1792, 5120, generator=generator) / 72).bfloat16().to('hpu')
    checks = []
    for generation in range(2):
        for tokens in (1, 2):
            for magnitude in (0., .03125, 1., 256.):
                value = (torch.randn(tokens, 5120, generator=generator) * magnitude).bfloat16().to('hpu')
                residual = (torch.randn(tokens, 4, 5120, generator=generator) * magnitude).bfloat16().to('hpu')
                packed = torch.rand(tokens, 24, generator=generator).to('hpu')
                post, comb = packed[:, 4:8].contiguous(), packed[:, 8:].reshape(tokens, 4, 4).contiguous()
                pre = packed[:, :4].contiguous()
                weight = (torch.rand(5120, generator=generator) + .5).bfloat16().to('hpu')
                args = (value, residual, post, comb, pre, weight, projection)
                expected = [x.cpu() for x in functions[0](*args)]
                observed = [x.cpu() for x in functions[1](*args)]
                for i, (actual, wanted) in enumerate(zip(observed, expected, strict=True)):
                    if diagnostic_path is not None and not torch.equal(actual, wanted):
                        torch.save(dict(tokens=tokens, magnitude=magnitude, output_index=i,
                                        args=[x.cpu() for x in args], expected=expected, observed=observed,
                                        passed=checks), diagnostic_path)
                    assert torch.equal(actual, wanted), (generation, tokens, magnitude, i,
                                                         (actual != wanted).sum().item())
                checks.append(dict(generation=generation, tokens=tokens, magnitude=magnitude, all_outputs_exact=True))
    return checks
