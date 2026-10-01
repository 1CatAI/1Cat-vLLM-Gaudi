# SPDX-License-Identifier: Apache-2.0
"""Exact activation/scale/FP8 checks before the complete expert consumer gate."""
import torch


def check_feature_silu_contract(diagnostic_path=None):
    op = torch.ops.custom_op
    # Distinct Python entries keep the two six-shape finite contracts from
    # competing for OpOverloadPacket.__call__'s shared Dynamo cache.
    def reference(product, ids, sx, channel, router):
        return op.custom_deepseek_v41_silu_quant_reference_gaudi2(product, ids, sx, channel, router)

    def candidate(product, ids, sx, channel, router):
        return op.custom_deepseek_v41_silu_quant_feature_gaudi2(product, ids, sx, channel, router)

    functions = [torch.compile(fn, backend='hpu_backend', fullgraph=True, dynamic=False)
                 for fn in (reference, candidate)]
    generator = torch.Generator().manual_seed(41135)
    channel = torch.exp2(torch.randint(-5, 5, (17, 5, 256), generator=generator).float()).bfloat16().to('hpu')
    cases = []
    for generation in range(3):
        for rows in (2, 6, 12):
            for scale_rows in (1, rows):
                for magnitude in (0., .03125, 1., 32.):
                    product = torch.randn(rows, 1, 1280, generator=generator) * magnitude
                    ids = (torch.arange(rows).reshape(1, rows) * 3 + generation) % 17
                    if generation == 2:
                        ids[0, 0] = -1
                        ids[0, -1] = 17
                    sx = torch.exp2(torch.randint(-5, 2, (scale_rows, 1), generator=generator).float())
                    router = torch.rand(1, rows, generator=generator) * 1.5
                    args = (product.to('hpu'), ids.int().to('hpu'), sx.to('hpu'), channel, router.to('hpu'))
                    expected = [x.cpu().view(torch.uint8) for x in functions[0](*args)]
                    actual = [x.cpu().view(torch.uint8) for x in functions[1](*args)]
                    for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
                        if not torch.equal(a, b) and diagnostic_path is not None:
                            torch.save(dict(args=[x.cpu() for x in args], expected=expected, actual=actual,
                                            passed=cases, output=i), diagnostic_path)
                        assert torch.equal(a, b), (generation, rows, scale_rows, magnitude, i,
                                                  int((a != b).sum()))
                    cases.append(dict(generation=generation, rows=rows, scale_rows=scale_rows,
                                      magnitude=magnitude, bytes_exact=True))
    # Do not infer that reorganizing maxima preserves special-value behavior.
    # Exercise the original kernel's actual NaN/Inf and signed-zero encoding.
    rows = 2
    product = torch.zeros(rows, 1, 1280)
    product[0, 0, ::129] = float('nan')
    product[0, 0, 4::133] = float('inf')
    product[1, 0, ::137] = -float('inf')
    product[1, 0, 3::139] = -0.
    args = (product.to('hpu'), torch.tensor([[0, 1]], dtype=torch.int32, device='hpu'),
            torch.ones(1, 1, device='hpu'), channel, torch.ones(1, rows, device='hpu'))
    expected = [x.cpu().view(torch.uint8) for x in functions[0](*args)]
    actual = [x.cpu().view(torch.uint8) for x in functions[1](*args)]
    for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
        if not torch.equal(a, b) and diagnostic_path is not None:
            torch.save(dict(args=[x.cpu() for x in args], expected=expected, actual=actual,
                            passed=cases, output=i, special_values=True), diagnostic_path)
        assert torch.equal(a, b), ('special_values', i, int((a != b).sum()))
    cases.append(dict(special_values=True, bytes_exact=True))
    return cases
