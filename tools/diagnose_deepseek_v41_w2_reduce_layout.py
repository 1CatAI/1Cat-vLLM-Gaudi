# SPDX-License-Identifier: Apache-2.0
"""Untimed lane sentinels for the rejected W2 consumer; no performance credit."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators

    torch.hpu.set_device(0)
    torch.set_num_threads(1)
    load_native_operators()
    op = torch.ops.custom_op.custom_deepseek_v41_w2_reduce_n256_gaudi2

    def chain(product, ids, scales, channel):
        result = op(product, ids, scales, channel)
        return result, result.float().square().sum(-1)

    function = torch.compile(chain, backend='hpu_backend', fullgraph=True, dynamic=False)
    reports = []
    for case, tokens in enumerate((2, 5, 6)):
        slots = tokens * 6
        ids = (torch.arange(slots, dtype=torch.int32) * 11 % 384).reshape(1, -1)
        product = ((torch.arange(slots * 5120, dtype=torch.float32).reshape(slots, 1, 5120) % 8192) / 32 - 128)
        channel = torch.ones((384, 20, 256), dtype=torch.bfloat16)
        if case:
            channel = torch.exp2((torch.arange(channel.numel()).reshape_as(channel) % 5 - 2).float()).bfloat16()
        scales = torch.ones((slots, 1)) if not case else (torch.arange(slots).float() % 4 + 1).reshape(-1, 1) / 4
        selected = channel.reshape(384, 5120)[ids.reshape(-1).long()].reshape(slots, 1, 5120).float()
        routed = ((product * selected) * scales.reshape(slots, 1, 1)).bfloat16().float()
        reference = torch.zeros((tokens, 1, 5120))
        for route in range(6):
            reference += routed.reshape(tokens, 6, 5120)[:, route:route + 1]
        reference = reference.bfloat16()
        actual, consumed = function(*(x.to('hpu') for x in (product, ids, scales, channel)))
        actual, consumed = actual.cpu(), consumed.cpu()
        difference = actual.float() - reference.float()
        bad = (actual.view(torch.int16) != reference.view(torch.int16)).reshape(-1).nonzero().reshape(-1)
        reports.append(dict(case=case, tokens=tokens, exact=not bad.numel(), max_abs=float(difference.abs().max()),
                            mismatched_elements=bad.numel(), first_indices=bad[:16].tolist(),
                            actual_first=actual.reshape(-1)[:32].float().tolist(),
                            reference_first=reference.reshape(-1)[:32].float().tolist(),
                            downstream_finite=bool(torch.isfinite(consumed).all())))
    args.output.write_text(json.dumps(dict(timed=False, production_inputs=False,
        purpose='Identify lane/layout mistakes only; production/native qualification still required',
        cases=reports, credited_e2e_ms=0), indent=2) + '\n')


if __name__ == '__main__':
    main()
