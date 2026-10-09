# SPDX-License-Identifier: Apache-2.0
"""Check a single BF16 PV against existing real C6/FP64 attention operands.

CPU-only numerical evidence; never qualifies performance. The checkpoint's
official sparse attention casts unnormalized exponents to BF16 before PV.
Evaluate that boundary and a normalized BF16 probability separately.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4, unpack_swa

    torch.set_num_threads(4)
    oracle = []
    for rank in range(4):
        cases = []
        for case in range(3):
            path = args.fixtures / f"attention-operands-rank{rank}-case{case}.pt"
            parent = torch.load(path, map_location="cpu", weights_only=True)[0]
            absolute = parent["positions"].unsqueeze(-1) - 127 + torch.arange(128)
            window = unpack_swa(parent["swa"])[absolute.remainder(256).long()]
            keys = torch.cat((window, unpack_fp4(parent["packed_main"], group=16)), dim=1)
            valid = torch.cat((absolute >= 0, parent["selected"] >= 0), dim=1)
            query = parent["query"]
            scores64 = query.double() @ keys.double().transpose(-1, -2)
            scores64 *= parent["scale"].double()
            scores64.masked_fill_(~valid.unsqueeze(1), -torch.inf)
            sink64 = parent["sink"].double().reshape(1, -1, 1).expand(scores64.shape[0], -1, -1)
            reference = torch.softmax(torch.cat((scores64, sink64), -1), -1)[..., :-1] @ keys.double()
            scores = query.float() @ keys.float().transpose(-1, -2)
            scores *= parent["scale"].float()
            scores.masked_fill_(~valid.unsqueeze(1), -torch.inf)
            sink = sink64.float()
            probability = torch.softmax(torch.cat((scores, sink), -1), -1)[..., :-1]
            normalized = (probability.bfloat16().float() @ keys.float()).bfloat16()
            maximum = torch.maximum(scores.amax(-1, keepdim=True), sink)
            exponential = torch.exp(scores - maximum)
            denominator = exponential.sum(-1, keepdim=True) + torch.exp(sink - maximum)
            official_boundary = ((exponential.bfloat16().float() @ keys.float()) / denominator).bfloat16()
            results = {}
            for name, actual in (("normalized_bf16", normalized), ("unnormalized_bf16", official_boundary)):
                delta = actual.double() - reference
                relative = float(delta.norm() / reference.norm().clamp_min(1e-30))
                finite = bool(torch.isfinite(actual).all())
                close = bool(torch.allclose(actual.double(), reference, atol=.5, rtol=.008))
                results[name] = dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                                     relative_l2=relative, finite=finite, official_tolerance_close=close,
                                     passed=finite and close and relative <= .002,
                                     parent_max_abs=float((actual.float() - parent["output"].float()).abs().max()))
            cases.append(dict(case=case, shape=list(query.shape), ratio=parent["ratio"], results=results,
                              fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        oracle.append(cases)
    report = dict(kind="CPU_attention_bf16_pv", oracle=oracle, performance_qualified=False,
                  performance_credit_ms=0, default_enabled=False,
                  normalized_passed=all(c["results"]["normalized_bf16"]["passed"] for r in oracle for c in r),
                  unnormalized_passed=all(c["results"]["unnormalized_bf16"]["passed"] for r in oracle for c in r))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "oracle"}))


if __name__ == "__main__":
    main()
