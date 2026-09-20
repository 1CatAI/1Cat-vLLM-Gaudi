# SPDX-License-Identifier: Apache-2.0
"""Validate and time the C8192 native prefill primitives on one Gaudi2.

This diagnostic compares complete producer/consumer operations at the same
shape.  It is intentionally independent of the serving process so a failed
large-M contract is found before the four-rank model is reloaded.
"""

import argparse
import json
import time
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _measure(fn, args, warmup, rounds):
    compiled = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
    for _ in range(warmup):
        compiled(*args)
    torch.hpu.synchronize()
    host_ms, device_ms = [], []
    for _ in range(rounds):
        begin = torch.hpu.Event(enable_timing=True)
        end = torch.hpu.Event(enable_timing=True)
        host_begin = time.perf_counter_ns()
        begin.record()
        compiled(*args)
        end.record()
        end.synchronize()
        host_ms.append((time.perf_counter_ns() - host_begin) / 1e6)
        device_ms.append(begin.elapsed_time(end))
    return compiled, {
        "device_mean_ms": float(np.mean(device_ms)),
        "device_p50_ms": float(np.median(device_ms)),
        "device_p95_ms": float(np.percentile(device_ms, 95)),
        "synchronized_host_mean_ms": float(np.mean(host_ms)),
        "samples": rounds,
    }


def _equal(actual, expected, *, rtol=0, atol=0):
    if isinstance(actual, torch.Tensor):
        actual, expected = (actual,), (expected,)
    for lhs, rhs in zip(actual, expected, strict=True):
        torch.testing.assert_close(lhs.cpu(), rhs.cpu(), rtol=rtol, atol=atol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(__import__("os").environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.manual_seed(160916)
    t = args.tokens
    report = {"tokens": t, "device": "Gaudi2", "results": {}}

    scores = torch.randn(t, 384, dtype=torch.float32, device="hpu")
    text = torch.randn(384, dtype=torch.float32, device="hpu")
    image = torch.randn(384, dtype=torch.float32, device="hpu")
    mask = (torch.arange(t, device="hpu") % 7 == 0)
    router = torch.ops.custom_op.custom_deepseek_v41_router_top6_gaudi2
    router_ref = lambda s, a, b, m: (lambda ids: (ids.int(), s.gather(1, ids) /
        (s.gather(1, ids).sum(-1, keepdim=True) + 1e-20) * 1.5))(
            torch.argsort(s + torch.where(m[:, None], b, a), dim=-1, descending=True, stable=True)[:, :6])
    router_compiled, router_time = _measure(router, (scores, text, image, mask), args.warmup, args.rounds)
    actual = router_compiled(scores, text, image, mask)
    expected = router_ref(scores, text, image, mask)
    _equal(actual, expected, rtol=2e-6, atol=2e-7)
    changed = scores.clone()
    changed[:, 0] += 100
    assert not torch.equal(actual[0].cpu(), router_compiled(changed, text, image, mask)[0].cpu())
    _, router_ref_time = _measure(router_ref, (scores, text, image, mask), args.warmup, args.rounds)
    report["results"]["router_top6"] = {
        "candidate": router_time, "generic_stable_sort_reference": router_ref_time,
        "ids_and_weights_match": True, "changed_input_consumed": True,
    }

    mixes = torch.randn(t, 24, dtype=torch.float32, device="hpu")
    rrms = torch.rand(t, 1, dtype=torch.float32, device="hpu")
    scale = torch.randn(3, dtype=torch.float32, device="hpu")
    base = torch.randn(24, dtype=torch.float32, device="hpu")
    gates = torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2
    sink = torch.ops.custom_op.custom_deepseek_v4_sinkhorn4_gaudi2

    def gates_reference(x, r, s, b):
        value = x * r
        pre = torch.sigmoid(value[:, :4] * s[0] + b[:4]) + 1e-6
        post = torch.sigmoid(value[:, 4:8] * s[1] + b[4:8]) * 2
        comb = torch.softmax(value[:, 8:].reshape(-1, 4, 4) * s[2] + b[8:].reshape(1, 4, 4), -1) + 1e-6
        return torch.cat((pre, post, sink(comb.contiguous()).flatten(1)), -1)

    gates_compiled, gates_time = _measure(gates, (mixes, rrms, scale, base), args.warmup, args.rounds)
    gates_ref_compiled, gates_ref_time = _measure(gates_reference, (mixes, rrms, scale, base),
                                                   args.warmup, args.rounds)
    _equal(gates_compiled(mixes, rrms, scale, base), gates_ref_compiled(mixes, rrms, scale, base),
           rtol=2e-5, atol=2e-6)
    report["results"]["mhc_gates_sinkhorn"] = {
        "candidate": gates_time, "torch_chain_reference": gates_ref_time, "values_match": True,
    }

    activation = torch.randn(t, 5120, dtype=torch.bfloat16, device="hpu")
    quant = torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_bf16_gaudi2
    from vllm_gaudi.ops.deepseek_v41_math import _pack_swa_torch, unpack_swa
    quant_ref = lambda x: unpack_swa(_pack_swa_torch(x), x.shape[-1])
    quant_compiled, quant_time = _measure(quant, (activation,), args.warmup, args.rounds)
    quant_ref_compiled, quant_ref_time = _measure(quant_ref, (activation,), args.warmup, args.rounds)
    _equal(quant_compiled(activation), quant_ref_compiled(activation))
    report["results"]["activation_quant_roundtrip"] = {
        "candidate": quant_time, "torch_pack_unpack_reference": quant_ref_time,
        "bitwise_match": True,
    }

    # Prefill must never fall back to the eager BF16->integer reinterpret
    # codec: that graph is unsupported by Synapse and previously wedged the
    # request after the first multi-token chat prompt.  Measure the public
    # dispatchers so this checks the same branch used by serving.
    from vllm_gaudi.ops.deepseek_v41_math import (
        _pack_fp4_torch,
        pack_fp4,
        pack_swa,
    )
    kv = torch.randn(t, 512, dtype=torch.bfloat16, device="hpu")
    codecs = (
        ("swa", pack_swa, _pack_swa_torch),
        ("fp4_g16", lambda x: pack_fp4(x, 16), lambda x: _pack_fp4_torch(x, 16)),
        ("fp4_g32", lambda x: pack_fp4(x, 32), lambda x: _pack_fp4_torch(x, 32)),
    )
    for name, candidate, reference in codecs:
        candidate_compiled, candidate_time = _measure(candidate, (kv,), args.warmup, args.rounds)
        reference_compiled, reference_time = _measure(reference, (kv,), args.warmup, args.rounds)
        _equal(candidate_compiled(kv), reference_compiled(kv))
        report["results"][f"prefill_{name}_pack"] = {
            "candidate": candidate_time,
            "compiled_torch_reference": reference_time,
            "bitwise_match": True,
        }

    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
