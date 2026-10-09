# SPDX-License-Identifier: Apache-2.0
"""Check changing nucleus coverage on the actual HPU compiled sampling path.

This is a backend capability check, never a production-input performance gate.
It deliberately alternates covered and fallback rows at unchanged addresses.
"""
import json
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radix", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    import habana_frameworks.torch.core  # noqa: F401
    import torch

    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
        bounded_sampling_parts, sample_bounded_or_full_distribution, sample_full_distribution)

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    bind_worker_cpu(rank)
    report = dict(production_micro_qualified=False, capability_passed=False, cases=[])
    if args.radix:
        torch.ops.load_library(os.environ["DSV41_UNIQUE_OPERATOR_LIBRARY"])
    try:
        with torch.inference_mode():
            for count in (6, 1):
                shape = (count, 129280)
                logits = torch.empty(shape, device="hpu", dtype=torch.float32)
                controls = torch.tensor([[1., .95, .13, -1.]]).repeat(count, 1).to("hpu")
                reference = torch.compile(sample_full_distribution, backend="hpu_backend",
                                          fullgraph=True, dynamic=False)
                def bounded(x, settings):
                    return sample_bounded_or_full_distribution(x, settings, radix=args.radix)

                candidate = torch.compile(bounded, backend="hpu_backend",
                                          fullgraph=True, dynamic=False)
                def certificate(x, settings):
                    return bounded_sampling_parts(x, settings, radix=args.radix)[2]

                compiled_certificate = torch.compile(certificate, backend="hpu_backend",
                                                      fullgraph=True, dynamic=False)
                # The final covered case must not inherit the fallback decision.
                for kind in ("covered", "fallback", "covered"):
                    generator = torch.Generator().manual_seed(42)
                    value = torch.randn(shape, generator=generator) * .1
                    if kind == "covered":
                        value[:, :8] += torch.arange(8).float() * .3 + 15
                    logits.copy_(value)
                    expected = tuple(x.cpu() for x in reference(logits, controls))
                    actual = tuple(x.cpu() for x in candidate(logits, controls))
                    covered = bounded_sampling_parts(value, controls.cpu())[2]
                    # An incorrect selector could silently force every input
                    # through the correct full-sort fallback. Check membership
                    # and the device certificate independently of final tokens.
                    if args.radix:
                        selected, _ = torch.ops.custom_op.custom_deepseek_v41_vocab_radix_topk_gaudi2(logits, 256)
                        selected = selected.cpu().sort(dim=-1, descending=True)[0]
                        torch.testing.assert_close(selected, value.topk(256, dim=-1)[0], rtol=0, atol=0)
                    device_covered = compiled_certificate(logits, controls).cpu()
                    if not torch.equal(device_covered, covered):
                        raise AssertionError("Device nucleus certificate differs from the reference")
                    exact = torch.equal(expected[0], actual[0])
                    error = float((expected[1] - actual[1]).abs().max())
                    report["cases"].append(dict(rows=count, kind=kind, covered=covered.reshape(-1).tolist(),
                                                tokens_equal=exact, probability_max_abs=error,
                                                fallback_bitwise=torch.equal(expected[1], actual[1])))
                    if not exact:
                        raise AssertionError("Bounded sampling changed the sampled token")
                    torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=2e-7)
                    if kind == "fallback" and not torch.equal(expected[1], actual[1]):
                        raise AssertionError("Uncovered nucleus did not execute the complete fallback")
                    bind_worker_helpers(rank)
            report["capability_passed"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        (root / "bounded-backend.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
