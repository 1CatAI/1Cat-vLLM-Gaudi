# SPDX-License-Identifier: Apache-2.0
"""Locate probability differences using actual request FP32 scores, no timing."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
        bounded_proposal_distribution, sample_full_distribution, speculative_sampling_draws)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers

    bind_worker_cpu(0)
    bind_worker_helpers(0)
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(scope="numerical root-cause diagnostic only", qualified_gain_ms=0, cases=[])

    def run(logits, controls):
        scaled = logits / controls[:, :1]
        values, ids = scaled[:, :32320].topk(64, dim=-1)
        gathered = scaled[:, :32320].gather(-1, ids)
        full_ids, full = sample_full_distribution(logits, controls)
        packet = torch.cat(tuple(local_nucleus_packet(logits[:, rank * 32320:(rank + 1) * 32320],
                                                     controls, rank, 64) for rank in range(4)), -1)
        bounded = torch.cat(tuple(bounded_proposal_distribution(packet, controls, tp_rank=rank,
                                                               tp_size=4, width=64, local_vocab=32320)[0]
                                  for rank in range(4)), -1)
        softmax = scaled.softmax(-1)
        exponential = (scaled - scaled.amax(-1, keepdim=True)).exp()
        normalized = exponential / exponential.sum(-1, keepdim=True)
        return values, gathered, full_ids, full, bounded, softmax, normalized

    compiled = torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)
    with torch.inference_mode():
        for case in range(3):
            data = [torch.load(args.fixtures / f"rank{rank}" / f"c6-{case}.pt", weights_only=True)
                    for rank in range(4)]
            logits = torch.cat([d["logits"] for d in data], -1)
            d = data[0]
            _, _, _, controls = speculative_sampling_draws(d["sampling_parameters"], d["sampling_seed"],
                                                          d["sampling_counter"].clone(), d["sampling_offsets"])
            values = tuple(v.cpu() for v in compiled(logits.to("hpu"), controls.to("hpu")))
            top, gathered, ids, full, bounded, softmax, normalized = values
            expected_top = logits[:, :32320].topk(64, dim=-1)[0]
            entry = dict(case=case, topk_gather_exact=torch.equal(top, gathered),
                         topk_cpu_max_abs=float((top - expected_top).abs().max()),
                         topk_gather_max_abs=float((top - gathered).abs().max()),
                         softmax_exp_max_abs=float((softmax - normalized).abs().max()),
                         full_bounded_max_abs=float((full - bounded).abs().max()))
            report["cases"].append(entry)
            torch.save(dict(logits=logits, controls=controls, values=values), root / f"case{case}.pt")
            (root / "sampling-primitives.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
