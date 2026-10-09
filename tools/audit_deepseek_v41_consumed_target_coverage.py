# SPDX-License-Identifier: Apache-2.0
"""Offline certificate audit from actual request hidden states and head weights.

CPU projection is an entropy/certificate diagnostic, not a device performance
or probability-equivalence qualification of the HPU projection.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
        bounded_proposal_distribution, sample_full_distribution, sample_speculative_prefix,
        speculative_sampling_draws, target_coverage_for_commit)

    torch.set_num_threads(8)
    allowed = os.sched_getaffinity(0)
    preferred = allowed & set(range(48, 56))
    if len(preferred) == 8:
        os.sched_setaffinity(0, preferred)
    cases = [[torch.load(args.fixtures / f"rank{r}/c6-{i}.pt", map_location="cpu", weights_only=True)
              for r in range(4)] for i in range(3)]
    logits = [[] for _ in cases]
    with torch.inference_mode():
        for rank in range(4):
            shard = PreparedV41Shard(args.prepared, 0, rank)
            weight = shard.tensor("head.weight", "cpu").float()
            for index, case in enumerate(cases):
                logits[index].append(torch.nn.functional.linear(case[rank]["hidden"].float(), weight))
            del weight
        report = dict(status="offline diagnostic", device_timing=False, cases=[], gain_credited_ms=0)
        for index, case in enumerate(cases):
            data = case[0]
            _, acceptance, correction, controls = speculative_sampling_draws(
                data["sampling_parameters"], data["sampling_seed"], data["sampling_counter"].clone(),
                data["sampling_offsets"])
            full_logits = torch.cat(logits[index], -1)
            _, p = sample_full_distribution(full_logits, controls)
            q = torch.cat([row["proposal"] for row in case], -1)
            ids = data["ids"][1:]
            reference, count, valid = sample_speculative_prefix(p, q, ids, acceptance, correction)
            widths = []
            for width in (64, 128, 256):
                packet = torch.cat([local_nucleus_packet(value, controls, r, width)
                                    for r, value in enumerate(logits[index])], -1)
                probabilities, covered = [], None
                for rank in range(4):
                    part, covered = bounded_proposal_distribution(packet, controls, tp_rank=rank,
                        tp_size=4, width=width, local_vocab=logits[index][rank].shape[-1])
                    probabilities.append(part)
                output, used, actual_valid = sample_speculative_prefix(
                    torch.cat(probabilities, -1), q, ids, acceptance, correction)
                widths.append(dict(width=width, covered=covered.flatten().tolist(),
                    all_rows_certified=bool(covered.all()),
                    consumed_rows_certified=bool(target_coverage_for_commit(covered, used)),
                    consumed_count=int(used), output_exact=bool(torch.equal(output, reference)),
                    valid=bool(actual_valid)))
            report["cases"].append(dict(case=index, reference_committed=int(count), reference_valid=bool(valid),
                target_nucleus_sizes=(p > 0).sum(-1).tolist(),
                previous_q_nucleus_sizes=(q > 0).sum(-1).tolist(), widths=widths))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
