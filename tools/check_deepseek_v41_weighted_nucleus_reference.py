# SPDX-License-Identifier: Apache-2.0
"""Screen weighted radix nucleus selection against saved production logits.

This is an offline arithmetic screen, never a performance qualification. The
proposed device graph reduces 16 mass bins per tile; the CPU implementation
below deliberately follows the same four-bit refinement. Ambiguous boundary
ties must retain the existing exact sort repair.
"""

import argparse
import json
from pathlib import Path

import torch


def weighted_cut(keys, probability, target, active):
    """Locate a descending-score mass quantile without sorting the vocabulary."""
    prefix = 0
    remaining = target.clone()
    eligible = active.clone()
    for shift in range(28, -1, -4):
        digit = (keys >> shift) & 15
        bins = torch.zeros(16, dtype=torch.float32)
        bins.scatter_add_(0, digit, torch.where(eligible, probability, 0.0))
        cumulative = bins.flip(0).cumsum(0)
        index = int((cumulative < remaining).sum().clamp_max(15))
        chosen = 15 - index
        if index:
            remaining = remaining - cumulative[index - 1]
        prefix |= chosen << shift
        eligible &= digit == chosen
    return prefix, eligible


def weighted_nucleus(logits, control):
    temperature, top_p, uniform, top_k = control
    if temperature <= 0 or top_p >= 1 or top_k > 0:
        raise ValueError("Screen requires official temperature/top_p, no top_k")
    scaled = logits.float() / temperature
    probability = scaled.softmax(-1)
    bits = scaled.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    keys = torch.where((bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)
    cutoff, boundary = weighted_cut(keys, probability, top_p, torch.ones_like(keys, dtype=torch.bool))
    if int(boundary.sum()) != 1:
        return None
    keep = keys >= cutoff
    kept = torch.where(keep, probability, 0.0)
    mass = kept.sum()
    _, winner = weighted_cut(keys, probability, uniform * mass, keep)
    if int(winner.sum()) != 1:
        return None
    return int(winner.nonzero()[0, 0]), kept / mass


def reference(logits, control):
    temperature, top_p, uniform, _ = control
    values, order = (logits.float() / temperature).sort(descending=True)
    probabilities = values.softmax(-1)
    cumulative = probabilities.cumsum(-1)
    keep = cumulative - probabilities < top_p
    kept = torch.where(keep, probabilities, 0.0)
    cumulative = kept.cumsum(-1)
    selected = int((cumulative < uniform * cumulative[-1]).sum().clamp_max(values.numel() - 1))
    q = torch.zeros_like(logits, dtype=torch.float32).scatter(0, order, kept / kept.sum())
    return int(order[selected]), q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = []
    for case in range(3):
        shards = [torch.load(args.fixtures / f"producer-case{case}-rank{rank}.pt", weights_only=True)
                  for rank in range(4)]
        logits = torch.cat([shard["native"][2] for shard in shards], dim=-1)
        controls = shards[0]["controls"]
        for row in range(logits.shape[0]):
            expected_id, expected_q = reference(logits[row], controls[row])
            result = weighted_nucleus(logits[row], controls[row])
            rows.append(dict(case=case, row=row, boundary_tie=result is None,
                             token_exact=result is not None and expected_id == result[0],
                             q_max_abs=None if result is None else float((expected_q - result[1]).abs().max()),
                             reference_nucleus=int((expected_q > 0).sum())))
    report = dict(scope="15 saved production biased C5 rows, arithmetic only", rows=rows,
                  qualified_micro=False, credited_ms=0,
                  all_tokens_exact=all(row["token_exact"] for row in rows),
                  q_max_abs=max(row["q_max_abs"] or 0 for row in rows))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
