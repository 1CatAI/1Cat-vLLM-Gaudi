# SPDX-License-Identifier: Apache-2.0
"""Real C5 bounds→bitmap→histogram correctness, without profiling/timing."""
import json
from pathlib import Path
import sys

import torch

from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators


def chain(scores, weights, prefix, shift):
    ops = torch.ops.custom_op
    bounds = ops.custom_deepseek_v41_weighted_score_bounds_gaudi2(scores, weights)
    mask = ops.custom_deepseek_v41_weighted_active_blocks_gaudi2(bounds, prefix, shift)
    bins = ops.custom_deepseek_v41_weighted_sparse_bins_gaudi2(scores, weights, prefix, mask, shift)
    parent = ops.custom_deepseek_v41_weighted_mass_bins_gaudi2(scores, weights, prefix, shift)
    return bounds, mask, bins, parent


def main():
    import habana_frameworks.torch.core  # noqa: F401

    load_native_operators()
    torch.set_num_threads(4)
    root = Path("/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving/sampled-nucleus-native-87")
    compiled = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)
    checks = []
    for case in range(3):
        shards = [torch.load(root / f"producer-case{case}-rank{rank}.pt", weights_only=True) for rank in range(4)]
        scores = torch.cat([shard["native"][2] for shard in shards], dim=-1).float().contiguous()
        weights = scores.softmax(-1)
        prefix = torch.zeros((5, 1), dtype=torch.int32)
        remaining = shards[0]["controls"][:, 1:2].clone()
        inputs = scores.to("hpu"), weights.to("hpu")
        bits = scores.view(torch.int32).long() & 0xFFFFFFFF
        keys = torch.where(bits > 0x7FFFFFFF, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)
        low = torch.where(weights > 0, keys, 0xFFFFFFFF).reshape(5, -1, 64).amin(-1)
        high = torch.where(weights > 0, keys, 0).reshape(5, -1, 64).amax(-1)
        expected_bounds = torch.stack((low, high), dim=1).to(torch.int32)
        for shift in range(28, -1, -4):
            values = tuple(value.cpu() for value in compiled(*inputs, prefix.to("hpu"), shift))
            mask = 0 if shift == 28 else (0xFFFFFFFF << (shift + 4)) & 0xFFFFFFFF
            start = prefix.long() & 0xFFFFFFFF
            active = (low <= high) & (high >= start) & (low <= (start | (~mask & 0xFFFFFFFF)))
            padded = torch.nn.functional.pad(active, (0, -active.shape[1] % 32)).reshape(5, -1, 32)
            expected_masks = (padded.long() << torch.arange(32)).sum(-1).to(torch.int32)
            if not torch.equal(values[1], expected_masks):
                torch.save(dict(actual=values, expected_masks=expected_masks, expected_bounds=expected_bounds,
                                case=case, shift=shift), Path(sys.argv[1]).with_suffix(".failure.pt"))
            assert torch.equal(values[0], expected_bounds), "Exact block bounds mismatch"
            assert torch.equal(values[1], expected_masks), "Vector bitmap lane mapping mismatch"
            assert torch.equal(values[2], values[3]), "Sparse histogram differs from the identical parent"
            checks.append(dict(case=case, shift=shift, bounds_masks_histograms_exact=True))
            histogram = values[3].sum(1).flip(-1)
            cumulative = histogram.cumsum(-1)
            selected = (cumulative < remaining).sum(-1, keepdim=True).clamp_max(15)
            prior = torch.where(selected > 0, cumulative.gather(1, (selected - 1).clamp_min(0)), 0)
            remaining -= prior
            prefix |= (15 - selected.int()) << shift
    Path(sys.argv[1]).write_text(json.dumps(dict(checks=checks, performance_credit_ms=0), indent=2) + "\n")


if __name__ == "__main__":
    main()
