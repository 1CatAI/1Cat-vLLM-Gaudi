# SPDX-License-Identifier: Apache-2.0
"""Actual C5 nucleus finish -> static positive tile mask -> eight draw histograms."""
import json
from pathlib import Path
import sys
import torch
from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators

def finish(scores, weights, prefix):
    ops = torch.ops.custom_op
    return (*ops.custom_deepseek_v41_weighted_finish_mask_gaudi2(scores, weights, prefix),
            *ops.custom_deepseek_v41_weighted_mass_finish_gaudi2(scores, weights, prefix))

def hist(scores, kept, prefix, mask, shift):
    ops = torch.ops.custom_op
    return (ops.custom_deepseek_v41_weighted_sparse_bins_gaudi2(scores, kept, prefix, mask, shift),
            ops.custom_deepseek_v41_weighted_mass_bins_gaudi2(scores, kept, prefix, shift))

def main():
    import habana_frameworks.torch.core  # noqa: F401
    load_native_operators()
    torch.set_num_threads(4)
    root = Path("/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving/sampled-nucleus-native-87")
    compiled_finish = torch.compile(finish, backend="hpu_backend", fullgraph=True, dynamic=False)
    compiled_hist = torch.compile(hist, backend="hpu_backend", fullgraph=True, dynamic=False)
    checks = []
    for case in range(3):
        shards = [torch.load(root / f"producer-case{case}-rank{rank}.pt", weights_only=True) for rank in range(4)]
        scores = torch.cat([shard["native"][2] for shard in shards], dim=-1).float().contiguous()
        weights = scores.softmax(-1)
        ordered, ids = scores.sort(descending=True)
        probs = weights.gather(1, ids)
        allowed = (probs.cumsum(-1) - probs) < shards[0]["controls"][:, 1:2]
        last = allowed.sum(-1, keepdim=True).clamp_min(1) - 1
        value = ordered.gather(1, last).contiguous()
        bits = value.view(torch.int32).long() & 0xFFFFFFFF
        prefix = torch.where(bits > 0x7FFFFFFF, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000).to(torch.int32)
        hs, hw = scores.to("hpu"), weights.to("hpu")
        outputs = compiled_finish(hs, hw, prefix.to("hpu"))
        values = [v.cpu() for v in outputs]
        assert all(torch.equal(values[i], values[i+4]) for i in range(3)), "Parent finish changed"
        active = torch.nn.functional.pad(values[0] > 0, (0, -scores.shape[1] % 2048))
        active = active.reshape(5, -1, 32, 64).any(-1)
        expected = (active.long() << torch.arange(32)).sum(-1).to(torch.int32)
        assert torch.equal(values[3], expected), "Static mask lane mismatch"
        prefix = torch.zeros((5, 1), dtype=torch.int32)
        remaining = shards[0]["controls"][:, 2:3] * values[0].sum(-1, keepdim=True)
        for shift in range(28, -1, -4):
            bins = [v.cpu() for v in compiled_hist(hs, outputs[0], prefix.to("hpu"), outputs[3], shift)]
            assert torch.equal(bins[0], bins[1]), "Sparse draw histogram changed"
            checks.append(dict(case=case, shift=shift, finish_mask_histogram_exact=True,
                               active_tile_fraction=float(active.float().mean())))
            cumulative = bins[1].sum(1).flip(-1).cumsum(-1)
            selected = (cumulative < remaining).sum(-1, keepdim=True).clamp_max(15)
            prior = torch.where(selected > 0, cumulative.gather(1, (selected-1).clamp_min(0)), 0)
            remaining -= prior
            prefix |= (15-selected.int()) << shift
    Path(sys.argv[1]).write_text(json.dumps(dict(checks=checks, performance_credit_ms=0), indent=2)+"\n")

if __name__ == "__main__":
    main()
