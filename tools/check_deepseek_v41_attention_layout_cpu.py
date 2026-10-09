# SPDX-License-Identifier: Apache-2.0
"""Real checkpoint contract for offline Q/wo_a channel permutations; no timing."""
import argparse
import json
from pathlib import Path

import torch

from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree
from vllm_gaudi.ops.deepseek_v41_attention_layout import (
    group32_membership, interleave_output_weight, interleave_query_weight, pack_heads, unpack_heads,
)
from vllm_gaudi.ops.deepseek_v41_math import quantize_activation
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    names = ('wq_b', 'wo_a')
    specs = {name: value for name, value in shard.specs.items()
             if any(name.startswith(f'layers.20.attn.{part}.') for part in names)}
    tree = _weight_tree(specs)
    load_weight_tree(shard, tree, 'cpu', specs)
    weights = tree.layers.get_submodule('20').attn
    wq, wo = weights.wq_b.weight, weights.wo_a.weight.reshape(2, 1024, 4096)
    new_q, new_o = interleave_query_weight(wq, 16), interleave_output_weight(wo)
    inverse_q = new_q.reshape(2, 512, 8, 1280).transpose(1, 2).reshape_as(wq)
    inverse_o = new_o.reshape(2, 1024, 512, 8).transpose(2, 3).reshape_as(wo)
    assert torch.equal(inverse_q.view(torch.int16), wq.view(torch.int16))
    assert torch.equal(inverse_o.view(torch.int16), wo.view(torch.int16))
    for heads in (16, 32):
        membership = group32_membership(heads)
        assert torch.equal(membership, membership[:, :, :1].expand_as(membership))
        assert membership.unique().numel() == heads * 16
    checks = []
    for path in sorted((args.fixtures / 'rank0').glob('c6-*.pt'))[:3]:
        saved = torch.load(path, weights_only=True, map_location='cpu')['groups'][4]
        # Actual residual channels exercise the complete width; these are not
        # claimed as post-projection attention fixtures or performance evidence.
        x = quantize_activation(saved['residual'][:, 0, :1280].contiguous())
        projected = torch.nn.functional.linear(x, wq).reshape(6, 16, 512)
        interleaved = torch.nn.functional.linear(x, new_q).reshape(6, 2, 512, 8)
        assert torch.equal(projected.view(torch.int16), unpack_heads(interleaved).view(torch.int16))
        # Include arbitrary actual generated values in all per-head codec
        # groups, preserving membership despite noncontiguous head dimensions.
        grouped = quantize_activation(projected.reshape(6, 2, 4096))
        packed = pack_heads(grouped.reshape(6, 16, 512)).reshape(6, 2, 4096)
        parent = torch.einsum('tgd,gnd->tgn', grouped.float(), wo.float())
        candidate = torch.einsum('tgd,gnd->tgn', packed.float(), new_o.float())
        delta = candidate - parent
        checks.append(dict(source=str(path), query_exact=True,
                           output_max_abs=float(delta.abs().max()),
                           output_relative_l2=float(delta.norm() / parent.norm().clamp_min(1e-30))))
    assert len(checks) == 3
    report = dict(status='cpu_layout_contract_passed_device_and_teacher_pending',
                  real_weight_layer=20, q_shape=list(wq.shape), output_shape=list(wo.shape),
                  weight_inverse_byte_exact=True, per_head_group32_membership_exact=True,
                  runtime_transpose_required=False, proposed_qk_transpose=(True, True),
                  proposed_pv_swapped_operands_transpose=(True, True),
                  output_reduction_order_can_change=True, checks=checks,
                  performance_vote=False, gain_credit_ms=0)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
