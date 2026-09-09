# SPDX-License-Identifier: Apache-2.0
"""Retain SWA/sparse compute and KV writes through whole-graph compilation."""
import json
import os
from pathlib import Path

import torch
from vllm.models.deepseek_v4.hw_agnostic.attention import attention  # noqa: F401


@torch.inference_mode()
def main():
    if os.environ.get("HLS_MODULE_ID") is None:
        raise RuntimeError("Select and reserve one HPU before this check")
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    output = Path(os.environ["DSV4_CHECK_OUTPUT"])
    output.mkdir(exist_ok=True)
    torch.manual_seed(18)
    q = torch.randn(1, 32, 512).bfloat16().to("hpu")
    kv = torch.randn(1, 512).bfloat16().to("hpu")
    phase = torch.arange(512).float()[:, None] * torch.arange(32).float()[None, :] / 100
    rotary = torch.cat((phase.cos(), phase.sin()), dim=1).to("hpu")
    geometry = torch.tensor([0, 128 * 584, 128, 4], dtype=torch.int32, device="hpu")
    slot = torch.zeros(1, dtype=torch.int32, device="hpu")
    indices = torch.arange(256, dtype=torch.int32, device="hpu").reshape(1, 256)
    lengths = torch.ones(1, dtype=torch.int32, device="hpu")
    topk = torch.zeros((1, 1), dtype=torch.int32, device="hpu")
    empty_lengths = torch.zeros(1, dtype=torch.int32, device="hpu")
    sink = torch.randn(64, dtype=torch.float32, device="hpu")
    rows = []
    for sparse in (False, True):
        def program(q, kv, cache, geometry, slot, rotary, indices, lengths, topk, empty_lengths, sink, sparse=sparse):
            value = q + 0.25
            ack = torch.ops.custom_op.custom_deepseek_v4_native_qnorm_rope_kv_pack_bf16_gaudi2(
                value, kv, cache, geometry, slot, slot, rotary)
            value = torch.where(ack.reshape(1, 1, 1) <= 1, value, torch.zeros_like(value))
            if sparse:
                result, _ = torch.ops.custom_op.custom_deepseek_v4_native_paged_sparse_attn_fp8_gaudi2(
                    value, cache, geometry, topk, empty_lengths, cache, geometry, indices, lengths, sink)
            else:
                result, _ = torch.ops.custom_op.custom_deepseek_v4_native_paged_swa_attn_fp8_gaudi2(
                    value, cache, geometry, indices, lengths, indices, lengths, lengths,
                    cache, geometry, indices, lengths, sink)
            return result[:, :32]

        compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
        reference_cache = torch.zeros(4 * 128 * 584, dtype=torch.uint8, device="hpu")
        candidate_cache = torch.zeros_like(reference_cache)
        for step in (1, 4, 127, 128):
            slot.fill_(step)
            lengths.fill_(step + 1)
            value = (q + 0.25).clone()
            torch.ops.custom_op.custom_deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2(
                value, kv, reference_cache, geometry, slot, slot, rotary)
            reference = torch.empty((1, 64, 512), dtype=q.dtype, device=q.device)
            if sparse:
                torch.ops.custom_op.custom_deepseek_v4_paged_sparse_attn_fp8_gaudi2(
                    value, reference_cache, geometry, topk, empty_lengths, reference_cache,
                    geometry, indices, lengths, sink, reference)
            else:
                torch.ops.custom_op.custom_deepseek_v4_paged_swa_attn_fp8_gaudi2(
                    value, reference_cache, geometry, indices, lengths, indices, lengths, lengths,
                    reference_cache, geometry, indices, lengths, sink, reference)
            actual = compiled(q, kv, candidate_cache, geometry, slot, rotary, indices,
                              lengths, topk, empty_lengths, sink).cpu()
            expected = reference[:, :32].cpu()
            row = dict(sparse=sparse, position=step, output_exact=torch.equal(expected, actual),
                       cache_exact=torch.equal(reference_cache.cpu(), candidate_cache.cpu()),
                       reference_nonzero=bool(expected.count_nonzero()),
                       changed_elements=int((expected != actual).sum()),
                       max_abs=float((expected.float() - actual.float()).abs().max()))
            rows.append(row)
            torch.save(dict(expected=expected, actual=actual), output / f"sparse{int(sparse)}-p{step}.pt")
            print(json.dumps(row), flush=True)
            (output / "result.json").write_text(json.dumps(rows, indent=2) + "\n")
    assert all(row["output_exact"] and row["cache_exact"] and row["reference_nonzero"] for row in rows)


if __name__ == "__main__":
    main()
