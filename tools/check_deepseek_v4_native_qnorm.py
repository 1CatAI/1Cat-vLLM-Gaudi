# SPDX-License-Identifier: Apache-2.0
"""Check the native QNorm mutation with persistent and graph-internal Q."""
import json
import os
from pathlib import Path

import torch
from vllm.models.deepseek_v4.hw_agnostic.attention import attention  # noqa: F401


@torch.inference_mode()
def main():
    if os.environ.get("HLS_MODULE_ID") is None:
        raise RuntimeError("Select and reserve one HPU module before this check")
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    output = Path(os.environ["DSV4_CHECK_OUTPUT"])
    output.mkdir(exist_ok=True)
    torch.manual_seed(17)
    q = torch.randn(1, 32, 512).bfloat16().to("hpu")
    kv = torch.randn(1, 512).bfloat16().to("hpu")
    phase = torch.arange(512).float()[:, None] * torch.arange(32).float()[None, :] / 100
    rotary = torch.cat((phase.cos(), phase.sin()), dim=1).to("hpu")
    geometry = torch.tensor([0, 128 * 584, 128, 4], dtype=torch.int32, device="hpu")
    rows = []
    for internal in (False, True):
        def program(q, kv, cache, geometry, slot, position, rotary, internal=internal):
            value = q + 0.25 if internal else q
            ack = torch.ops.custom_op.custom_deepseek_v4_native_qnorm_rope_kv_pack_bf16_gaudi2(
                value, kv, cache, geometry, slot, position, rotary)
            return torch.where(ack.reshape(1, 1, 1) <= 1, value, torch.zeros_like(value))

        compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
        for step in (3, 4, 127):
            slot = torch.tensor([step], dtype=torch.int32, device="hpu")
            reference_q = (q + 0.25 if internal else q).clone()
            reference_cache = torch.zeros(4 * 128 * 584, dtype=torch.uint8, device="hpu")
            candidate_cache = torch.zeros_like(reference_cache)
            torch.ops.custom_op.custom_deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2(
                reference_q, kv, reference_cache, geometry, slot, slot, rotary)
            candidate_q = compiled(q.clone(), kv, candidate_cache, geometry, slot, slot, rotary)
            expected, actual = reference_q.cpu(), candidate_q.cpu()
            cache_equal = torch.equal(reference_cache.cpu(), candidate_cache.cpu())
            rows.append(dict(internal=internal, position=step, q_exact=torch.equal(expected, actual),
                             changed_q_elements=int((expected != actual).sum()), cache_exact=cache_equal,
                             max_abs=float((expected.float() - actual.float()).abs().max())))
            torch.save(dict(expected=expected, actual=actual), output / f"q-internal{int(internal)}-p{step}.pt")
            print(json.dumps(rows[-1]), flush=True)
    (output / "result.json").write_text(json.dumps(rows, indent=2))
    if not all(row["q_exact"] and row["cache_exact"] for row in rows):
        raise AssertionError("Native QNorm differs from the existing physical kernel")


if __name__ == "__main__":
    main()
