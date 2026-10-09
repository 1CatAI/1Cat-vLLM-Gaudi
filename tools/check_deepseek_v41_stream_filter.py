# SPDX-License-Identifier: Apache-2.0
"""Real-logit selector correctness diagnostic; no performance qualification."""

import os
import json
from pathlib import Path


def main():
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment, load_native_operators

    prepare_environment(
        Path("/opt/optane/prepared/DeepSeek-V4.1-Flash-dba1be0-tp4-pp1-q16v2"),
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
    )
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    r = Path("/opt/ssd960/builds/dsv41-tp4-dspark-round-chain-v1/production-c6-request-fixtures-02-with-history/rank0")
    results = []
    with (
        set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4))),
        torch.inference_mode(),
    ):
        load_native_operators()
        torch.hpu.set_device(0)

        def select(scores, cut):
            return torch.ops.custom_op.custom_deepseek_v41_vocab_filter_gaudi2(scores, cut, 64)

        call = torch.compile(select, backend="hpu_backend", fullgraph=True, dynamic=False)
        for case in range(3):
            data = torch.load(r / f"c6-{case}.pt", weights_only=True)
            x = data["logits"].float()
            maximum = x.amax(-1, keepdim=True)
            total = (x - maximum).exp().sum(-1, keepdim=True)
            cut = maximum + total.log() + torch.tensor(0.05 / 64).log()
            scores = torch.nn.functional.pad(x, (0, 32768 - 32320), value=-float("inf")).reshape(48, 4096).contiguous()
            actual = tuple(v.cpu() for v in call(scores.to("hpu"), cut.to("hpu")))
            expected = []
            for row in range(48):
                ids = (scores[row] > cut[row // 8]).nonzero().flatten()
                n = min(ids.numel(), 65)
                ids = ids[:64]
                values = torch.full((64,), -float("inf"))
                indices = torch.zeros(64, dtype=torch.int32)
                values[: ids.numel()] = scores[row, ids]
                indices[: ids.numel()] = ids.int()
                expected.append((values, indices, n))
            ref = (
                torch.stack([a[0] for a in expected]),
                torch.stack([a[1] for a in expected]),
                torch.tensor([a[2] for a in expected], dtype=torch.int32).reshape(48, 1),
            )
            checks = [torch.equal(a, b) for a, b in zip(actual, ref, strict=True)]
            results.append(
                dict(
                    case=case,
                    checks=checks,
                    actual_counts=actual[2].flatten().tolist(),
                    reference_counts=ref[2].flatten().tolist(),
                )
            )
            torch.save(dict(scores=scores, cutoff=cut, actual=actual, reference=ref), root / f"selector-case{case}.pt")
            (root / "selector-checks.json").write_text(json.dumps(results, indent=2) + "\n")
            if not all(checks):
                raise AssertionError("TPC compaction differs from real-logit CPU reference")


if __name__ == "__main__":
    main()
