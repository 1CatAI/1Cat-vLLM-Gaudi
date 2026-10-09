# SPDX-License-Identifier: Apache-2.0
"""Three actual probability fixtures; primitive correctness, no speed claim."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--teacher-logits", type=Path, required=True)
    args = parser.parse_args()
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators

    load_native_operators(required=("custom_deepseek_v41_probability_draw_gaudi2",))
    torch.hpu.set_device(0)

    def operation(q, controls):
        return torch.ops.custom_op.custom_deepseek_v41_probability_draw_gaudi2(q, controls)

    compiled = torch.compile(operation, backend="hpu_backend", fullgraph=True, dynamic=False)
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import weighted_sampling_distribution
    from check_deepseek_v41_teacher_forced_acceptance import distribution

    def parent(logits, controls):
        return weighted_sampling_distribution(logits, controls)

    def candidate(logits, controls):
        return weighted_sampling_distribution(logits, controls)

    functions = [torch.compile(f, backend="hpu_backend", fullgraph=True, dynamic=False)
                 for f in (parent, candidate)]
    rows = []
    for case in range(3):
        shards = [torch.load(args.fixtures / f"fixed-prefix-case{case}-rank{rank}.pt",
                             map_location="cpu", weights_only=True) for rank in range(4)]
        if len({s["prefix_sha256"] for s in shards}) != 1:
            raise ValueError("Only common real prefixes qualify")
        q = torch.cat([s["draft_probabilities"] for s in shards], -1).contiguous()
        controls = torch.tensor([[1., .95, .13 + .17 * i, 0.] for i in range(5)], dtype=torch.float32)
        actual = compiled(q.to("hpu"), controls.to("hpu")).cpu()
        cdf = q.double().cumsum(-1)
        desired = controls[:, 2:3].double() * cdf[:, -1:]
        expected = (cdf < desired).sum(-1).int()
        if not torch.equal(actual[:, 0], expected):
            raise AssertionError((case, actual.tolist(), expected.tolist()))
        if not actual[:, 1].bool().all():
            raise AssertionError("These actual draws are away from CDF boundaries")
        rows.append(dict(case=case, prefix_sha256=shards[0]["prefix_sha256"],
                         token_ids=actual[:, 0].tolist(), oracle="FP64 vocabulary-order CDF", exact=True))
        teachers = [torch.load(args.teacher_logits / f"mtp-prefix-case{case}-rank{rank}.pt",
                               map_location="cpu", weights_only=True) for rank in range(4)]
        if any(d["prefix_sha256"] != shards[0]["prefix_sha256"] for d in teachers):
            raise ValueError("The teacher Target and draft prefixes differ")
        scores = torch.cat([d["reference_draft_logits"] for d in teachers], -1).contiguous()
        probabilities = []
        for arm, fn in enumerate(functions):
            os.environ["VLLM_HPU_DSV41_DSPARK_DRAFT_VOCAB_CDF"] = str(arm)
            result = fn(scores.to("hpu"), controls.to("hpu"))
            probabilities.append(result[1].cpu())
        if not torch.equal(*probabilities):
            raise AssertionError("Changing the draw must not change the fixed-prefix q")
        p = distribution(torch.cat([s["reference_target_logits"] for s in shards], -1), 1., .95)
        alpha = [torch.minimum(p, q.double()).sum(-1) for q in probabilities]
        rows[-1].update(teacher_forced_q_exact=True, alpha_reference=alpha[0].tolist(),
                        alpha_candidate=alpha[1].tolist(), alpha_delta=(alpha[1] - alpha[0]).tolist())
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    (root / "probability-draw.json").write_text(json.dumps(dict(
        status="passed", cases=rows, probability_construction_unchanged=True,
        teacher_forced=True, mean_alpha_delta=0., performance_measured=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
