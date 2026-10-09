# SPDX-License-Identifier: Apache-2.0
"""Cold correctness diagnosis only; never a performance qualification."""

import json
import os
from pathlib import Path

os.environ["VLLM_HPU_DSV41_DSPARK_NUCLEUS_MASS"] = "1"
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment, load_native_operators

prepare_environment(
    "/opt/optane/prepared/DeepSeek-V4.1-Flash-dba1be0-tp4-pp1-q16v2", tensor_parallel_size=4, pipeline_parallel_size=1
)
import habana_frameworks.torch.core  # noqa: F401,E402
import torch  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_speculative_sampling import sample_full_distribution  # noqa: E402

load_native_operators()
torch.hpu.set_device(0)
root = Path("/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving")
sampler = torch.compile(
    torch.ops.custom_op.custom_deepseek_v41_nucleus_mass_sample_gaudi2,
    backend="hpu_backend",
    fullgraph=True,
    dynamic=False,
)
reports = []
with torch.inference_mode():
    for case in range(3):
        data = [
            torch.load(root / "sampled-native-bf16-head-68" / f"producer-case{case}-rank{r}.pt", weights_only=True)
            for r in range(4)
        ]
        logits = torch.cat([d["reference"][2] for d in data], -1)
        controls = data[0]["controls"]
        expected_token, expected = sample_full_distribution(logits, controls)
        probability, token, certificate = (x.cpu() for x in sampler(logits.to("hpu"), controls.to("hpu")))
        torch.save(
            dict(
                logits=logits,
                controls=controls,
                reference=expected,
                probability=probability,
                expected_token=expected_token,
                token=token,
                certificate=certificate,
            ),
            Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"case{case}.pt",
        )
        reports.append(
            dict(
                case=case,
                token=token.tolist(),
                expected_token=expected_token.tolist(),
                certificate=certificate.tolist(),
                max_abs=float((probability - expected).abs().max()),
                actual_nonzero=(probability > 0).sum(-1).tolist(),
                reference_nonzero=(expected > 0).sum(-1).tolist(),
                actual_sum=probability.sum(-1).tolist(),
            )
        )
(Path(os.environ["DSV41_RUN_EVIDENCE"]) / "values.json").write_text(json.dumps(reports, indent=2) + "\n")
print(json.dumps(reports), flush=True)
