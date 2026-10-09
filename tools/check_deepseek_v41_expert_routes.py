# SPDX-License-Identifier: Apache-2.0
"""Device descriptor correctness only; does not qualify conditional expert GEMM."""
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401

from vllm_gaudi.ops.deepseek_v41_expert_routes import pack_expert_routes
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
from vllm_gaudi.distributed import tp2_fused_ar_norm  # noqa: F401


def main():
    bind_worker_cpu(0)
    cases = []
    compiled = torch.compile(pack_expert_routes, backend="hpu_backend", fullgraph=True, dynamic=False)
    for rows in (1, 2, 6):
        for shared in (False, True):
            for changed in (False, True):
                ids = torch.arange(rows * 6).reshape(rows, 6)
                if shared:
                    ids %= 6
                if changed:
                    ids = 383 - ids.flip((0, 1))
                ids = ids.to(torch.int32)
                expected = pack_expert_routes(ids)
                actual = tuple(value.cpu() for value in compiled(ids.to("hpu")))
                bind_worker_helpers(0)
                exact = all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
                cases.append(dict(rows=rows, shared=shared, changed=changed, byte_exact=exact))
                assert exact, cases[-1]
    Path(os.environ["DSV41_RUN_EVIDENCE"], "DEVICE_ROUTE_DESCRIPTOR_CONTRACT.json").write_text(
        json.dumps(dict(status="passed", cases=cases, timing_measured=False,
                        conditional_mme_dispatch_qualified=False, performance_gain_qualified=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
