# SPDX-License-Identifier: Apache-2.0
"""Check prepared wo_a storage against the existing ordered BF16 MME path."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from vllm_gaudi.ops.deepseek_v41_attention import CSA2Attention  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402


def reference(value, weight):
    return torch.einsum("tgd,grd->tgr", value, weight.reshape(4, 1024, 4096)).flatten(1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    args = parser.parse_args()
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    bind_worker_cpu(0)
    obj = CSA2Attention.__new__(CSA2Attention)
    torch.nn.Module.__init__(obj)
    obj.heads, obj.groups, obj.prepared_output = 32, 4, True
    obj.weights = torch.nn.Module()
    obj.weights.wo_a = torch.nn.Module()
    obj.weights.wo_a.register_buffer("weight", torch.empty(0))
    original = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    candidate = torch.compile(obj.project_output, backend="hpu_backend", fullgraph=True, dynamic=False)
    records = []
    torch.manual_seed(41)
    for pp, tp, layer in ((0, 0, 2), (1, 1, 20)):
        shard = PreparedV41Shard(args.prepared, pp, tp)
        name = f"layers.{layer}.attn.wo_a.weight"
        weight = shard.dense(name, "hpu")
        obj.weights.wo_a.weight = weight
        obj.prepare_output_weight()
        prepared = obj.weights.wo_a.weight
        assert prepared.shape == (4, 4096, 1024) and prepared.is_contiguous()
        assert torch.equal(prepared.cpu().transpose(1, 2).contiguous().view(torch.int16),
                           weight.cpu().reshape(4, 1024, 4096).view(torch.int16))
        for count in (1, 2, 6, 1):
            value = torch.randn(count, 4, 4096).bfloat16().to("hpu")
            expected = original(value, weight).cpu()
            actual = candidate(value).cpu()
            assert torch.equal(expected.view(torch.int16), actual.view(torch.int16)), (pp, tp, count)
            records.append({"pp": pp, "tp": tp, "layer": layer, "tokens": count,
                            "bitwise_equal": True,
                            "output_sha256": hashlib.sha256(actual.view(torch.int16).numpy().tobytes()).hexdigest()})
    (evidence / "result.json").write_text(json.dumps({
        "status": "component_pass", "cases": records, "model_performance_claim": False,
        "precision": "BF16 MME; full K preserved; preparation permutes storage bits only",
        "peak_device_bytes": torch.hpu.max_memory_allocated(),
    }, indent=2) + "\n")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
