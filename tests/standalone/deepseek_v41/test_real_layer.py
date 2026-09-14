# SPDX-License-Identifier: Apache-2.0
"""Local-TP layer execution before starting the complete four-rank model."""

import json
import os
from pathlib import Path

import pytest

from vllm_gaudi.entrypoints.deepseek_v4 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401

from vllm_gaudi.models.deepseek_v41_program import (  # noqa: E402
    PreparedDecoderLayer, _weight_tree, load_weight_tree,
)
from vllm_gaudi.ops.deepseek_v41_attention import CSA2SharedState  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
torch.ops.load_library(str(next((ROOT / "vllm_gaudi/lib").glob("hpu_dsv4_sparse_attn_pt2*.so"))))
pytestmark = pytest.mark.skipif(os.getenv("DSV41_TEST_HPU") != "1" or not os.getenv("DSV41_PREPARED_TEST_PATH"),
                                reason="Requires an explicit HPU lease and verified rank-local checkpoint")


def test_real_full_csa2_layer_consumes_positions_and_all_resident_experts():
    directory = Path(os.environ["DSV41_PREPARED_TEST_PATH"])
    config = json.loads((directory / "config.json").read_text())["text_config"]
    shard = PreparedV41Shard(directory, 0, 0)
    specs = {key: value for key, value in shard.specs.items() if key.startswith("layers.2.")}
    weights = _weight_tree(specs)
    load_weight_tree(shard, weights, "hpu", specs)
    assert weights.layers.get_submodule("2").ffn.experts.w13_q16.shape[0] == 384
    shared = CSA2SharedState(config, 0, 20, "hpu")
    layer = PreparedDecoderLayer(weights.layers.get_submodule("2"), config, 2, shared,
                                 shard.manifest["normal_scales"]["layers.2.ffn.experts"][0],
                                 mxfp4_bf16_lut(torch.device("hpu")), lambda value: value,
                                 lambda value, dim: value, "hpu")
    # No distributed reduction is claimed in this isolated local-TP check.
    compiled = torch.compile(layer, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(41)
    value = torch.randn(6, 4, 5120).bfloat16().to("hpu")
    pre = torch.tensor([[1., 0., 0., 0.]] * 6, device="hpu")
    mask = torch.zeros(6, dtype=torch.bool, device="hpu")
    outputs = []
    for start in (0, 6):
        positions = torch.arange(start, start + 6, dtype=torch.int32, device="hpu")
        output, next_pre, aux = compiled(value, pre, positions, mask)
        output, next_pre = output.cpu(), next_pre.cpu()
        assert aux is None and output.shape == (6, 4, 5120) and next_pre.shape == (6, 4)
        assert torch.isfinite(output).all() and torch.isfinite(next_pre).all()
        cache = layer.attention.swa.cpu()
        assert cache[start:start + 6].count_nonzero() > 0
        selection = shared.topk["2"].indices.cpu()[start:start + 6]
        for row, position in enumerate(range(start, start + 6)):
            count = (position + 1) // 2
            assert selection[row, :count].tolist() == list(range(count))
            assert (selection[row, count:] == -1).all()
        outputs.append(output)
    assert not torch.equal(*outputs)
    evidence = Path(os.environ["HABANA_LOGS"]).parent
    torch.save({"outputs": outputs, "pre": next_pre}, evidence / "real-layer-output.pt")
    (evidence / "weight-loading.json").write_text(json.dumps({
        "prepared_rank_sha256": shard.manifest["rank_files"]["pp0-tp0"]["sha256"],
        "layer": 2, "resident_experts": 384, "tokens": 6, "tp_reduction": "identity, local check only",
        "host_temporary_bytes": shard.max_host_chunk_bytes,
        "device_max_allocated_bytes": torch.hpu.max_memory_allocated(),
    }, indent=2) + "\n")
