# SPDX-License-Identifier: Apache-2.0
"""Isolate C6/C1 draft state transitions from PP and prepared target capture."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_gaudi.entrypoints.deepseek_v4 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401

from vllm_gaudi.models.deepseek_v41_program import PreparedDraft, _weight_tree, load_weight_tree  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_attention import CSA2SharedState  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(os.getenv("DSV41_TEST_HPU") != "1" or not os.getenv("DSV41_PREPARED_TEST_PATH"),
                                reason="Requires an explicit HPU lease and frozen prepared checkpoint")


@torch.inference_mode()
def test_local_draft_context_changes_complete_host_sampling():
    torch.ops.load_library(str(next((ROOT / "vllm_gaudi/lib").glob("hpu_dsv4_sparse_attn_pt2*.so"))))
    directory = Path(os.environ["DSV41_PREPARED_TEST_PATH"])
    config = json.loads((directory / "config.json").read_text())
    shard = PreparedV41Shard(directory, 1, 0)
    specs = {name: spec for name, spec in shard.specs.items() if name.startswith("mtp.") or name == "head.weight"}
    tree = _weight_tree(specs)
    load_weight_tree(shard, tree, "hpu", specs)
    stage = SimpleNamespace(weights=tree, config=config, tp_rank=0, shard=shard,
                            shared=CSA2SharedState(config["text_config"], 40, 43, "hpu"),
                            reduce=lambda value: value, all_gather=lambda value, dim: value)
    draft = PreparedDraft(stage, mxfp4_bf16_lut(torch.device("hpu")), "hpu")
    insert = torch.compile(draft.insert_context, backend="hpu_backend", fullgraph=True, dynamic=False)
    forward = torch.compile(draft, backend="hpu_backend", fullgraph=True, dynamic=False)
    sample = torch.compile(draft.sample_greedy, backend="hpu_backend", fullgraph=True, dynamic=False)
    outputs = []
    for step, count in enumerate((6, 1, 1, 6)):
        print(f"STEP {step} C{count} context", flush=True)
        values = torch.randn(count, 15360, device="cpu", dtype=torch.bfloat16).to("hpu")
        positions = torch.arange(count, device="hpu", dtype=torch.int32)
        insert(values, positions)
        first = torch.tensor([step + 1], device="hpu", dtype=torch.int64)
        draft_positions = torch.arange(count, count + 5, device="hpu", dtype=torch.int32)
        hidden, logits = forward(first, draft_positions)
        print(f"STEP {step} sampling", flush=True)
        tokens, confidence = sample(first, hidden, logits)
        tokens, confidence = tokens.cpu(), confidence.cpu()
        assert tokens.shape == (5,) and (tokens >= 0).all() and (tokens < 64640).all()
        assert torch.isfinite(confidence).all()
        outputs.append({"context_count": count, "tokens": tokens.tolist()})
        print(f"STEP {step} complete", flush=True)
    evidence = Path(os.environ["HABANA_LOGS"]).parent
    (evidence / "draft-pipeline.json").write_text(json.dumps({"tp": "local identity diagnostic only",
        "outputs": outputs, "peak_device_bytes": torch.hpu.max_memory_allocated()}, indent=2) + "\n")
