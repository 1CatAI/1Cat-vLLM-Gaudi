# SPDX-License-Identifier: Apache-2.0
"""Recipe debug caches share a basename; archived graph identities must not."""
import hashlib
import importlib
import json
from pathlib import Path


def test_serialized_cold_graphs_keep_distinct_paths_and_matching_hashes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    module = importlib.import_module("restore_deepseek_v41_cached_trace_metadata")
    run, analysis = tmp_path / "run", tmp_path / "analysis"
    rank = analysis / "rank0"
    rank.mkdir(parents=True)
    (analysis / "collection.json").write_text(json.dumps(dict(ranks=[dict(rank=0, pid=42)])))
    nodes = []
    for rid in (1, 2):
        nodes.append(dict(recipe=f"{rid}@:", raw_context_id="0", engine="TPC"))
        cold = run / "recipes" / "rank0" / f"{rid}.recipe_debug_files" / "graph.post.json"
        cold.parent.mkdir(parents=True)
        node = dict(engine="TPC",
                    context_id=0,
                    name=f"node{rid}",
                    guid=f"kernel{rid}",
                    id=rid,
                    input_tensors=[],
                    output_tensors=[])
        graph = dict(recipe_debug_id=rid, name=f"cold{rid}", tensors=[], nodes=[node])
        cold.write_text(json.dumps(dict(graphs=[graph])))
    (rank / "inventory.json").write_text(json.dumps(dict(format="raw_hltv_cpu", hardware_events=2, nodes=nodes)))
    (rank / "recipe-symbols.json").write_text(json.dumps(dict(recipes=[])))
    (rank / "node-contracts.json").write_text("[]")
    (rank / "raw-graph-manifest.json").write_text("[]")
    module.restore(run, analysis, None)
    graphs = json.loads((rank / "raw-graph-manifest.json").read_text())
    assert len({record["path"] for record in graphs}) == 2
    for record in graphs:
        assert hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == record["sha256"]
    restored = json.loads((rank / "inventory.json").read_text())
    assert [node["kernel"] for node in restored["nodes"]] == ["kernel1", "kernel2"]
