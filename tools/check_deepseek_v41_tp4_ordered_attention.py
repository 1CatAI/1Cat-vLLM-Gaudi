# SPDX-License-Identifier: Apache-2.0
"""Quick TP2 selector port gate through TP4 scoring and two MLA consumers.

This is a one-card component decision gate with synthetic operands at actual
decode shapes. It is not a model performance result or gain-ledger entry.
The historical TP2 cutoff rule is checked independently; equality with the
current generic TopK is reported, never assumed for tied scores.
"""

import json
import os
from pathlib import Path
import time


def main():
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    if "{rank}" in os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", ""):
        os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].replace("{rank}", "0")
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from habana_frameworks.torch.hpu.metrics import metric_global
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4
    from vllm_gaudi.ops.deepseek_v41_ordered_selection import ordered_index_ids
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention

    torch.hpu.set_device(0)
    bind_worker_cpu(0)
    bind_worker_helpers(0)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    ops = torch.ops.custom_op
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(status="running", scope=__doc__, checks=[], timings={}, gain_ledger_credit=False,
                  new_full_model_requests=0, new_traces=0,
                  comparison_reason=("No archived timing covers this score-to-two-MLA-consumer fixture; "
                                     "measure its missing local reference once."))

    def save():
        (root / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")

    def cpu_ordered(scores, positions, candidates, ratio, reindex=False, blocks=False):
        result = torch.full((scores.shape[0], 2048 if blocks else 512), -1, dtype=torch.int32)
        for batch in range(scores.shape[0]):
            visible = (int(positions[batch]) + 1) // ratio
            limit = ((visible + 7) // 8 if blocks else
                     min((visible + 7) // 8, 2048) * 8 if reindex else visible)
            values = scores[batch, :limit]
            selected = torch.argsort(values, descending=True, stable=True)[:result.shape[1]]
            selected = selected[values[selected] > -torch.inf].sort().values
            if reindex and not blocks:
                selected = candidates[batch, selected // 8] * 8 + selected % 8
            result[batch, :selected.numel()] = selected.int()
        return result

    save()
    with torch.inference_mode():
        # First check the changed tie contract, full publication and reindex
        # mapping without loading weights or running a long model request.
        gen = torch.Generator().manual_seed(7301)
        for batch, ratio, columns, visible, reindex, blocks in (
                (1, 2, 10240, 8257, False, False),
                (2, 1, 20480, 16515, False, False),
                (6, 1, 2560, 16515, False, True),
                (1, 1, 16384, 16515, True, False)):
            candidates = torch.arange(2048, dtype=torch.int32).expand(batch, -1).contiguous()
            if reindex:
                candidates[:, -1] = (visible - 1) // 8
            pos = torch.full((batch,), visible * ratio - 1, dtype=torch.int32)

            def select(scores, positions, pool, ratio=ratio, reindex=reindex, blocks=blocks):
                return ordered_index_ids(scores, positions, pool, ratio, reindex=reindex, blocks=blocks)

            call = torch.compile(select, backend="hpu_backend", fullgraph=True, dynamic=False)
            for tied in (False, True):
                scores = (torch.zeros(batch, columns) if tied else
                          torch.randn(batch, columns, generator=gen).bfloat16().float())
                if not reindex:
                    limit = (visible + 7) // 8 if blocks else visible
                    scores[:, limit:] = -torch.inf
                else:
                    logical = (candidates[..., None] * 8 + torch.arange(8)).flatten(1)
                    scores.masked_fill_(logical >= visible, -torch.inf)
                if blocks:
                    scores[:, (visible - 1) // 8] = torch.inf
                expected = cpu_ordered(scores, pos, candidates, ratio, reindex, blocks)
                actual = call(scores.to("hpu"), pos.to("hpu"), candidates.to("hpu")).cpu()
                assert torch.equal(actual, expected), (batch, ratio, reindex, blocks, tied)
                report["checks"].append(dict(batch=batch, ratio=ratio, reindex=reindex, blocks=blocks,
                                             tied=tied, tp2_ordered_ids_exact=True))
                save()

        ratio, columns, position = 2, 10240, 16514
        q_cpu = (torch.randn(1, 16, 512, generator=gen) / 32).bfloat16()
        keys_cpu = unpack_fp4(pack_fp4(torch.randn(columns, 128, generator=gen).bfloat16(), 32), 128, 32)
        weights_cpu = (torch.randn(1, 32, generator=gen) * .02).bfloat16()
        main_cpu = torch.randint(0, 256, (columns, 288), generator=gen, dtype=torch.uint8)
        main_cpu[:, 256:] = 120
        rings = [torch.randint(0, 127, (256, 528), generator=gen, dtype=torch.uint8) for _ in range(2)]
        for ring in rings:
            ring[:, 512:] = 119
        pages_cpu = torch.full((8192,), -1, dtype=torch.int32)
        pages_cpu[:columns // 64] = torch.arange(columns // 64, dtype=torch.int32)
        seed = q_cpu.to("hpu")
        fixed = tuple(t.to("hpu") for t in (
            keys_cpu, weights_cpu, main_cpu, *rings, pages_cpu,
            torch.tensor([position], dtype=torch.int32), torch.arange(columns, dtype=torch.int32),
            torch.zeros(1, 2048, dtype=torch.int32), torch.zeros(16),
            torch.tensor([512**-.5]), torch.tensor([640], dtype=torch.int32)))

        class Chain(torch.nn.Module):
            def __init__(self, ordered):
                super().__init__()
                self.ordered = ordered

            def forward(self, query, keys, weights, main, swa0, swa1, pages, positions, rows, pool,
                        sink, scale, lengths):
                index_query = query.flatten(1)[:, :4096].reshape(1, 32, 128).contiguous()
                scores = ops.custom_deepseek_v41_prefill_index_scores_gaudi2(
                    index_query, weights, keys, positions, rows, ratio, 8)
                if self.ordered:
                    ids = ordered_index_ids(scores, positions, pool, ratio)
                else:
                    values = ids = None
                    for start in range(0, columns, 2048):
                        values, ids = PagedCSA2Attention._merge_topk(
                            values, ids, scores[:, start:start + 2048], rows[start:start + 2048], 512)
                    ids = torch.where(ids < ((positions + 1) // ratio)[:, None], ids, 1048576).sort(-1).values
                    ids = torch.where(ids < 1048576, ids, -1).int()
                first, decoded, mask = ops.custom_deepseek_v41_main_publish_mla_gaudi2(
                    query, swa0, main, ids, positions, pages, sink, scale, lengths, ratio)
                output = ops.custom_deepseek_v41_main_reuse_mla_gaudi2(
                    first, swa1, decoded, mask, positions, sink, scale, lengths)
                return output, ids, scores

        functions = [torch.compile(Chain(flag), backend="hpu_backend", fullgraph=True, dynamic=False)
                     for flag in (False, True)]
        old, new = [fn(seed, *fixed) for fn in functions]
        expected_ids = cpu_ordered(new[2].cpu(), fixed[6].cpu(), fixed[8].cpu(), ratio)
        assert torch.equal(new[1].cpu(), expected_ids)
        assert torch.equal(old[2].cpu(), new[2].cpu()), "TP4 scoring changed"
        report["first_step"] = dict(scores_exact=True, tp2_ordered_ids_exact=True,
                                    current_ids_equal=torch.equal(old[1].cpu(), new[1].cpu()),
                                    current_output_equal=torch.equal(old[0].cpu(), new[0].cpu()),
                                    current_output_max_abs=float((old[0].float() - new[0].float()).abs().max().cpu()))
        # Verify the actual next consumer using the independently selected IDs.
        def consume(query, ids, main, swa0, swa1, pages, pos, sink, scale, lengths):
            first, decoded, mask = ops.custom_deepseek_v41_main_publish_mla_gaudi2(
                query, swa0, main, ids, pos, pages, sink, scale, lengths, ratio)
            return ops.custom_deepseek_v41_main_reuse_mla_gaudi2(
                first, swa1, decoded, mask, pos, sink, scale, lengths)
        consumer = torch.compile(consume, backend="hpu_backend", fullgraph=True, dynamic=False)
        expected_output = consumer(seed, expected_ids.to("hpu"), fixed[2], fixed[3], fixed[4], fixed[5],
                                   fixed[6], fixed[9], fixed[10], fixed[11])
        assert torch.equal(new[0].cpu(), expected_output.cpu()), "Ordered selection downstream mismatch"
        report["first_step"]["ordered_downstream_exact"] = True
        save()

        compile_metric = metric_global("graph_compilation")
        fallback_metric = metric_global("cpu_fallback")
        for name, fn in zip(("current", "tp2_ordered"), functions, strict=True):
            output = seed.clone()
            for _ in range(32):
                output = fn(output, *fixed)[0]
            torch.hpu.synchronize()
            before = (dict(compile_metric.stats())["TotalNumber"], dict(fallback_metric.stats())["TotalNumber"])
            start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
            start.record()
            begin = time.perf_counter_ns()
            for _ in range(64):
                output = fn(output, *fixed)[0]
            end.record()
            end.synchronize()
            torch.hpu.synchronize()
            elapsed = (time.perf_counter_ns() - begin) / 1e6 / 64
            after = (dict(compile_metric.stats())["TotalNumber"], dict(fallback_metric.stats())["TotalNumber"])
            assert before == after, (before, after)
            assert torch.isfinite(output).all().item()
            torch.save(output.cpu(), root / f"{name}-feedback-output.pt")
            report["timings"][name] = dict(host_ms=elapsed, device_ms=start.elapsed_time(end) / 64,
                                            warm_steps=32, timed_steps=64, no_hot_compile_or_fallback=True)
            save()
        report["status"] = "ordered_contract_and_local_score_mla_chain_measured_model_gate_pending"
        save()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
