# SPDX-License-Identifier: Apache-2.0
"""Check shared TP head reduction and selector -> two MLA consumers."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-reference", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace("{rank}", "0")
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from habana_frameworks.torch.hpu.metrics import metric_global
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4
    from vllm_gaudi.ops.deepseek_v41_indexer import _bounded_mme_scores, runtime_index_select
    torch.hpu.set_device(0)
    bind_worker_cpu(0); bind_worker_helpers(0)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    ops = torch.ops.custom_op
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    reference = json.loads((args.reuse_reference / "RESULT.json").read_text())
    report = dict(status="running", reference=str(args.reuse_reference),
                  reference_ms=reference["timings"]["current"]["host_ms"], checks=[],
                  completion_boundary="selector -> publish MLA -> reuse MLA -> next query feedback; device drain",
                  formal_target_met=False)

    def save():
        (root / "RESULT.json").write_text(json.dumps(report, indent=2)+"\n")

    def head_sum(raw, weights, positions, ratio, local_heads):
        total = torch.zeros(raw.shape[0], raw.shape[2])
        for first in range(0, 32, local_heads):
            partial = torch.zeros_like(total)
            for h in range(first, first + local_heads):
                partial += (raw[:, h].relu() * weights[:, h, None]).bfloat16().float()
            total += partial.bfloat16().float()
        total = total.bfloat16().float()
        return total.masked_fill(torch.arange(raw.shape[-1])[None] >= ((positions+1)//ratio)[:, None], -torch.inf)

    def ordered(scores, pos, ratio):
        result = torch.full((scores.shape[0], 512), -1, dtype=torch.int32)
        for b in range(scores.shape[0]):
            count = (int(pos[b])+1)//ratio
            offsets = scores[b, :count].argsort(descending=True, stable=True)[:512]
            offsets = offsets[scores[b, offsets] > -torch.inf].sort().values
            result[b, :offsets.numel()] = offsets.int()
        return result

    save()
    with torch.inference_mode():
        gen = torch.Generator().manual_seed(3001)
        for local_heads in (16, 8):
            for batch in (1, 2):
                raw = (torch.randn(batch, 32, 1280, generator=gen)*.3).bfloat16()
                gains = (torch.randn(batch, 32, generator=gen)*.05).bfloat16()
                positions = torch.tensor([1299 + b for b in range(batch)], dtype=torch.int32)
                def reduce(raw, gains, positions, local_heads=local_heads):
                    return ops.custom_deepseek_v41_index_reduce_bf16_gaudi2(raw, gains, positions, 2, local_heads)
                call = torch.compile(reduce, backend="hpu_backend", fullgraph=True, dynamic=False)
                actual = call(raw.to("hpu"), gains.to("hpu"), positions.to("hpu")).cpu()
                expected = head_sum(raw, gains, positions, 2, local_heads)
                assert torch.equal(actual, expected), (local_heads, batch, "head sum")
                report["checks"].append(dict(local_heads=local_heads, batch=batch, head_sum_exact=True))
                save()
        # Recreate the preserved74 fixture's RNG state; do not remeasure B.
        gen = torch.Generator().manual_seed(7301)
        for count in (10240, 40960, 15360, 16384):
            torch.randn(count, generator=gen)
        ratio, columns, position = 2, 10240, 16514
        seed_cpu = (torch.randn(1, 16, 512, generator=gen)/32).bfloat16()
        keys_cpu = unpack_fp4(pack_fp4(torch.randn(columns, 128, generator=gen).bfloat16(), 32), 128, 32)
        weights_cpu = (torch.randn(1, 32, generator=gen)*.02).bfloat16()
        main_cpu = torch.randint(0, 256, (columns, 288), generator=gen, dtype=torch.uint8)
        main_cpu[:, 256:] = 120
        rings = [torch.randint(0, 127, (256, 528), generator=gen, dtype=torch.uint8) for _ in range(2)]
        for ring in rings:
            ring[:, 512:] = 119
        pages = torch.full((8192,), -1, dtype=torch.int32)
        pages[:columns//64] = torch.arange(columns//64, dtype=torch.int32)
        fixed = tuple(t.to("hpu") for t in (
            keys_cpu, weights_cpu, main_cpu, *rings, pages,
            torch.tensor([position], dtype=torch.int32), torch.zeros(1, 2048, dtype=torch.int32),
            torch.zeros(16), torch.tensor([512**-.5]), torch.tensor([640], dtype=torch.int32)))

        def chain(query, keys, weights, main, swa0, swa1, pages, pos, pool, sink, scale, lengths):
            index_query = query.flatten(1)[:, :4096].reshape(1, 32, 128).contiguous()
            ids, _ = runtime_index_select(
                index_query, weights, main, pages, pos, pool, ratio=ratio, capacity=524288,
                local_heads=8, search_rows=columns, decoded_keys=keys)
            first, decoded, mask = ops.custom_deepseek_v41_main_publish_mla_gaudi2(
                query, swa0, main, ids, pos, pages, sink, scale, lengths, ratio)
            return ops.custom_deepseek_v41_main_reuse_mla_gaudi2(
                first, swa1, decoded, mask, pos, sink, scale, lengths), ids

        seed = seed_cpu.to("hpu")
        call = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)
        def scores(query, keys, weights, main, swa0, swa1, pages, pos, pool, sink, scale, lengths):
            q = query.flatten(1)[:, :4096].reshape(1, 32, 128).contiguous()
            return _bounded_mme_scores(q, weights, main, pages, pos, pool, ratio, 8, columns, False, keys)
        score_call = torch.compile(scores, backend="hpu_backend", fullgraph=True, dynamic=False)
        expected_ids = ordered(score_call(seed, *fixed).cpu(), fixed[6].cpu(), ratio)
        output, ids = call(seed, *fixed)
        assert torch.equal(ids.cpu(), expected_ids), "TP2 ordered protocol"
        report["checks"].append(dict(tp2_ordered_protocol_exact=True))
        # Use independently selected IDs through the real consumers.
        def consume(query, ids, main, swa0, swa1, pages, pos, sink, scale, lengths):
            first, decoded, mask = ops.custom_deepseek_v41_main_publish_mla_gaudi2(
                query, swa0, main, ids, pos, pages, sink, scale, lengths, ratio)
            return ops.custom_deepseek_v41_main_reuse_mla_gaudi2(
                first, swa1, decoded, mask, pos, sink, scale, lengths)
        consumer = torch.compile(consume, backend="hpu_backend", fullgraph=True, dynamic=False)
        expected_output = consumer(seed, expected_ids.to("hpu"), fixed[2], fixed[3], fixed[4], fixed[5],
                                   fixed[6], fixed[8], fixed[9], fixed[10])
        assert torch.equal(output.cpu(), expected_output.cpu()), "shared downstream"
        for _ in range(32):
            output = call(output, *fixed)[0]
        torch.hpu.synchronize()
        compile_metric, fallback_metric = metric_global("graph_compilation"), metric_global("cpu_fallback")
        before = (dict(compile_metric.stats())["TotalNumber"], dict(fallback_metric.stats())["TotalNumber"])
        start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
        start.record(); begin = time.perf_counter_ns()
        for _ in range(64):
            output = call(output, *fixed)[0]
        end.record(); end.synchronize(); torch.hpu.synchronize()
        elapsed = (time.perf_counter_ns()-begin)/1e6/64
        after = (dict(compile_metric.stats())["TotalNumber"], dict(fallback_metric.stats())["TotalNumber"])
        assert before == after and torch.isfinite(output).all().item()
        report.update(status="small_chain_measured_16layer_pending", host_ms=elapsed,
                      device_ms=start.elapsed_time(end)/64, warm_steps=32, timed_steps=64,
                      no_hot_compilation_or_fallback=True, downstream_exact=True)
        torch.save(output.cpu(), root / "feedback-output.pt")
        save(); print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
