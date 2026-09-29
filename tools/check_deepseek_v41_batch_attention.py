# SPDX-License-Identifier: Apache-2.0
"""Real rank-local Attention: independent C1 calls versus slot-addressed batch.

Includes input projections, compressor/history writes, index selection, packed
KV, MLA and output projections. TP is an identity/synthetic second-head shard;
this is a single-rank component contract, not a TP2/PP2 throughput result.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import FunctionType

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prepared", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, choices=(2, 4, 8, 16, 32, 64), default=8)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--reference-timing", action="store_true", help="Only for an unmeasured component reference")
    a = p.parse_args()
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi import envs
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree, linear
    from vllm_gaudi.ops.deepseek_v41_batch_state import BatchLayerState
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar, precision_config
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention, PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar, layer_selection

    torch.set_num_threads(1)
    torch.manual_seed(4912)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    shard = PreparedV41Shard(a.prepared, 0, 0)
    config = json.loads((a.prepared / "config.json").read_text())["text_config"]
    specs = {k: v for k, v in shard.specs.items() if k.startswith("layers.2.attn.")}
    weights = _weight_tree(specs)
    dc = precision_config(envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG) if envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8 else None
    wc = layer_selection(envs.VLLM_HPU_DSV41_WO_A_FP8_CONFIG) if envs.VLLM_HPU_DSV41_WO_A_FP8 else {"layers": []}
    ds = DenseFP8Sidecar(envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR, shard) if dc else None
    ws = WoaFP8Sidecar(envs.VLLM_HPU_DSV41_WO_A_FP8_SIDECAR, shard) if wc["layers"] else None
    load_weight_tree(shard,
                     weights,
                     "hpu",
                     specs,
                     woa_sidecar=ws,
                     woa_layers=wc["layers"],
                     dense_sidecar=ds,
                     dense_config=dc)
    batch, capacity = a.batch, 64
    shared = PagedCSA2SharedState(config, 2, 3, "hpu", 1048576)
    cache = shared.sources["2"]
    cache.main = torch.zeros((capacity * 8 + 1) * 64, 288, dtype=torch.uint8, device="hpu")
    cache.index = torch.zeros((capacity * 8 + 1) * 64, 68, dtype=torch.uint8, device="hpu")
    reduce = lambda x, **kwargs: x
    gather = lambda x, dim: torch.cat((x, x), dim)
    attention = PagedCSA2Attention(
        weights.layers.get_submodule("2").attn, config, 2, shared, linear, reduce, gather, "hpu")
    attention.woa_fp8 = 2 in wc["layers"]
    attention.prepare_output_weight()
    attention.prepare_qkv_input_weight()
    attention.prepare_compressor_input_weight()
    attention.set_search_length(1048576)
    attention.batch_state = BatchLayerState(capacity, "hpu", compressor=True)
    states = attention.batch_state
    slots_host = torch.randperm(capacity)[:batch].int()
    pos_host = torch.tensor([(126, 127, 254, 255, 510, 511, 766, 999)[i % 8] for i in range(batch)], dtype=torch.int32)
    pages_host = torch.zeros(batch, 8192, dtype=torch.int32)
    for i, slot in enumerate(slots_host):
        pages_host[i, :8] = torch.arange(8) + 1 + int(slot) * 8
    pages, positions, slots = [x.to("hpu") for x in (pages_host, pos_host, slots_host)]
    value = torch.randn(batch, 5120).bfloat16().to("hpu")
    selected = torch.full((batch, 512), -1, dtype=torch.int32, device="hpu")
    candidates = torch.full((batch, 2048), -1, dtype=torch.int32, device="hpu")
    done = torch.zeros(batch, dtype=torch.int32, device="hpu")
    # Random finite canonical prior cache/history exercises actual reads.
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa
    states.swa.copy_(pack_swa(torch.randn(capacity * 256, 512).bfloat16()).to("hpu"))
    states.kv_history.copy_(torch.randn(capacity * 8, 512))
    states.score_history.copy_(torch.randn(capacity * 8, 512))
    cache.main.copy_(pack_fp4(torch.randn(cache.main.shape[0], 512).bfloat16(), 16).to("hpu"))
    cache.index.copy_(pack_fp4(torch.randn(cache.index.shape[0], 128).bfloat16(), 32).to("hpu"))
    buffers = (states.swa, states.kv_history, states.score_history, cache.main, cache.index)
    original = [x.clone() for x in buffers]

    def restore():
        for dest, source in zip(buffers, original):
            dest.copy_(source)

    def candidate(x, pos, owner, page, ids, pool, main_ready, index_ready):
        return attention.forward_batch(x, pos, owner, page, ids, pool, main_ready, index_ready)

    entry = FunctionType(candidate.__code__.replace(co_name=f"batch_attention_{batch}"),
                         candidate.__globals__,
                         closure=candidate.__closure__)
    compiled = torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)
    fixed = (value, positions, slots, pages, selected, candidates, done, done)
    references = []
    for i, slot in enumerate(slots_host.tolist()):
        # Distinct entry functions but shared immutable weights. Buffer views
        # bind each request's own history, as the old serial runner does.
        old = PagedCSA2Attention(
            weights.layers.get_submodule("2").attn, config, 2, shared, linear, reduce, gather, "hpu")
        old.woa_fp8 = attention.woa_fp8
        old.swa = states.swa[slot * 256:(slot + 1) * 256]
        old.kv_history = states.kv_history[slot * 8:(slot + 1) * 8]
        old.score_history = states.score_history[slot * 8:(slot + 1) * 8]
        old._fused_qkv_weight = attention._fused_qkv_weight
        old._fused_qkv_quantized = attention._fused_qkv_quantized
        old._fused_compressor_weight = attention._fused_compressor_weight
        old._fused_compressor_kv_width = attention._fused_compressor_kv_width
        old.set_search_length(1048576)

        def reference(x, pos, page, module=old):
            module.shared.block_table = page
            return module(x, pos, decode=True)

        ref = FunctionType(reference.__code__.replace(co_name=f"serial_attention_{i}"),
                           reference.__globals__,
                           argdefs=reference.__defaults__,
                           closure=reference.__closure__)
        references.append(torch.compile(ref, backend="hpu_backend", fullgraph=True, dynamic=False))

    def serial():
        return torch.cat([fn(value[i:i + 1], positions[i:i + 1], pages[i]) for i, fn in enumerate(references)], 0)

    report = {"batch": batch, "scope": __doc__, "checks": [], "timings": {}}
    with torch.inference_mode():
        for change in range(2):
            if change:
                value.copy_(torch.randn(batch, 5120).bfloat16())
            restore()
            reference = serial().cpu()
            ref_buffers = [x.cpu() for x in buffers]
            restore()
            actual = compiled(*fixed)[0].cpu()
            checks = dict(change=change,
                          output_exact=torch.equal(reference, actual),
                          max_abs=float((reference.float() - actual.float()).abs().max()),
                          relative_l2=float((reference.float() - actual.float()).norm() / reference.float().norm()),
                          state_exact=[
                              torch.equal(x.cpu()[64:] if j >= 3 else x.cpu(), ref[64:] if j >= 3 else ref)
                              for j, (x, ref) in enumerate(zip(buffers, ref_buffers))
                          ])
            report["checks"].append(checks)
            torch.save({
                "input": value.cpu(),
                "reference": reference,
                "actual": actual
            }, a.output.parent / f"batch-attention-output-{change}.pt")
            a.output.write_text(json.dumps(report, indent=2))
            if not all(checks["state_exact"]) or not checks["output_exact"]:
                raise RuntimeError(f"Batch Attention contract mismatch: {checks}")
        for name, fn in (("serial", serial), ("batch", lambda: compiled(*fixed)[0])):
            if name == "serial" and not a.reference_timing:
                continue
            for _ in range(2):
                fn()
            torch.hpu.synchronize()
            samples = []
            for _ in range(a.repeat):
                value.copy_(torch.randn(batch, 5120).bfloat16())
                torch.hpu.synchronize()
                resident = torch.hpu.memory_allocated()
                torch.hpu.reset_peak_memory_stats()
                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                start = time.perf_counter()
                begin.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(
                    dict(device_ms=begin.elapsed_time(end),
                         wall_ms=(time.perf_counter() - start) * 1000,
                         extra_peak_bytes=torch.hpu.max_memory_allocated() - resident))
            report["timings"][name] = samples
            a.output.write_text(json.dumps(report, indent=2))
            print(name, statistics.median(x["device_ms"] for x in samples), flush=True)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
