# SPDX-License-Identifier: Apache-2.0
"""Complete C1 projection chains on immutable rank-local weights.

Component references are measured once when no comparable record exists. This
is a compiled-recipe diagnostic, not a substitute for native stage replay or
TP/PP end-to-end timing. It never enables a serving feature.
"""
import argparse
import json
import os
from pathlib import Path
import time

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_sampling import local_greedy_candidate  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import rms_norm  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_qkv import concatenate_static_weights  # noqa: E402


def summarize(values):
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": max(values),
        "samples": values
    }


def benchmark(name,
              reference,
              candidate,
              inputs,
              old_weights,
              new_weights,
              output,
              rounds,
              recorder,
              candidate_only=False,
              ordinary_validator=None,
              profile_only=False):
    # Each sweep visits every layer once. Distinct real matrices exceed on-chip
    # storage; weights are never copied inside the timed region. Compilation is
    # fullgraph and recipes persist across input/weight changes.
    functions = {
        "reference": torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False),
        "candidate": torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
    }
    weights = {"reference": old_weights, "candidate": new_weights}
    if candidate_only:
        functions.pop("reference")
    record = {
        "name": name,
        "execution": "torch.compile hpu_backend fullgraph cached recipe replay",
        "scope": "complete local producer-consumer projection, no TP/PP communication",
        "layers_per_sweep": len(inputs),
        "input_source": "seeded changing BF16; real checkpoint weights",
        "micro_not_native_stage_qualification": True,
        "weight_bytes": {
            arm: sum(t.numel() * t.element_size() for row in rows for t in row)
            for arm, rows in weights.items()
        },
        "rounds": [],
        "checks": []
    }
    for arm, fn in functions.items():
        for _ in range(3):
            for x, w in zip(inputs, weights[arm], strict=True):
                fn(x, *w)
        torch.hpu.synchronize()
        # Same new algorithm must consume changed inputs through cached recipes.
        x, w = inputs[0], weights[arm][0]
        original = fn(x, *w)
        original = tuple(t.cpu() for t in original) if isinstance(original, tuple) else (original.cpu(), )
        x.mul_(1.25)
        actual = fn(x, *w)
        actual = tuple(t.cpu() for t in actual) if isinstance(actual, tuple) else (actual.cpu(), )
        ordinary = (reference if arm == "reference" else candidate)(x, *w)
        ordinary = tuple(t.cpu() for t in ordinary) if isinstance(ordinary, tuple) else (ordinary.cpu(), )
        equal = all(torch.equal(a, b) for a, b in zip(actual, ordinary, strict=True))
        if not equal:
            torch.save({
                "compiled": actual,
                "ordinary": ordinary,
                "input": x.cpu()
            }, output / f"{name}-{arm}-ordinary-difference.pt")
            if ordinary_validator is not None:
                record["checks"].append(ordinary_validator(arm, x, w, actual, ordinary))
            elif name == "head":
                assert torch.equal(actual[0][:, 1], ordinary[0][:, 1]), "head token differs"
                torch.testing.assert_close(actual[0][:, 0], ordinary[0][:, 0], rtol=2e-6, atol=2e-5)
            elif name != "router":
                raise AssertionError((name, arm, "replay mismatch"))
            # The unchanged F32 gate/softplus compiled chain already differs
            # from eager execution. Isolate selection at identical score bytes.
            if name == "router":
                assert torch.equal(actual[0], ordinary[0]), "Router expert IDs differ"
                torch.testing.assert_close(actual[1], ordinary[1], rtol=2e-6, atol=2e-7)
                gate, text, image, mask = w if name == "router" else (None, ) * 4
                scores = torch.compile(lambda a, b: F.softplus(F.linear(a.float(), b)).sqrt(),
                                       backend="hpu_backend",
                                       fullgraph=True,
                                       dynamic=False)(x, gate)
                if arm == "candidate":
                    selector = torch.ops.custom_op.custom_deepseek_v41_router_top6_gaudi2
                    selected = selector(scores, text, image, mask)
                    replayed = torch.compile(selector, backend="hpu_backend", fullgraph=True,
                                             dynamic=False)(scores, text, image, mask)
                    assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(selected, replayed, strict=True))
        changed = any(not torch.equal(a, b) for a, b in zip(original, actual, strict=True))
        assert changed, (name, arm, "changed activation was not consumed")
        record["checks"].append({"arm": arm, "changed_input_consumed": changed, "ordinary_compiled_equal": equal})
    replays = {arm: recorder.prepare(fn, inputs, weights[arm]) for arm, fn in functions.items()}
    record["execution"] = ("hpu_backend native compute command capture/replay; "
                           "all segments staged before one final queue publication")
    record["recipes_per_sweep"] = {arm: replay.recipes for arm, replay in replays.items()}
    for arm, replay in replays.items():
        before = [t.cpu().clone() for t in replay.outputs]
        replay()
        torch.hpu.synchronize()
        assert all(torch.equal(a, b.cpu())
                   for a, b in zip(before, replay.outputs, strict=True)), "native recipe replay changed results"
        for _ in range(32):
            replay()
        torch.hpu.synchronize()
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_helpers
    bind_worker_helpers(0)
    record["thread_affinity"] = {
        p.name: sorted(os.sched_getaffinity(int(p.name)))
        for p in Path(f"/proc/{os.getpid()}/task").iterdir()
    }
    if profile_only:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]) as profiler:
            for arm, replay in replays.items():
                for sample in range(4):
                    with torch.profiler.record_function(f"chain-{arm}-{sample}"):
                        replay()
                        torch.hpu.synchronize()
        profiler.export_chrome_trace(str(output / "chain-trace.json.gz"))
        record["profile_only"] = True
        record["native_compute_info"] = {arm: replay.native.info() for arm, replay in replays.items()}
        for replay in replays.values():
            replay.close()
        return record
    events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)) for _ in range(32)]
    for round_id in range(rounds):
        # Alternate arm order without adding an end-to-end baseline request.
        order = (("candidate", ) if candidate_only else (("reference", "candidate") if round_id % 2 == 0 else
                                                         ("candidate", "reference")))
        for arm in order:
            fn = functions[arm]
            for x in inputs:
                x.mul_(0.99 if round_id % 2 else 1.01)
            torch.hpu.synchronize()
            expected = fn(inputs[0], *weights[arm][0])
            expected = tuple(t.cpu() for t in expected) if isinstance(expected, tuple) else (expected.cpu(), )
            replays[arm]()
            torch.hpu.synchronize()
            assert all(
                torch.equal(a, b.cpu()) for a, b in zip(expected, replays[arm].outputs[:len(expected)],
                                                        strict=True)), "native replay did not consume changed inputs"
            host, device = [], []
            for begin, end in events:
                start = time.perf_counter_ns()
                begin.record()
                replays[arm]()
                end.record()
                end.synchronize()
                host.append((time.perf_counter_ns() - start) / 1e6)
                device.append(begin.elapsed_time(end))
            torch.hpu.synchronize()
            item = {
                "round": round_id,
                "arm": arm,
                "device_sweep": summarize(device),
                "synchronized_host_sweep": summarize(host),
                "mean_device_per_layer_ms": float(np.mean(device)) / len(inputs)
            }
            record["rounds"].append(item)
            (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
            print(name, arm, round_id, "device sweep", np.mean(device), "host sweep", np.mean(host), flush=True)
    record["native_compute_info"] = {arm: replay.native.info() for arm, replay in replays.items()}
    (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    for replay in replays.values():
        replay.close()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--dense-sidecar", type=Path)
    parser.add_argument("--components",
                        nargs="+",
                        choices=("woa", "woa-output", "woa-roundtrip", "woa-rope",
                                 "router", "head", "compressor-input"),
                        default=["woa", "router", "head"])
    parser.add_argument("--rounds", type=int, default=3)
    arms = parser.add_mutually_exclusive_group()
    arms.add_argument("--candidate-only",
                      action="store_true",
                      default=True,
                      help="Reuse the archived component reference (default)")
    arms.add_argument("--include-reference",
                      action="store_false",
                      dest="candidate_only",
                      help="Measure a missing component reference once; never repeat a valid archived reference")
    args = parser.parse_args()
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.manual_seed(140913)
    from deepseek_v41_micro_replay import RecipeRecorder
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    bind_worker_cpu(0)
    recorder = RecipeRecorder(output)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    with torch.inference_mode():
        if "woa-roundtrip" in args.components:
            if args.dense_sidecar is None:
                raise ValueError("--dense-sidecar is required for woa-roundtrip")
            from vllm_gaudi.ops.deepseek_v41_math import quantize_activation, rotary_table
            woa_sidecar = WoaFP8Sidecar(args.sidecar, shard)
            dense_sidecar = DenseFP8Sidecar(args.dense_sidecar, shard)
            config = json.loads((args.prepared / "config.json").read_text())["text_config"]
            old, new, inputs = [], [], []
            positions = torch.tensor([127], dtype=torch.int32, device="hpu")
            for layer in range(20):
                prefix = f"layers.{layer}.attn."
                ratio, scaling = config["compress_ratios"][layer], config["rope_scaling"]
                table = rotary_table(64, 512, config["compress_rope_theta"] if ratio else config["rope_theta"],
                                     scaling["original_max_position_embeddings"] if ratio else 0,
                                     scaling["factor"], scaling["beta_fast"], scaling["beta_slow"])
                inverse = torch.cat((table[..., 0], -table[..., 1]), -1).contiguous().to("hpu")
                weights = (woa_sidecar.tensor(prefix + "wo_a.weight", "hpu"),
                           woa_sidecar.tensor(prefix + "wo_a.channel_scale", "hpu"),
                           dense_sidecar.tensor(prefix + "wo_b.weight", "hpu"),
                           dense_sidecar.tensor(prefix + "wo_b.channel_scale", "hpu"), positions, inverse)
                old.append(weights)
                new.append(weights)
                inputs.append(torch.randn(1, 32, 512).bfloat16().to("hpu"))

            def reference(x, woa, woa_scale, wob, wob_scale, positions, inverse):
                value = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(x, positions, inverse)
                value = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_gaudi2(
                    value.reshape(1, 4, 4096), woa, woa_scale)
                value = quantize_activation(value)
                return torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(value, wob, wob_scale)

            def candidate(x, woa, woa_scale, wob, wob_scale, positions, inverse):
                value = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(x, positions, inverse)
                value = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2(
                    value.reshape(1, 4, 4096), woa, woa_scale)
                return torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(value, wob, wob_scale)

            benchmark("woa-roundtrip", reference, candidate, inputs, old, new, output, args.rounds, recorder,
                      args.candidate_only)
            del old, new, inputs
        if "woa-rope" in args.components:
            from vllm_gaudi.ops.deepseek_v41_math import quantize_activation, rotary_table
            sidecar = WoaFP8Sidecar(args.sidecar, shard)
            config = json.loads((args.prepared / "config.json").read_text())["text_config"]
            old, new, inputs = [], [], []
            positions = torch.tensor([2051], dtype=torch.int32, device="hpu")
            for layer in range(20):
                prefix = f"layers.{layer}.attn."
                consumer = shard.dense(prefix + "wo_b.weight", "hpu")
                ratio, scaling = config["compress_ratios"][layer], config["rope_scaling"]
                table = rotary_table(64, 2560, config["compress_rope_theta"] if ratio else config["rope_theta"],
                                     scaling["original_max_position_embeddings"] if ratio else 0, scaling["factor"],
                                     scaling["beta_fast"], scaling["beta_slow"])
                phase = table.reshape(-1, 64).contiguous().to("hpu")
                inverse = torch.cat((table[..., 0], -table[..., 1]), -1).contiguous().to("hpu")
                weight = sidecar.tensor(prefix + "wo_a.weight", "hpu")
                scale = sidecar.tensor(prefix + "wo_a.channel_scale", "hpu")
                old.append((weight, scale, consumer, positions, inverse))
                new.append((weight, scale, consumer, positions, phase))
                inputs.append(torch.randn(1, 32, 512).bfloat16().to("hpu"))

            def separate(x, weight, scale, consumer, positions, inverse):
                value = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(x, positions, inverse)
                value = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_gaudi2(
                    value.reshape(1, 4, 4096), weight, scale)
                return F.linear(quantize_activation(value), consumer)

            def fused(x, weight, scale, consumer, positions, phase):
                value = torch.ops.custom_op.custom_deepseek_v41_rope_woa_fp8_gaudi2(
                    x, weight, scale, positions, phase)
                return F.linear(quantize_activation(value), consumer)

            benchmark("woa-rope", separate, fused, inputs, old, new, output, args.rounds, recorder,
                      args.candidate_only)
            del old, new, inputs
        if "woa-output" in args.components:
            from vllm_gaudi.ops.deepseek_v41_math import quantize_activation, rotary_table
            sidecar = WoaFP8Sidecar(args.sidecar, shard)
            config = json.loads((args.prepared / "config.json").read_text())["text_config"]
            old, new, inputs = [], [], []
            positions = torch.tensor([127], dtype=torch.int32, device="hpu")
            for layer in range(20):
                prefix = f"layers.{layer}.attn."
                dense = shard.dense(prefix + "wo_a.weight", "hpu").reshape(4, 1024, 4096).transpose(1, 2).contiguous()
                consumer = shard.dense(prefix + "wo_b.weight", "hpu")
                ratio, scaling = config["compress_ratios"][layer], config["rope_scaling"]
                table = rotary_table(64, 512, config["compress_rope_theta"] if ratio else config["rope_theta"],
                                     scaling["original_max_position_embeddings"] if ratio else 0, scaling["factor"],
                                     scaling["beta_fast"], scaling["beta_slow"])
                inverse = torch.cat((table[..., 0], -table[..., 1]), -1).contiguous().to("hpu")
                old.append((dense, consumer, positions, inverse))
                new.append((sidecar.tensor(prefix + "wo_a.weight",
                                           "hpu"), sidecar.tensor(prefix + "wo_a.channel_scale",
                                                                  "hpu"), consumer, positions, inverse))
                inputs.append(torch.randn(1, 32, 512).bfloat16().to("hpu"))

            def reference(x, weight, consumer, positions, inverse):
                value = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(x, positions, inverse)
                value = torch.einsum("tgd,gdr->tgr", value.reshape(1, 4, 4096), weight).flatten(1)
                return F.linear(quantize_activation(value), consumer)

            def candidate(x, weight, scale, consumer, positions, inverse):
                value = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(x, positions, inverse)
                value = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_gaudi2(value.reshape(1, 4, 4096), weight, scale)
                return F.linear(quantize_activation(value), consumer)

            benchmark("woa-output", reference, candidate, inputs, old, new, output, args.rounds, recorder,
                      args.candidate_only)
            del old, new, inputs
        if "woa" in args.components:
            sidecar = WoaFP8Sidecar(args.sidecar, shard)
            old, new, inputs = [], [], []
            for layer in range(20):
                prefix = f"layers.{layer}.attn.wo_a."
                dense = shard.dense(prefix + "weight", "hpu").reshape(4, 1024, 4096).transpose(1, 2).contiguous()
                old.append((dense, ))
                new.append((sidecar.tensor(prefix + "weight", "hpu"), sidecar.tensor(prefix + "channel_scale", "hpu")))
                inputs.append(torch.randn(1, 4, 4096).bfloat16().to("hpu"))

            def reference(x, weight):
                return torch.einsum("tgd,gdr->tgr", x, weight).flatten(1)

            benchmark("woa", reference, torch.ops.custom_op.custom_deepseek_v41_woa_fp8_gaudi2, inputs, old, new,
                      output, args.rounds, recorder, args.candidate_only)
            del old, new, inputs
        if "woa" not in args.components:
            for _ in range(20):
                torch.randn(1, 4, 4096)
        if "router" in args.components:
            weights, inputs = [], []
            for layer in range(20):
                prefix = f"layers.{layer}.ffn.gate."
                weights.append((shard.tensor(prefix + "weight", "hpu").float(), shard.tensor(prefix + "bias", "hpu"),
                                shard.tensor(prefix + "bias_vl", "hpu"), torch.zeros(1, dtype=torch.bool,
                                                                                     device="hpu")))
                inputs.append(torch.randn(1, 5120).bfloat16().to("hpu"))

            def reference(x, weight, text, image, mask):
                scores = F.softplus(F.linear(x.float(), weight)).sqrt()
                ids = torch.topk(scores + torch.where(mask[:, None], image, text), 6, sorted=True).indices
                values = scores.gather(1, ids)
                return ids.int(), values / (values.sum(-1, keepdim=True) + 1e-20) * 1.5

            def candidate(x, weight, text, image, mask):
                scores = F.softplus(F.linear(x.float(), weight)).sqrt()
                return torch.ops.custom_op.custom_deepseek_v41_router_top6_gaudi2(scores, text, image, mask)

            benchmark("router", reference, candidate, inputs, weights, weights, output, args.rounds, recorder,
                      args.candidate_only)
            del weights, inputs
        if "router" not in args.components:
            for _ in range(20):
                torch.randn(1, 5120)
        if "head" in args.components:
            shard = PreparedV41Shard(args.prepared, 1, 0)
            weight = shard.tensor("head.weight", "hpu")
            x = torch.randn(1, 5120).bfloat16().to("hpu")

            def reference(x, weight):
                return local_greedy_candidate(F.linear(x.float(), weight), 0)

            def candidate(x, weight):
                return local_greedy_candidate(torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(x, weight),
                                              0)

            benchmark("head", reference, candidate, [x], [(weight.float(), )], [(weight, )], output, args.rounds,
                      recorder, args.candidate_only)
        if "compressor-input" in args.components:
            shard = PreparedV41Shard(args.prepared, 0, 0)
            inputs, old, new = [], [], []
            for layer in (2, 8, 14):
                prefix = f"layers.{layer}.attn."
                wkv = shard.tensor(prefix + "compressor.wkv.weight", "hpu").float()
                wgate = shard.tensor(prefix + "compressor.wgate.weight", "hpu").float()
                norm = shard.tensor(prefix + "compressor.norm.weight", "hpu")
                index_weight = shard.tensor(prefix + "indexer.wk.weight", "hpu")
                index_norm = shard.tensor(prefix + "indexer.k_norm.weight", "hpu")
                previous_kv = torch.randn(1, wkv.shape[0], dtype=torch.float32, device="hpu")
                previous_score = torch.randn_like(previous_kv)
                fused = concatenate_static_weights(wkv, wgate).contiguous()
                common = (previous_kv, previous_score, norm, index_weight, index_norm)
                old.append((wkv, wgate, *common))
                new.append((fused, *common))
                inputs.append(torch.randn(1, wkv.shape[1], dtype=torch.bfloat16, device="hpu"))

            def consume(kv, score, previous_kv, previous_score, norm, index_weight, index_norm):
                gates = torch.stack((previous_score, score), 1).softmax(1)
                latent = (previous_kv * gates[:, 0] + kv * gates[:, 1]).to(torch.bfloat16)
                latent = rms_norm(latent, norm, 1e-6)
                index = rms_norm(F.linear(latent, index_weight), index_norm, 1e-6)
                return latent, index

            def reference(x, wkv, wgate, previous_kv, previous_score, norm, index_weight, index_norm):
                value = x.float()
                kv = F.linear(value, wkv)
                score = F.linear(value, wgate)
                return consume(kv, score, previous_kv, previous_score, norm, index_weight, index_norm)

            def candidate(x, fused, previous_kv, previous_score, norm, index_weight, index_norm):
                projected = F.linear(x.float(), fused)
                kv, score = projected.chunk(2, dim=-1)
                return consume(kv, score, previous_kv, previous_score, norm, index_weight, index_norm)

            ref_projection = torch.compile(lambda x, a, b: (F.linear(x.float(), a), F.linear(x.float(), b)),
                                           backend="hpu_backend",
                                           fullgraph=True,
                                           dynamic=False)
            fused_projection = torch.compile(lambda x, w: F.linear(x.float(), w).chunk(2, dim=-1),
                                             backend="hpu_backend",
                                             fullgraph=True,
                                             dynamic=False)
            compiled_reference = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
            compiled_candidate = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
            for x, reference_weights, candidate_weights in zip(inputs, old, new, strict=True):
                expected = ref_projection(x, reference_weights[0], reference_weights[1])
                actual = fused_projection(x, candidate_weights[0])
                assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(expected, actual, strict=True))
                expected = reference(x, *reference_weights)
                actual = candidate(x, *candidate_weights)
                assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(expected, actual, strict=True))
                expected = compiled_reference(x, *reference_weights)
                actual = compiled_candidate(x, *candidate_weights)
                assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(expected, actual, strict=True))

            def validate_existing_compile_delta(arm, x, weights, compiled, ordinary):
                del x, weights
                deltas = []
                for actual, expected in zip(compiled, ordinary, strict=True):
                    delta = (actual.float() - expected.float()).abs()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=1 / 128)
                    deltas.append({
                        "max_abs": delta.max().item(),
                        "mean_abs": delta.mean().item(),
                        "different": int(torch.count_nonzero(delta).item()),
                        "elements": delta.numel(),
                    })
                return {"arm": arm, "existing_compile_eager_bf16_delta": deltas}

            benchmark("compressor-input",
                      reference,
                      candidate,
                      inputs,
                      old,
                      new,
                      output,
                      args.rounds,
                      recorder,
                      args.candidate_only,
                      ordinary_validator=validate_existing_compile_delta)
    torch.hpu.synchronize()
    torch.distributed.destroy_process_group()
    (output / "memory.json").write_text(
        json.dumps(
            {
                "max_allocated_bytes": torch.hpu.max_memory_allocated(),
                "max_reserved_bytes": torch.hpu.max_memory_reserved()
            },
            indent=2))


if __name__ == "__main__":
    main()
