# SPDX-License-Identifier: Apache-2.0
"""Compare CPU and device startup preparation through the real MoE consumer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import time


def qualify_loader(shard, fixtures, consumer, recorder, root):
    """Exercise the actual shared loader and its resident tensor attributes."""
    import torch
    from vllm_gaudi.ops.deepseek_v41_expert_n256 import load_projection
    from vllm_gaudi.ops.deepseek_v41_fp8 import read_expert
    from vllm_gaudi.ops.deepseek_v41_device_prepare import compiled_preparation, read_source_batch

    config = json.loads((shard.directory / "config.json").read_text())["text_config"]
    started = time.perf_counter()
    # The one-time preparation graphs are shared by all forty layers. Preserve
    # their cold cost separately; startup accounting must include this cost.
    for name in ("w13", "w2"):
        prefix = f"layers.20.ffn.experts.{name}"
        q, s = (read_source_batch(shard.catalog[prefix + suffix], 0, 16) for suffix in ("_q16", "_s16"))
        active = config["moe_intermediate_size"] // shard.tensor_parallel_size if name == "w2" else q.shape[-1] // 32
        output = compiled_preparation(True, active)(torch.from_numpy(q).to("hpu"),
                                                    torch.from_numpy(s.view("<i2")).to("hpu"))
    torch.hpu.synchronize()
    initialization_ms = (time.perf_counter() - started) * 1000
    del output, q, s

    records = []
    for case, layer in enumerate((20, 21, 22)):
        prefixes = [f"layers.{layer}.ffn.experts.{name}" for name in ("w13", "w2")]
        # File I/O remains in the timer, but both paths start with equally warm
        # source pages. No whole-checkpoint buffer is retained.
        for prefix in prefixes:
            for expert in range(shard.catalog[prefix + "_q16"].shape[0]):
                for suffix in ("_q16", "_s16"):
                    read_expert(shard.catalog[prefix + suffix], expert, keep_file_cache=True)
        fixture = torch.load(fixtures / f"layer20-case{case}.pt", weights_only=True, map_location="cpu")
        x = (fixture["residual"].float() * fixture["pre"].unsqueeze(-1)).sum(1).bfloat16().to("hpu")
        os.environ["VLLM_HPU_DSV41_N256_LOAD_WORKERS"] = "4"
        os.environ["VLLM_HPU_DSV41_N256_DEVICE_PREPARE"] = "0"
        started = time.perf_counter()
        reference = [load_projection(shard, prefix, "hpu") for prefix in prefixes]
        torch.hpu.synchronize()
        cpu_ms = (time.perf_counter() - started) * 1000
        os.environ["VLLM_HPU_DSV41_N256_DEVICE_PREPARE"] = "1"
        started = time.perf_counter()
        candidate = [load_projection(shard, prefix, "hpu") for prefix in prefixes]
        torch.hpu.synchronize()
        device_ms = (time.perf_counter() - started) * 1000
        for old_projection, new_projection in zip(reference, candidate, strict=True):
            for old, new in zip(old_projection, new_projection, strict=True):
                for first in range(0, old.shape[0], 16):
                    if not torch.equal(old[first:first + 16].cpu().view(torch.int16), new[first:first + 16].cpu().view(
                            torch.int16)):
                        raise RuntimeError("Shared loader changed prepared bytes")
            for attribute in ("dsv41_sat_eligible", "dsv41_active_k"):
                if getattr(old_projection[0], attribute, None) != getattr(new_projection[0], attribute, None):
                    raise RuntimeError("Shared loader changed qualification metadata")
        expected = consumer(x, *reference[0], *reference[1]).cpu()
        actual = consumer(x, *candidate[0], *candidate[1]).cpu()
        if not torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
            raise RuntimeError("Shared loader changed the real expert consumer")
        replay = recorder.prepare(consumer, [x], [(*candidate[0], *candidate[1])])
        replay()
        torch.hpu.synchronize()
        if not torch.equal(actual.view(torch.int16), replay.outputs[0].cpu().view(torch.int16)):
            raise RuntimeError("Shared loader broke the native consumer")
        replay.close()
        records.append(
            dict(layer=layer,
                 cpu_load_ms=cpu_ms,
                 device_load_ms=device_ms,
                 exact_weights=True,
                 exact_attributes=True,
                 exact_native_consumer=True))
        del reference, candidate
    report = dict(scope="actual shared projection loader, file reads, upload and native consumer; not full startup",
                  cold_device_initialization_ms=initialization_ms,
                  pairs=records,
                  cpu_median_ms=statistics.median(p["cpu_load_ms"] for p in records),
                  device_median_ms=statistics.median(p["device_load_ms"] for p in records),
                  all_three_faster=all(p["device_load_ms"] < p["cpu_load_ms"] for p in records))
    (root / "device-loader-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1, choices=(1, 16))
    parser.add_argument("--reference-workers", type=int, default=1, choices=(1, 4))
    parser.add_argument("--pack-library", type=Path)
    parser.add_argument("--projection-loader", action="store_true")
    args = parser.parse_args()
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import habana_frameworks.torch.core  # noqa: F401
    import numpy as np
    import torch
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators

    load_native_operators(required=("custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2", ))
    from deepseek_v41_micro_replay import RecipeRecorder
    from vllm_gaudi.ops.deepseek_v41_device_prepare import prepare_expert_device
    from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert
    from vllm_gaudi.ops.deepseek_v41_fp8 import read_expert
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    recorder = RecipeRecorder(root)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    active = config["moe_intermediate_size"] // 4
    pack_q16 = None
    if args.pack_library:
        from vllm_gaudi.ops.deepseek_v41_startup_pack import native_q16_packer

        pack_q16 = native_q16_packer(args.pack_library)
        os.environ["VLLM_HPU_DSV41_N256_PACK_LIBRARY"] = str(args.pack_library)
    # Use the maintained lookup's exact sign/exponent encodings.
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut

    lut = mxfp4_bf16_lut("hpu")
    ids = torch.zeros((6, 6), dtype=torch.int32, device="hpu")
    route = torch.full((6, 6), 1 / 6, dtype=torch.float32, device="hpu")

    def conversion(q, s, active_k):
        return prepare_expert_device(q, s, compact_scales=True, active_k=active_k)

    def consumer(x, q13, p13, c13, q2, p2, c2):
        return torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2(
            x, ids, route, q13, q2, p13, p2, lut, c13, c2, True)

    conversion = torch.compile(conversion, backend="hpu_backend", fullgraph=True, dynamic=False)
    consumer = torch.compile(consumer, backend="hpu_backend", fullgraph=True, dynamic=False)
    pairs = []
    with torch.inference_mode():
        if args.projection_loader:
            qualify_loader(shard, args.fixtures, consumer, recorder, root)
            return
        for case, expert in enumerate((13, 171, 337) if args.batch_size == 1 else (0, 160, 320)):
            source = {}
            for name in ("w13", "w2"):
                prefix = f"layers.20.ffn.experts.{name}"
                source[name] = tuple(
                    np.stack([
                        read_expert(shard.catalog[prefix + suffix], index, keep_file_cache=True)
                        for index in range(expert, expert + args.batch_size)
                    ]) for suffix in ("_q16", "_s16"))
            fixture = torch.load(args.fixtures / f"layer20-case{case}.pt", weights_only=True, map_location="cpu")
            x = (fixture["residual"].float() * fixture["pre"].unsqueeze(-1)).sum(1).bfloat16().to("hpu")
            # Cold graph preparation is not part of repeated loading cost.
            for name, (q, s) in source.items():
                conversion(
                    torch.from_numpy(q).to("hpu"),
                    torch.from_numpy(s.view(np.int16)).to("hpu"), q.shape[-1] // 32 if name == "w13" else active)
            torch.hpu.synchronize()
            started = time.perf_counter()
            reference = {}
            for name, (q, s) in source.items():

                def prepare(index, q=q, s=s):
                    return prepare_expert(q[index], s[index], compact_scales=True, pack_q16=pack_q16)[:3]

                with ThreadPoolExecutor(max_workers=args.reference_workers) as pool:
                    values = list(pool.map(prepare, range(args.batch_size)))
                arrays = [np.stack([value[index] for value in values]) for index in range(3)]
                reference[name] = (torch.from_numpy(arrays[0]).to("hpu"), torch.from_numpy(arrays[1]).to("hpu"),
                                   torch.from_numpy(arrays[2].view(np.int16)).view(torch.bfloat16).to("hpu"))
            torch.hpu.synchronize()
            cpu_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            candidate, checks = {}, []
            for name, (q, s) in source.items():
                packed, planes, channel, certificate = conversion(
                    torch.from_numpy(q).to("hpu"),
                    torch.from_numpy(s.view(np.int16)).to("hpu"), q.shape[-1] // 32 if name == "w13" else active)
                assert all(value.device.type == "hpu" for value in (packed, planes, channel, certificate))
                candidate[name] = packed, planes, channel.view(torch.bfloat16)
                checks.append(certificate)
            torch.hpu.synchronize()
            device_ms = (time.perf_counter() - started) * 1000
            assert all(bool((check[:, 0] == 1).all().cpu()) for check in checks)
            exact = all(
                torch.equal(a.cpu().view(torch.int16),
                            b.cpu().view(torch.int16)) for name in source
                for a, b in zip(reference[name], candidate[name], strict=True))
            if not exact:
                raise RuntimeError("Device preparation changed compressed weight or channel bytes")
            ids.fill_(case * 5 if args.batch_size > 1 else 0)
            expected = consumer(x, *reference["w13"], *reference["w2"]).cpu()
            actual = consumer(x, *candidate["w13"], *candidate["w2"]).cpu()
            if not torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
                raise RuntimeError("Device preparation changed the MoE consumer output")
            replay = recorder.prepare(consumer, [x], [(*candidate["w13"], *candidate["w2"])])
            replay()
            torch.hpu.synchronize()
            if not torch.equal(actual.view(torch.int16), replay.outputs[0].cpu().view(torch.int16)):
                raise RuntimeError("Prepared weight consumer did not preserve native replay")
            replay.close()
            pairs.append(
                dict(expert=expert,
                     cpu_prepare_upload_ms=cpu_ms,
                     device_upload_prepare_ms=device_ms,
                     exact_weights=True,
                     exact_consumer=True,
                     native_consumer=True))
        record = dict(scope="real expert W13/W2 preparation, upload and native MoE consumer; no whole startup claim",
                      batch_size=args.batch_size,
                      reference_workers=args.reference_workers,
                      reference_native_pack=bool(args.pack_library),
                      pairs=pairs,
                      cpu_median_ms=statistics.median(p["cpu_prepare_upload_ms"] for p in pairs),
                      device_median_ms=statistics.median(p["device_upload_prepare_ms"] for p in pairs),
                      all_three_faster=all(p["device_upload_prepare_ms"] < p["cpu_prepare_upload_ms"] for p in pairs))
        (root / "device-weight-result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
