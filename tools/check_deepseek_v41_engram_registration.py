# SPDX-License-Identifier: Apache-2.0
"""Measure production table registration, separately from shared backing creation."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment, load_native_operators

    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import numpy as np
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch.core  # noqa: F401
    from transformers import AutoTokenizer
    from vllm_gaudi.ops.deepseek_v41_completion import _control_bridge
    from vllm_gaudi.ops.deepseek_v41_engram import (
        EngramHashLayout, EngramTokenHistory, build_compressed_token_map,
    )
    from vllm_gaudi.ops.deepseek_v41_host import device_engram_parameters
    from vllm_gaudi.ops.deepseek_v41_residency import shared_host_region, table_regions

    report = dict(status="running", scope="one production TP shard, layer 1, C6", rounds=[])
    output = Path(os.environ["DSV41_RUN_EVIDENCE"]) / "result.json"
    descriptors, producer = [], None
    try:
        torch.hpu.set_device(0)
        load_native_operators()
        dist.init_process_group("hccl", store=dist.HashStore(), rank=0, world_size=1)
        ready = torch.ones(1, dtype=torch.bfloat16, device="hpu")
        dist.all_reduce(ready)
        ready.cpu()
        bridge = _control_bridge()
        backend = dist.group.WORLD._get_backend(torch.device("hpu"))
        config = json.loads((args.prepared / "config.json").read_text())["text_config"]
        layout = EngramHashLayout.from_config(config)
        mapping, compressed_size = build_compressed_token_map(
            AutoTokenizer.from_pretrained(args.prepared, local_files_only=True))
        token_map = np.asarray(mapping, dtype=np.int64)
        assert compressed_size == layout.vocab_size
        reference = EngramTokenHistory(layout, token_map)
        parameters = torch.tensor(device_engram_parameters(layout, 1, 0, reference.pad_id, 4),
                                  dtype=torch.int32, device="hpu")
        device_map = torch.tensor(token_map.astype(np.int32), device="hpu")
        regions = next(owner["regions"] for owner in table_regions(args.prepared)
                       if owner["tp"] == 0 and owner["layer"] == 1)
        shard = layout.head_shard(1, 0, 4)
        rows = shard["row_stop"] - shard["row_start"]
        tables = [np.memmap(region["file"], dtype=np.uint8, mode="r", offset=region["offset"],
                            shape=(rows, width)) for region, width in zip(regions, (256, 8), strict=True)]
        fixtures = [torch.load(args.fixtures / "rank0" / f"c6-{index}.pt", map_location="cpu",
                               weights_only=False) for index in range(3)]
        shared = []
        started = time.monotonic()
        for region in regions:
            descriptor, binding = shared_host_region(region)
            descriptors.append(descriptor)
            shared.append(dict(file=binding["file"], offset=0, length=binding["length"]))
        report["shared_backing_creation_seconds"] = time.monotonic() - started
        report["mapped_bytes"] = sum(region["length"] for region in regions)
        print(json.dumps({key: value for key, value in report.items() if key != "rounds"}), flush=True)
        raw = torch.empty(6, dtype=torch.int32, device="hpu")
        history = torch.empty(3, dtype=torch.int32, device="hpu")
        next_history = torch.empty((7, 3), dtype=torch.int32, device="hpu")
        decoded = torch.empty((6, 6, 256), dtype=torch.bfloat16, device="hpu")
        for round_index in range(3):
            for label, source in (("private", regions), ("shared", shared)):
                torch.hpu.synchronize()
                started = time.monotonic()
                producer = bridge.DeviceEngramProducer(
                    backend, source[0]["file"], source[0]["offset"], source[0]["length"],
                    source[1]["file"], source[1]["offset"], source[1]["length"], rows,
                    device_map, parameters, shared_checkpoint=label == "shared", local_heads=6, tokens=6)
                elapsed = time.monotonic() - started
                for fixture in fixtures:
                    ids = fixture["ids"].to(torch.int32).reshape(6)
                    cursor = fixture["cursor_history"].to(torch.int32).reshape(3)
                    reference.reset("registration")
                    reference.history = np.array(list(reversed(cursor.tolist())), dtype=np.int64)
                    batch = reference.prepare("registration", ids.tolist(), image_mask=ids.ge(129264).tolist())
                    indices = batch.hash_ids[:, layout.layer_ids.index(1), :6] - shard["row_start"]
                    values = torch.from_numpy(tables[0][indices].copy()).view(torch.float8_e4m3fn).float()
                    scales = torch.exp2(torch.from_numpy(tables[1][indices].astype(np.int32)) - 127)
                    expected = (values * scales.repeat_interleave(32, -1)).to(torch.bfloat16)
                    raw.copy_(ids)
                    history.copy_(cursor)
                    producer.launch(raw, history, next_history, decoded)
                    assert torch.equal(decoded.cpu(), expected), (round_index, label, "decoded rows")
                    reference.pending = None
                started = time.monotonic()
                producer.close()
                producer = None
                result = dict(round=round_index, arm=label, registration_seconds=elapsed,
                              release_seconds=time.monotonic() - started, real_cases=3, exact=True)
                report["rounds"].append(result)
                print(json.dumps(result), flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if producer is not None:
            producer.close()
        for descriptor in descriptors:
            os.close(descriptor)
        output.write_text(json.dumps(report, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
