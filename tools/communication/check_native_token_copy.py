# SPDX-License-Identifier: Apache-2.0
"""Exercise asynchronous sampler D2H with changing compiled producers."""
import importlib.util
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch  # noqa: F401


@torch.inference_mode()
def main():
    if os.environ.get("HLS_MODULE_ID") is None:
        raise RuntimeError("Reserve and select a device before this check")
    path = Path(os.environ["VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE"])
    spec = importlib.util.spec_from_file_location("tp2_fused_ar_norm_bridge", path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    output = Path(os.environ["DSV4_CHECK_OUTPUT"])
    output.mkdir(exist_ok=True)
    # Device-side inputs and compiled arithmetic change on every call. Queue
    # many copies before consuming results to expose stale output reuse.
    rows = []
    for dtype in (torch.int32, torch.int64):
        def producer(value):
            return (value * 13 + 17).reshape(1, 1)
        compiled = torch.compile(producer, backend="hpu_backend", fullgraph=True, dynamic=False)
        pending = []
        for index in range(64):
            value = torch.tensor([index * 127 - 1], dtype=dtype).to("hpu", non_blocking=True)
            result = compiled(value)
            host, event = bridge.copy_sampled_tokens_to_host(result)
            pending.append((index, host, event))
            event.query()
        for index, host, event in pending:
            event.synchronize()
            assert event.query()
            expected = (index * 127 - 1) * 13 + 17
            actual = host.tolist()[0][0]
            rows.append(dict(dtype=str(dtype), generation=index, expected=expected,
                             actual=actual, exact=actual == expected))
        # A stable input allocation is also overwritten through the ordinary
        # stream dependency path while copies are outstanding.
        value = torch.zeros(1, dtype=dtype, device="hpu")
        pending = []
        for index in range(32):
            value.fill_(index * 31)
            host, event = bridge.copy_sampled_tokens_to_host(compiled(value))
            pending.append((index, host, event))
        for index, host, event in pending:
            event.synchronize()
            expected = index * 31 * 13 + 17
            actual = host.tolist()[0][0]
            rows.append(dict(dtype=str(dtype), overwrite_generation=index, expected=expected,
                             actual=actual, exact=actual == expected))
        destination = torch.empty((1, 1), dtype=dtype, device="hpu")
        pending = []
        for index in range(32):
            value.fill_(index * 53)
            destination.copy_(compiled(value))
            host, event = bridge.copy_sampled_tokens_to_host(destination)
            pending.append((index, host, event))
        for index, host, event in pending:
            event.synchronize()
            expected = index * 53 * 13 + 17
            actual = host.tolist()[0][0]
            rows.append(dict(dtype=str(dtype), fixed_output_generation=index, expected=expected,
                             actual=actual, exact=actual == expected))
    (output / "result.json").write_text(json.dumps(rows, indent=2) + "\n")
    assert all(row["exact"] for row in rows)
    print(json.dumps(dict(passed=True, generations=len(rows))), flush=True)


if __name__ == "__main__":
    main()
