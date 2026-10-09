# SPDX-License-Identifier: Apache-2.0
"""Compare C1 and C6 logits after one identical, teacher-forced real prefix.

Isolated numerical diagnosis, never a serving performance qualification.
Uses the loaded worker's normal target/input/Engram transaction methods.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import time
import traceback


def compare_worker(worker, prompt, reference, index, destination):
    import torch
    from vllm import SamplingParams
    from vllm_gaudi.ops.deepseek_v41_replay import _Snapshot, stage_state_tensors
    from vllm_gaudi.ops.deepseek_v41_state import PagedStageState
    from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState, target_chunks

    runner = worker.model_runner
    if runner.requests or getattr(runner, "device_round_queue", None) is not None:
        raise RuntimeError("Prefix diagnosis requires an idle owned worker")
    if not runner.pp.group.is_first_rank or not runner.pp.group.is_last_rank:
        raise RuntimeError("Prefix diagnosis requires one sampling TP group and no PP boundary")
    if not isinstance(runner.state, PagedStageState) or runner.request_slots_enabled:
        raise RuntimeError("Prefix diagnosis requires the current single-request paged state")
    root = Path(destination)
    rank = runner.model.tp_rank
    root.mkdir(parents=True, exist_ok=True)
    program = runner.model.program
    req_id = "prefix-logits-diagnostic"
    start = len(prompt) + index - 1
    pages = (start + 6 + 127) // 128
    if pages + 1 > runner.state.blocks:
        raise RuntimeError("Prefix diagnosis exceeds the initialized physical page pool")
    request = RequestState(req_id, list(prompt), [], SamplingParams(temperature=0, max_tokens=1),
                           (list(range(1, pages + 1)),), output=list(reference[:index]))
    # Startup uses temporary owners. Retire them before installing this one,
    # so the state manager does not retain an unrelated full working snapshot.
    if runner.state.active is not None:
        runner.state.release(runner.state.active)
    runner._bind_request(request)
    report = dict(status="preparing_identical_prefix", rank=rank, predicted_index=index,
                  target_input_position=start, prompt_tokens=len(prompt),
                  source_reference_token=reference[index], formal_qualified=False,
                  reference_precision=program.runtime_precision, comparisons=[])

    def save():
        (root / f"logits-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    with torch.inference_mode():
        for offset, chunk in target_chunks(prompt, runner.prefill_capacity):
            runner._forward(req_id, chunk, offset, decode=False, reset=offset == 0, request=request)
            runner.model.complete_step(len(chunk))
            request.num_computed_tokens = offset + len(chunk)
        # Prompt's last input predicts output[0]. To predict output[index],
        # the anchor is output[index-1]; consume only its preceding inputs.
        for offset, token in enumerate(reference[:index - 1]):
            runner._forward(req_id, [token], len(prompt) + offset, decode=True, request=request)
            runner.model.complete_step(1)
            request.num_computed_tokens = len(prompt) + offset + 1
        torch.hpu.synchronize()
        limits = {}
        for cache in program.shared.sources.values():
            for name in ("main", "index"):
                value = getattr(cache, name)
                limits[id(value)] = (pages + 1) * 128 // cache.ratio
            mirror = getattr(cache, "index_mirror", None)
            if mirror is not None:
                limits[id(mirror)] = min(mirror.shape[0], (start + 6 + cache.ratio - 1) // cache.ratio)
        tensors = [value[:limits[id(value)]] if id(value) in limits else value
                   for value in stage_state_tensors(program)]
        size = sum(value.numel() * value.element_size() for value in tensors)
        if size > 512 * 1024**2:
            raise RuntimeError(f"Unexpected diagnostic snapshot size {size}; no full-pool clone attempted")
        snapshot = _Snapshot(tensors)
        report.update(status="comparing", snapshot_bytes=size)
        save()
        values = {}
        anchors = list(reference[index - 1:index + 5])
        if len(anchors) != 6:
            raise ValueError("Reference must contain the anchor and five future tokens")
        alternatives = [anchors[0], *[7 + i for i in range(5)]]
        for name, tokens in (("c1", anchors[:1]), ("c6", anchors),
                             ("c1_repeat", anchors[:1]), ("c6_changed_future", alternatives)):
            snapshot.restore()
            torch.hpu.synchronize()
            hidden = runner._forward(req_id, tokens, start, decode=True, request=request)
            # Keep the head's actual M=1/M=6 geometry too. Truncating hidden
            # first would hide numerical differences in the final projection.
            full_logits = program.logits(hidden).cpu().clone()
            logits = full_logits[:1]
            # The diagnostic transaction consumes no input: keep Engram's
            # identical prefix. Device recurrent/cache writes are restored
            # above before every arm, preserving all captured addresses.
            runner.model.complete_step(0)
            values[name] = logits
            torch.save(full_logits, root / f"{name}-rank{rank}.pt")
            top = logits.topk(2)
            report["comparisons"].append(dict(arm=name, count=len(tokens),
                                              top_ids=top.indices[0].tolist(),
                                              top_values=top.values[0].tolist()))
            save()
        if not torch.equal(values["c1"], values["c1_repeat"]):
            raise RuntimeError("State/Engram restoration did not reproduce C1 logits exactly")
        delta = (values["c1"] - values["c6"]).abs()
        future = (values["c6"] - values["c6_changed_future"]).abs()
        top = values["c1"].topk(2).values[0]
        report.update(status="completed_numerical_diagnostic", max_abs_diff=float(delta.max()),
                      mean_abs_diff=float(delta.mean()), c1_top1_top2_margin=float(top[0] - top[1]),
                      c1_reference_top1_reproduced=int(values["c1"].argmax()) == reference[index],
                      c6_future_sensitivity_max_abs=float(future.max()),
                      c1_repeated_byte_exact=True,
                      conclusion="unclassified until source-reference reproduction and future sensitivity reviewed")
        snapshot.restore()
        torch.hpu.synchronize()
        save()
    runner.state.release(req_id)
    runner.model.engram_host.release_request(req_id)
    runner.active_request = None
    return {k: report[k] for k in ("rank", "status", "max_abs_diff", "c1_top1_top2_margin",
                                   "c1_reference_top1_reproduced", "c6_future_sensitivity_max_abs")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("prompt_ids", type=Path)
    parser.add_argument("reference_ids", type=Path)
    parser.add_argument("--index", type=int, default=98)
    args = parser.parse_args()
    if args.index < 1:
        parser.error("A decode discrepancy must follow the first sampled token")
    # This owned diagnostic passes a trusted local callable through the
    # normal collective RPC. Verify transport before loading model weights.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from vllm.v1.serial_utils import MsgpackEncoder
    MsgpackEncoder().encode(compare_worker)
    prompt = json.loads(args.prompt_ids.read_text())
    reference = json.loads(args.reference_ids.read_text())
    if len(reference) < args.index + 5:
        parser.error("Reference does not contain the full anchor/future comparison")
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    from vllm import LLM
    llm = LLM(model=str(args.prepared), tensor_parallel_size=4, pipeline_parallel_size=1,
              trust_remote_code=True, dtype="bfloat16", generation_config="vllm",
              load_format="dsv41_prepared", mm_encoder_tp_mode="data", async_scheduling=False,
              speculative_config={"method": "dspark", "num_speculative_tokens": 5},
              shutdown_timeout=120, max_model_len=1048576,
              max_num_batched_tokens=8192, max_num_seqs=32, block_size=128,
              enable_prefix_caching=False,
              gpu_memory_utilization=1.0, num_gpu_blocks_override=8193,
              additional_config={"dsv41_native_warmup_searches": [512, 1024, 32768, 65536]})
    destination = Path(os.environ["DSV41_RUN_EVIDENCE"])
    function = compare_worker
    for attempt in range(3):
        try:
            result = llm.collective_rpc(function, timeout=1800,
                                        args=(prompt, reference, args.index, str(destination)))
            break
        except Exception:
            # Retain this owned diagnostic model while a concrete helper
            # failure is corrected. A retry submits a new isolated callable;
            # it does not replace model methods or qualify serving defaults.
            (destination / f"RPC_FAILURE_{attempt}.txt").write_text(traceback.format_exc())
            if attempt == 2:
                raise
            retry = destination / f"RETRY_PREFIX_{attempt}.json"
            print(f"Diagnostic failed; owned model retained for local retry {retry}", flush=True)
            deadline = time.monotonic() + 900
            while not retry.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("No corrected diagnostic helper received") from None
                time.sleep(1)
            spec = json.loads(retry.read_text())
            source = Path(spec["source"])
            if source.name != "check_deepseek_v41_prefix_logits.py":
                raise ValueError("Retry must use the prefix diagnostic helper") from None
            function = runpy.run_path(str(source))["compare_worker"]
    Path(os.environ["DSV41_RUN_EVIDENCE"], "QUALITY_LOGITS_RESULT.json").write_text(
        json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
