# SPDX-License-Identifier: Apache-2.0
"""Opt-in paired attention diagnosis after the native performance gate."""
import json
from pathlib import Path

import torch
import torch.distributed as dist


def compare(expected, actual):
    expected, actual = expected.detach().cpu(), actual.detach().cpu()
    difference = (expected.float() - actual.float()).abs()
    return dict(exact=torch.equal(expected, actual), shape=list(actual.shape),
                changed=int((expected != actual).sum()), max_abs=float(difference.max()),
                mean_abs=float(difference.mean()), finite=bool(actual.isfinite().all()))


@torch.inference_mode()
def audit_decoder_replay(decoder, roots, groups, directory, step):
    """Compare identical full-decoder inputs/state via host and native replay.

    This explicit diagnosis restores the capture write ranges before every
    trial and before the real invocation. It never handles execution errors by
    retrying the user call, and its runs are excluded from speed qualification.
    """
    from vllm_gaudi.ops.tp2_model_adapter import DEEPSEEK_V4
    from vllm_gaudi.ops.tp2_prepared_plan import (
        _modules, collect_prepared_group_replays, prepared_group_stats,
        record_native_decoder_outputs, replay_native_decoder)

    owner, metadata = decoder.owner(), roots["metadata"]
    output = Path(directory) / f"rank{dist.get_rank()}" / f"step{step}"
    output.mkdir(parents=True, exist_ok=False)
    snapshot = decoder.snapshot(metadata)
    initial_stats = prepared_group_stats()
    group_plans = sorted((key[1], plan) for module in tuple(_modules)
                         for key, plan in zip(module.plan_owners, module.plans, strict=True)
                         if key is not None and key[0] == id(owner) and key[2] == decoder.generation)
    if [index for index, _ in group_plans] != list(range(6)):
        raise RuntimeError("Replay audit requires the six existing decoder plans")
    samples = {}
    report = dict(audit_version=3,
                  scope="same-runtime replay diagnosis; not production or performance qualification",
                  step=step, position=roots["positions"].cpu().tolist(),
                  snapshot_bytes=snapshot.bytes, comparisons={}, initial_stats=initial_stats)
    torch.save({key: roots[key].cpu() for key in ("hidden_states", "positions", "input_ids", "metadata_pack")},
               output / "inputs.pt")
    try:
        # Native must run before any ordinary launch can refresh a persistent
        # intermediate that an incomplete captured program might omit.
        for name in ("native0", "native1", "host0", "host1", "native2"):
            snapshot.restore()
            if name.startswith("host"):
                # Ordinary execution needs its metadata staged explicitly.
                # Native trials must exercise their own staging transaction.
                decoder.fixed_metadata_pack.copy_(roots["metadata_pack"])
                hidden, residual, post, comb = roots["hidden_states"], None, None, None
                with collect_prepared_group_replays(owner=owner, adapter=DEEPSEEK_V4,
                                                   diagnostic_host_replay=True, **roots) as context:
                    for index, (chunk, inputs) in enumerate(zip(owner._hpu_compiled_layer_chunks, groups, strict=True)):
                        context["group_index"] = index
                        hidden, residual, post, comb = chunk(
                            hidden, roots["positions"], roots["input_ids"], post, comb, residual, inputs)
                    record_native_decoder_outputs(hidden)
            else:
                result = replay_native_decoder(owner, **roots)
                if result is None:
                    raise RuntimeError("Replay audit invalidated a ready native plan")
                hidden = result[0]
            torch.hpu.synchronize()
            samples[name] = dict(output=hidden.cpu(), state=[page.cpu() for page, _ in snapshot.saved],
                                 groups=[[tensor.cpu() for tensor in plan.outputs()] for _, plan in group_plans])
            torch.save(samples[name], output / f"{name}.pt")
        for left, right in (("host0", "host1"), ("host0", "native0"),
                            ("native0", "native1"), ("native0", "native2")):
            a, b = samples[left], samples[right]
            report["comparisons"][f"{left}_vs_{right}"] = dict(
                output=compare(a["output"], b["output"]),
                changed_state_pages=[i for i, (x, y) in enumerate(zip(a["state"], b["state"], strict=True))
                                     if not torch.equal(x, y)],
                groups=[[compare(x, y) for x, y in zip(xs, ys, strict=True)]
                        for xs, ys in zip(a["groups"], b["groups"], strict=True)])
        report["final_stats"] = prepared_group_stats()
        if report["final_stats"]["prepares"] != initial_stats["prepares"]:
            raise RuntimeError("Replay audit unexpectedly prepared a new tensor program")
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("DSV4_REPLAY_AUDIT " + json.dumps(dict(rank=dist.get_rank(), step=step,
              outputs={key: value["output"] for key, value in report["comparisons"].items()})), flush=True)
    finally:
        snapshot.restore()
        torch.hpu.synchronize()


@torch.inference_mode()
def audit_attention(decoder, hidden_states, positions, input_ids, metadata, directory):
    from vllm_gaudi.compilation.deepseek_v4 import make_backend

    owner = decoder.owner()
    output = Path(directory) / f"rank{dist.get_rank()}"
    output.mkdir(parents=True, exist_ok=True)
    report = dict(reference="existing attention implementation on the same runtime; not production qualification",
                  layers=[])
    reference_embedding = owner.embed_input_ids(input_ids)
    report["embedding"] = compare(reference_embedding, hidden_states)
    torch.save(dict(reference=reference_embedding.cpu(), actual=hidden_states.cpu(),
                    positions=positions.cpu(), input_ids=input_ids.cpu()), output / "embedding.pt")
    sample = reference_embedding.unsqueeze(-2).repeat(1, owner.hc_mult, 1)
    layer = owner.layers[0]
    projected, _, _ = layer.hc_pre(sample, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
    activation = layer.attn_norm(projected)
    seen = set()
    for index, layer in enumerate(owner.layers):
        wrapper = layer.attn.mla_attn
        if wrapper.compress_ratio in seen:
            continue
        seen.add(wrapper.compress_ratio)
        native = wrapper._hpu_native_decode
        bindings = native.bind_metadata(metadata)
        snapshot = decoder.snapshot(metadata)
        try:
            reference = torch.empty((1, wrapper.padded_heads, 512), dtype=activation.dtype, device=activation.device)
            wrapper.attention_impl(activation, positions, reference)
            reference_cpu = reference[:, :wrapper.n_local_heads].cpu()
            reference_pages = [page.cpu() for page, _ in snapshot.saved]
            snapshot.restore()
            eager = native(activation, positions, bindings)
            eager_cpu = eager[:, :wrapper.n_local_heads].cpu()
            eager_pages = [page.cpu() for page, _ in snapshot.saved]
            snapshot.restore()
            compiled = torch.compile(native, backend=make_backend(), fullgraph=True, dynamic=False)
            actual = compiled(activation, positions, bindings)
            actual_cpu = actual[:, :wrapper.n_local_heads].cpu()
            actual_pages = [page.cpu() for page, _ in snapshot.saved]
            row = dict(layer=index, compress_ratio=wrapper.compress_ratio,
                       eager=compare(reference_cpu, eager_cpu), compiled=compare(reference_cpu, actual_cpu),
                       eager_pages=[i for i, (a, b) in enumerate(zip(reference_pages, eager_pages, strict=True))
                                    if not torch.equal(a, b)],
                       compiled_pages=[i for i, (a, b) in enumerate(zip(reference_pages, actual_pages, strict=True))
                                       if not torch.equal(a, b)])
            report["layers"].append(row)
            torch.save(dict(reference=reference_cpu, eager=eager_cpu, compiled=actual_cpu,
                            reference_pages=reference_pages, eager_pages=eager_pages, compiled_pages=actual_pages),
                       output / f"attention-layer{index}.pt")
            print("DSV4_ATTENTION_AUDIT " + json.dumps(dict(rank=dist.get_rank(), **row)), flush=True)
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            snapshot.restore()
            torch.hpu.synchronize()
        if len(seen) == 3:
            break
