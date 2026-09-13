# SPDX-License-Identifier: Apache-2.0
"""Capture real Qwen residual/norm/FP8 projection inputs without changing outputs.

This is a diagnostic eager model run, not an end-to-end performance benchmark.
Forward hooks are removed after capture. No candidate route is enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def install_capture(model, directory, layer_ids):
    import torch
    from vllm.forward_context import get_forward_context
    from vllm_gaudi.ops.hpu_layernorm import HPUGemmaRMSNorm, HPURMSNorm

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "_flashinfer_projection_capture"):
        raise RuntimeError("Projection capture is already installed")
    pending, captured, handles, inventory = {}, set(), [], []
    modules = dict(model.named_modules())
    selected = [(name, norm) for name, norm in modules.items()
                if name.endswith(".post_attention_layernorm") and any(f".layers.{index}." in name
                                                                      for index in layer_ids)]
    if len(selected) != len(layer_ids):
        raise RuntimeError(f"Expected {len(layer_ids)} captured layers, found {len(selected)}")
    for name, norm in selected:
        projection_name = name.removesuffix("post_attention_layernorm") + "mlp.gate_up_proj"
        projection = modules[projection_name]
        if not isinstance(norm, (HPUGemmaRMSNorm, HPURMSNorm)):
            raise RuntimeError(f"Unrecognized normalization contract: {type(norm).__name__}")
        if getattr(norm, "_hpu_tp2_fused_ar_norm", False):
            raise RuntimeError("Capture is TP1 only; do not bypass deferred all-reduces")
        if projection.weight.dtype != torch.float8_e4m3fn or projection.weight_scale_inv.ndim != 1:
            raise RuntimeError("Capture requires the postprocessed channel-FP8 model weights")
    # Validate the entire topology before installing any hook.
    for name, norm in selected:
        projection_name = name.removesuffix("post_attention_layernorm") + "mlp.gate_up_proj"
        projection = modules[projection_name]
        prefix = name.replace(".", "-")
        gamma = norm.weight.detach() + 1.0 if isinstance(norm, HPUGemmaRMSNorm) else norm.weight.detach()
        weights = {
            "gamma": gamma.cpu(),
            "weight": projection.weight.detach().cpu(),
            "weight_scale": projection.weight_scale_inv.detach().cpu()
        }
        torch.save(weights, directory / f"{prefix}-weights.pt")
        inventory.append({
            "norm": name,
            "projection": projection_name,
            "prefix": prefix,
            "gamma_dtype": str(gamma.dtype),
            "weight_shape": list(projection.weight.shape),
            "epsilon": norm.variance_epsilon
        })

        def before_norm(module, args, kwargs, prefix=prefix):
            x = args[0] if args else kwargs["x"]
            residual = args[1] if len(args) > 1 else kwargs.get("residual")
            if residual is None:
                return
            rows = x.numel() // x.shape[-1]
            metadata = get_forward_context().attn_metadata
            phase = "prefill" if metadata is not None and bool(getattr(metadata, "is_prompt", False)) else "decode"
            key = f"{prefix}-{phase}-{rows}"
            if rows not in (1, 8, 32, 2048) or key in captured:
                return
            pending[prefix] = (key, {
                "x": x.detach().reshape(-1, x.shape[-1]).cpu(),
                "residual": residual.detach().reshape(-1, x.shape[-1]).cpu(),
                "source_shape": list(x.shape),
                "phase": phase
            })

        def after_norm(module, args, output, prefix=prefix):
            if prefix in pending:
                pending[prefix][1]["normed"] = output[0].detach().reshape(-1, output[0].shape[-1]).cpu()
                pending[prefix][1]["summed"] = output[1].detach().reshape(-1, output[1].shape[-1]).cpu()

        def after_projection(module, args, output, prefix=prefix):
            if prefix not in pending:
                return
            key, tensors = pending.pop(prefix)
            result = output[0] if isinstance(output, tuple) else output
            tensors["projected"] = result.detach().reshape(-1, result.shape[-1]).cpu()
            torch.save(tensors, directory / f"{key}.pt")
            captured.add(key)

        handles.extend((norm.register_forward_pre_hook(before_norm, with_kwargs=True),
                        norm.register_forward_hook(after_norm), projection.register_forward_hook(after_projection)))
    object.__setattr__(model, "_flashinfer_projection_capture", (handles, captured))
    (directory / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory


def remove_capture(model):
    handles, captured = model._flashinfer_projection_capture
    for handle in handles:
        handle.remove()
    delattr(model, "_flashinfer_projection_capture")
    return sorted(captured)


class ProjectionCaptureWorkerExtension:
    """Use named control RPCs without enabling pickle-based serialization."""

    def install_projection_capture(self, directory, layer_ids):
        return install_capture(self.get_model(), directory, layer_ids)

    def remove_projection_capture(self):
        return remove_capture(self.get_model())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,3,63")
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--decode-tokens",
                        type=int,
                        default=64,
                        help="Keep requests active long enough for the requested concurrent decode buckets")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory to preserve previous model evidence")
    args.output.mkdir(parents=True)
    import habana_frameworks.torch  # noqa: F401
    from vllm import LLM, SamplingParams

    layers = tuple(map(int, args.layers.split(",")))
    llm = LLM(model=args.model,
              tensor_parallel_size=1,
              dtype="bfloat16",
              max_model_len=4096,
              max_num_seqs=32,
              max_num_batched_tokens=2048,
              gpu_memory_utilization=.70,
              enforce_eager=True,
              enable_prefix_caching=False,
              generation_config="vllm",
              worker_extension_cls="tools.capture_flashinfer_model_projection.ProjectionCaptureWorkerExtension",
              seed=31)
    directory = str(args.output.resolve())
    report = {
        "model": args.model,
        "model_config_sha256": hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest(),
        "enforce_eager": True,
        "candidate_enabled": False,
        "performance_measurement": False,
        "requests": []
    }
    report["inventory"] = llm.collective_rpc("install_projection_capture", args=(directory, layers))
    try:
        tokenizer = llm.get_tokenizer()
        text = "Explain how residual connections and normalization help a language model. "
        unit = tokenizer.encode(text, add_special_tokens=False)
        tokens = (unit * (2048 // len(unit) + 1))[:2048]
        for batch in map(int, args.batches.split(",")):
            prompt_tokens = tokens if batch == 1 else tokens[:128]
            outputs = llm.generate([{
                "prompt_token_ids": prompt_tokens
            }] * batch,
                                   SamplingParams(temperature=0,
                                                  max_tokens=3 if batch == 1 else args.decode_tokens,
                                                  ignore_eos=True),
                                   use_tqdm=False)
            report["requests"].append({
                "batch": batch,
                "prompt_tokens": len(prompt_tokens),
                "output_token_ids": [list(output.outputs[0].token_ids) for output in outputs]
            })
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        report["captured"] = llm.collective_rpc("remove_projection_capture")
        expected = {
            f"{layer['prefix']}-decode-{batch}"
            for layer in report["inventory"][0]
            for batch in map(int, args.batches.split(","))
        }
        if "1" in args.batches.split(","):
            expected.update(f"{layer['prefix']}-prefill-2048" for layer in report["inventory"][0])
        report["missing_captures"] = sorted(expected - set(report["captured"][0]))
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["missing_captures"]:
        raise RuntimeError(f"Requested decode buckets were not observed: {report['missing_captures']}")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
