# SPDX-License-Identifier: Apache-2.0
"""Capture-time constants for immutable DSpark projection operands.

This capability changes compiler tensor ownership, not replay transport or
numerical precision. Request/KV/Engram state and routed expert banks are never
marked. Copies isolate the baseline's StorageImpl from constant repacking.
"""
from itertools import count
import os
from pathlib import Path

import torch

_constant_ids = count(1000000000)
_constant_directory = None


def immutable_bindings(layer):
    """Use only the checkpoint tree and explicit prepared projection buffers."""
    result = []
    for name, tensor in layer.weights.named_buffers(remove_duplicate=False):
        if name.startswith("ffn.experts.") or name.startswith("engram."):
            continue
        path, _, leaf = name.rpartition(".")
        owner = layer.weights.get_submodule(path) if path else layer.weights
        result.append((owner, leaf, tensor, "weights." + name))
    for name in ("hc_attn_fn_packed", "hc_ffn_fn_packed", "hc_attn_fn_mme", "hc_ffn_fn_mme"):
        tensor = getattr(layer, name, None)
        if tensor is not None:
            result.append((layer, name, tensor, name))
    for name in ("fused_wqa_wkv", "fused_qkv_channel", "fused_compressor_wkv_wgate"):
        tensor = layer.attention._buffers.get(name)
        if tensor is not None:
            result.append((layer.attention, name, tensor, "attention." + name))
    moe = getattr(layer, "moe", None)
    if moe is not None:
        for name in ("shared_gate_up_weight", "shared_gate_up_channel", "shared_down_weight"):
            tensor = moe._buffers.get(name)
            if tensor is not None:
                result.append((moe, name, tensor, "moe." + name))
    return result


def prepare_layer_constants(layer):
    """Prepare before capture, with baseline-safe copies and bounded ownership."""
    if hasattr(layer, "recipe_constant_contract"):
        raise RuntimeError("Recipe constants must be prepared once on a fresh loaded layer")
    from habana_frameworks.torch import _core_C

    bindings = immutable_bindings(layer)
    if not bindings:
        raise RuntimeError("No immutable checkpoint projections to prepare")
    skipped, admitted = [], []
    for binding in bindings:
        _, _, tensor, name = binding
        if tensor.requires_grad:
            raise ValueError(f"Recipe constant is not an inference HPU operand: {name}")
        # Bound the capability to live two-dimensional MME matrices. The
        # installed cache first failed to restore a folded norm
        # constant; the narrowed matrix variant also failed cache reuse. Keep the shared FP8
        # encoding/scales and custom TPC scalar/vector inputs untouched.
        legacy_control = name in ("weights.hc_attn_fn", "weights.hc_ffn_fn",
                                  "hc_attn_fn_packed", "hc_ffn_fn_packed")
        if (tensor.ndim != 2 or min(tensor.shape) <= 1
                or tensor.dtype not in (torch.bfloat16, torch.float32) or legacy_control):
            skipped.append(dict(name=name, device=str(tensor.device),
                                reason="not a live BF16/FP32 MME matrix"))
            continue
        if (tensor.device.type != "hpu" or not tensor.is_contiguous() or tensor.storage_offset() != 0
                or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()):
            # Meta originals of prepared shared projections are intentionally
            # released. Mark their live joined buffer, never their placeholder.
            skipped.append(dict(name=name, device=str(tensor.device), reason="placeholder or storage view"))
            continue
        admitted.append(binding)
    if not admitted:
        raise RuntimeError("No full-storage resident immutable projection operands")
    torch.hpu.enable_inference_mode()
    # Keep the original logical tensor bytes for every compiled shape/layout.
    # Recipe-owned repacking can replace the device storage. Marking without
    # the stock original-constant serializer is not a reusable capture contract.
    global _constant_directory
    cache = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").split(",", 1)[0]
    if not cache:
        raise RuntimeError("Recipe constants require the isolated production recipe cache")
    directory = str(Path(cache) / "immutable-constant-sections")
    if _constant_directory is None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.hpu.enable_const_section_serialization(directory, False, False)
        _constant_directory = directory
    elif _constant_directory != directory:
        raise RuntimeError("Constant cache ownership changed in a loaded process")
    copies, records = {}, []
    for owner, leaf, tensor, name in admitted:
        key = id(tensor)
        if key not in copies:
            original = _core_C.get_tensor_extra_meta(tensor)
            original_mark = original.is_const_tensor, original.const_id
            prepared = tensor.clone()
            metadata = _core_C.get_tensor_extra_meta(prepared)
            metadata.const_id = next(_constant_ids)
            metadata.is_const_tensor = True
            if (original.is_const_tensor, original.const_id) != original_mark:
                raise RuntimeError("Constant clone unexpectedly changed baseline tensor metadata")
            copies[key] = prepared
        prepared = copies[key]
        setattr(owner, leaf, prepared)
        metadata = _core_C.get_tensor_extra_meta(prepared)
        records.append(dict(name=name, bytes=prepared.numel() * prepared.element_size(),
                            shape=list(prepared.shape), dtype=str(prepared.dtype),
                            constant_id=metadata.const_id, marked=metadata.is_const_tensor))
    # The attention entry reads the Python alias as well as its buffer owner.
    fused = layer.attention._buffers.get("fused_wqa_wkv")
    if fused is not None:
        layer.attention._fused_qkv_weight = fused
    compressor = layer.attention._buffers.get("fused_compressor_wkv_wgate")
    if compressor is not None:
        layer.attention._fused_compressor_weight = compressor
    layer.recipe_constant_contract = dict(
        layer=layer.layer, tensors=records, skipped=skipped,
        copied_bytes=sum(value.numel() * value.element_size() for value in copies.values()),
        excludes="all request/KV/Engram state and routed expert banks",
        numerical_path="unchanged dtypes, weights, operations and scalar parameters",
        original_constant_serialization=directory,
        compiled_constant_sections_verified=False,
    )
    return layer.recipe_constant_contract
