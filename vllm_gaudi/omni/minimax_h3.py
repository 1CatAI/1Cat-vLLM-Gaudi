# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax H3 checkpoint, FastH3, and HPU compatibility for Omni."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

_H3_SOURCE_PREFIXES = ("transformer.", "transformers_ref.", "text_encoder.")
_H3_PHASE_OFFLOAD_ENV = "VLLM_GAUDI_H3_PHASE_OFFLOAD"
_H3_DIT_TRACE_DIR_ENV = "VLLM_GAUDI_H3_DIT_TRACE_DIR"
_H3_DIT_TRACE_ARM_FILE_ENV = "VLLM_GAUDI_H3_DIT_TRACE_ARM_FILE"
_H3_DIT_TRACE_STEP_ENV = "VLLM_GAUDI_H3_DIT_TRACE_STEP"
_H3_BASE_OUTPUT_SHORT_EDGE = 768
_LIGHTX2V_REF_OUTPUT_SHORT_EDGE = 544
_PATCHED = False


class _EncoderOnlyOffloadConfig:
    """Delegate an Omni config while hiding non-DiT offload flags."""

    enable_cpu_offload = False
    enable_layerwise_offload = False
    enable_distributed_layerwise_offload = False

    def __init__(self, config: object) -> None:
        self._config = config

    def __getattr__(self, name: str) -> Any:
        return getattr(self._config, name)


def _validate_fasth3_quant_config(fusion: object | None, quant_config: object | None) -> None:
    if fusion is not None and bool(getattr(quant_config, "is_checkpoint_fp8_serialized", False)):
        raise ValueError("FastH3 must be fused into a BF16 MiniMax-H3 transformer; a serialized FP8 checkpoint "
                         "cannot be reconstructed with its old scales")


def _replace_projection(
    name: str,
    sources: tuple[tuple[str, str, str | int | None], ...],
) -> tuple[str, str | int | None] | None:
    for source, target, shard_id in sources:
        marker = f".{source}."
        if marker in name:
            return name.replace(marker, f".{target}."), shard_id
    return None


_DIT_PROJECTIONS = (
    ("attn.to_q", "attn.qkv_proj", "q"),
    ("attn.to_k", "attn.qkv_proj", "k"),
    ("attn.to_v", "attn.qkv_proj", "v"),
    ("attn.to_out.0", "attn.out_proj", None),
    ("attn.norm_q", "attn.q_norm", None),
    ("attn.norm_k", "attn.k_norm", None),
    ("ff.net.0.proj", "mlp.fc1", None),
    ("ff.net.2", "mlp.fc2", None),
)

_DIT_TOP_LEVEL = (
    ("audio_proj_in.", "audio_patch_proj."),
    ("audio_proj_out.", "final_layer.audio_out."),
    ("context_embedder.", "condition_proj."),
    ("norm_out.linear.", "final_layer.adaln_proj.linear."),
    ("norm_out.norm.", "final_layer.norm."),
    ("proj_in.", "video_patch_proj."),
    ("proj_out.", "final_layer.video_out."),
    ("time_embedder.linear_1.", "time_embedder.proj_in."),
    ("time_embedder.linear_2.", "time_embedder.proj_out."),
)


def map_minimax_h3_dit_weight(name: str) -> tuple[str, str | int | None]:
    """Map a Diffusers H3 DiT tensor to its vLLM parameter and shard."""

    for source, target in _DIT_TOP_LEVEL:
        if name.startswith(source):
            return target + name[len(source):], None

    if name.startswith("transformer_blocks."):
        name = "blocks." + name[len("transformer_blocks."):]
    elif name.startswith("token_refiner.refiner_blocks."):
        name = "token_refiner.blocks." + name[len("token_refiner.refiner_blocks."):]

    mapped = _replace_projection(name, _DIT_PROJECTIONS)
    return mapped if mapped is not None else (name, None)


_ENCODER_PROJECTIONS = (
    ("self_attn.q_proj", "self_attn.qkv_proj", "q"),
    ("self_attn.k_proj", "self_attn.qkv_proj", "k"),
    ("self_attn.v_proj", "self_attn.qkv_proj", "v"),
    ("mlp.gate_proj", "mlp.gate_up_proj", 0),
    ("mlp.up_proj", "mlp.gate_up_proj", 1),
)


def map_minimax_h3_encoder_weight(name: str) -> tuple[str, str | int | None] | None:
    """Map Qwen3-VL source tensors, including ModelOpt scale vectors."""

    if name in {"lm_head.weight", "model.language_model.norm.weight"}:
        return None
    if name.startswith("model.visual."):
        return "vision." + name[len("model.visual."):], None
    if not name.startswith("model.language_model."):
        return None

    rest = name[len("model.language_model."):]
    if rest.startswith("embed_tokens."):
        return "text_model." + rest, None
    match = re.match(r"layers\.(\d+)\.", rest)
    if match is not None:
        # H3 consumes the first 50 decoder layers and intentionally discards
        # any later layers from compatible Qwen3-VL exports.
        from vllm_omni.diffusion.models.minimax_h3.encoder import (
            MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER, )

        if int(match.group(1)) >= MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER:
            return None
    mapped = _replace_projection(rest, _ENCODER_PROJECTIONS)
    if mapped is not None:
        target, shard_id = mapped
        return "text_model." + target, shard_id
    return "text_model." + rest, None


def map_minimax_h3_pipeline_weight(name: str) -> tuple[str, str | int | None] | None:
    """Map a source tensor carrying its pipeline component prefix."""

    for prefix in _H3_SOURCE_PREFIXES:
        if not name.startswith(prefix):
            continue
        local_name = name[len(prefix):]
        if prefix == "text_encoder.":
            mapped = map_minimax_h3_encoder_weight(local_name)
        else:
            mapped = map_minimax_h3_dit_weight(local_name)
        if mapped is None:
            return None
        target, shard_id = mapped
        return prefix + target, shard_id
    return None


def _map_encoder_exclude_prefix(name: str) -> str:
    if name.startswith("model.language_model."):
        name = "text_encoder.text_model." + name[len("model.language_model."):]
    elif name.startswith("model.visual."):
        name = "text_encoder.vision." + name[len("model.visual."):]
    replacements = (
        (".self_attn.q_proj", ".self_attn.qkv_proj"),
        (".self_attn.k_proj", ".self_attn.qkv_proj"),
        (".self_attn.v_proj", ".self_attn.qkv_proj"),
        (".mlp.gate_proj", ".mlp.gate_up_proj"),
        (".mlp.up_proj", ".mlp.gate_up_proj"),
    )
    for source, target in replacements:
        name = name.replace(source, target)
    return name


def _map_dit_exclude_prefix(name: str) -> str:
    top_level = (
        ("audio_proj_in", "audio_patch_proj"),
        ("audio_proj_out", "final_layer.audio_out"),
        ("context_embedder", "condition_proj"),
        ("norm_out.linear", "final_layer.adaln_proj.linear"),
        ("norm_out.norm", "final_layer.norm"),
        ("proj_in", "video_patch_proj"),
        ("proj_out", "final_layer.video_out"),
        ("time_embedder.linear_1", "time_embedder.proj_in"),
        ("time_embedder.linear_2", "time_embedder.proj_out"),
    )
    for source, target in top_level:
        if name == source or name.startswith(source + ".") or name.startswith(source + "*"):
            return target + name[len(source):]

    if name.startswith("transformer_blocks."):
        name = "blocks." + name[len("transformer_blocks."):]
    elif name.startswith("token_refiner.refiner_blocks."):
        name = "token_refiner.blocks." + name[len("token_refiner.refiner_blocks."):]
    replacements = (
        (".attn.to_q", ".attn.qkv_proj"),
        (".attn.to_k", ".attn.qkv_proj"),
        (".attn.to_v", ".attn.qkv_proj"),
        (".attn.to_out.0", ".attn.out_proj"),
        (".attn.to_out", ".attn.out_proj"),
        (".attn.norm_q", ".attn.q_norm"),
        (".attn.norm_k", ".attn.k_norm"),
        (".ff.net.0.proj", ".mlp.fc1"),
        (".ff.net.0", ".mlp.fc1"),
        (".ff.net.2", ".mlp.fc2"),
        (".ff", ".mlp"),
    )
    for source, target in replacements:
        name = name.replace(source, target)
    return name


def _map_h3_dit_quant_config(quant_config: object | None) -> object | None:
    if quant_config is None or getattr(quant_config, "get_name", lambda: None)() != "modelopt":
        return quant_config
    if getattr(quant_config, "_vllm_gaudi_h3_names_mapped", False):
        return quant_config
    quant_config.exclude_modules = [_map_dit_exclude_prefix(str(name)) for name in quant_config.exclude_modules]
    quant_config._vllm_gaudi_h3_names_mapped = True
    return quant_config


def _resolve_h3_encoder_disk_quant_config(
    model_path: str,
    fallback: object | None,
) -> object | None:
    """Build the encoder's serialized ModelOpt config from its partition."""

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return fallback
    with config_path.open(encoding="utf-8") as config_file:
        disk_config = json.load(config_file).get("quantization_config")
    if not isinstance(disk_config, dict):
        return fallback
    method = str(disk_config.get("quant_method", "")).lower()
    algorithm = str(disk_config.get("quant_algo", "")).upper()
    if method != "modelopt" or algorithm != "FP8_PER_CHANNEL_PER_TOKEN":
        return fallback

    # The Omni general plugin can run before vLLM invokes the HPU ops plugin.
    # Importing this module here makes the ModelOpt class replacement explicit
    # before the encoder creates any LinearBase parameters.
    import vllm_gaudi.ops.hpu_modelopt  # noqa: F401
    from vllm.model_executor.layers.quantization import modelopt

    mapped_config = dict(disk_config)
    mapped_config["ignore"] = [_map_encoder_exclude_prefix(str(name)) for name in disk_config.get("ignore", [])]
    quant_config = modelopt.ModelOptFp8Config.from_config(mapped_config)
    logger.info_once(
        "MiniMax H3 text encoder uses its native %s checkpoint config with %d full-precision prefixes",
        algorithm,
        len(mapped_config["ignore"]),
    )
    return quant_config


def _load_minimax_h3_dit_weights(
    self: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load raw Diffusers or already-normalized H3 DiT tensors."""

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        _reorder_grouped_qkv_to_qkv, )

    params = dict(self.named_parameters())
    params.update(dict(self.named_buffers()))
    loaded: set[str] = set()
    loaded_shards: dict[str, set[str | int]] = {}

    for source_name, loaded_weight in weights:
        name, shard_id = map_minimax_h3_dit_weight(source_name)
        param = params.get(name)
        if param is None:
            logger.warning("Skipping MiniMax H3 weight not present in model: %s", source_name)
            continue
        weight_loader = getattr(param, "weight_loader", default_weight_loader)

        if shard_id is not None:
            weight_loader(param, loaded_weight, shard_id)
            loaded_shards.setdefault(name, set()).add(shard_id)
        elif name.endswith(".attn.qkv_proj.weight"):
            # This branch accepts an already-fused legacy checkpoint. Native
            # H3 releases use the separate Q/K/V branch above.
            loaded_weight = _reorder_grouped_qkv_to_qkv(
                loaded_weight,
                num_query_groups=self.arch.num_attention_heads,
                heads_per_group=1,
                head_dim=self.arch.attention_head_dim,
            )
            weight_loader(param, loaded_weight)
        elif name.endswith((".mlp.fc1.weight", ".mlp.fc1.weight_scale")):
            if loaded_weight.shape[0] % 2:
                raise ValueError("MiniMax H3 fc1 checkpoint rows must split evenly into "
                                 f"gate/up tensors, got {tuple(loaded_weight.shape)}")
            gate, up = loaded_weight.chunk(2, dim=0)
            weight_loader(param, gate, 0)
            weight_loader(param, up, 1)
            loaded_shards.setdefault(name, set()).update((0, 1))
        else:
            weight_loader(param, loaded_weight)
        loaded.add(name)

    for name, shards in loaded_shards.items():
        expected: set[str | int]
        if ".attn.qkv_proj." in name:
            expected = {"q", "k", "v"}
        elif ".mlp.fc1." in name:
            expected = {0, 1}
        else:
            continue
        if shards != expected:
            raise RuntimeError(f"MiniMax H3 fused parameter {name!r} has shards {sorted(map(str, shards))}; "
                               f"expected {sorted(map(str, expected))}")
    return loaded


def _load_minimax_h3_encoder_weights(
    self: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load Qwen3-VL weights and validate every fused weight and scale shard."""

    params = dict(self.named_parameters())
    loaded: set[str] = set()
    expected_fused: dict[str, set[str | int]] = {}
    for module_name, module in self.named_modules():
        class_name = type(module).__name__
        if class_name == "MiniMaxH3Qwen3VLQKVParallelLinear":
            shards: set[str | int] = {"q", "k", "v"}
        elif class_name == "MiniMaxH3Qwen3VLMergedColumnParallelLinear":
            shards = {0, 1}
        else:
            continue
        expected_fused[f"{module_name}.weight"] = shards
        if f"{module_name}.weight_scale" in params:
            expected_fused[f"{module_name}.weight_scale"] = shards
    loaded_fused = {name: set() for name in expected_fused}

    for source_name, tensor in weights:
        mapped = map_minimax_h3_encoder_weight(source_name)
        if mapped is None:
            continue
        param_name, shard_id = mapped
        param = params.get(param_name)
        if param is None:
            logger.warning("MiniMax H3 text encoder weight %s has no target parameter", source_name)
            continue
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        if shard_id is None:
            weight_loader(param, tensor)
        else:
            weight_loader(param, tensor, shard_id)
        loaded.add(param_name)
        if shard_id is not None and param_name in loaded_fused:
            loaded_fused[param_name].add(shard_id)

    missing_shards = {
        name: sorted(map(str, expected - loaded_fused[name]))
        for name, expected in expected_fused.items() if expected - loaded_fused[name]
    }
    if missing_shards:
        details = "; ".join(f"{name}: {shards}" for name, shards in sorted(missing_shards.items()))
        raise RuntimeError(f"MiniMax H3 text encoder fused tensors are missing source shards: {details}")
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(f"MiniMax H3 text encoder weights not loaded: {len(missing)} params: {missing}")
    return loaded


def _h3_row_parallel_weight_loader(
    self: nn.Module,
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: str | None = None,
) -> None:
    """Load row-parallel weights and their unsharded output-channel scales."""

    del loaded_shard_id
    if param.ndim == 1:
        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"MiniMax H3 row-parallel scale shape mismatch: {tuple(loaded_weight.shape)} != {tuple(param.shape)}")
        param.data.copy_(loaded_weight)
        return
    shard_size = self.input_size_per_partition
    start_idx = self._tp_rank * shard_size
    param.data.copy_(loaded_weight.narrow(1, start_idx, shard_size))


def _install_checkpoint_adapter() -> None:
    from vllm_omni.diffusion.model_loader import checkpoint_adapters
    from vllm_omni.diffusion.model_loader.checkpoint_adapters.modelopt import (
        ModelOptFp8CheckpointAdapter, )

    original = checkpoint_adapters.get_checkpoint_adapter

    class MiniMaxH3ModelOptFp8CheckpointAdapter(ModelOptFp8CheckpointAdapter):
        """Resolve H3 component names before ModelOpt dequantization."""

        def _resolve_target_and_output_names(self, name: str) -> tuple[str | None, str]:
            mapped = map_minimax_h3_pipeline_weight(name)
            if mapped is None:
                return None, name
            target_name, _ = mapped
            if target_name in self._loadable_tensors:
                # Keep the source name so the component loader can route each
                # Q/K/V and gate/up shard independently.
                return target_name, name
            return None, name

    def get_checkpoint_adapter(
        model: nn.Module,
        source: object,
        quant_config: object | None,
        use_safetensors: bool,
    ) -> Any:
        prefix = str(getattr(source, "prefix", ""))
        is_h3 = type(model).__module__.startswith("vllm_omni.diffusion.models.minimax_h3")
        if (is_h3 and prefix.startswith(_H3_SOURCE_PREFIXES)
                and ModelOptFp8CheckpointAdapter._is_checkpoint_quant_config(quant_config) and use_safetensors):
            return MiniMaxH3ModelOptFp8CheckpointAdapter(model, source)
        return original(model, source, quant_config, use_safetensors)

    checkpoint_adapters.get_checkpoint_adapter = get_checkpoint_adapter
    # diffusers_loader imports the function into its module namespace.
    from vllm_omni.diffusion.model_loader import diffusers_loader

    diffusers_loader.get_checkpoint_adapter = get_checkpoint_adapter


def _install_h3_modulation_ops() -> None:
    """Replace H3's CUDA-Triton modulation bindings with HPU operators."""

    from vllm_gaudi.omni import minimax_h3_ops as hpu_ops
    from vllm_omni.diffusion.attention.ops import minimax_h3_modulation
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer

    for name in hpu_ops.__all__:
        implementation = getattr(hpu_ops, name)
        # The Transformer imports these functions by name, so replace both
        # the source module and its already-bound globals.
        setattr(minimax_h3_modulation, name, implementation)
        setattr(minimax_h3_transformer, name, implementation)


def _hpu_fused_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    """Run Habana's memory-efficient SDPA without a host fallback."""

    if any(tensor.device.type != "hpu" for tensor in (query, key, value)):
        raise RuntimeError("MiniMax H3 HPU FusedSDPA requires HPU query, key, and value tensors")
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
    from vllm_gaudi.omni.attention import HPU_SDPA_SOFTMAX_MODE

    return FusedSDPA.apply(
        query,
        key,
        value,
        None,
        0.0,
        is_causal,
        scale,
        HPU_SDPA_SOFTMAX_MODE,
        True,
    )


def _install_h3_encoder_attention_hooks() -> None:
    """Use HPU FusedSDPA for H3's Qwen3-VL vision and text attention."""

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    vision_class = encoder.MiniMaxH3Qwen3VLVisionAttention
    original_vision_forward = vision_class.forward
    original_text_attention = encoder._scaled_dot_product_attention

    def vision_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if hidden_states.device.type != "hpu":
            return original_vision_forward(self, hidden_states, cu_seqlens, position_embeddings)
        if position_embeddings is None:
            raise ValueError("MiniMax H3 vision attention requires rotary position embeddings")

        seq_length = int(hidden_states.shape[0])
        query, key, value = (self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2,
                                                                                                        3).unbind(0))
        cos, sin = position_embeddings
        query, key = encoder._apply_rotary_pos_emb_vision(query, key, cos, sin)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to("cpu").tolist()
        splits = [torch.split(tensor, lengths, dim=2) for tensor in (query, key, value)]
        outputs = [_hpu_fused_sdpa(q, k, v, is_causal=False, scale=self.scaling) for q, k, v in zip(*splits)]
        output = torch.cat(outputs, dim=2).transpose(1, 2).contiguous().reshape(seq_length, -1).contiguous()
        return self.proj(output)

    def text_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if query.device.type != "hpu":
            return original_text_attention(query, key, value)
        groups = int(query.shape[1]) // int(key.shape[1])
        if groups != 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        return _hpu_fused_sdpa(query, key, value, is_causal=True, scale=None)

    vision_class.forward = vision_forward
    encoder._scaled_dot_product_attention = text_attention
    logger.info_once("Enabled HPU FusedSDPA for MiniMax H3 Qwen3-VL vision and text attention")


def _lightx2v_ref_output_canvas(aspect_ratio: float) -> tuple[int, int]:
    """Resolve LightX2V Ref2V's published 544p canvas on a 32px grid."""

    from vllm_omni.errors import OmniClientError
    from vllm_omni.model_executor.models.minimax_h3.preprocessing import MINIMAX_H3_OUTPUT_MAX_PIXELS

    if not math.isfinite(float(aspect_ratio)) or float(aspect_ratio) <= 0:
        raise OmniClientError(f"MiniMax H3 canvas aspect ratio must be positive, got {aspect_ratio!r}")
    if aspect_ratio >= 1.0:
        width = float(_LIGHTX2V_REF_OUTPUT_SHORT_EDGE) * aspect_ratio
        height = float(_LIGHTX2V_REF_OUTPUT_SHORT_EDGE)
    else:
        width = float(_LIGHTX2V_REF_OUTPUT_SHORT_EDGE)
        height = float(_LIGHTX2V_REF_OUTPUT_SHORT_EDGE) / aspect_ratio
    area = width * height
    if area > MINIMAX_H3_OUTPUT_MAX_PIXELS:
        resize = (MINIMAX_H3_OUTPUT_MAX_PIXELS / area)**0.5
        width *= resize
        height *= resize

    def align(value: float) -> int:
        return max(32, int(round(value / 32)) * 32)

    return align(height), align(width)


def _validate_h3_output_short_edge(partition: str, task: str, short_edge: object) -> None:
    """Keep the 544p LightX2V release scoped to the Ref2VA partition."""

    if short_edge != _LIGHTX2V_REF_OUTPUT_SHORT_EDGE:
        return
    if partition == "ref2va" and task == "ref2va":
        return
    from vllm_omni.errors import OmniClientError

    raise OmniClientError("MiniMax H3 target.short_edge=544 requires the Ref2VA partition and task")


def _install_h3_ref2va_canvas_hook() -> None:
    """Allow the official LightX2V Ref2V 544p profile in pinned Omni."""

    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline

    original_canvas = pipeline._resolve_output_canvas
    original_resolve_shape = pipeline.MiniMaxH3Pipeline._resolve_shape

    def resolve_output_canvas(aspect_ratio: float, short_edge: int) -> tuple[int, int]:
        if short_edge == _LIGHTX2V_REF_OUTPUT_SHORT_EDGE:
            return _lightx2v_ref_output_canvas(aspect_ratio)
        return original_canvas(aspect_ratio, short_edge)

    def resolve_shape(
        self: nn.Module,
        task: str,
        sampling: Any,
        image: Any | None,
    ) -> tuple[int, int, int, int, int]:
        extra = sampling.extra_args or {}
        target = extra.get("target")
        target = target if isinstance(target, Mapping) else {}
        short_edge = target.get("short_edge", extra.get("short_edge", _H3_BASE_OUTPUT_SHORT_EDGE))
        _validate_h3_output_short_edge(str(self.partition), task, short_edge)
        return original_resolve_shape(self, task, sampling, image)

    pipeline._resolve_output_canvas = resolve_output_canvas
    pipeline.MiniMaxH3Pipeline._resolve_shape = resolve_shape
    logger.info_once("Enabled MiniMax H3 Ref2VA 544p canvas support for the LightX2V four-step profile")


def _parse_ffmpeg_video_encoders(output: str) -> frozenset[str]:
    """Extract video encoder names from ``ffmpeg -encoders`` output."""

    encoders = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and len(fields[0]) == 6 and fields[0].startswith("V"):
            encoders.add(fields[1])
    return frozenset(encoders)


def _select_h3_reference_video_codec(encoders: Iterable[str]) -> str:
    """Prefer upstream RGB H.264, with Habana's lossless FFV1 as fallback."""

    available = frozenset(encoders)
    if "libx264rgb" in available:
        return "libx264rgb"
    if "ffv1" in available:
        return "ffv1"
    raise RuntimeError("MiniMax H3 reference-video preparation requires the libx264rgb or ffv1 ffmpeg encoder")


def _h3_ffv1_reference_video_command(
    source: str,
    *,
    target_width: int,
    target_height: int,
    target_frame_count: int,
    workdir: str,
    fps: float,
    start_time_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> tuple[str, list[str]]:
    """Build Habana ffmpeg's lossless RGB reference-video preparation."""

    output = str(Path(workdir) / "prepared.mkv")
    duration_args = ["-t", f"{float(duration_seconds):.6f}"] if duration_seconds is not None else []
    frame_count_args = ["-frames:v", str(int(target_frame_count))] if target_frame_count > 0 else []
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{float(start_time_seconds):.6f}",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps={fps:g},scale={target_width}:{target_height}:flags=lanczos,setsar=1,format=bgr0",
        *duration_args,
        *frame_count_args,
        "-metadata:s:v:0",
        "rotate=0",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-coder",
        "1",
        "-context",
        "1",
        "-g",
        "1",
        "-slicecrc",
        "1",
        "-pix_fmt",
        "bgr0",
        output,
    ]
    return output, command


def _install_h3_reference_video_transcode_hook() -> None:
    """Use FFV1 when Habana ffmpeg omits the upstream libx264rgb encoder."""

    from vllm_omni.model_executor.models.minimax_h3 import reference_video

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
        codec = _select_h3_reference_video_codec(_parse_ffmpeg_video_encoders(result.stdout))
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        logger.warning_once("Could not qualify MiniMax H3 reference-video ffmpeg encoder: %s", exc)
        return
    if codec == "libx264rgb":
        return

    def transcode_reference_video(
        source: str,
        *,
        target_width: int,
        target_height: int,
        target_frame_count: int,
        workdir: str,
        start_time_seconds: float = 0.0,
        duration_seconds: float | None = None,
    ) -> str:
        output, command = _h3_ffv1_reference_video_command(
            source,
            target_width=target_width,
            target_height=target_height,
            target_frame_count=target_frame_count,
            workdir=workdir,
            fps=reference_video.MINIMAX_H3_FPS,
            start_time_seconds=start_time_seconds,
            duration_seconds=duration_seconds,
        )
        subprocess.run(command, check=True)
        return output

    reference_video._transcode_reference_video = transcode_reference_video
    logger.info_once("Enabled lossless FFV1 MiniMax H3 reference-video preparation for Habana ffmpeg")


def _install_fasth3_hooks() -> None:
    """Keep FastH3 fusion on a BF16 source and allow encoder offload."""

    from vllm_omni.diffusion.models.minimax_h3 import fasth3
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline
    from vllm_omni.diffusion.offloader.config import DIT_COMPONENT, resolve_offload

    original_resolve = pipeline.resolve_fasth3_fusion
    original_contract = fasth3.FastH3WeightFusion.check_serving_contract

    def resolve_fasth3_fusion(od_config: object, transformer: nn.Module):
        fusion = original_resolve(od_config, transformer)
        quant_config = getattr(transformer, "_vllm_gaudi_quant_config", None)
        _validate_fasth3_quant_config(fusion, quant_config)
        return fusion

    def check_serving_contract(
        self,
        *,
        partition: str,
        od_config: object,
        video_shift: float,
        audio_shift: float,
    ) -> None:
        resolved = resolve_offload(od_config)
        if not resolved.offloads(DIT_COMPONENT):
            # Upstream rejects the boolean offload flags because a DiT host
            # weight plan bypasses load_weights(), where FastH3 is fused. H3's
            # component-aware plan can offload only the text encoder while the
            # DiT still follows that ordinary streaming path.
            od_config = _EncoderOnlyOffloadConfig(od_config)
        original_contract(
            self,
            partition=partition,
            od_config=od_config,
            video_shift=video_shift,
            audio_shift=audio_shift,
        )

    pipeline.resolve_fasth3_fusion = resolve_fasth3_fusion
    fasth3.resolve_fasth3_fusion = resolve_fasth3_fusion
    fasth3.FastH3WeightFusion.check_serving_contract = check_serving_contract


@lru_cache(maxsize=4)
def _minimax_h3_unpatchify_indices(
    batch: int,
    t: int,
    h: int,
    w: int,
    channel: int,
    pt: int,
    ph: int,
    pw: int,
) -> torch.Tensor:
    """Build the flat source indices for the H3 token-to-latent layout."""

    source = torch.arange(batch * t * h * w * channel * pt * ph * pw, dtype=torch.long)
    source = source.reshape(batch, t, h, w, channel, pt, ph, pw)
    return source.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().reshape(-1)


def _minimax_h3_unpatchify_video_tokens_hpu(
    rows: torch.Tensor,
    *,
    latent_shape: Sequence[int],
    patch_size: Sequence[int],
) -> torch.Tensor:
    """Unpatchify H3 rows without HPU's incorrect permutation-only einsum."""

    if rows.ndim != 2:
        raise ValueError(f"video token rows must be rank 2, got shape={list(rows.shape)}")
    if len(latent_shape) != 4 or len(patch_size) != 3:
        raise ValueError("latent_shape and patch_size must contain four and three values")
    t, h, w, channel = (int(value) for value in latent_shape)
    pt, ph, pw = (int(value) for value in patch_size)
    if min(t, h, w, channel, pt, ph, pw) <= 0:
        raise ValueError("latent_shape and patch_size values must be positive")
    expected_dim = channel * pt * ph * pw
    if int(rows.shape[1]) != expected_dim:
        raise ValueError(f"video token dim {int(rows.shape[1])} != patch volume * channel {expected_dim}")
    rows_per_sample = t * h * w
    if int(rows.shape[0]) % rows_per_sample:
        raise ValueError(f"video rows {int(rows.shape[0])} must be divisible by t*h*w {rows_per_sample}")
    batch = int(rows.shape[0]) // rows_per_sample
    indices = _minimax_h3_unpatchify_indices(batch, t, h, w, channel, pt, ph, pw).to(rows.device)
    latent = rows.reshape(-1).index_select(0, indices)
    return latent.reshape(batch, channel, t * pt, h * ph, w * pw).contiguous()


def _h3_phase_offload_enabled() -> bool:
    value = os.environ.get(_H3_PHASE_OFFLOAD_ENV, "0").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{_H3_PHASE_OFFLOAD_ENV} must be a boolean value, got {value!r}")


def _move_module_tensors(module: nn.Module, device: torch.device) -> tuple[int, int]:
    """Move one H3 component without recursively following module references."""

    seen: set[int] = set()
    tensor_count = 0
    byte_count = 0
    for tensor in (*module.parameters(), *module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        if tensor.device == device:
            continue
        byte_count += tensor.numel() * tensor.element_size()
        tensor.data = tensor.data.to(device=device, non_blocking=False)
        tensor_count += 1
    return tensor_count, byte_count


def _record_h3_phase_duration(owner: object, name: str, duration: float) -> None:
    durations = getattr(owner, "_stage_durations", None)
    lock = getattr(owner, "_profiler_lock", None)
    if not isinstance(durations, dict) or lock is None:
        return
    metric = f"MiniMaxH3Pipeline.hpu_phase.{name}"
    with lock:
        durations[metric] = float(durations.get(metric, 0.0)) + duration


def _move_h3_phase_component(
    owner: nn.Module,
    component: nn.Module,
    device: torch.device,
    *,
    metric: str,
) -> None:
    """Synchronously move a phase component and expose transfer time to Omni."""

    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.synchronize()
    started = time.perf_counter()
    tensor_count, byte_count = _move_module_tensors(component, device)
    current_omni_platform.synchronize()
    if device.type == "cpu" and tensor_count:
        current_omni_platform.empty_cache()
    duration = time.perf_counter() - started
    _record_h3_phase_duration(owner, metric, duration)
    if tensor_count:
        logger.info(
            "MiniMax H3 phase transfer %s: %d tensors, %.3f GiB, %.3f seconds",
            metric,
            tensor_count,
            byte_count / (1024**3),
            duration,
        )


def _install_h3_phase_offload_hooks() -> None:
    """Keep VAEs on CPU and swap the resident DiT once per request phase."""

    if not _h3_phase_offload_enabled():
        return

    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline

    pipeline_class = pipeline.MiniMaxH3Pipeline
    video_vae_class = pipeline.MiniMaxH3VideoVAE
    audio_vae_class = pipeline.MiniMaxH3AudioVAE
    original_video_vae_init = video_vae_class.__init__
    original_audio_vae_init = audio_vae_class.__init__
    original_uses_manual_offload = pipeline_class._uses_manual_component_offload
    original_component_context = pipeline_class._component_on_device
    original_encode_prompt = pipeline_class.encode_prompt
    original_diffuse = pipeline_class.diffuse

    def video_vae_init(
        self: nn.Module,
        component_path: str,
        *,
        device: torch.device,
        load_device: torch.device | None = None,
    ) -> None:
        del load_device
        original_video_vae_init(self, component_path, device=device, load_device=torch.device("cpu"))

    def audio_vae_init(
        self: nn.Module,
        component_path: str,
        *,
        device: torch.device,
        load_device: torch.device | None = None,
    ) -> None:
        del load_device
        original_audio_vae_init(self, component_path, device=device, load_device=torch.device("cpu"))

    def uses_manual_component_offload(self: nn.Module, component: nn.Module) -> bool:
        if component in (getattr(self, "video_vae", None), getattr(self, "audio_vae", None)):
            return True
        return original_uses_manual_offload(self, component)

    @contextmanager
    def component_on_device(self: nn.Module, component: nn.Module):
        is_phase_vae = component in (getattr(self, "video_vae", None), getattr(self, "audio_vae", None))
        if not is_phase_vae:
            with original_component_context(self, component) as value:
                yield value
            return

        component_name = "video_vae" if component is getattr(self, "video_vae", None) else "audio_vae"
        manager = original_component_context(self, component)
        started = time.perf_counter()
        value = manager.__enter__()
        _record_h3_phase_duration(self, f"load_{component_name}", time.perf_counter() - started)
        try:
            yield value
        except BaseException as exc:
            started = time.perf_counter()
            suppress = manager.__exit__(type(exc), exc, exc.__traceback__)
            _record_h3_phase_duration(self, f"offload_{component_name}", time.perf_counter() - started)
            if not suppress:
                raise
        else:
            started = time.perf_counter()
            manager.__exit__(None, None, None)
            _record_h3_phase_duration(self, f"offload_{component_name}", time.perf_counter() - started)

    def encode_prompt(self: nn.Module, *args: Any, **kwargs: Any):
        task = kwargs.get("task")
        if task is None:
            raise TypeError("MiniMax H3 phase-offloaded encode_prompt requires task as a keyword")
        # The first request starts with the DiT on HPU. Ref2VA's 2048-short-edge
        # visual encoder needs more transient HBM than can coexist with that
        # 66 GB transformer, so establish the same CPU-resident state that all
        # later requests inherit after diffuse().
        transformer = self._transformer_for_task(task)
        _move_h3_phase_component(self, transformer, torch.device("cpu"), metric="offload_dit_for_encode")
        return original_encode_prompt(self, *args, **kwargs)

    def diffuse(self: nn.Module, *args: Any, **kwargs: Any):
        task = kwargs.get("task")
        if task is None:
            raise TypeError("MiniMax H3 phase-offloaded diffuse requires task as a keyword")
        transformer = self._transformer_for_task(task)
        _move_h3_phase_component(self, transformer, self.device, metric="load_dit")
        try:
            return original_diffuse(self, *args, **kwargs)
        finally:
            _move_h3_phase_component(self, transformer, torch.device("cpu"), metric="offload_dit")

    video_vae_class.__init__ = video_vae_init
    audio_vae_class.__init__ = audio_vae_init
    pipeline_class._uses_manual_component_offload = uses_manual_component_offload
    pipeline_class._component_on_device = component_on_device
    pipeline_class.encode_prompt = encode_prompt
    pipeline_class.diffuse = diffuse
    for target in ("decode_to_mp4", "video_vae.decode_with_chunks"):
        if target not in pipeline_class._PROFILER_TARGETS:
            pipeline_class._PROFILER_TARGETS.append(target)
    logger.info_once("MiniMax H3 single-HPU phase offload is enabled for DiT and both VAEs")


def _minimax_h3_initial_noise(
    self: nn.Module,
    *,
    seed: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw video then audio noise from FastVideo's shared CPU generator."""

    del self
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent, )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    video = torch.randn(
        1,
        24,
        latent_t,
        latent_h,
        latent_w,
        generator=generator,
        dtype=torch.float32,
    )
    video_rows = minimax_h3_patchify_video_latent(
        video,
        patch_size=(1, 2, 2),
    )
    audio_rows = torch.randn(
        audio_t * 2,
        32,
        generator=generator,
        dtype=torch.float32,
    )
    return video_rows, audio_rows


def _install_h3_layout_and_noise_hooks() -> None:
    """Install HPU-safe latent layout and the official H3 RNG sequence."""

    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline
    from vllm_omni.diffusion.models.minimax_h3 import packed_tokens

    original_unpatchify = packed_tokens.minimax_h3_unpatchify_video_tokens

    def unpatchify_video_tokens(
        rows: torch.Tensor,
        *,
        latent_shape: Sequence[int],
        patch_size: Sequence[int],
    ) -> torch.Tensor:
        if rows.device.type != "hpu":
            return original_unpatchify(rows, latent_shape=latent_shape, patch_size=patch_size)
        return _minimax_h3_unpatchify_video_tokens_hpu(
            rows,
            latent_shape=latent_shape,
            patch_size=patch_size,
        )

    packed_tokens.minimax_h3_unpatchify_video_tokens = unpatchify_video_tokens
    pipeline.minimax_h3_unpatchify_video_tokens = unpatchify_video_tokens
    pipeline.MiniMaxH3Pipeline._initial_noise = _minimax_h3_initial_noise


def _summarize_debug_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Copy a compact numerical summary for an opt-in H3 trajectory trace."""

    values = tensor.detach().float()
    finite = torch.isfinite(values)
    safe = torch.where(finite, values, 0.0)
    scalars = torch.stack((
        safe.amin(),
        safe.amax(),
        safe.mean(),
        safe.std(),
        safe.square().mean().sqrt(),
        finite.float().mean(),
    )).cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(scalars[0]),
        "max": float(scalars[1]),
        "mean": float(scalars[2]),
        "std": float(scalars[3]),
        "rms": float(scalars[4]),
        "finite_fraction": float(scalars[5]),
    }


def _install_h3_trajectory_debug_hook() -> None:
    """Record H3 denoiser values when an external evidence path is set."""

    output_value = os.environ.get("VLLM_GAUDI_H3_TRAJECTORY_DIR")
    if not output_value:
        return

    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline

    output_dir = Path(output_value).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_loop = pipeline.minimax_h3_denoise_loop
    request_index = 0

    def debug_loop(**kwargs: Any):
        nonlocal request_index
        request_index += 1
        current_index = request_index
        model = kwargs["model"]
        original_on_step = kwargs.get("on_step")
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "request_index": current_index,
            "pid": os.getpid(),
            "sigmas_video": list(kwargs["sigmas_video"]),
            "sigmas_audio": list(kwargs["sigmas_audio"]),
            "initial_video_rows": _summarize_debug_tensor(kwargs["initial_video_rows"]),
            "initial_audio_rows": _summarize_debug_tensor(kwargs["initial_audio_rows"]),
            "model_outputs": [],
            "steps": [],
        }
        retained: dict[str, Any] = {
            "initial_video_rows": kwargs["initial_video_rows"].detach().float().cpu(),
            "initial_audio_rows": kwargs["initial_audio_rows"].detach().float().cpu(),
        }

        def observed_model(**model_kwargs: Any):
            result = model(**model_kwargs)
            video, audio = result
            report["model_outputs"].append({
                "step":
                len(report["model_outputs"]),
                "video":
                _summarize_debug_tensor(video),
                "audio":
                _summarize_debug_tensor(audio),
                "unique_timesteps":
                model_kwargs["unique_timesteps"].detach().float().cpu().tolist(),
            })
            return result

        def observed_step(step: int, video: torch.Tensor, audio: torch.Tensor) -> None:
            report["steps"].append({
                "step": step,
                "video_rows": _summarize_debug_tensor(video),
                "audio_rows": _summarize_debug_tensor(audio),
            })
            retained[f"video_rows_step_{step}"] = video.detach().float().cpu()
            retained[f"audio_rows_step_{step}"] = audio.detach().float().cpu()
            if original_on_step is not None:
                original_on_step(step, video, audio)

        observed_kwargs = dict(kwargs)
        observed_kwargs["model"] = observed_model
        observed_kwargs["on_step"] = observed_step
        stem = f"trajectory-{os.getpid()}-{current_index:03d}"
        try:
            result = original_loop(**observed_kwargs)
            report["final_video_rows"] = _summarize_debug_tensor(result[0])
            report["final_audio_rows"] = _summarize_debug_tensor(result[1])
            retained["final_video_rows"] = result[0].detach().float().cpu()
            retained["final_audio_rows"] = result[1].detach().float().cpu()
            report["status"] = "pass"
            return result
        except BaseException as exc:
            report["status"] = "fail"
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            torch.save(retained, output_dir / f"{stem}.pt")

    pipeline.minimax_h3_denoise_loop = debug_loop
    logger.warning("MiniMax H3 trajectory evidence is enabled at %s", output_dir)


def _write_h3_trace_metadata(path: Path, report: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summarize_h3_trace_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Mapping):
        return {str(key): _summarize_h3_trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_h3_trace_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def _install_h3_dit_trace_hook() -> None:
    """Capture one armed H3 DiT call with CPU and HPU hardware events."""

    output_value = os.environ.get(_H3_DIT_TRACE_DIR_ENV)
    if not output_value:
        return

    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline

    output_dir = Path(output_value).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_value = os.environ.get(_H3_DIT_TRACE_ARM_FILE_ENV)
    arm_file = Path(arm_value).expanduser().resolve() if arm_value else output_dir / "ARM"
    try:
        target_step = int(os.environ.get(_H3_DIT_TRACE_STEP_ENV, "0"))
    except ValueError as exc:
        raise ValueError(f"{_H3_DIT_TRACE_STEP_ENV} must be a non-negative integer") from exc
    if target_step < 0:
        raise ValueError(f"{_H3_DIT_TRACE_STEP_ENV} must be a non-negative integer")

    original_loop = pipeline.minimax_h3_denoise_loop

    def trace_loop(**kwargs: Any):
        claim_file = output_dir / f"armed-{os.getpid()}-{time.time_ns()}"
        try:
            arm_file.replace(claim_file)
        except FileNotFoundError:
            return original_loop(**kwargs)

        original_model = kwargs["model"]
        call_index = 0
        trace_stem = f"dit-step-{os.getpid()}-{time.time_ns()}-step{target_step}"
        trace_path = output_dir / f"{trace_stem}.json"
        metadata_path = output_dir / f"{trace_stem}.metadata.json"
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "armed",
            "pid": os.getpid(),
            "target_step": target_step,
            "trace_path": str(trace_path),
            "arm_file": str(arm_file),
            "claim_file": str(claim_file),
            "claimed_at": datetime.now(timezone.utc).isoformat(),
            "sigmas_video": list(kwargs["sigmas_video"]),
            "sigmas_audio": list(kwargs["sigmas_audio"]),
        }
        _write_h3_trace_metadata(metadata_path, report)

        def traced_model(**model_kwargs: Any):
            nonlocal call_index
            current_step = call_index
            call_index += 1
            if current_step != target_step:
                return original_model(**model_kwargs)

            report["status"] = "capturing"
            report["model_inputs"] = _summarize_h3_trace_value(model_kwargs)
            report["capture_started_at"] = datetime.now(timezone.utc).isoformat()
            _write_h3_trace_metadata(metadata_path, report)
            torch.hpu.synchronize()
            try:
                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.HPU,
                        ],
                        record_shapes=True,
                        profile_memory=False,
                        with_stack=False,
                ) as profiler, torch.profiler.record_function(f"minimax_h3_dit_model_step_{current_step}"):
                    started = time.perf_counter()
                    result = original_model(**model_kwargs)
                    torch.hpu.synchronize()
                    report["model_wall_seconds"] = time.perf_counter() - started
                report["status"] = "exporting"
                report["capture_stopped_at"] = datetime.now(timezone.utc).isoformat()
                _write_h3_trace_metadata(metadata_path, report)
                profiler.export_chrome_trace(str(trace_path))
                report["status"] = "pass"
                report["trace_bytes"] = trace_path.stat().st_size
                report["export_finished_at"] = datetime.now(timezone.utc).isoformat()
                _write_h3_trace_metadata(metadata_path, report)
                logger.warning("MiniMax H3 DiT hardware trace exported to %s", trace_path)
                return result
            except BaseException as exc:
                report["status"] = "fail"
                report["error"] = f"{type(exc).__name__}: {exc}"
                report["failed_at"] = datetime.now(timezone.utc).isoformat()
                _write_h3_trace_metadata(metadata_path, report)
                raise

        traced_kwargs = dict(kwargs)
        traced_kwargs["model"] = traced_model
        try:
            result = original_loop(**traced_kwargs)
        except BaseException as exc:
            if report["status"] == "armed":
                report["status"] = "fail"
                report["error"] = f"{type(exc).__name__}: {exc}"
                report["failed_at"] = datetime.now(timezone.utc).isoformat()
                _write_h3_trace_metadata(metadata_path, report)
            raise
        if report["status"] == "armed":
            report["status"] = "fail"
            report["error"] = f"target step {target_step} was not executed; observed {call_index} DiT calls"
            report["failed_at"] = datetime.now(timezone.utc).isoformat()
            _write_h3_trace_metadata(metadata_path, report)
        return result

    pipeline.minimax_h3_denoise_loop = trace_loop
    logger.warning(
        "MiniMax H3 one-step hardware tracing is enabled at %s; create %s to arm the next request",
        output_dir,
        arm_file,
    )


def install_minimax_h3_patches() -> None:
    """Install idempotent H3 hooks at Omni worker startup."""

    global _PATCHED
    if _PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder as h3_encoder
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel, )

    MiniMaxH3Qwen3VLEncoder = h3_encoder.MiniMaxH3Qwen3VLEncoder
    MiniMaxH3Qwen3VLRowParallelLinear = h3_encoder.MiniMaxH3Qwen3VLRowParallelLinear
    original_dit_init = MiniMaxH3DiTModel.__init__

    def dit_init(
        self: nn.Module,
        od_config: object,
        quant_config: object | None = None,
    ) -> None:
        quant_config = _map_h3_dit_quant_config(quant_config)
        original_dit_init(self, od_config, quant_config)
        self._vllm_gaudi_quant_config = quant_config

    def encoder_init(
        self: nn.Module,
        model_path: str,
        *,
        device: torch.device,
        load_model: bool,
        encoder_group: Any | None = None,
        quant_config: object | None = None,
    ) -> None:
        encoder_quant_config = _resolve_h3_encoder_disk_quant_config(model_path, quant_config)
        nn.Module.__init__(self)
        self.device_target = device
        self.encoder_group = encoder_group
        self.quant_config = encoder_quant_config
        h3_encoder._validate_encoder_quant_config(encoder_quant_config)
        self.image_token_id = 151655
        self.video_token_id = 151656
        self._tp_size = 1
        if not load_model:
            return

        from transformers import Qwen3VLConfig

        config = Qwen3VLConfig.from_pretrained(model_path, trust_remote_code=False)
        self.image_token_id = int(config.image_token_id)
        self.video_token_id = int(config.video_token_id)
        self._tp_size = int(encoder_group.world_size) if encoder_group is not None else 1
        dtype = torch.bfloat16
        self.vision = h3_encoder.MiniMaxH3Qwen3VLVisionModel(config.vision_config)
        self.vision.to(dtype=dtype)
        self.text_model = h3_encoder.MiniMaxH3Qwen3VLTextModel(
            encoder_group,
            config.text_config,
            h3_encoder.MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
            dtype,
            quant_config=encoder_quant_config,
        )
        logger.info(
            "MiniMax H3 Qwen3-VL encoder: %d retained decoder layers, text_encoder_tp_size=%d, vision replicated",
            h3_encoder.MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
            self._tp_size,
        )

    MiniMaxH3DiTModel.__init__ = dit_init
    MiniMaxH3DiTModel.load_weights = _load_minimax_h3_dit_weights
    MiniMaxH3Qwen3VLEncoder.__init__ = encoder_init
    MiniMaxH3Qwen3VLEncoder._map_weight_name = staticmethod(map_minimax_h3_encoder_weight)
    MiniMaxH3Qwen3VLEncoder.load_weights = _load_minimax_h3_encoder_weights
    MiniMaxH3Qwen3VLRowParallelLinear.weight_loader = _h3_row_parallel_weight_loader
    _install_checkpoint_adapter()
    _install_h3_modulation_ops()
    _install_h3_encoder_attention_hooks()
    _install_h3_ref2va_canvas_hook()
    _install_h3_reference_video_transcode_hook()
    _install_fasth3_hooks()
    _install_h3_layout_and_noise_hooks()
    _install_h3_phase_offload_hooks()
    _install_h3_trajectory_debug_hook()
    _install_h3_dit_trace_hook()
    _PATCHED = True
    logger.info_once("Installed MiniMax H3 FP8, FastH3, and HPU operator hooks for Gaudi")


__all__ = [
    "install_minimax_h3_patches",
    "map_minimax_h3_dit_weight",
    "map_minimax_h3_encoder_weight",
    "map_minimax_h3_pipeline_weight",
]
