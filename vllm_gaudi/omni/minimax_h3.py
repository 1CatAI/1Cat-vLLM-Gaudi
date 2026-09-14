# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax H3 native ModelOpt FP8 checkpoint compatibility for Omni."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

_H3_SOURCE_PREFIXES = ("transformer.", "transformers_ref.", "text_encoder.")
_PATCHED = False


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
        original_dit_init(self, od_config, _map_h3_dit_quant_config(quant_config))

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
    _PATCHED = True
    logger.info_once("Installed MiniMax H3 native FP8 and HPU operator hooks for Gaudi")


__all__ = [
    "install_minimax_h3_patches",
    "map_minimax_h3_dit_weight",
    "map_minimax_h3_encoder_weight",
    "map_minimax_h3_pipeline_weight",
]
