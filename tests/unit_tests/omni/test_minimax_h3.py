# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import torch

from vllm_gaudi.omni.minimax_h3 import (
    _h3_row_parallel_weight_loader,
    _load_minimax_h3_dit_weights,
    _load_minimax_h3_encoder_weights,
    _map_dit_exclude_prefix,
    _resolve_h3_encoder_disk_quant_config,
    install_minimax_h3_patches,
    map_minimax_h3_dit_weight,
    map_minimax_h3_encoder_weight,
)


def test_dit_mapping_covers_fused_projection_scales():
    assert map_minimax_h3_dit_weight("transformer_blocks.3.attn.to_k.weight_scale") == (
        "blocks.3.attn.qkv_proj.weight_scale",
        "k",
    )
    assert map_minimax_h3_dit_weight("token_refiner.refiner_blocks.1.ff.net.0.proj.weight_scale") == (
        "token_refiner.blocks.1.mlp.fc1.weight_scale",
        None,
    )
    assert map_minimax_h3_dit_weight("norm_out.linear.weight_scale") == (
        "final_layer.adaln_proj.linear.weight_scale",
        None,
    )
    assert map_minimax_h3_dit_weight("transformer_blocks.9.attn.norm_q.weight") == (
        "blocks.9.attn.q_norm.weight",
        None,
    )


def test_dit_modelopt_exclusions_follow_vllm_names():
    assert _map_dit_exclude_prefix("context_embedder") == "condition_proj"
    assert _map_dit_exclude_prefix("transformer_blocks.28.ff.net.0*") == "blocks.28.mlp.fc1*"
    assert (_map_dit_exclude_prefix("transformer_blocks.31.attn.to_out*") == "blocks.31.attn.out_proj*")


def test_encoder_mapping_covers_fused_projection_scales():
    assert map_minimax_h3_encoder_weight("model.language_model.layers.7.self_attn.v_proj.weight_scale") == (
        "text_model.layers.7.self_attn.qkv_proj.weight_scale", "v")
    assert map_minimax_h3_encoder_weight("model.language_model.layers.8.mlp.up_proj.weight_scale") == (
        "text_model.layers.8.mlp.gate_up_proj.weight_scale",
        1,
    )
    assert map_minimax_h3_encoder_weight("model.visual.blocks.2.attn.qkv.weight") == (
        "vision.blocks.2.attn.qkv.weight",
        None,
    )


class _RecordingParameter(torch.nn.Parameter):
    pass


def _recording_parameter(shape: tuple[int, ...], calls: list[tuple[tuple[int, ...], object]]):
    param = _RecordingParameter(torch.empty(shape), requires_grad=False)

    def loader(_param, tensor, shard_id=None):
        calls.append((tuple(tensor.shape), shard_id))

    param.weight_loader = loader
    return param


def test_dit_loader_routes_qkv_and_fc1_weight_scales():
    qkv_calls: list[tuple[tuple[int, ...], object]] = []
    qkv_scale_calls: list[tuple[tuple[int, ...], object]] = []
    fc1_calls: list[tuple[tuple[int, ...], object]] = []
    fc1_scale_calls: list[tuple[tuple[int, ...], object]] = []
    params = {
        "blocks.0.attn.qkv_proj.weight": _recording_parameter((12, 4), qkv_calls),
        "blocks.0.attn.qkv_proj.weight_scale": _recording_parameter((12, ), qkv_scale_calls),
        "blocks.0.mlp.fc1.weight": _recording_parameter((8, 4), fc1_calls),
        "blocks.0.mlp.fc1.weight_scale": _recording_parameter((8, ), fc1_scale_calls),
    }

    model = SimpleNamespace(
        named_parameters=lambda: params.items(),
        named_buffers=lambda: (),
        arch=SimpleNamespace(num_attention_heads=1, attention_head_dim=4),
    )
    source = []
    for projection in ("q", "k", "v"):
        source.extend([
            (f"transformer_blocks.0.attn.to_{projection}.weight", torch.empty(4, 4)),
            (f"transformer_blocks.0.attn.to_{projection}.weight_scale", torch.empty(4)),
        ])
    source.extend([
        ("transformer_blocks.0.ff.net.0.proj.weight", torch.empty(8, 4)),
        ("transformer_blocks.0.ff.net.0.proj.weight_scale", torch.empty(8)),
    ])

    loaded = _load_minimax_h3_dit_weights(model, source)

    assert qkv_calls == [((4, 4), "q"), ((4, 4), "k"), ((4, 4), "v")]
    assert qkv_scale_calls == [((4, ), "q"), ((4, ), "k"), ((4, ), "v")]
    assert fc1_calls == [((4, 4), 0), ((4, 4), 1)]
    assert fc1_scale_calls == [((4, ), 0), ((4, ), 1)]
    assert set(params) == loaded


def test_encoder_loader_uses_two_arguments_for_unfused_parameters():
    qkv_calls: list[tuple[tuple[int, ...], object]] = []
    embedding = torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)
    qkv = _recording_parameter((6, 3), qkv_calls)
    params = {
        "text_model.embed_tokens.weight": embedding,
        "text_model.layers.0.self_attn.qkv_proj.weight": qkv,
    }
    qkv_module = type("MiniMaxH3Qwen3VLQKVParallelLinear", (), {})()
    model = SimpleNamespace(
        named_parameters=lambda: params.items(),
        named_modules=lambda: [("text_model.layers.0.self_attn.qkv_proj", qkv_module)],
    )
    checkpoint_embedding = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = [("model.language_model.embed_tokens.weight", checkpoint_embedding)]
    source.extend((
        f"model.language_model.layers.0.self_attn.{projection}_proj.weight",
        torch.empty(2, 3),
    ) for projection in ("q", "k", "v"))

    loaded = _load_minimax_h3_encoder_weights(model, source)

    assert torch.equal(embedding, checkpoint_embedding)
    assert qkv_calls == [((2, 3), "q"), ((2, 3), "k"), ((2, 3), "v")]
    assert set(params) == loaded


def test_text_row_parallel_scale_is_not_input_sharded():
    layer = SimpleNamespace(input_size_per_partition=2, _tp_rank=1)
    scale = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    checkpoint_scale = torch.arange(1, 5, dtype=torch.float32)

    _h3_row_parallel_weight_loader(layer, scale, checkpoint_scale)

    assert torch.equal(scale, checkpoint_scale)


def test_encoder_uses_its_own_native_fp8_config(tmp_path):
    config = {
        "quantization_config": {
            "quant_method":
            "modelopt",
            "quant_algo":
            "FP8_PER_CHANNEL_PER_TOKEN",
            "ignore": [
                "model.language_model.embed_tokens",
                "model.language_model.layers.17.mlp.down_proj",
                "model.language_model.layers.24.self_attn.o_proj",
            ],
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    resolved = _resolve_h3_encoder_disk_quant_config(str(tmp_path), None)

    assert resolved.quant_method == "FP8_PER_CHANNEL_PER_TOKEN"
    assert resolved.LinearMethodCls.__name__ == "HPUModelOptFp8PcPtLinearMethod"
    assert resolved.exclude_modules == [
        "text_encoder.text_model.embed_tokens",
        "text_encoder.text_model.layers.17.mlp.down_proj",
        "text_encoder.text_model.layers.24.self_attn.o_proj",
    ]


def test_h3_adapter_dequantizes_an_ignored_fp8_weight():
    install_minimax_h3_patches()
    from vllm_omni.diffusion.model_loader import checkpoint_adapters

    class MiniMaxH3Fake(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.target = torch.nn.Parameter(torch.empty(2, 2, dtype=torch.bfloat16))

        def named_parameters(self, *args, **kwargs):
            del args, kwargs
            yield "transformer.blocks.31.mlp.fc2.weight", self.target

    MiniMaxH3Fake.__module__ = "vllm_omni.diffusion.models.minimax_h3.fake"

    class Config:
        is_checkpoint_fp8_serialized = True

        @staticmethod
        def get_name():
            return "modelopt"

    model = MiniMaxH3Fake()
    adapter = checkpoint_adapters.get_checkpoint_adapter(
        model,
        SimpleNamespace(prefix="transformer.", subfolder="transformer"),
        Config(),
        True,
    )
    source_weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]]).to(torch.float8_e4m3fn)
    source_scale = torch.tensor([0.5, 0.25])
    output = list(
        adapter.adapt([
            ("transformer.transformer_blocks.31.ff.net.2.weight", source_weight),
            ("transformer.transformer_blocks.31.ff.net.2.weight_scale", source_scale),
        ]))

    assert [name for name, _ in output] == ["transformer.transformer_blocks.31.ff.net.2.weight"]
    expected = source_weight.float() * source_scale[:, None]
    assert torch.equal(output[0][1].float(), expected.to(torch.bfloat16).float())
