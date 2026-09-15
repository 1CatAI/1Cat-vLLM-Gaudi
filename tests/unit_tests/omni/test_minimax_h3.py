# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.omni.minimax_h3 import (
    _h3_ffv1_reference_video_command,
    _h3_phase_offload_enabled,
    _h3_row_parallel_weight_loader,
    _hpu_fused_sdpa,
    _install_h3_dit_stagers,
    _lightx2v_ref_output_canvas,
    _load_minimax_h3_dit_weights,
    _load_minimax_h3_encoder_weights,
    _map_dit_exclude_prefix,
    _minimax_h3_initial_noise,
    _minimax_h3_unpatchify_video_tokens_hpu,
    _resolve_h3_encoder_disk_quant_config,
    _select_h3_reference_video_codec,
    _stage_h3_dit_component,
    _summarize_h3_trace_value,
    _validate_h3_output_short_edge,
    _validate_fasth3_quant_config,
    install_minimax_h3_patches,
    map_minimax_h3_dit_weight,
    map_minimax_h3_encoder_weight,
)
from vllm_gaudi.omni.minimax_h3_vae import (
    _h3_vae_persist_bf16_weights,
    _h3_vae_tile_batch_size,
    _install_h3_vae_decode_tile_batching,
    _materialize_h3_vae_decoder_linear_weights,
)


def test_h3_reference_video_codec_prefers_x264rgb_then_habana_ffv1():
    assert _select_h3_reference_video_codec({"ffv1", "libx264rgb"}) == "libx264rgb"
    assert _select_h3_reference_video_codec({"ffv1"}) == "ffv1"
    with pytest.raises(RuntimeError, match="libx264rgb or ffv1"):
        _select_h3_reference_video_codec({"mjpeg"})


def test_h3_habana_ffv1_reference_video_command_is_lossless_rgb(tmp_path):
    output, command = _h3_ffv1_reference_video_command(
        "source.mp4",
        target_width=1344,
        target_height=768,
        target_frame_count=107,
        workdir=str(tmp_path),
        fps=24.0,
        duration_seconds=4.458333,
    )

    assert output == str(tmp_path / "prepared.mkv")
    assert command[command.index("-c:v") + 1] == "ffv1"
    assert command[command.index("-pix_fmt") + 1] == "bgr0"
    assert command[command.index("-frames:v") + 1] == "107"
    assert command[command.index("-t") + 1] == "4.458333"
    assert "libx264rgb" not in command
    assert "-crf" not in command


def test_lightx2v_ref_canvas_uses_published_544p_grid():
    assert _lightx2v_ref_output_canvas(16 / 9) == (544, 960)
    assert _lightx2v_ref_output_canvas(9 / 16) == (960, 544)
    assert _lightx2v_ref_output_canvas(21 / 9) == (544, 1280)
    assert _lightx2v_ref_output_canvas(4.0) == (512, 2016)


def test_lightx2v_544p_canvas_is_ref2va_only():
    _validate_h3_output_short_edge("ref2va", "ref2va", 544)
    _validate_h3_output_short_edge("fl2va", "fl2va", 768)

    with pytest.raises(ValueError, match="requires the Ref2VA partition and task"):
        _validate_h3_output_short_edge("fl2va", "fl2va", 544)


def test_lightx2v_canvas_patch_keeps_shared_base_preprocessing_strict():
    install_minimax_h3_patches()
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3
    from vllm_omni.model_executor.models.minimax_h3 import preprocessing

    assert pipeline_minimax_h3._resolve_output_canvas(16 / 9, 544) == (544, 960)
    with pytest.raises(ValueError, match="must be 768"):
        preprocessing.resolve_minimax_h3_output_canvas(16 / 9, 544)


def test_h3_encoder_fused_sdpa_disallows_implicit_cpu_fallback():
    tensor = torch.empty(1, 2, 3, 4)
    with pytest.raises(RuntimeError, match="requires HPU query, key, and value"):
        _hpu_fused_sdpa(tensor, tensor, tensor, is_causal=False, scale=0.5)


def test_h3_phase_offload_environment_is_strict(monkeypatch):
    monkeypatch.delenv("VLLM_GAUDI_H3_PHASE_OFFLOAD", raising=False)
    assert not _h3_phase_offload_enabled()

    monkeypatch.setenv("VLLM_GAUDI_H3_PHASE_OFFLOAD", "yes")
    assert _h3_phase_offload_enabled()

    monkeypatch.setenv("VLLM_GAUDI_H3_PHASE_OFFLOAD", "sometimes")
    with pytest.raises(ValueError, match="must be a boolean"):
        _h3_phase_offload_enabled()


def test_h3_dit_stager_reuses_immutable_cpu_master(monkeypatch):
    events = []

    class FakeStager:

        def __init__(self, component, device, *, pin_memory):
            events.append(("create", component, device, pin_memory))

        def load(self):
            events.append(("load", ))

        def offload(self):
            events.append(("offload", ))

    platform = SimpleNamespace(
        synchronize=lambda: events.append(("synchronize", )),
        empty_cache=lambda: events.append(("empty_cache", )),
    )
    import vllm_omni.diffusion.offloader.module_residency as module_residency
    import vllm_omni.platforms as omni_platforms

    monkeypatch.setattr(module_residency, "PinnedModuleStager", FakeStager)
    monkeypatch.setattr(omni_platforms, "current_omni_platform", platform)
    component = torch.nn.Linear(8, 4)
    owner = SimpleNamespace(
        device=torch.device("hpu"),
        transformer=component,
        _dit_modules=["transformer"],
        _stage_durations={},
        _profiler_lock=threading.Lock(),
    )

    _install_h3_dit_stagers(owner)
    component.register_buffer("runtime_cache", torch.ones(1))
    _install_h3_dit_stagers(owner)
    _stage_h3_dit_component(owner, component, load=True, metric="load_dit")
    _stage_h3_dit_component(owner, component, load=False, metric="offload_dit")

    created_groups = [event for event in events if event[0] == "create"]
    assert len(created_groups) == 1
    _, parameter_group, device, pin_memory = created_groups[0]
    assert device == torch.device("hpu")
    assert pin_memory is True
    assert tuple(parameter_group.parameters()) == tuple(component.parameters())
    assert tuple(parameter_group.buffers()) == ()
    assert events.count(("load", )) == 1
    assert events.count(("offload", )) == 1
    assert events.count(("empty_cache", )) == 1
    assert owner._stage_durations["MiniMaxH3Pipeline.hpu_phase.load_dit"] >= 0
    assert owner._stage_durations["MiniMaxH3Pipeline.hpu_phase.offload_dit"] >= 0


def test_h3_vae_tile_batch_environment_is_strict(monkeypatch):
    monkeypatch.delenv("VLLM_GAUDI_H3_VAE_TILE_BATCH_SIZE", raising=False)
    assert _h3_vae_tile_batch_size() == 4

    monkeypatch.setenv("VLLM_GAUDI_H3_VAE_TILE_BATCH_SIZE", "7")
    assert _h3_vae_tile_batch_size() == 7

    for invalid in ("0", "-1", "auto"):
        monkeypatch.setenv("VLLM_GAUDI_H3_VAE_TILE_BATCH_SIZE", invalid)
        with pytest.raises(ValueError, match="positive integer"):
            _h3_vae_tile_batch_size()


def test_h3_vae_persist_bf16_weights_environment_is_strict(monkeypatch):
    monkeypatch.delenv("VLLM_GAUDI_H3_VAE_PERSIST_BF16_WEIGHTS", raising=False)
    assert _h3_vae_persist_bf16_weights()

    monkeypatch.setenv("VLLM_GAUDI_H3_VAE_PERSIST_BF16_WEIGHTS", "off")
    assert not _h3_vae_persist_bf16_weights()

    monkeypatch.setenv("VLLM_GAUDI_H3_VAE_PERSIST_BF16_WEIGHTS", "sometimes")
    with pytest.raises(ValueError, match="must be a boolean"):
        _h3_vae_persist_bf16_weights()


def test_h3_vae_materializes_only_decoder_linears_in_bf16():
    decoder = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.LayerNorm(16),
        torch.nn.Sequential(torch.nn.Linear(16, 8)),
    )

    assert _materialize_h3_vae_decoder_linear_weights(decoder) == 2
    assert decoder[0].weight.dtype == torch.bfloat16
    assert decoder[0].bias.dtype == torch.bfloat16
    assert decoder[1].weight.dtype == torch.float32
    assert decoder[2][0].weight.dtype == torch.bfloat16


class _FakeTiledVAE(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder_tiling = True
        self.stack_tiling = False
        self.decode_batch_sizes: list[int] = []
        self.reference_calls = 0

    def decode(self, value):
        self.decode_batch_sizes.append(int(value.shape[0]))
        return value + 10

    def _run_tile_tasks(self, tiles, tile_indices, forward_fn, stack_tiling, cls_agg=None):
        del stack_tiling, cls_agg
        self.reference_calls += 1
        return [forward_fn(tiles[index]) for index in tile_indices]


def test_h3_vae_decode_tile_batching_preserves_order_and_tail():
    model = _FakeTiledVAE()
    tiles = [torch.tensor([[value]]) for value in range(5)]

    assert _install_h3_vae_decode_tile_batching(model, 2)
    output = model._run_tile_tasks(tiles, list(range(5)), model.decode, False)

    assert model.decode_batch_sizes == [2, 2, 1]
    assert [int(value.item()) for value in output] == [10, 11, 12, 13, 14]
    assert model.reference_calls == 0

    passthrough = model._run_tile_tasks(tiles, [1, 3], lambda value: value * 2, False)
    assert [int(value.item()) for value in passthrough] == [2, 6]
    assert model.reference_calls == 1


def test_h3_trace_summary_keeps_only_tensor_contract():
    value = {
        "hidden_states": torch.empty((1, 7, 16), dtype=torch.bfloat16),
        "nested": [torch.empty((3, ), dtype=torch.float32), None],
        "opaque": object(),
    }

    assert _summarize_h3_trace_value(value) == {
        "hidden_states": {
            "shape": [1, 7, 16],
            "dtype": "torch.bfloat16",
            "device": "cpu",
        },
        "nested": [{
            "shape": [3],
            "dtype": "torch.float32",
            "device": "cpu",
        }, None],
        "opaque": "object",
    }


def test_hpu_unpatchify_gather_matches_video_latent_layout():
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_patchify_video_latent

    latent = torch.arange(2 * 3 * 4 * 6 * 8, dtype=torch.float32).reshape(2, 3, 4, 6, 8)
    rows = minimax_h3_patchify_video_latent(latent, patch_size=(2, 2, 4))

    output = _minimax_h3_unpatchify_video_tokens_hpu(
        rows,
        latent_shape=(2, 3, 2, 3),
        patch_size=(2, 2, 4),
    )

    assert output.is_contiguous()
    assert torch.equal(output, latent)


def test_initial_noise_uses_one_generator_for_video_then_audio():
    seed = 2101
    latent_shape = (3, 4, 6)
    audio_t = 5
    video_rows, audio_rows = _minimax_h3_initial_noise(
        None,
        seed=seed,
        latent_t=latent_shape[0],
        latent_h=latent_shape[1],
        latent_w=latent_shape[2],
        audio_t=audio_t,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    video = torch.randn(1, 24, *latent_shape, generator=generator, dtype=torch.float32)
    expected_audio = torch.randn(audio_t * 2, 32, generator=generator, dtype=torch.float32)

    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_patchify_video_latent

    assert torch.equal(video_rows, minimax_h3_patchify_video_latent(video, patch_size=(1, 2, 2)))
    assert torch.equal(audio_rows, expected_audio)


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


def test_fasth3_rejects_serialized_transformer_but_accepts_bf16_source():
    fusion = object()

    _validate_fasth3_quant_config(fusion, SimpleNamespace(is_checkpoint_fp8_serialized=False))
    with pytest.raises(ValueError, match="fused into a BF16"):
        _validate_fasth3_quant_config(fusion, SimpleNamespace(is_checkpoint_fp8_serialized=True))


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
