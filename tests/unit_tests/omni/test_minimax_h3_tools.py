# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from safetensors.torch import save_file
import torch

from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

_REPO_ROOT = Path(__file__).parents[3]


def _load_tool(name: str):
    path = _REPO_ROOT / "tools" / "minimax_h3" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


request_video = _load_tool("request_video")
serve_single_hpu = _load_tool("serve_single_hpu")
download_modelscope = _load_tool("download_modelscope")
download_flashgen_modelscope = _load_tool("download_flashgen_modelscope")
download_fasth3_modelscope = _load_tool("download_fasth3_modelscope")
download_lightx2v_modelscope = _load_tool("download_lightx2v_modelscope")
prepare_fasth3_modelscope = _load_tool("prepare_fasth3_modelscope")
prepare_ref2va_modelscope = _load_tool("prepare_ref2va_modelscope")
benchmark_t2va = _load_tool("benchmark_t2va")


def _request_args(**overrides):
    values = {
        "task": "t2va",
        "prompt": "A fox walks through snow.",
        "duration": 5.0,
        "fps": 24,
        "sigma_points": None,
        "flow_shift": None,
        "audio_flow_shift": None,
        "width": 1344,
        "height": 768,
        "aspect_ratio": "16:9",
        "short_edge": 768,
        "audio_url": None,
        "frame_indices": None,
        "seed": 2101,
        "flashgen_lora": None,
        "fasth3_4step": False,
        "lightx2v_lora": None,
        "lora_scale": 1.0,
        "preencode_mp4": False,
        "preencode_batch_frames": 17,
    }
    values.update(overrides)
    return Namespace(**values)


def _write_fp8_partition(root: Path, partition: str) -> Path:
    target = root / partition
    target.mkdir(parents=True)
    (target / "model_index.json").write_text("{}\n", encoding="utf-8")
    quantization = {
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
            "producer": {
                "name": "modelopt",
                "version": "0.45"
            },
        }
    }
    for component in ("transformer", "text_encoder"):
        path = target / component
        path.mkdir()
        (path / "config.json").write_text(json.dumps(quantization), encoding="utf-8")
    return target


def test_pinned_omni_base_schedule_has_50_points_and_49_joint_forwards():
    video = minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=12.0)
    audio = minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=3.0)

    assert len(video) == len(audio) == 50
    assert video[0] == audio[0] == 1.0
    assert video[-1] == audio[-1] == 0.0
    assert len(video) - 1 == 49


def test_request_omits_sampling_fields_for_explicit_pinned_omni_reference():
    args = _request_args()
    request_video._validate(args, [])
    data = request_video._request_data(args)

    assert "num_inference_steps" not in data
    assert "flow_shift" not in data
    assert json.loads(data["extra_params"]) == {"task": "t2va", "duration": 5.0}


def test_request_cli_defaults_to_pinned_omni_base_schedule():
    assert request_video._resolve_sigma_points(None, None, False) is None
    assert request_video._resolve_sigma_points(None, 8, False) == 9
    assert request_video._resolve_sigma_points(7, None, False) == 7
    assert request_video._resolve_sigma_points(None, None, True) is None


def _benchmark_args(**overrides):
    values = {
        "api_url": "http://127.0.0.1:8097/v1/videos/sync",
        "prompt": "A fox walks through snow.",
        "timeout": 3600,
        "sigma_points": 21,
        "pinned_omni_reference": False,
        "flashgen_lora": None,
        "fasth3_4step": False,
        "lightx2v_lora": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_benchmark_can_name_an_explicit_base_grid(tmp_path):
    command = benchmark_t2va._request_command(_benchmark_args(), tmp_path)

    assert command[command.index("--sigma-points") + 1] == "21"
    assert "--pinned-omni-reference" not in command


def test_flashgen_request_uses_adapter_interval_contract():
    path = Path("/models/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors")
    args = _request_args(flashgen_lora=path, sigma_points=None)

    request_video._validate(args, [])
    data = request_video._request_data(args)
    contract = request_video._sampling_contract(args)

    assert data["num_inference_steps"] == "4"
    assert json.loads(data["lora"]) == {
        "name": "h3-flashgen-v1.0",
        "path": str(path),
        "scale": 1.0,
    }
    assert contract["resolved_sigma_points"] == 5
    assert contract["expected_joint_dit_forwards"] == 4
    assert contract["base_schedule"] == [1.0, 0.7, 0.4, 0.15, 0.0]


def test_flashgen_request_rejects_non_t2va():
    args = _request_args(
        task="fl2va",
        flashgen_lora=Path("minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"),
    )
    with pytest.raises(ValueError, match="T2VA only"):
        request_video._validate(args, [(Path("first.png"), "image/png")])


def test_fasth3_request_uses_exact_four_forward_contract():
    args = _request_args(fasth3_4step=True)

    request_video._validate(args, [])
    data = request_video._request_data(args)
    contract = request_video._sampling_contract(args)

    assert data["num_inference_steps"] == "4"
    assert "lora" not in data
    assert contract["resolved_sigma_points"] == 5
    assert contract["expected_joint_dit_forwards"] == 4
    assert contract["base_schedule"] == [0.999, 0.749, 0.5, 0.25, 0.0]


def test_lightx2v_request_uses_five_sigma_points_and_four_forwards():
    path = Path("/models/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors")
    args = _request_args(lightx2v_lora=path)

    request_video._validate(args, [])
    data = request_video._request_data(args)
    contract = request_video._sampling_contract(args)

    assert data["num_inference_steps"] == "5"
    assert data["flow_shift"] == "6.0"
    assert json.loads(data["extra_params"])["audio_flow_shift"] == 3.0
    assert json.loads(data["lora"]) == {
        "name": path.stem,
        "path": str(path),
        "scale": 1.0,
    }
    assert contract["resolved_sigma_points"] == 5
    assert contract["expected_joint_dit_forwards"] == 4
    assert contract["precision"] == "bf16_base_plus_bf16_lora"


def test_lightx2v_request_rejects_manual_shifts_and_ref2va():
    path = Path("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors")
    with pytest.raises(ValueError, match="owns its exact"):
        request_video._validate(_request_args(lightx2v_lora=path, flow_shift=6.0), [])
    with pytest.raises(ValueError, match="supports FL2VA, T2VA only"):
        request_video._validate(_request_args(task="ref2va", lightx2v_lora=path), [])


def test_lightx2v_ref2va_request_uses_544p_four_step_profile():
    path = Path("minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors")
    args = _request_args(
        task="ref2va",
        lightx2v_lora=path,
        short_edge=544,
        width=None,
        height=None,
    )
    references = [(Path("fox.png"), "image/png")]

    request_video._validate(args, references)
    data = request_video._request_data(args)
    contract = request_video._sampling_contract(args)

    assert data["num_inference_steps"] == "5"
    assert data["flow_shift"] == "12.0"
    assert data["short_edge"] == "544"
    assert json.loads(data["extra_params"])["audio_flow_shift"] == 3.0
    assert contract["profile"] == "lightx2v_ref2v_turbo_4step_v0.1_544p_bf16"
    assert contract["expected_joint_dit_forwards"] == 4
    assert contract["output_short_edge"] == 544

    with pytest.raises(ValueError, match="short_edge must be 544"):
        request_video._validate(_request_args(task="ref2va", lightx2v_lora=path), references)


def test_request_enables_worker_side_chunked_mp4():
    args = _request_args(preencode_mp4=True, preencode_batch_frames=11)

    data = request_video._request_data(args)

    extra = json.loads(data["extra_params"])
    assert extra["preencode_mp4"] is True
    assert extra["preencode_batch_frames"] == 11


def test_benchmark_fasth3_command_uses_server_fused_preset(tmp_path):
    command = benchmark_t2va._request_command(_benchmark_args(fasth3_4step=True), tmp_path)

    assert "--fasth3-4step" in command
    assert "--sigma-points" not in command
    assert command[command.index("--preencode-batch-frames") + 1] == "17"
    assert "--preencode-mp4" in command


def test_benchmark_flashgen_command_uses_exact_preset(tmp_path):
    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    command = benchmark_t2va._request_command(_benchmark_args(sigma_points=None, flashgen_lora=path), tmp_path)

    assert command[command.index("--flashgen-4step-lora") + 1] == str(path)
    assert "--sigma-points" not in command


def test_benchmark_lightx2v_command_uses_exact_preset(tmp_path):
    path = tmp_path / download_lightx2v_modelscope.FILENAME
    args = _benchmark_args(sigma_points=None, lightx2v_lora=path)
    command = benchmark_t2va._request_command(args, tmp_path)

    assert command[command.index("--lightx2v-4step-lora") + 1] == str(path)
    assert benchmark_t2va._sampling_plan(args) == (
        5,
        4,
        "lightx2v_fl2v_turbo_4step_v1.0_768p_bf16",
    )
    assert "--sigma-points" not in command


def test_flashgen_modelscope_metadata_contract():
    download_flashgen_modelscope._validate_metadata(dict(download_flashgen_modelscope.EXPECTED_METADATA))
    invalid = dict(download_flashgen_modelscope.EXPECTED_METADATA, base_schedule="1,0")
    with pytest.raises(ValueError, match="metadata mismatch"):
        download_flashgen_modelscope._validate_metadata(invalid)


def test_fasth3_modelscope_contracts():
    download_fasth3_modelscope._validate_metadata(dict(download_fasth3_modelscope.EXPECTED_METADATA))
    manifest = {
        "sampling": {
            "transformer_forwards": 4
        },
        "variants": [{
            "slug": "dense-datafree",
            "adapter_path": str(download_fasth3_modelscope.RELATIVE_PATH),
            "adapter_size_bytes": download_fasth3_modelscope.EXPECTED_BYTES,
            "adapter_sha256": download_fasth3_modelscope.EXPECTED_SHA256,
            "requires_vsa": False,
        }],
    }
    download_fasth3_modelscope._validate_bundle_manifest(manifest)

    manifest["sampling"]["transformer_forwards"] = 49
    with pytest.raises(ValueError, match="four transformer forwards"):
        download_fasth3_modelscope._validate_bundle_manifest(manifest)


def test_lightx2v_modelscope_contract_is_official_bf16_four_step():
    download_lightx2v_modelscope._validate_metadata(dict(download_lightx2v_modelscope.EXPECTED_METADATA))
    ref2va = download_lightx2v_modelscope.PROFILES["ref2v-4step-544p"]
    download_lightx2v_modelscope._validate_metadata(
        dict(ref2va.expected_metadata),
        ref2va.expected_metadata,
    )
    assert download_lightx2v_modelscope.MODEL_ID == "lightx2v/Minimax-h3-Turbo"
    assert download_lightx2v_modelscope.SIGMA_POINTS == 5
    assert download_lightx2v_modelscope.DENOISER_FORWARDS == 4
    assert download_lightx2v_modelscope.VIDEO_FLOW_SHIFT == 6.0
    assert download_lightx2v_modelscope.AUDIO_FLOW_SHIFT == 3.0
    assert download_lightx2v_modelscope.EXPECTED_METADATA["floating_dtype"] == "bfloat16"
    assert ref2va.task_family == "ref2va"
    assert ref2va.video_flow_shift == 12.0
    assert ref2va.audio_flow_shift == 3.0
    assert ref2va.alpha == 8.0


def test_fasth3_base_preparation_uses_official_bf16_compute_components(tmp_path):
    partition = tmp_path / "FL2VA"
    component_bytes = {}
    for component, shard_count in (("transformer", 13), ("text_encoder", 14)):
        component_path = partition / component
        component_path.mkdir(parents=True)
        (component_path / "config.json").write_text("{}\n", encoding="utf-8")
        weight_map = {}
        for index in range(1, shard_count + 1):
            shard = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            save_file({f"layer.{index}": torch.ones(1, dtype=torch.bfloat16)}, component_path / shard)
            weight_map[f"layer.{index}"] = shard
        (component_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}),
            encoding="utf-8",
        )
        component_bytes[component] = sum(path.stat().st_size for path in component_path.glob("*.safetensors"))

    transformer = prepare_fasth3_modelscope._validate_bf16_transformer(partition)
    text_encoder = prepare_fasth3_modelscope._validate_bf16_text_encoder(partition)

    assert transformer == {
        "format": "bf16",
        "shards": 13,
        "bytes": component_bytes["transformer"],
        "tensor_dtypes": {
            "BF16": 13
        },
    }
    assert text_encoder == {
        "format": "bf16",
        "shards": 14,
        "bytes": component_bytes["text_encoder"],
        "tensor_dtypes": {
            "BF16": 14
        },
    }
    assert "FL2VA/text_encoder/**" in prepare_fasth3_modelscope.BASE_PATTERNS
    assert "text_encoder" not in prepare_fasth3_modelscope.REUSED_COMPONENTS


def test_ref2va_preparation_reuses_only_sha_identical_modelscope_components(tmp_path):
    source = tmp_path / "FL2VA"
    remote_files = []
    expected_bytes = 0
    for component in prepare_ref2va_modelscope.SHARED_COMPONENTS:
        path = source / component / "payload.bin"
        path.parent.mkdir(parents=True)
        payload = f"official-{component}".encode()
        path.write_bytes(payload)
        sha256 = hashlib.sha256(payload).hexdigest()
        expected_bytes += len(payload)
        for partition in ("FL2VA", "Ref2VA"):
            remote_files.append(
                SimpleNamespace(
                    path=f"{partition}/{component}/payload.bin",
                    size=len(payload),
                    sha256=sha256,
                ))

    contract = prepare_ref2va_modelscope._validate_remote_shared_components(remote_files)
    summary = prepare_ref2va_modelscope._validate_shared_source(source, contract)

    assert set(summary) == set(prepare_ref2va_modelscope.SHARED_COMPONENTS)
    assert sum(item["bytes"] for item in summary.values()) == expected_bytes

    remote_files[-1].sha256 = "0" * 64
    with pytest.raises(ValueError, match="differs between FL2VA and Ref2VA"):
        prepare_ref2va_modelscope._validate_remote_shared_components(remote_files)


def test_single_hpu_launcher_validates_and_preloads_flashgen(tmp_path):
    lora = tmp_path / download_flashgen_modelscope.FILENAME
    save_file(
        {"placeholder": torch.zeros(1)},
        lora,
        metadata=dict(download_flashgen_modelscope.EXPECTED_METADATA),
    )
    resolved = serve_single_hpu._resolve_flashgen_lora(tmp_path, "fl2va")
    args = serve_single_hpu._parse_args(["/models/h3", "--partition", "FL2VA", "--flashgen-4step-lora", str(tmp_path)])
    args.flashgen_lora = resolved
    command = serve_single_hpu._command(args, Path("/models/h3/FL2VA"), "fl2va")

    assert resolved == lora
    assert command[command.index("--lora-backend") + 1] == "peft"
    assert command[command.index("--lora-path") + 1] == str(lora)
    offload = json.loads(command[command.index("--diffusion-offload-config") + 1])
    assert offload["components"] == ["text_encoder"]


def test_single_hpu_launcher_fuses_fasth3_in_official_bf16_by_default(tmp_path, monkeypatch):
    model = _write_fp8_partition(tmp_path / "model", "FL2VA")
    for component in ("transformer", "text_encoder"):
        (model / component / "config.json").write_text("{}\n", encoding="utf-8")
    adapter_root = tmp_path / "adapter"
    adapter = adapter_root / "dense-datafree" / "adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    save_file(
        {"placeholder": torch.zeros(1, dtype=torch.bfloat16)},
        adapter,
        metadata=dict(download_fasth3_modelscope.EXPECTED_METADATA),
    )
    monkeypatch.setattr(serve_single_hpu, "_FASTH3_BYTES", adapter.stat().st_size)
    monkeypatch.setattr(serve_single_hpu, "_FASTH3_TENSORS", 1)
    with adapter.open("rb") as stream:
        monkeypatch.setattr(serve_single_hpu, "_FASTH3_SHA256", hashlib.file_digest(stream, "sha256").hexdigest())

    resolved = serve_single_hpu._resolve_fasth3_adapter(adapter_root, "fl2va")
    args = serve_single_hpu._parse_args(
        [str(model.parent), "--partition", "FL2VA", "--fasth3-4step-adapter",
         str(adapter_root)])
    args.fasth3_adapter = resolved
    command = serve_single_hpu._command(args, model, "fl2va")

    assert serve_single_hpu._checkpoint_formats(model) == {
        "transformer": "bf16",
        "text_encoder": "bf16",
    }
    assert command[command.index("--lora-path") + 1] == str(adapter)
    assert "--lora-backend" not in command
    assert "--diffusion-quantization-config" not in command
    assert args.phase_offload is True

    resident_args = serve_single_hpu._parse_args([
        str(model.parent),
        "--partition",
        "FL2VA",
        "--fasth3-4step-adapter",
        str(adapter_root),
        "--no-phase-offload",
    ])
    assert resident_args.phase_offload is False

    online_args = serve_single_hpu._parse_args([
        str(model.parent),
        "--partition",
        "FL2VA",
        "--fasth3-4step-adapter",
        str(adapter_root),
        "--online-fp8",
    ])
    online_args.fasth3_adapter = resolved
    online_command = serve_single_hpu._command(online_args, model, "fl2va")
    quantization = json.loads(online_command[online_command.index("--diffusion-quantization-config") + 1])
    assert quantization == {"method": "fp8"}


def test_single_hpu_launcher_validates_and_preloads_lightx2v(tmp_path, monkeypatch):
    lora = tmp_path / download_lightx2v_modelscope.FILENAME
    save_file(
        {"placeholder": torch.zeros((2, 3), dtype=torch.bfloat16)},
        lora,
        metadata=dict(download_lightx2v_modelscope.EXPECTED_METADATA),
    )
    monkeypatch.setattr(serve_single_hpu, "_LIGHTX2V_TENSORS", 1)
    monkeypatch.setattr(serve_single_hpu, "_LIGHTX2V_SHAPES", {(2, 3): 1})
    with lora.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    monkeypatch.setitem(
        serve_single_hpu._LIGHTX2V_ARTIFACTS,
        lora.name,
        serve_single_hpu._LightX2VArtifact(
            expected_bytes=lora.stat().st_size,
            expected_sha256=sha256,
            expected_metadata=download_lightx2v_modelscope.EXPECTED_METADATA,
            task_type="fl2va",
        ),
    )

    resolved = serve_single_hpu._resolve_lightx2v_lora(tmp_path, "fl2va")
    args = serve_single_hpu._parse_args(["/models/h3", "--partition", "FL2VA", "--lightx2v-4step-lora", str(tmp_path)])
    args.lightx2v_lora = resolved
    command = serve_single_hpu._command(args, Path("/models/h3/FL2VA"), "fl2va")

    assert resolved == lora
    assert command[command.index("--lora-backend") + 1] == "peft"
    assert command[command.index("--lora-path") + 1] == str(lora)
    assert args.phase_offload is True


def test_single_hpu_launcher_accepts_ref2va_lightx2v_artifact(tmp_path, monkeypatch):
    profile = download_lightx2v_modelscope.PROFILES["ref2v-4step-544p"]
    lora = tmp_path / profile.filename
    metadata = {key: value for key, value in profile.expected_metadata.items() if value is not None}
    save_file({"placeholder": torch.zeros((2, 3), dtype=torch.bfloat16)}, lora, metadata=metadata)
    monkeypatch.setattr(serve_single_hpu, "_LIGHTX2V_TENSORS", 1)
    monkeypatch.setattr(serve_single_hpu, "_LIGHTX2V_SHAPES", {(2, 3): 1})
    with lora.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    monkeypatch.setitem(
        serve_single_hpu._LIGHTX2V_ARTIFACTS,
        lora.name,
        serve_single_hpu._LightX2VArtifact(
            expected_bytes=lora.stat().st_size,
            expected_sha256=sha256,
            expected_metadata=profile.expected_metadata,
            task_type="ref2va",
        ),
    )

    resolved = serve_single_hpu._resolve_lightx2v_lora(lora, "ref2va")

    assert resolved == lora
    with pytest.raises(ValueError, match="requires the REF2VA partition"):
        serve_single_hpu._resolve_lightx2v_lora(lora, "fl2va")


def test_single_hpu_launcher_resolves_data_volume_temp_dir(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("VLLM_GAUDI_H3_TMPDIR", str(scratch))

    from_environment = serve_single_hpu._parse_args(["/models/h3", "--partition", "FL2VA"])
    explicit = serve_single_hpu._parse_args(
        ["/models/h3", "--partition", "FL2VA", "--temp-dir",
         str(tmp_path / "explicit")])

    assert from_environment.temp_dir == scratch
    assert explicit.temp_dir == tmp_path / "explicit"


def test_single_hpu_launcher_configures_bounded_vae_tile_batching():
    default = serve_single_hpu._parse_args(["/models/h3", "--partition", "FL2VA"])
    explicit = serve_single_hpu._parse_args([
        "/models/h3",
        "--partition",
        "FL2VA",
        "--vae-tile-batch-size",
        "7",
        "--no-vae-persist-bf16-weights",
    ])

    assert default.vae_tile_batch_size == 4
    assert default.vae_persist_bf16_weights is True
    assert default.vae_fused_sdpa is True
    assert explicit.vae_tile_batch_size == 7
    assert explicit.vae_persist_bf16_weights is False


def test_single_hpu_launcher_prepends_media_tools(tmp_path):
    media_bin = tmp_path / "habana-media"
    media_bin.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        (media_bin / name).touch()
    args = serve_single_hpu._parse_args([
        "/models/h3",
        "--partition",
        "Ref2VA",
        "--media-bin",
        str(media_bin),
    ])
    resolved = serve_single_hpu._resolve_media_bin(args.media_bin)
    environment = {"PATH": "/usr/bin"}

    serve_single_hpu._prepend_media_path(environment, resolved)

    assert resolved == media_bin.resolve()
    assert environment["PATH"] == f"{media_bin.resolve()}:/usr/bin"

    (media_bin / "ffprobe").unlink()
    with pytest.raises(FileNotFoundError, match="ffprobe"):
        serve_single_hpu._resolve_media_bin(media_bin)


def test_formal_benchmark_requires_explicit_pinned_omni_reference(tmp_path):
    command = benchmark_t2va._request_command(
        _benchmark_args(sigma_points=None, pinned_omni_reference=True),
        tmp_path,
    )

    assert "--pinned-omni-reference" in command
    assert "--sigma-points" not in command


def test_request_can_name_sigma_points_explicitly():
    data = request_video._request_data(_request_args(sigma_points=9, flow_shift=6.0, audio_flow_shift=3.0))

    assert data["num_inference_steps"] == "9"
    assert data["flow_shift"] == "6.0"
    assert json.loads(data["extra_params"])["audio_flow_shift"] == 3.0


def test_sync_response_metrics_are_typed():
    metrics = request_video._response_metrics({
        "X-Request-Id": "video_sync-1",
        "X-Model": "MiniMax-H3",
        "X-Inference-Time-S": "12.375",
        "X-Stage-Durations": '{"MiniMaxH3Pipeline.diffuse":10.25}',
        "X-Peak-Memory-MB": "92160.5",
    })

    assert metrics == {
        "request_id": "video_sync-1",
        "model": "MiniMax-H3",
        "server_inference_seconds": 12.375,
        "stage_durations_seconds": {
            "MiniMaxH3Pipeline.diffuse": 10.25
        },
        "peak_device_memory_mb": 92160.5,
    }


def test_request_rejects_one_sigma_point():
    with pytest.raises(ValueError, match="at least two sigma"):
        request_video._validate(_request_args(sigma_points=1), [])


def test_modelscope_validation_and_single_hpu_partition_resolution(tmp_path):
    partition = _write_fp8_partition(tmp_path, "FL2VA")

    validation = download_modelscope._validate_native_fp8(tmp_path, ["FL2VA"])
    resolved, task_type = serve_single_hpu._resolve_partition(tmp_path, "FL2VA")

    assert validation["status"] == "pass"
    assert resolved == partition
    assert task_type == "fl2va"
    assert download_modelscope._patterns(["FL2VA"])[-1] == "FL2VA/**"


def test_modelscope_validation_rejects_non_native_fp8(tmp_path):
    partition = _write_fp8_partition(tmp_path, "Ref2VA")
    config_path = partition / "transformer" / "config.json"
    config_path.write_text(json.dumps({"quantization_config": {"quant_algo": "FP8"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="not the required native ModelOpt"):
        download_modelscope._validate_native_fp8(tmp_path, ["Ref2VA"])


def test_single_hpu_cli_accepts_options_after_model_and_forwards_only_after_separator():
    args = serve_single_hpu._parse_args(
        ["/models/h3", "--partition", "FL2VA", "--port", "9000", "--", "--stage-init-timeout", "1800"])

    assert args.model == Path("/models/h3")
    assert args.partition == "FL2VA"
    assert args.port == 9000
    assert args.vllm_args == ["--stage-init-timeout", "1800"]


def test_benchmark_media_contract_and_completed_progress(tmp_path):
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1344,
                "height": 768,
                "r_frame_rate": "24/1",
                "nb_read_frames": "120",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "32000",
                "channels": 2,
            },
        ],
        "format": {
            "duration": "5.000000"
        },
    }
    server_log = tmp_path / "server.log"
    server_log.write_text("0%| | 0/49\r100%|########| 49/49\n", encoding="utf-8")

    result = benchmark_t2va._validate_media(probe, width=1344, height=768, frames=120, exact_duration=5.0)

    assert result["status"] == "pass"
    assert benchmark_t2va._progress_total(server_log, 0) == 49
