# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import pytest

from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

_REPO_ROOT = Path(__file__).parents[3]


def _load_tool(name: str):
    path = _REPO_ROOT / "tools" / "minimax_h3" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


request_video = _load_tool("request_video")
serve_single_hpu = _load_tool("serve_single_hpu")
download_modelscope = _load_tool("download_modelscope")
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


def test_request_cli_resolves_to_20_forward_base_schedule():
    assert request_video._resolve_sigma_points(None, None, False) == 21
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
    }
    values.update(overrides)
    return Namespace(**values)


def test_benchmark_defaults_to_20_forward_base_schedule(tmp_path):
    command = benchmark_t2va._request_command(_benchmark_args(), tmp_path)

    assert command[command.index("--sigma-points") + 1] == "21"
    assert "--pinned-omni-reference" not in command


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
