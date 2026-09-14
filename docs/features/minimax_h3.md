# MiniMax H3 native FP8 on Gaudi

MiniMax H3 support is part of this `vllm_gaudi` project. The package owns the
HPU platform, attention, ModelOpt FP8 loading, H3 weight mapping, launch tools,
and tests under `vllm_gaudi.omni`. A pinned vLLM Omni checkout supplies the
model-independent serving pipeline through the optional dependency group; it
is not a separately maintained deployment.

## Qualified dependency set

| Dependency | Qualified revision or version |
| --- | --- |
| vLLM | `680e2177` |
| vLLM Omni | `767cc7977e04dc1f1ae7307e630429e9169af622` |
| PyTorch | 2.11 |
| Habana framework plugin | 1.24.1 |
| Transformers | 5.14.1 |
| Diffusers | 0.40.0 |
| ModelScope | 1.40.0 |
| ImageIO FFmpeg | 0.6.0 |

Install the multimodal dependencies only where H3 serving is needed:

```bash
python -m pip install -e '.[omni]'
```

Text-only installations do not pull the audio/video stack.

## ModelScope-only artifacts

The normal launcher accepts local paths only and forces Hugging Face,
Transformers, Diffusers, and datasets offline. The pinned upstream Omni recipe
publishes an online-FP8 mode for the official BF16 Base checkpoint; it does not
name an official serialized ModelOpt H3 checkpoint. The native-FP8 path in this
project accepts a local or privately mirrored ModelScope derivative and audits
both the DiT and encoder before serving it:

```bash
python tools/minimax_h3/download_modelscope.py \
  ORG/MiniMax-H3-ModelOpt-Dynamic-FP8 \
  --local-dir /data/models/MiniMax-H3-FP8 \
  --partition FL2VA \
  --manifest /data/models/MiniMax-H3-FP8/modelscope-download-manifest.json
```

Replace `ORG/...` with the actual ModelScope repository ID. The official
[`MiniMax/MiniMax-H3`](https://modelscope.cn/models/MiniMax/MiniMax-H3)
repository contains BF16 Base weights and is intentionally rejected by this
native-checkpoint helper. Both the DiT and Qwen encoder must declare ModelOpt
`FP8_PER_CHANNEL_PER_TOKEN`.

For the recommended four-forward T2VA schedule, download the exact FlashGen
adapter from ModelScope. It is 1.26 GB and carries its own sampler contract:

```bash
python tools/minimax_h3/download_flashgen_modelscope.py \
  --local-dir /data/models/Minimax-H3-4step-lora-flashgen \
  --manifest /data/models/Minimax-H3-4step-lora-flashgen/modelscope-download-manifest.json
```

Audit an already present checkpoint without loading its large tensors:

```bash
python tools/minimax_h3/validate_checkpoint.py \
  /data/models/MiniMax-H3-FP8/FL2VA \
  /data/models/MiniMax-H3-FP8/Ref2VA \
  --output checkpoint-audit.json
```

## Single-Gaudi launch

The launcher takes a free-device lock, checks that the card is idle, binds the
server to the card's NUMA CPUs, and sets both `HABANA_VISIBLE_MODULES` and
`HLS_MODULE_ID`. Existing accelerator jobs are left untouched.

```bash
VLLM_GAUDI_LOCK_DIR=/data/gaudi-locks \
python tools/minimax_h3/serve_single_hpu.py \
  /data/models/MiniMax-H3-FP8 \
  --partition FL2VA \
  --flashgen-4step-lora \
  /data/models/Minimax-H3-4step-lora-flashgen/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors \
  --module 0 \
  --port 8097
```

The default single-card placement keeps the selected DiT and both VAEs on the
HPU and applies layerwise staged offload to the text encoder. Use
`--offload-component dit` as well only when the resident DiT does not fit; it
streams the 50 DiT blocks every denoising interval and is much slower. Extra
vLLM arguments go after an explicit `--`.

Start the reference partition by changing `--partition Ref2VA`. Only one
partition is loaded, and shared components are instantiated once.

## Sampling workflows

H3 has several public sampling contracts. Weight precision
does not select one: INT8 or FP8 describes linear-layer storage and compute,
while the sampler and distilled adapter determine the number of denoiser calls.

| Workflow | Sampler contract | Actual joint DiT calls |
| --- | --- | ---: |
| [ComfyUI Base template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json) | `res_multistep` + `simple` | 20 |
| [Pinned Omni Base reference](https://github.com/vllm-project/vllm-omni/blob/767cc7977e04dc1f1ae7307e630429e9169af622/recipes/MiniMaxAI/MiniMax-H3.md) | uniform `euler_eta0`, 50 sigma points | 49 |
| [FlashGen native LoRA](https://modelscope.cn/models/FlashGen/Minimax-H3-4step-lora-flashgen) | adapter-owned `1.0,0.7,0.4,0.15,0.0` | 4 |
| [Matching LightX2V Turbo LoRA](https://github.com/ModelTC/LightX2V/) | artifact-owned distilled schedule | 4 or 8 |

The ComfyUI Base template and pinned Omni Base reference use different
numerical solvers. A Base checkpoint cannot be turned into a four-step model by
changing one request integer; the four-step schedule requires its matching
distilled adapter.

Video and audio latents occupy one packed sequence and are predicted in the
same transformer call. H3 Base is CFG-distilled, so each denoising step uses one
joint branch rather than separate video, audio, positive, and negative calls.
The released Base shifts remain 12 for video and 3 for audio.

The recommended single-Gaudi T2VA profile uses FlashGen and makes exactly four
joint DiT calls. Its API field is an interval count (`num_inference_steps=4`),
while Base and LightX2V's uniform-grid contracts count sigma points. The helper
keeps these meanings separate. Without a distilled adapter, omission follows
the pinned Omni Base reference (50 points, 49 calls); `--sigma-points` remains
an expert-only Base experiment.

## Requests

Generate 768p T2VA with the ModelScope FlashGen four-forward preset:

```bash
python tools/minimax_h3/request_video.py \
  --task t2va \
  --prompt 'A fox walks through snow with synchronized footsteps and winter wind.' \
  --width 1344 --height 768 --aspect-ratio 16:9 \
  --duration 5 --seed 2101 \
  --flashgen-4step-lora \
  /data/models/Minimax-H3-4step-lora-flashgen/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors \
  --output t2va-internal-124f.mp4 \
  --metadata t2va.json
```

Omit the FlashGen option to run the pinned Omni Base quality reference with 49
actual DiT forwards. FlashGen v1.0 serves T2VA only; FL2VA and Ref2VA examples
below use their Base partitions unless a matching task-family adapter is added.

On an FL2VA server, select first frame, last frame, or both:

```bash
# First frame
python tools/minimax_h3/request_video.py --task fl2va \
  --prompt 'Continue this scene with natural motion and matching ambience.' \
  --reference first.png --frame-indices 0 --duration 5 --output first.mp4

# Last frame
python tools/minimax_h3/request_video.py --task fl2va \
  --prompt 'Move naturally toward the supplied final composition.' \
  --reference last.png --frame-indices -1 --duration 5 --output last.mp4

# First and last frames, in that order
python tools/minimax_h3/request_video.py --task fl2va \
  --prompt 'Create a coherent transition with synchronized ambience.' \
  --reference first.png --reference last.png --frame-indices 0 -1 \
  --duration 5 --output first-last.mp4
```

On a Ref2VA server, upload mixed local references in their prompt order:

```bash
python tools/minimax_h3/request_video.py --task ref2va \
  --prompt 'Use <Picture 1> for the subject, <Video 1> for motion, and <Audio 1> for sound.' \
  --reference subject.png --reference motion.mp4 --reference voice.wav \
  --aspect-ratio 16:9 --duration 5 --output ref2va.mp4
```

Ref2VA accepts at most nine images, three videos, three audio clips, and twelve
files in total. It requires at least one image or video; audio-only requests
are rejected. FL2VA accepts `[0]`, `[-1]`, or `[0, -1]` keyframe indices.

## Reproduce the 5-second single-card measurement

After a fresh server reaches `/health`, find its worker PID in the startup log
and run:

```bash
python tools/minimax_h3/benchmark_t2va.py \
  --output-dir /data/evidence/h3-t2va-gaudi2 \
  --server-log /data/evidence/h3-server/server.log \
  --server-pid "$SERVER_PID" \
  --module 0 \
  --flashgen-4step-lora \
  /data/models/Minimax-H3-4step-lora-flashgen/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
```

The tool makes one first request followed by three measured requests. The
command above checks the server log for exactly four DiT forwards. Use
`--pinned-omni-reference` only for a deliberate 49-forward Base reproduction,
or set `--sigma-points` explicitly for another controlled Base experiment.
Each request uses 1344x768, 24 FPS, five seconds, seed 2101, and validates a
124-frame internal MP4. It retains that file, trims a
separate delivery file to exactly 120 frames/five seconds, runs full video and
audio decode through the Habana FFmpeg build, samples HBM and host memory, and
reports the warmed median and range. Profiling is disabled for these main
measurements.

For a separate phase diagnostic, restart the same launcher with the pinned
Omni profiler enabled. Do not mix this request into the four-run result above:

```bash
python tools/minimax_h3/serve_single_hpu.py \
  /data/models/MiniMax-H3-FP8 --partition FL2VA --module 0 --port 8097 \
  -- --enable-diffusion-pipeline-profiler

python tools/minimax_h3/request_video.py \
  --task t2va --prompt 'A fox walks through snow.' \
  --width 1344 --height 768 --duration 5 --seed 2101 \
  --output profiled.mp4 --metadata profiled.json
```

The metadata decodes the sync endpoint headers into typed server inference,
pipeline-stage, and peak-memory fields. H3 reports prompt encoding, joint
denoising, video VAE, and audio VAE separately. The server log reports MP4/AAC
encoding time; the remaining server interval contains tensor transfer and
request orchestration. Profiler synchronization changes the timing, so these
numbers explain the phase split and do not replace the unprofiled wall-clock
result.

The HPU ModelOpt path dynamically scales each activation row, applies stored
per-output-channel weight scales, and dispatches `hpu.fp8_gemm_v2`. Selected
checkpoint linears therefore execute FP8 MME work; BF16/FP32 ignore layers,
biases, norms, and VAE operations keep their declared precision. Host work is
limited to tokenization, media preparation, request scheduling, and MP4
encoding.
