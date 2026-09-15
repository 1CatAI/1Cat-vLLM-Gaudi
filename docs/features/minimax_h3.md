# MiniMax H3 on Gaudi

MiniMax H3 support is part of this `vllm_gaudi` project. The package owns the
HPU platform, attention, BF16 and optional ModelOpt FP8 loading, H3 weight
mapping, launch tools, and tests under `vllm_gaudi.omni`. A pinned vLLM Omni
checkout supplies the model-independent serving pipeline through the optional
dependency group; it is not a separately maintained deployment.

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
Transformers, Diffusers, and datasets offline. FastH3 publishes BF16
Dense-DataFree adapter tensors for the official BF16 MiniMax H3 base. The
qualified default therefore fuses the adapter into the BF16 DiT and leaves the
model in BF16. To avoid downloading duplicate unchanged assets, prepare the
official BF16 DiT and text encoder while reusing existing VAE, tokenizer, and
processor files:

```bash
python tools/minimax_h3/prepare_fasth3_modelscope.py \
  --local-dir /data/models/MiniMax-H3-FastH3-HPU \
  --reuse-components-from /data/models/MiniMax-H3-FP8

python tools/minimax_h3/download_fasth3_modelscope.py \
  --local-dir /data/models/FastVideo-FastH3-4-step-Preview-v1-LoRA

python tools/minimax_h3/download_lightx2v_modelscope.py \
  --local-dir /data/models/LightX2V-Minimax-h3-Turbo

python tools/minimax_h3/download_lightx2v_modelscope.py \
  --local-dir /data/models/LightX2V-Minimax-h3-Turbo \
  --profile ref2v-4step-544p

python tools/minimax_h3/prepare_ref2va_modelscope.py \
  --local-dir /data/models/MiniMax-H3-LightX2V-HPU \
  --reuse-components-from /data/models/MiniMax-H3-FastH3-HPU
```

All downloads use ModelScope exclusively. The preparation helper downloads
the official FL2VA BF16 transformer and BF16 text encoder. It reuses only the
unchanged video/audio VAE, tokenizer, and processor files. At startup the
Dense-DataFree adapter is fused into BF16 and model computation remains BF16 by
default. The qualified LightX2V artifact is the exact Diffusers-layout
`minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`: rank 128, alpha 128,
and BF16 throughout. Its downloader refuses renamed, truncated, wrong-layout,
or non-BF16 files and records a SHA256 manifest.

The Ref2VA preparation command first checks ModelScope's content hashes for
every shared file. The BF16 text encoder, video/audio VAEs, tokenizer, and
processor are byte-identical across the official FL2VA and Ref2VA partitions,
so the command links the already verified local files and downloads only the
66.3 GB Ref2VA transformer. The matching
`minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` is BF16, rank 128,
alpha 8, trained for four forwards at 544p with video/audio shifts 12/3.

Online FP8 remains available as an explicit A/B experiment. Add `--online-fp8`
to the launch command only when testing that path. It runs after FastH3 fusion
and quantizes eligible BF16 linears to per-output-channel weights with dynamic
per-token activations on HPU; it is not the official FastH3 inference default.

The serialized native-FP8 path still accepts a local or privately mirrored
ModelScope derivative and audits both the DiT and encoder before serving it:

```bash
python tools/minimax_h3/download_modelscope.py \
  ORG/MiniMax-H3-ModelOpt-Dynamic-FP8 \
  --local-dir /data/models/MiniMax-H3-FP8 \
  --partition FL2VA \
  --manifest /data/models/MiniMax-H3-FP8/modelscope-download-manifest.json
```

Replace `ORG/...` with the actual ModelScope repository ID. The official
[`MiniMax/MiniMax-H3`](https://modelscope.cn/models/MiniMax/MiniMax-H3)
repository contains BF16 Base weights; the native-checkpoint downloader remains
strict and accepts only derivatives where both the DiT and Qwen encoder declare
ModelOpt `FP8_PER_CHANNEL_PER_TOKEN`.

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
  /data/models/MiniMax-H3-FastH3-HPU \
  --partition FL2VA \
  --fasth3-4step-adapter \
  /data/models/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  --module 0 \
  --temp-dir /data/tmp/vllm-h3 \
  --port 8097
```

The default single-card placement applies layerwise staged offload to the text
encoder and initializes both VAEs on the host. On the first request, after any
static LoRA has been activated, it creates an immutable pinned CPU master for
the selected DiT. Prompt and reference encoding run with the DiT on the host;
the denoise phase loads it once and keeps it resident for every denoising
forward. After denoising, offload rebinds the parameters to their CPU master and
releases device storage without copying the immutable weights back from HPU.
This placement is required for Ref2VA's 2048-short-edge visual conditioning to
fit on one Gaudi 2. The first CPU snapshot and every later host-to-device load
remain included in request wall time. Use `--no-phase-offload` only for a
resident-memory comparison. Base and dynamic LoRA profiles can also add
`--offload-component dit`, although that streams the 50 DiT blocks every
denoising interval and is much slower. FastH3 rejects that layerwise DiT mode
because fusion must pass through the ordinary DiT weight stream; the
once-per-phase transfer keeps the fused weights intact.
Extra vLLM arguments go after an explicit `--`.
Use `--temp-dir` (or `VLLM_GAUDI_H3_TMPDIR`) to keep preprocessing scratch
files on the data volume. The launcher exports the resolved directory through
`TMPDIR`, `TMP`, and `TEMP` before the worker starts.

Video decode keeps the checkpoint's native temporal chunks, spatial tile size,
overlap, and stitching order. On HPU the launcher decodes four independent
spatial tiles per VAE call and stores decoder Linear parameters in BF16, the
same dtype selected by HPU autocast. This avoids repeated small launches and
weight casts while preserving the sequential decoder's output bytes. Use
`--vae-tile-batch-size 1` to reproduce sequential tile execution, or
`--no-vae-persist-bf16-weights` to retain FP32 parameter storage.

The launcher also validates `ffmpeg` and `ffprobe` before starting a Ref2VA
worker. `--media-bin` defaults to `/opt/habanalabs/media/ffmpeg/bin` and is
prepended to `PATH`; set it explicitly when the Habana media tools live
elsewhere. For reference-video normalization, the H3 patch uses Omni's
`libx264rgb` path when that encoder is present. Habana builds without
`libx264rgb` automatically use lossless FFV1 in Matroska, preserving the
reference frames without changing the pinned Omni source tree.

To run the qualified LightX2V workflow instead, keep the same official BF16
FL2VA base and replace the FastH3 option with:

```bash
python tools/minimax_h3/serve_single_hpu.py \
  /data/models/MiniMax-H3-FastH3-HPU \
  --partition FL2VA \
  --lightx2v-4step-lora \
  /data/models/LightX2V-Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --module 0 --temp-dir /data/tmp/vllm-h3 --port 8097
```

This starts Omni's dynamic PEFT LoRA manager and keeps phase offload enabled.
Layerwise DiT offload and online FP8 are rejected for this qualification
profile so the executed graph stays on the published BF16 base-plus-LoRA path.

Start the four-step LightX2V reference service from the assembled Ref2VA
partition:

```bash
python tools/minimax_h3/serve_single_hpu.py \
  /data/models/MiniMax-H3-LightX2V-HPU \
  --partition Ref2VA \
  --lightx2v-4step-lora \
  /data/models/LightX2V-Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --module 0 --temp-dir /data/tmp/vllm-h3-ref2va --port 8097
```

Only the selected partition is loaded, and every shared component is
instantiated once. The Gaudi integration admits the release's 544-pixel canvas
only for the Ref2VA partition and task. Its Qwen3-VL vision and text stacks use
Habana FusedSDPA, including one attention segment per uploaded reference, so
the 2048-short-edge reference encoder does not materialize quadratic attention
matrices.

## Sampling workflows

H3 has several public sampling contracts. Weight precision
does not select one: INT8 or FP8 describes linear-layer storage and compute,
while the sampler and distilled adapter determine the number of denoiser calls.

| Workflow | Sampler contract | Actual joint DiT calls |
| --- | --- | ---: |
| [ComfyUI Base template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json) | `res_multistep` + `simple` | 20 |
| [Pinned Omni Base reference](https://github.com/vllm-project/vllm-omni/blob/767cc7977e04dc1f1ae7307e630429e9169af622/recipes/MiniMaxAI/MiniMax-H3.md) | uniform `euler_eta0`, 50 sigma points | 49 |
| [FastH3 Dense-DataFree](https://modelscope.cn/models/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA) | adapter-owned `0.999,0.749,0.5,0.25,0.0` | 4 |
| [FlashGen native LoRA](https://modelscope.cn/models/FlashGen/Minimax-H3-4step-lora-flashgen) | adapter-owned `1.0,0.7,0.4,0.15,0.0` | 4 |
| [LightX2V FL2V Turbo v1.0 768p BF16](https://modelscope.cn/models/lightx2v/Minimax-h3-Turbo) | 5 sigma points, video shift 6, audio shift 3 | 4 |
| [LightX2V Ref2V Turbo v0.1 544p BF16](https://modelscope.cn/models/lightx2v/Minimax-h3-Turbo) | 5 sigma points, video shift 12, audio shift 3 | 4 |

The ComfyUI Base template and pinned Omni Base reference use different
numerical solvers. A Base checkpoint cannot be turned into a four-step model by
changing one request integer; the four-step schedule requires its matching
distilled adapter.

Video and audio latents occupy one packed sequence and are predicted in the
same transformer call. H3 Base is CFG-distilled, so each denoising step uses one
joint branch rather than separate video, audio, positive, and negative calls.
The released Base shifts remain 12 for video and 3 for audio.

The first single-Gaudi qualification profile uses FastH3 Dense-DataFree and
makes exactly four joint DiT calls. Its API field is an interval count
(`num_inference_steps=4`). FlashGen uses the same interval count with a different
adapter-owned schedule, while Base and LightX2V contracts use their own grid
semantics. The helper keeps these meanings separate. Without a distilled
adapter, omission follows the pinned Omni Base reference (50 points, 49 calls);
`--sigma-points` remains an expert-only Base experiment.

For the qualified LightX2V file, `num_inference_steps=5` means five sigma grid
points including terminal zero, which bound exactly four DiT evaluations. This
is an API convention of the LightX2V integration; it is not a fifth denoising
step. The request helper owns all three values and refuses manual overrides.

## Requests

Generate T2VA through a server with FastH3 fused at startup:

```bash
python tools/minimax_h3/request_video.py \
  --task t2va \
  --prompt 'A fox walks through snow with synchronized footsteps and winter wind.' \
  --width 1344 --height 768 --aspect-ratio 16:9 \
  --duration 5 --seed 2101 --fasth3-4step \
  --preencode-mp4 --preencode-batch-frames 17 \
  --output t2va-internal-124f.mp4 \
  --metadata t2va.json
```

The FlashGen four-forward preset remains available as a request-time LoRA:

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

Generate T2VA through the LightX2V server with its exact four-step contract:

```bash
python tools/minimax_h3/request_video.py \
  --task t2va \
  --prompt 'A fox walks through snow with synchronized footsteps and winter wind.' \
  --width 1344 --height 768 --aspect-ratio 16:9 \
  --duration 5 --seed 2101 \
  --lightx2v-4step-lora \
  /data/models/LightX2V-Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --preencode-mp4 --preencode-batch-frames 17 \
  --output lightx2v-t2va-internal-124f.mp4 \
  --metadata lightx2v-t2va.json
```

The helper sends five sigma points, flow shifts 6/3, and dynamic LoRA scale 1.
The matching LightX2V file also supports FL2VA: change `--task` to `fl2va` and
add the first and/or last image arguments shown below.

Omit all distilled options to run the pinned Omni Base quality reference with
49 actual DiT forwards. FastH3 preview v1 and FlashGen v1.0 serve T2VA only;
use the matching LightX2V file for distilled FL2VA or Ref2VA requests.

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

On the LightX2V Ref2VA server, upload mixed local references in their prompt
order. Its published four-step v0.1 profile uses a 544-pixel short edge:

```bash
python tools/minimax_h3/request_video.py --task ref2va \
  --prompt 'Use <Picture 1> for the subject, <Video 1> for motion, and <Audio 1> for sound.' \
  --reference subject.png --reference motion.mp4 --reference voice.wav \
  --aspect-ratio 16:9 --short-edge 544 --duration 5 \
  --lightx2v-4step-lora \
  /data/models/LightX2V-Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --preencode-mp4 --output ref2va.mp4 --metadata ref2va.json
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
  --fasth3-4step
```

For the matching LightX2V server, replace the final option with:

```bash
--lightx2v-4step-lora \
/data/models/LightX2V-Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
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

The qualified Gaudi 2 run at 1344x768 produced the following BF16 baselines.
Both used four actual joint DiT forwards, generated 124 internal frames, and
delivered a separately trimmed 120-frame/five-second H.264 + stereo AAC file.
All files passed full FFmpeg decoding.

| Profile | Placement | Startup | First request | Next-three median | Range | Sampled peak HBM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FastH3 Dense-DataFree | resident-DiT baseline | 174.01 s | 277.02 s | 256.75 s | 250.48-276.79 s | 96,895 MiB |
| LightX2V FL2V Turbo 768p | default phase offload | 75.49 s | 278.14 s | 301.03 s | 300.51-302.07 s | 97,421 MiB |
| LightX2V FL2V Turbo 768p | parameter staging + batched VAE | 74.56 s | 264.46 s | 249.10 s | 246.12-258.94 s | 97,461 MiB |

The placement differs, so the table records usable baselines rather than an
isolated adapter speed comparison. Start FastH3 with `--no-phase-offload` to
reproduce its resident-DiT row exactly; the normal launcher default uses the
safer phase placement. A separate LightX2V warmed profiler request
measured 17.39 s prompt encoding, 236.42 s for the inclusive denoise phase
(21.58 s DiT load, 193.91 s four-forward schedule and setup, 20.93 s DiT
offload), 0.21 s audio VAE, 56.83 s video VAE, and 58.18 s inclusive decode and
MP4 packaging. Profiler synchronization is excluded from the main table.

With parameter staging and the batched VAE enabled, a separate warmed phase
diagnostic measured 5.51 s prompt encoding, 187.52 s for the inclusive denoise
phase (2.87 s DiT load, 184.63 s for the four forwards and scheduler work, and
0.017 s release), 0.21 s audio VAE, 41.14 s video VAE, and 1.01 s for VAE
transfers plus MP4 completion. The corresponding total was 235.38 s. The formal
four-request row above includes resource polling and was recorded while other
modules on the same host were active; the component gate below provides the
isolated causal measurement.

The HPU VAE policy was selected with a production-shape component gate before
the end-to-end run. The input latent was `[1, 24, 37, 48, 84]`; both arms
decoded 124 frames at 1344x768 using the checkpoint's 28 spatial tiles and
seven temporal chunks. Four-tile batching plus persistent BF16 Linear operands
reduced synchronized decode time from 53.96 s to 42.90 s (20.5%) while keeping
the prepared uint8 output byte-identical. Its measured allocator peak was
25.38 GiB, leaving enough room for the single-card phase placement.

A hardware trace of one decoder call explains the improvement. Moving from one
tile to four tiles reduced the idle share from 19.45% to 2.86%, raised TPC
active time from 58.27% to 67.47%, and raised MME active time from 18.96% to
27.83%. The policy keeps four as the default because larger batches provided
little latency benefit while increasing peak HBM sharply. Reproduce the
component sweep and the two trace arms with:

```bash
HABANA_VISIBLE_MODULES=0 HLS_MODULE_ID=0 PT_HPU_LAZY_MODE=0 \
python tools/minimax_h3/benchmark_vae_tile_batch.py \
  /data/models/MiniMax-H3/FL2VA/video_vae \
  --batch-sizes 1,2,4,7,14,28 \
  --reference-raw /data/evidence/h3-vae/reference-124f.rgb \
  --output /data/evidence/h3-vae/tile-batch-sweep.json

HABANA_VISIBLE_MODULES=0 HLS_MODULE_ID=0 PT_HPU_LAZY_MODE=0 \
python tools/minimax_h3/trace_vae_decoder_batch.py \
  /data/models/MiniMax-H3/FL2VA/video_vae --batch-size 1 \
  --reference-npy /data/evidence/h3-vae/first-tile.npy \
  --output-dir /data/evidence/h3-vae/trace-batch1

HABANA_VISIBLE_MODULES=0 HLS_MODULE_ID=0 PT_HPU_LAZY_MODE=0 \
python tools/minimax_h3/trace_vae_decoder_batch.py \
  /data/models/MiniMax-H3/FL2VA/video_vae --batch-size 4 \
  --reference-npy /data/evidence/h3-vae/first-tile.npy \
  --output-dir /data/evidence/h3-vae/trace-batch4
```

The same LightX2V BF16 integrations completed the conditioning workflows below.
These are functional qualification requests rather than the warmed T2VA
benchmark: the FL2VA and Ref2VA release contracts generated 107 frames for a
four-second request, and each still executed exactly four joint DiT forwards.

| Task and references | Output | Wall time | Sampled peak HBM | Validation |
| --- | --- | ---: | ---: | --- |
| FL2VA first + last frame | 1344x768, 107 frames | 300.01 s | 97,064 MiB | endpoint SSIM 0.9425 / 0.8810; full decode |
| Ref2VA image | 960x544, 107 frames | 243.58 s | 96,543 MiB | video + audio; full decode |
| Ref2VA image + video + audio | 960x544, 107 frames | 699.85 s | 96,959 MiB | video + audio; full decode |

The mixed Ref2VA request is slower because its 2048-short-edge reference-video
encoding produced a much longer conditioning sequence. It remains within one
Gaudi 2 through phase offload and segmented HPU FusedSDPA; no model-compute CPU
fallback is used.

For a separate phase diagnostic, restart the same launcher with the pinned
Omni profiler enabled. Do not mix this request into the four-run result above:

```bash
python tools/minimax_h3/serve_single_hpu.py \
  /data/models/MiniMax-H3-FastH3-HPU --partition FL2VA \
  --fasth3-4step-adapter \
  /data/models/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  --module 0 --port 8097 \
  -- --enable-diffusion-pipeline-profiler

python tools/minimax_h3/request_video.py \
  --task t2va --prompt 'A fox walks through snow.' \
  --width 1344 --height 768 --duration 5 --seed 2101 --fasth3-4step \
  --output profiled.mp4 --metadata profiled.json
```

The metadata decodes the sync endpoint headers into typed server inference,
pipeline-stage, and peak-memory fields. H3 reports prompt encoding, joint
denoising, video VAE, audio VAE, MP4 completion, and each HPU phase transfer
separately. Profiler synchronization changes the timing, so these numbers
explain the phase split and do not replace the unprofiled wall-clock result.

To capture hardware events without asking the Synapse profiler to retain a
complete video request, arm the built-in one-DiT-call trace after a warmup:

```bash
HABANA_PROFILE=profile_api_light \
VLLM_GAUDI_H3_DIT_TRACE_DIR=/data/evidence/h3-dit-trace \
python tools/minimax_h3/serve_single_hpu.py \
  /data/models/MiniMax-H3-FastH3-HPU --partition FL2VA \
  --lightx2v-4step-lora \
  /data/models/LightX2V-Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --module 0 --port 8097

# Run one unprofiled warmup, then arm exactly the next request.
touch /data/evidence/h3-dit-trace/ARM
```

The default target is DiT step zero. Set
`VLLM_GAUDI_H3_DIT_TRACE_STEP=1` before server startup to select another step,
or `VLLM_GAUDI_H3_DIT_TRACE_ARM_FILE` to place the arm file elsewhere. The
worker atomically claims the file, records CPU shapes plus HPU MME/TPC events,
and writes the Chrome trace and a metadata sidecar. Profiling adds substantial
synchronization and export overhead, so use the resulting request only for
operator analysis and keep it out of latency results.

The official FastH3 profile keeps its DiT and text-encoder model computation in
BF16. If `--online-fp8` is selected, eligible linears dynamically scale each
activation row, apply per-output-channel weight scales, and dispatch
`hpu.fp8_gemm_v2`; ignored linears, biases, norms, and VAE operations keep their
declared precision. The serialized native ModelOpt FP8 path uses the same HPU
GEMM contract. Host work is limited to tokenization, media preparation, request
scheduling, and MP4 encoding.
