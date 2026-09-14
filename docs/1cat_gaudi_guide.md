# Gaudi 工程指南

本指南配合[项目首页](../README.md)使用。命令面向 Linux / Bash 和已经安装 Gaudi 软件栈的环境；占位路径需要替换为真实路径。执行源码以 `ac14567637ca1d8f3d4434e62dbab3317ebe9fbe` 为基准。

[环境与安装](#installation) · [V4](#deepseek-v4) · [V4.1](#deepseek-v41) · [运行配置](#runtime-profile) · [Qwen](#qwen-dflash2) · [验证](#validation) · [排障](#troubleshooting)

<a id="installation"></a>

## 环境与安装

### 版本与依赖

| 项目 | 要求 |
|---|---|
| 加速器 | 本分支专用路线的目标为 Gaudi2 |
| Gaudi 软件 / PyTorch | DeepSeek 基线为 Gaudi Software 1.24.1 与配套 Gaudi PyTorch 2.11 |
| Python | 专用工具选用匹配软件栈的 3.11 / 3.12；部分脚本依赖 `hashlib.file_digest` 等 3.11+ 接口 |
| 原生开发条件 | 与所选 runtime 匹配的 Bridge 源码、生成后的 build headers、SDK 开发库与 TPC 编译器 |
| 构建工具 | Git、CMake、C/C++ 工具链、Python 构建依赖；V4.1 上游审计另需可用的 `gh` |
| 环境隔离 | 在独立 Python 环境、构建目录和 recipe cache 中准备每套候选 |

先检查目标环境：

```bash
hl-smi
python --version
python -m pip --version

python - <<'PY'
import torch
import habana_frameworks.torch.core
print("PyTorch:", torch.__version__)
print("HPU available:", torch.hpu.is_available())
print("HPU count:", torch.hpu.device_count())
assert torch.hpu.is_available(), "请先检查 Gaudi 环境与设备可见性"
PY

python -m pip install 'packaging>=24.2' 'setuptools>=77.0.3,<85.0.0' \
  'setuptools-scm>=8.0' wheel jinja2
```

硬件可见只代表环境探测成功。开始服务前记录插件、引擎、驱动、固件、PyTorch、Bridge 版本及实际设备集合。

### 插件与引擎配对

`vllm_gaudi` 是硬件插件，`vllm` 引擎需要单独安装。V4 使用下一节的固定引擎补丁链；V4.1 需要独立的配套引擎；Qwen / 通用 HPU 应选择已验证支持目标模型的引擎版本。

通用插件安装方式参考[上游安装文档](https://docs.vllm.ai/projects/gaudi/en/latest/getting_started/installation.html)。上游的 last-good-commit 属于其验证组合，不应自动视为本分支每条实验路线的模型验证结果。

已有 Gaudi torch 应保持与软件栈配对。对需要 torchaudio 的模型，在确认版本匹配后选用 CPU wheel，并避免依赖解析另装 CUDA torch。

<a id="deepseek-v4"></a>

## DeepSeek V4 Flash

### 固定引擎并构建

从插件仓库根目录执行：

```bash
git clone https://github.com/vllm-project/vllm.git ../vllm-dsv4
git -C ../vllm-dsv4 checkout fe755c88995ad468882517b6c4bdd60138d46a3a
python tools/prepare_deepseek_v4_engine.py ../vllm-dsv4

python -m pip install -r <(sed '/^torch/d' ../vllm-dsv4/requirements/build/cuda.txt)
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e ../vllm-dsv4
python -m pip install --no-build-isolation -e .

export GAUDI_PYTORCH_BRIDGE_ROOT=/path/to/matching/gaudi-pytorch-bridge
export GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=/path/to/matching/bridge/build

python tools/build_deepseek_v4.py --jobs "$(nproc)"
ctest --test-dir build/deepseek_v4/kernels --output-on-failure
```

默认库目录为 `vllm_gaudi/lib`，应包含 kernel database、一个匹配的 PT2 扩展和构建 manifest。仅使用 `--kernel-only` 不能满足完整入口要求。

### 启动与默认值

```bash
HABANA_VISIBLE_MODULES=0,1 \
python -m vllm_gaudi.entrypoints.deepseek_v4 \
  /path/to/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --host 127.0.0.1 --port 8000
```

| 参数 | 专用入口值 |
|---|---|
| TP / PP | 2 / 1 |
| 最大总上下文 / 最大 batch tokens | 512 / 512 |
| 最大序列数 | 1 |
| dtype / KV dtype | bfloat16 / fp8 |
| Prefix caching / async scheduling | 关闭 / 开启 |
| GPU memory utilization | **0.09** |

`0.09` 是代码中的实际值；它不代表全部运行时分配的硬上限。Attention 投影缓存使用 BF16 计算。普通入口使用区域编译，完整 native decoder 另需相应 runtime、通信计划和显式启用。

源码说明：[V4 源码集成](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/docs/features/deepseek_v4_flash.md) · [V4 native decoder](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/docs/features/deepseek_v4_native_decode.md)。

<a id="deepseek-v41"></a>

## DeepSeek V4.1 Flash

### 准备条件

这是一条 **Gaudi2、TP2×PP2、单请求、512-token 总上下文、贪心**的专用实验路线。当前仓库有模型集成、权重准备和 runtime 补丁；V4.1 的可安装引擎锁定包尚未完整归档。因此本节从已验证的配套引擎环境开始，不能把某个上游 PR 的最新 head 当成经过验证的安装版本。

需要成套保留：

| 制品 | 对应要求 |
|---|---|
| V4.1 引擎 | 真实 commit、实际补丁和安装路径；包含模型所需的 `vllm.models.deepseek_v4_1` 等接口 |
| 插件 | 固定 commit 与对应原生扩展 |
| Native replay runtime | Bridge、Synapse、HCL 源码与补丁指纹、构建产物 |
| TP2 扩展 | 对应 runtime 的扩展及 `.abi.json` |
| Prepared 模型 | 四个 rank 文件、模型 manifest、Engram host-shard manifests |
| Sidecar | `sidecars/wo_a_fp8`、`sidecars/attention_dense_fp8` 及其 manifests |
| 运行配置 | 对真实库、配置文件计算指纹的 runtime profile |

Native runtime 的源码仓库与 base commit 已列于[manifest](../tools/communication/patches/native-runtime/manifest.json)，补丁与构建顺序见[runtime 文档](../tools/communication/patches/native-runtime/README.md)。该 Synapse 路线使用旧公开源码及兼容补丁；ABI 匹配不等于与已安装 SDK 的编译器和数值完全等价。

`audit_deepseek_v41_upstream.py` 收集上游 PR 的状态与 diff 证据，**不安装引擎，也不生成已验证的引擎兼容锁**。设备运行工具会另行归档实际 `engine_commit` 与 `engine.patch`，这些记录才对应真实运行环境。

### 审计与权重准备

```bash
export MODEL_DIR=/data/models/DeepSeek-V4.1-Flash-source
export PREPARED_DIR=/data/models/DeepSeek-V4.1-Flash-prepared
export CHECKPOINT_AUDIT_DIR=/data/evidence/dsv41-checkpoint
export UPSTREAM_AUDIT_DIR=/data/evidence/dsv41-upstream

python tools/audit_deepseek_v41_upstream.py "$UPSTREAM_AUDIT_DIR"
python tools/sync_deepseek_v41_checkpoint.py "$MODEL_DIR" \
  --evidence "$CHECKPOINT_AUDIT_DIR"

python tools/prepare_deepseek_v41_shards.py "$MODEL_DIR" "$PREPARED_DIR" \
  --checkpoint-audit "$CHECKPOINT_AUDIT_DIR/complete.json" \
  --upstream-lock "$UPSTREAM_AUDIT_DIR/upstream-lock.json" \
  --plan-only

python tools/prepare_deepseek_v41_shards.py "$MODEL_DIR" "$PREPARED_DIR" \
  --checkpoint-audit "$CHECKPOINT_AUDIT_DIR/complete.json" \
  --upstream-lock "$UPSTREAM_AUDIT_DIR/upstream-lock.json"

python tools/prepare_deepseek_v41_woa_fp8.py "$PREPARED_DIR"
python tools/prepare_deepseek_v41_dense_fp8.py "$PREPARED_DIR"
```

上游审计输出目录应为新目录。checkpoint 同步工具会写入目标模型目录，使用独立路径。普通 shard 准备的 `--checkpoint-audit` 接收 **complete.json 文件**；服务入口接收**审计目录**。

`--plan-only` 会写入准备计划并输出 payload 与临时空间预算。生成 rank 文件时，工具要求剩余 payload 空间外加 8 GiB 预留；这不含源 checkpoint、sidecar 和其他构建目录，不能当成整机磁盘需求。完成过的 prepared checkpoint 不可原地覆盖。

**Engram 原始源文件必须保留。** 四个 rank 文件不包含完整主机表；部署容量需要同时考虑源文件、prepared、sidecar、主机页缓存和构建缓存。当前未给出经过验证的通用最低 RAM 数字。

### 原生扩展

```bash
export GAUDI_PYTORCH_BRIDGE_ROOT=/path/to/matching/gaudi-pytorch-bridge
export GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=/path/to/matching/bridge/build
export VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR=/data/build/dsv41-native/lib

python tools/build_deepseek_v41.py \
  --build-root /data/build/dsv41-native/build \
  --output-dir "$VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR" \
  --jobs "$(nproc)"
```

该工具构建 DeepSeek kernel / Bridge 扩展与 Engram host gather。patched Bridge / Synapse / HCL 和 TP2 通信扩展需要依照前述 runtime 文档另行构建；复制一个 `.so` 无法代替整套产物。

### 服务状态

PP0 拥有 target 层 0–19、embedding、vision 和两个 Engram 主机表；PP1 拥有层 20–39、输出 norm / head，以及可选的三层 DSpark draft。PP 边界同时传递 hidden states 和后续需要的 mHC mixing coefficients。

专用入口默认启用普通 C1 组合，自动查找两个 sidecar。普通 C1 不加载 `mtp.*`，不创建 draft state，也不执行 proposal / verification。通用 vLLM 入口仍使用原始显式开关；DSpark 默认关闭。

`VLLM_HPU_DSV41_DEFAULT_FASTPATHS=0` 只阻止默认值注入，不清理已有变量，也不恢复参考环境。对照或 DSpark 实验应使用完整、独立的归档配置。

启动命令见[首页](../README.md#quick-start)；默认总上下文与 batch tokens 均为 512，序列数为 1，loader 为 `dsv41_prepared`，block size 为 512，prefix caching 与 async scheduling 均关闭。接口与状态细节见[V4.1 功能文档](features/deepseek_v41.md)。

<a id="runtime-profile"></a>

## Runtime profile 与设备选择

### 配置结构与生成

profile 记录**已构建 runtime**的选择方式与指纹，不能用它代替构建或完成模型验证。其入口消费的结构为：

```json
{
  "environment": {
    "PT_HPU_LAZY_MODE": "0",
    "VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR": "/absolute/path/to/native/lib",
    "VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE": "/absolute/path/to/tp2_extension.so"
  },
  "additional_libraries": [
    {"path": "/absolute/path/to/library.so", "sha256": "<实际 SHA256>"}
  ],
  "configuration_files": [
    {"path": "/absolute/path/to/tp2_extension.abi.json", "sha256": "<实际 SHA256>"}
  ]
}
```

上面只展示结构，实际 `environment` 应来自已验证的 runtime 配置，并包含必要的库选择与搜索路径。扩展 ABI manifest 会进一步校验依赖；profile 的指纹检查不会代替这些检查。

记录方法：把经过验证的环境变量保存为 `runtime-environment.json`，把实际所用库与配置文件的绝对路径分别逐行放入 `runtime-libraries.txt`、`runtime-configs.txt`，然后计算指纹：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

environment = json.loads(Path("runtime-environment.json").read_text())
if not isinstance(environment, dict) or not environment:
    raise ValueError("需要已有 runtime 的非空环境配置")
if not all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items()):
    raise ValueError("environment 的键和值必须是字符串")

def records(filename):
    result = []
    for line in Path(filename).read_text().splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"需要真实文件的绝对路径: {path}")
        path = path.resolve()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result.append({"path": str(path), "sha256": digest})
    if not result:
        raise ValueError(f"{filename} 不能是空清单")
    return result

profile = {
    "environment": environment,
    "additional_libraries": records("runtime-libraries.txt"),
    "configuration_files": records("runtime-configs.txt"),
}
with Path("verified-runtime-profile.json").open("x") as stream:
    json.dump(profile, stream, indent=2)
    stream.write("\n")
PY
```

库清单应覆盖实际使用的 Bridge / HCCL binding / Synapse / HCL / TP2 及模型原生库；配置清单应覆盖 ABI 与模型扩展 build manifests。文件变更后重新生成 profile，并重新完成相应验证。

### 环境优先级

普通入口依次合并 **进程环境 → profile.environment → 用户配置 environment**。因此 profile 或用户配置包含 `HABANA_VISIBLE_MODULES` 时，命令前的选卡值可能被覆盖。用户配置路径为 `~/.config/1cat-vllm/deepseek-v41.json`。

首页采用 `tools/run_deepseek_v41.py`：先加载 profile，再选择并锁定模块，写入实际设备与证据路径；专用入口会保留该启动器拥有的值。多个任务需要使用同一个共享 lock 目录。脚本还检查实际设备占用；已有其他进程占用的模块不可强行复用。

<a id="qwen-dflash2"></a>

## Qwen GDN 与 DFlash2

GDN 默认 recurrent state 为 FP32。重点 prefill 形状为 TP1 的 Q/K heads 16、V heads 48，或 TP2 每 rank 的 8 / 24；K / V dimension 均为 128，BF16 输入、chunk 128、单条均匀序列。其他形状由分派逻辑选择相应实现。

在已配对并完成目标模型验证的环境中，DFlash2 启动示例为：

```bash
export VLLM_HPU_FLASHINFER_GDN=1
export VLLM_HPU_FLASHINFER_DFLASH2=1
export VLLM_COMPACT_GDN=1
export FLASHINFER_GAUDI_BACKEND=auto

HABANA_VISIBLE_MODULES=0 \
vllm serve /path/to/Qwen3.8-27B-FP8 \
  --served-model-name qwen-dflash2 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --max-num-seqs 16 \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"dflash","model":"/path/to/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}' \
  --host 127.0.0.1 --port 8000
```

使用 `temperature=0` 的文本请求，关闭 LoRA。当前 DFlash2 限定 TP1 / PP1 / DP1，不接收多模态输入。它提出七个 draft tokens，target 验证八 token block，并保留卷积与 GDN 状态供回滚。

检查启动时实际 KV 容量：draft Attention cache 也占空间，客户端并发 16 个请求不证明设备正在执行 16 并发。首页并发 32 的 Qwen 历史记录没有确认采用 DFlash2，不能用来扩展 DFlash2 的支持范围。

更多 API、原生候选与状态条件见[FlashInfer-Gaudi](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/docs/features/flashinfer_gaudi.md)。

<a id="api-contract"></a>

## V4.1 API 条件

| 请求选项 | 当前要求 |
|---|---|
| 温度 | `temperature=0` |
| 返回 logprobs | `logprobs` 与 `prompt_logprobs` 均不支持 |
| 惩罚参数 | presence / frequency penalty 为 0；repetition penalty 为 1 |
| Token 约束 | 不附加 allowed token IDs、bad words、logit bias 或非零 min_tokens |
| 结构化输出 | 暂不支持 |
| 上下文 | 模板和系统提示也计入 512-token 总预算 |

依据：[采样校验代码](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/vllm_gaudi/ops/deepseek_v41_config.py#L16)。普通客户端若自动附带上述选项，需要相应关闭。

<a id="validation"></a>

## 验证与性能记录

按环境、算子数值、完整模型输出、请求性能、重复请求与退出顺序验证。不同层次的通过结果不能代替其他层次。

```bash
python -m pytest -q tests/unit_tests/test_deepseek_v41_entrypoint.py

python tools/benchmark_flashinfer_gaudi_gdn_prefill.py \
  --tokens 2048,4096,16384 --iterations 10 --waves 7

python tools/benchmark_flashinfer_gaudi_gdn.py \
  --batches 1,8,16,32 --timing-mode both \
  --reference-layout legacy --backend pytorch

python tools/benchmark_flashinfer_gaudi_dflash2.py \
  --batches 1,2,4,8,16 --warmups 20 --wave-iterations 100 --waves 7

FLASHINFER_GAUDI_RUN_HARDWARE_TESTS=1 \
python -m pytest -q tests/unit_tests/ops/test_flashinfer_norm_quant_hardware.py
```

这些工具各有软件、原生构建及设备条件；先阅读 `--help`，在已保留的设备上运行。它们分别测量自己的契约，不能汇总成一个全模型加速倍数。

记录插件/引擎 SHA、模型 revision、runtime 指纹、设备数与 TP/PP/DP、精度、输入输出长度、实际并发、采样、warmup、CPU/NUMA、测量窗口及汇总方式。Profiler 请求与计分请求分开；SSE 事件和 draft 提议数不能充当实际输出 token 数。

已归档结果见[DeepSeek 记录](benchmarks/deepseek-records.md)。Qwen 对照的缺失字段单独列在[数据说明](benchmarks/qwen38-a800-gaudi2.md)中。

<a id="troubleshooting"></a>

## 常见问题

| 现象 | 检查方向 |
|---|---|
| 要求先构建 native libraries | 核对所选库目录、kernel database、唯一 PT2 扩展和 manifests |
| sidecar 缺失或身份不符 | 检查标准目录、模型 manifest 和对应 recipe；更换权重需重新准备 |
| ABI / fingerprint 不匹配 | 使用同一套源码、构建产物和依赖；重建后重新生成 profile |
| 选到意外模块 | 检查 profile 与用户配置优先级，使用显式 `--modules` 和共享设备锁 |
| 关闭总开关后无法运行 | 总开关仅停止默认注入；恢复完整参考配置 |
| 搬迁模型后 Engram 无法读取 | 保留源文件及可读路径，并核对 manifest、偏移和审计记录；不能直接删掉原 checkpoint |
| 大量主机内存/页缓存占用 | mmap 主机表依然有 RAM、页缓存与 I/O 成本；默认只要求 staging buffer pinned |
| strict native 拒绝，auto 可运行 | 严格策略要求完整原生契约，auto 可以选择已建立的编译图 |
| API 返回错误 | 先检查模型名、512-token 总预算及采样参数，再看四个 rank 的异常 |
| 退出异常或资源未释放 | 保留完整 worker 与 runtime 日志；已有 lifecycle 问题仍在验证中 |

重绑定缓存地址、重载权重或更换通信器需要使旧 replay 计划失效。状态写入后发生错误，应终止该次执行并保留证据。
