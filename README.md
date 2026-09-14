<!-- markdownlint-disable MD041 -->

<p align="center">
  <img src="./assets/1cat-gaudi-logo.png" alt="1Cat vLLM for Intel® Gaudi® logo" width="420">
</p>

# 1Cat-vLLM-Gaudi

## Make Gaudi Fast Again

### 面向 Intel® Gaudi® 的现代大模型推理与原生执行优化

<strong>DeepSeek V4 / V4.1 · FlashInfer-Gaudi · GDN / DFlash2 · TPC / MME · Native Replay</strong>

> 我们不满足于：
>
> “这个模型能在 Gaudi 上启动。”
>
> **我们要让模型、算子、内存和通信真正协同起来，把 Gaudi 的算力转化成完整推理请求的性能。**

**1Cat-vLLM-Gaudi** 是 1CatAI 基于 **vLLM-Gaudi** 维护的推理工程分支，沿用 vLLM 的服务与调度基础，将 **Gaudi2** 作为新增原生执行和模型优化的重点目标。

从模型加载到专家权重布局，从稀疏 Attention 到 Engram 主机表，从 TPC / MME 计算到多卡通信与原生 replay——我们优化的不只是某一个 kernel，而是模型实际经过的整条执行路径。

在已归档的 **DeepSeek V4.1 Flash · 4× Gaudi2 · TP2 × PP2 · 普通 C1 decode** 配置中，[PR #26](https://github.com/1CatAI/1cat-vllm-gaudi/pull/26) 记录的三次完整模型运行，相对其 trace 驱动的优化起点，实现了：

# 稳态 Decode 延迟降低约 1/3

**不依赖 DSpark。普通单 token decode 本身，就是优化对象。**

> 这是对应 PR 中保留的相对测量总结，不是统一的 `tok/s` 成绩，也不是所有模型、长度和并发的性能承诺。当前专用配置限定单请求、最多 512 tokens 总上下文；完整生产质量与长时间 replay 验证仍在进行。

[性能记录](#performance) · [核心技术](#engineering) · [模型与边界](#models) · [源码安装](#installation) · [启动服务](#serving) · [完整工程手册](docs/1cat_gaudi_guide.md) · [问题反馈](https://github.com/1CatAI/1cat-vllm-gaudi/issues)

---

<a id="performance"></a>
# 📊 性能优先，完整请求优先

更低的算子延迟值得记录。

**但只有收益穿过模型、调度和 API，才是用户真正得到的收益。**

本页区分普通 decode、speculative transaction、专家组件和完整请求，不把它们拼成一个“综合加速倍数”。以下数字均来自对应 PR 的归档总结，并非本文重新测量。

## DeepSeek V4.1：普通 C1 与 DSpark 分开看

| 执行路径 | 保留的测量口径 | 记录结果 | 证据与状态 |
|---|---|---:|---|
| **普通 C1 · TP2 × PP2 · DSpark 关闭** | 三次完整模型运行的稳态 decode 延迟，相对 trace 优化起点 | **降低约 1/3** | [#26](https://github.com/1CatAI/1cat-vllm-gaudi/pull/26)；专用入口默认选择，仍属实验配置 |
| **DSpark · TP2 × PP2 · C6** | 完整 C6 transaction 延迟，相对保留的候选参考 | **降低 29.7%** | [#25](https://github.com/1CatAI/1cat-vllm-gaudi/pull/25)；独立实验路径 |
| 同一 DSpark 工作负载 | 首 token 之后的 decode 延迟 | **降低 28.5%** | [#25](https://github.com/1CatAI/1cat-vllm-gaudi/pull/25)；不含 TTFT |
| 同一 DSpark 工作负载 | Expert 组件延迟 | **降低 40.5%** | [#25](https://github.com/1CatAI/1cat-vllm-gaudi/pull/25)；组件结果 |
| **同一 DSpark 工作负载，完整请求** | 包含 prompt / prefill 的整体结果 | **仍慢约 0.8%** | [#25](https://github.com/1CatAI/1cat-vllm-gaudi/pull/25)；N256 prefill 兼容路径尚有开销，因此未默认启用 |

**C1 的约三分之一，与 C6 的 29.7%，不是同一组基准，也不能相加。**

`C1` 在这里指单 token decode 步骤；`C6` 指 DSpark 的六 token 验证工作负载。它们不是六倍并发，也不表示每轮一定接受六个输出 token。

## 为什么把回退结果也放在首页？

因为下面两件事可以同时发生：

```text
Decode 变快
    +
Prefill 兼容路径仍有额外开销
    ↓
完整请求未必变快
```

[PR #25](https://github.com/1CatAI/1cat-vllm-gaudi/pull/25) 正是这样的例子。

它交付了有价值的 DSpark 原生执行工作，但没有用局部收益掩盖完整请求的回退。

> **优化目标是完整服务，不是挑出最好看的那一段。**

---

# 🚀 已合并的关键工程进展

| 方向 | 交付内容 | 验证记录 | 当前边界 |
|---|---|---|---|
| **V4.1 普通 C1 默认组合** | FP8 sidecar 自动发现、Attention norm / dense projection、Q scaling + RoPE、mHC gates、expert finalize 等集成 | [#26](https://github.com/1CatAI/1cat-vllm-gaudi/pull/26)：61 项单元测试通过，14 项硬件依赖测试在 CPU 运行中跳过；另有针对性 HPU 检查 | 默认值只作用于专用入口；生产质量及长 replay 未完成 |
| **V4.1 DSpark device replay** | 设备侧 verify / commit 状态、持久 native plan、PP / Engram 交接、N256 / FP8 expert 与 CSA2 / MLA 路径 | [#25](https://github.com/1CatAI/1cat-vllm-gaudi/pull/25)：相关单元测试 494 通过、58 跳过；记录覆盖 28 项针对性 Gaudi 硬件检查 | 完整请求仍有回退；DSpark 默认关闭 |
| **V4 prepared native decode** | Q16 / S16 专家布局、完整 decoder replay、普通 TP2 reductions、显式 Attention / KV / compressor 依赖 | [#21](https://github.com/1CatAI/1cat-vllm-gaudi/pull/21)：96 项 CPU / meta 测试通过，16 项硬件门控测试跳过；原生拓扑回归通过 | 生产参考 token / logprob 一致性与独立异步链退出问题未解决 |

这些是 **PR 作者记录的不同测试集合**，不是本次文档生成执行的测试，也不是可以直接相加的全仓库覆盖率。`skipped` 不计为通过。

---

<a id="engineering"></a>
# 🔥 不只是给 Gaudi 增加几个算子

## 我们在重构真实模型的数据流

模型推理不是孤立的矩阵乘：

```text
Checkpoint / prepared weights
        ↓
模型分片与输入准备
        ↓
Attention / MoE / GDN / mHC
        ↓
KV、recurrent state 与 Engram 数据访问
        ↓
图编译、TPC / MME 执行、多卡通信
        ↓
采样、状态提交、API 输出
```

只优化其中一层，瓶颈可能马上移动到下一层。

本仓库把模型适配、布局转换、原生 kernel、编译边界、通信依赖和验证工具放在同一条源码链路里。各层实现可从 [源码导航](#source-map) 继续阅读。

---

# Layer 1 — 权重先准备好，不在热路径里反复展开

DeepSeek 的低精度 checkpoint，不能只用“loader 能识别 dtype”来衡量支持程度。

本仓库的 prepared 路径处理的是：

```text
原始 MXFP4 / FP8 checkpoint
        ↓
校验模型 revision 与文件身份
        ↓
按目标 TP / PP 拓扑生成分片
        ↓
整理专家布局、scale 编码与对齐
        ↓
以运行时需要的形式直接装入目标 rank
```

V4 / V4.1 的专家路径包含 **Q16 / S16 prepared 布局**、明确的 K 对齐和压缩权重访问。V4.1 在默认 C1 组合中进一步使用 N256 FP8 expert 及对应的融合准备路径。

设计重点是保留压缩专家权重的单设备分配，避免为了每一步 decode 反复构造长期重复的展开副本。它不意味着没有临时张量，也不意味着所有 prefill 路径都已经最优。

**低精度存储格式与实际计算精度，必须分开说明。**

| 路径 | 实际含义 |
|---|---|
| MXFP4 prepared | 按目标执行布局保留压缩专家权重；计算路线由具体算子决定 |
| Block-FP8 linear | 可采用 TPC 解量化后接 **BF16 MME**；不能称为 FP8 MME |
| V4.1 FP8 sidecar | 为指定投影准备独立的精度与布局制品，必须与模型和 recipe 配对 |
| `--dtype bfloat16` | 不代表所有权重存储、缓存、scale、累加与 logits 都采用 BF16 |

详见 [V4 native decode](docs/features/deepseek_v4_native_decode.md)、[V4.1 prepared execution](docs/features/deepseek_v41.md) 和 [原生 mixed-engine 契约](docs/features/flashinfer_native_backend.md)。

---

# Layer 2 — Attention 的代价，不只有 QK 和 PV

DeepSeek Attention 路径还涉及投影、归一化、RoPE、稀疏选择、物理缓存位置和状态写入。

如果这些步骤不断制造临时结果、切开编译区域，或者丢失 producer / consumer 依赖，单个 GEMM 再快也不够。

本分支围绕真实模型组织这些步骤：

| 位置 | 已集成或独立验证中的工作 |
|---|---|
| 输入端 | Q / KV 输入投影融合、Attention norm、Q scaling + RoPE |
| Cache producer | packed KV / FP4 cache write、SWA 写入与固定位置元数据 |
| 稀疏访问 | selected-row valid-only、decoded KV state、CSA2 / MLA 路径 |
| 矩阵计算 | shared-KV MME Attention、特定投影的 FP8 sidecar |
| 输出端 | 输出投影、状态依赖保留及 native replay 交接 |

这些不是面向任意 Attention shape 的全局替换。选择条件由模型、请求形状、精度和 profile 共同决定。

**计算必须完整保留，依赖必须清楚，收益必须回到整模型。**

实现与配置见 [V4.1 文档](docs/features/deepseek_v41.md)、[入口默认组合](vllm_gaudi/entrypoints/deepseek_v41.py) 和 [PR #26](https://github.com/1CatAI/1cat-vllm-gaudi/pull/26)。

---

# Layer 3 — TPC 与 MME 协同，而不是“全换成自定义 kernel”

TPC 负责适合专用实现和融合的张量处理，MME 承担相应矩阵计算。关键在于让两者之间的数据布局和执行边界更合适。

本仓库包含的原生路线举例：

```text
压缩权重
    → TPC 解码 / 解量化
    → MME 矩阵计算

残差 + RMSNorm
    → 融合 scale 计算与 FP8 packing
    → 后续投影消费

专家输出
    → 融合 scale / finalize reduction
    → 后续模型层
```

**一个 Synapse recipe 不一定只有一个 device kernel。**

**一个原生算子存在，也不意味着替换编译图一定更快。**

[FlashInfer 原生后端文档](docs/features/flashinfer_native_backend.md) 保留了完整性能门槛未通过的 mixed-engine / quantization 候选；有些 shape 的既有编译图已经很好，额外阶段反而增加调度代价。

我们的目标不是自定义算子数量，而是正确数据流下更低的完整执行成本。

---

# Layer 4 — Engram 留在主机，按需送到设备

V4.1 的 Engram 不是简单地把一大块 embedding 表搬进 HBM。

prepared 模型将主机表按完整 hash heads 分片，在 manifest 中保存源文件及字节偏移。执行时使用主机映射与原生 gather，再把当前步骤需要的数据送入设备。

```text
原始只读 checkpoint 文件
        ↓
Engram mmap / 主机页缓存
        ↓
按当前 token 历史 gather
        ↓
小型 HPU-pinned staging buffer
        ↓
DMA / 模型消费 / completion ownership
```

V4.1 C1 组合包含 native Engram preparation 和 packet 路径。实现还维护 generation 与完成事件，避免前一个步骤仍在消费时就覆写传输缓冲区。

这也决定了两个实际部署要求：

**prepared 生成后，原始 Engram 源文件仍然需要保留。**

**减少 HBM 占用，不等于没有主机内存、页缓存或磁盘成本。**

默认不要求把全部主机表强制锁进物理内存。源文件迁移、memlock、NUMA 和读写权限见 [完整工程手册](docs/1cat_gaudi_guide.md#deepseek-v41) 与 [Engram 实现](vllm_gaudi/ops/deepseek_v41_host.py)。

---

# Layer 5 — Native Replay：把重复提交移出热路径

对固定形状 decode，框架层反复准备和提交碎片化执行片段，也会成为延迟来源。

原生 replay 路径将计算与通信组织为预先准备的执行计划，同时保留输入变化、状态更新和完成顺序：

```text
准备权重、固定输入与状态地址
        ↓
编译并捕获 recipe / 通信计划
        ↓
恢复 capture 期间改变的状态
        ↓
复制本轮真实输入，保留 producer 依赖
        ↓
Replay 计算与通信
        ↓
消费输出，提交状态，等待安全复用
```

V4 native decoder 的限定路径覆盖 **43 层、86 次 decoder reductions**，并保留最后的 mHC / head / norm 计算；embedding reduction、最终输出投影和采样仍有各自的外围路径。见 [V4 native decode](docs/features/deepseek_v4_native_decode.md)。

V4.1 则围绕 **TP2 × PP2** 的 stage replay、PP 状态交接、mHC / TP 依赖和可选 DSpark 验证组织执行。见 [V4.1 prepared execution](docs/features/deepseek_v41.md)。

> **一次 replay 入口调用，不等于一条硬件命令。**
>
> 缓存重分配、权重重载、通信器变化或状态地址重新绑定，都需要使旧计划失效。状态写入后发生错误，不能静默切换另一实现重跑。

---

# 🧩 FlashInfer-Gaudi

## 兼容部分 API 语义，为 Gaudi 重新选择执行方式

`flashinfer_gaudi` 是独立命名空间，部分 API 对齐 FlashInfer **0.6.18**，重点覆盖 **Gated Delta Rule、prefill / decode、MTP 状态与相关推理原语**。

它不是把 CUDA 的 FlashInfer 二进制放到 HPU 上，也不是整个 FlashInfer 库已经完成原生移植。

| 类别 | 代表接口 |
|---|---|
| GDN prefill | `chunk_gated_delta_rule` |
| GDN decode | `flashinfer_gaudi.gdn_decode` |
| Fused decode | `gdn_fused_decode_step` |
| MTP / rollback | `gated_delta_rule_mtp_packed`、`gated_delta_rule_mtp_rollback` |
| Activation / quant | `silu_and_mul`、`silu_and_mul_quant` |
| Residual / norm / quant | `fused_add_rmsnorm_quant` |
| Block-FP8 | `block_fp8_dequant`、`block_fp8_linear` |
| 能力查询与后端选择 | `get_capabilities`、`set_backend_policy` |

其中一些是 **Gaudi 专用扩展**，不能直接视为官方 FlashInfer 的同名契约。

## 先查询能力，再决定后端

```python
import json
from flashinfer_gaudi import get_capabilities

print(json.dumps(get_capabilities(), indent=2, ensure_ascii=False, default=str))
```

| 策略 | 行为 |
|---|---|
| `auto` | 依据形状和 tactic 选择；保留已建立的服务编译图策略 |
| `pytorch` | 参考路径；参考实现本身也可以经过编译 |
| `native` | 要求操作满足完整原生契约，不满足则拒绝 |
| `public` | 只接受对应 public 原生路径 |
| `bridge` | 要求对应 Bridge 原生实现与匹配 ABI |

当前 GDN 的部分 TPC 原型仍有数值 prologue，严格原生策略会拒绝不完整路径。不能把 `prototype loaded` 当成 `production qualified`。

更多说明：[兼容算子文档](docs/features/flashinfer_gaudi.md) · [原生资格与精度契约](docs/features/flashinfer_native_backend.md) · [能力报告代码](flashinfer_gaudi/_capabilities.py)。

---

# 🧠 GDN：减少状态搬运与重复计算

GDN 是 recurrent state 的计算问题，也是数据布局问题。

本仓库的 Qwen GDN 路线包含紧凑 Q / K heads、KKT 与 causal-decay 复用、静态三角 mask、分块求解和 fused direct-state decode，将有用的 FlashQLA 代数组织到 Gaudi 编译图中。

矩阵计算仍由相应 MME 路径执行，外围张量计算由编译图融合；这不是把所有工作改写为 TPC 循环。

## 当前重点形状

| 路径 | Q / K heads | V heads | K / V dimension |
|---|---:|---:|---:|
| TP1 | 16 | 48 | 128 / 128 |
| TP2，每 rank | 8 | 24 | 128 / 128 |

该 prefill tactic 还限制单条均匀序列、BF16 输入与 chunk size 128。correctness-first 图保留 FP32 计算，默认 recurrent state 精度也不是因为 BF16 接口存在就自动降低。

重点 fused decode buckets：

```text
1 / 2 / 4 / 8 / 16 / 32
```

其他 bucket、indexed state、padding、prefix caching 和 speculative 状态需要各自的路径判断。

## 启用适配

```bash
export VLLM_HPU_FLASHINFER_GDN=1
export FLASHINFER_GAUDI_BACKEND=auto

# 只有目标为 TP2 且满足对应局部形状时才另外开启。
# export VLLM_HPU_FLASHINFER_GDN_TP2=1
```

Q / K 的归一化契约为：

```text
x * rsqrt(sum(x * x) + 1e-6)
```

加 epsilon 与 clamp 范数不是同一种数学操作。相关变化需要重新建立数值和性能基线，而不是只检查输出“看起来正常”。

依据：[FlashInfer-Gaudi GDN 文档](docs/features/flashinfer_gaudi.md) 与 [归一化验证说明](docs/features/flashinfer_native_backend.md)。

---

# ⚡ DFlash2：验证一个 block，而不是只加一个 draft

Qwen DFlash2 实验路径一次产生七个 draft tokens，由 target 验证八 token block，并保留每一步卷积与 GDN checkpoint，以支持接受和回滚。

```text
Draft 一次产生候选
        ↓
HPU top-16 selector
        ↓
Target block verification
        ↓
选择接受前缀，恢复对应状态
        ↓
继续下一轮
```

初始配置范围为 **贪心文本、TP1 / PP1 / DP1、最多 16 个序列、compact GDN state、关闭 prefix caching 和 LoRA**。即使 target 带视觉塔，当前这条 DFlash2 路径也不接收多模态输入。

它与 V4.1 的 **DSpark** 是不同的模型与执行集成，不能互换开关，也不能共用一组性能结论。

DFlash2 的原生 MTP、selector、top-k 和 score-select 候选有独立验证门槛；图策略可用，不等于这些原生候选已全部进入默认服务。

实现与测试入口见 [DFlash2 文档](docs/features/flashinfer_gaudi.md)，启动示例见 [下方服务部分](#dflash2-serving)。

---

<a id="models"></a>
# 🎯 模型与执行边界

下表描述本页快照中的专用路线，不是所有 Gaudi 型号和参数组合的认证矩阵。

| 路线 | 主要配置边界 | 默认状态 | 下一步 |
|---|---|---|---|
| **DeepSeek V4 Flash 普通路径** | Gaudi2；TP2 / PP1；单请求；512-token 总上下文；原始混合 MXFP4 / FP8 checkpoint | 专用入口选择普通 decode 与区域编译 | [构建与启动](#v4-serving) |
| **DeepSeek V4 native decoder** | 同代硬件；限定 C1；需要独立 native runtime | **默认关闭** | [原生指南](docs/features/deepseek_v4_native_decode.md) |
| **DeepSeek V4.1 Flash prepared C1** | Gaudi2；TP2 × PP2；单请求；512-token 总上下文；贪心 | **专用入口默认开启 C1 组合**，仍属实验 | [准备与启动](#v41-serving) |
| **V4.1 DSpark** | 独立 prepared / draft / replay 配置 | **默认关闭** | [完整配置说明](docs/1cat_gaudi_guide.md#deepseek-v41) |
| **Qwen GDN** | 已列明的 TP1 / TP2 局部形状与状态布局 | 父开关显式启用；TP2 另有开关 | [GDN 文档](docs/features/flashinfer_gaudi.md) |
| **Qwen DFlash2** | TP1 / PP1 / DP1；贪心文本；最多 16 序列 | **默认关闭** | [启动示例](#dflash2-serving) |
| 通用 HPU 模型 | 取决于上游引擎、插件、模型和软件版本配对 | 按原有插件配置 | [继承的功能说明](docs/features/supported_features.md) |

V4.1 代码有 PP0 视觉塔与图像 span 集成，但不能据此宣称任意图像尺寸和多模态工作负载都已完成验证。

**512 tokens 指总上下文预算，不是输入 512 再额外输出 512。** 改大 CLI 上限不会自动完成缓存、metadata、warmup 与 replay 的适配。

---

<a id="installation"></a>
# 📦 从源码安装

## 先固定运行环境

本文 DeepSeek 源码集成采用的环境基线为：

```text
Intel Gaudi2
Gaudi Software 1.24.1
匹配的 Gaudi PyTorch 2.11 环境
匹配的 Bridge / 开发头文件 / TPC 编译工具
```

通用插件与不同私有 native adapter 的依赖并不完全相同。部分专用工具使用 Python 3.11+ 接口，请在匹配的软件栈中选择 Python 3.11 / 3.12；不要只根据包元数据的宽范围判断整套工具都可运行。

先在已保留的空闲设备和目标 Python 环境中检查：

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
```

## 获取本分支

```bash
git clone https://github.com/1CatAI/1cat-vllm-gaudi.git
cd 1cat-vllm-gaudi

# 复现本页时使用此源码快照。
git checkout ac14567637ca1d8f3d4434e62dbab3317ebe9fbe

python -m pip install 'packaging>=24.2' 'setuptools>=77.0.3,<85.0.0' \
  'setuptools-scm>=8.0' wheel jinja2
```

**仓库名不等于 PyPI 包名。** 当前 Python distribution 仍为 `vllm_gaudi`，本文不假设存在 `pip install 1cat-vllm-gaudi` 或已经发布的 1Cat Gaudi 一键镜像。

| 目标 | 引擎与构建方式 |
|---|---|
| DeepSeek V4 | 下方固定提交 + `prepare_deepseek_v4_engine.py` + 原生算子构建 |
| DeepSeek V4.1 | 配套的 V4.1 引擎、prepared 权重、sidecar 和 native replay runtime |
| 通用 HPU / Qwen 路线 | 与目标模型和插件完成配对的引擎；[通用源码安装模板](docs/1cat_gaudi_guide.md#installation) |

不要无条件安装最新 vLLM，也不要用 V4 的固定补丁链代替 V4.1 的引擎准备。参考：[V4 构建](docs/features/deepseek_v4_flash.md) · [V4.1 构建](docs/features/deepseek_v41.md)。

---

<a id="serving"></a>
<a id="v4-serving"></a>
# ▶ DeepSeek V4 Flash · 2× Gaudi2

## 固定引擎并安装

以下从本仓库根目录执行，前提是上一节的 Gaudi 环境和构建依赖已经准备好：

```bash
git clone https://github.com/vllm-project/vllm.git ../vllm-dsv4
git -C ../vllm-dsv4 checkout fe755c88995ad468882517b6c4bdd60138d46a3a

python tools/prepare_deepseek_v4_engine.py ../vllm-dsv4

# 使用 Bash；保留已有的 Gaudi torch，不另装 CUDA torch。
python -m pip install -r <(sed '/^torch/d' ../vllm-dsv4/requirements/build/cuda.txt)
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e ../vllm-dsv4
python -m pip install --no-build-isolation -e .
```

## 构建原生库

```bash
export GAUDI_PYTORCH_BRIDGE_ROOT=/path/to/matching/gaudi-pytorch-bridge
export GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=/path/to/matching/bridge/build

python tools/build_deepseek_v4.py --jobs "$(nproc)"
ctest --test-dir build/deepseek_v4/kernels --output-on-failure
```

路径必须指向匹配 Bridge 的真实源码与生成后的 build headers。只安装 Python 包，或仅使用 `--kernel-only`，都不能代替完整原生构建。

## 启动普通 TP2 服务

```bash
HABANA_VISIBLE_MODULES=0,1 \
python -m vllm_gaudi.entrypoints.deepseek_v4 \
  /path/to/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --host 127.0.0.1 \
  --port 8000
```

入口自动选择的关键参数：

```text
TP / PP                 2 / 1
max_model_len           512
max_num_seqs            1
max_num_batched_tokens  512
dtype                   bfloat16
KV dtype                fp8
prefix caching          off
async scheduling        on
gpu_memory_utilization  0.09
```

**`0.09` 是该入口真实写入的值，不是 `0.9` 的排版错误。** 它也不表示所有运行时分配都被严格限制为设备内存的 9%。

默认使用普通 HPU worker 与区域编译，不自动开启整段 native decoder replay，也不启用 speculative decoding。注意力投影缓存使用 BF16 计算。

原生 decoder 需另外准备 ABI 锁定的 runtime 和通信计划；详见 [原生 decode 指南](docs/features/deepseek_v4_native_decode.md)。

---

<a id="v41-serving"></a>
# ▶ DeepSeek V4.1 Flash · 4× Gaudi2

## Prepared TP2 × PP2

```text
PP0：目标层 0–19、embedding、vision、两个 Engram 主机表
PP1：目标层 20–39、输出 norm / head、可选三层 DSpark draft

PP 边界：hidden states + 后续依赖的 mHC mixing coefficients
```

普通 C1 不加载 `mtp.*` draft 张量、不创建 draft state，也不执行 proposal / verification。prepared 文件里存在 draft，并不意味着服务启动了 DSpark。

## 准备流程

必须先准备 **匹配的 V4.1 引擎、native replay runtime、TP2 扩展和运行时指纹配置**。下面展示权重与服务步骤，不能用来替代这些前置构建。完整流程见 [工程手册](docs/1cat_gaudi_guide.md#deepseek-v41)。

```bash
export MODEL_DIR=/data/models/DeepSeek-V4.1-Flash-source
export PREPARED_DIR=/data/models/DeepSeek-V4.1-Flash-prepared
export CHECKPOINT_AUDIT_DIR=/data/evidence/dsv41-checkpoint
export UPSTREAM_AUDIT_DIR=/data/evidence/dsv41-upstream

# 上游审计需要可用的 gh；输出目录应为新目录。
python tools/audit_deepseek_v41_upstream.py "$UPSTREAM_AUDIT_DIR"
python tools/sync_deepseek_v41_checkpoint.py "$MODEL_DIR" \
  --evidence "$CHECKPOINT_AUDIT_DIR"

# 先查看空间计划；再生成 immutable rank files。
python tools/prepare_deepseek_v41_shards.py "$MODEL_DIR" "$PREPARED_DIR" \
  --checkpoint-audit "$CHECKPOINT_AUDIT_DIR/complete.json" \
  --upstream-lock "$UPSTREAM_AUDIT_DIR/upstream-lock.json" \
  --plan-only

python tools/prepare_deepseek_v41_shards.py "$MODEL_DIR" "$PREPARED_DIR" \
  --checkpoint-audit "$CHECKPOINT_AUDIT_DIR/complete.json" \
  --upstream-lock "$UPSTREAM_AUDIT_DIR/upstream-lock.json"

# 默认普通 C1 组合需要的两个 FP8 sidecar。
python tools/prepare_deepseek_v41_woa_fp8.py "$PREPARED_DIR"
python tools/prepare_deepseek_v41_dense_fp8.py "$PREPARED_DIR"
```

上游审计只保存证据，不会替你安装引擎。同步工具会写入并校验目标 checkpoint 目录，建议使用独立目录，不要指向其他服务正在使用的模型。

## 构建 V4.1 原生扩展

```bash
export GAUDI_PYTORCH_BRIDGE_ROOT=/path/to/matching/gaudi-pytorch-bridge
export GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=/path/to/matching/bridge/build
export VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR=/data/build/dsv41-native/lib

python tools/build_deepseek_v41.py \
  --build-root /data/build/dsv41-native/build \
  --output-dir "$VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR" \
  --jobs "$(nproc)"
```

该工具构建 DeepSeek kernel / Bridge 扩展与 Engram host gather，**不替代完整 patched Bridge / Synapse / HCL replay runtime 的构建**。原生库、manifest 和运行时必须成套匹配。

## 启动默认普通 C1

先保留四个空闲模块；设备编号和文件路径均应改为本机实际配置：

```bash
HABANA_VISIBLE_MODULES=0,1,2,3 \
python -m vllm_gaudi.entrypoints.deepseek_v41 "$PREPARED_DIR" \
  --checkpoint-audit "$CHECKPOINT_AUDIT_DIR" \
  --runtime-profile /path/to/fingerprinted-v41-runtime-profile.json \
  --host 127.0.0.1 \
  --port 8000
```

`runtime-profile` 指向真实、已构建并记录指纹的配置，不是仓库自动附带的示例文件。

| 默认配置 | 值 |
|---|---|
| TP / PP | `2 / 2` |
| 总上下文 / 最大 batch tokens | `512 / 512` |
| 最大序列数 | `1` |
| Loader / block size | `dsv41_prepared / 512` |
| Prefix caching / async scheduling | **均关闭** |
| 服务模型名 | `DeepSeek-V4.1-Flash` |
| 普通 C1 fast-path 组合 | **专用入口默认开启** |
| DSpark | **默认关闭** |

### 三个容易踩坑的地方

**审计参数。** 普通分片准备传 `complete.json` **文件**；服务入口传审计**目录**。

**模型迁移。** prepared 的四个 `.safetensors` 不包含全部 Engram 源数据，不能生成后删除原 checkpoint。

**关闭默认组合。** `VLLM_HPU_DSV41_DEFAULT_FASTPATHS=0` 只阻止注入默认值，不会清理已有变量，也不会自动补齐参考环境。做对照应恢复完整归档配置；启用 DSpark 同样需要独立的完整配置。

入口逻辑与边界：[专用入口代码](vllm_gaudi/entrypoints/deepseek_v41.py) · [功能文档](docs/features/deepseek_v41.md)。

---

<a id="dflash2-serving"></a>
# ▶ Qwen GDN + DFlash2

以下示例需要与目标模型配对的引擎、插件与足够设备容量；使用独立于上述 DeepSeek 服务的环境 / 进程：

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
  --host 127.0.0.1 \
  --port 8000
```

使用 `temperature=0` 的文本请求，不启用 LoRA，也不附加该路径尚未支持的采样变换。

**检查启动日志中的实际 KV 容量。** DFlash2 还需要 draft Attention cache；客户端同时发出 16 个请求，不证明设备端正在以 16 并发执行。

更完整的状态与验证条件见 [DFlash2 文档](docs/features/flashinfer_gaudi.md)。

---

# 🌐 API 调用

使用对应服务的实际模型名。以下对默认 V4.1 服务发起一个短请求：

```bash
curl --fail --show-error http://127.0.0.1:8000/health
curl --fail --show-error http://127.0.0.1:8000/v1/models

curl --fail --show-error --no-buffer \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "请用一句话解释张量并行。"}],
    "temperature": 0,
    "max_tokens": 64,
    "stream": true
  }'
```

API 可调用只证明服务链路连通，不等于模型质量、长上下文或生命周期验证通过。

示例绑定 localhost。对外部署前应补齐认证、访问控制和资源限制，不要直接暴露实验服务。

---

# ✅ 正确性与质量，不能被性能表替代

本仓库明确区分：

```text
CPU / meta contract
        ↓
HPU 算子数值与状态
        ↓
完整模型输出与固定参考
        ↓
端到端性能
        ↓
重复请求、长 replay、资源释放和退出
```

## 当前需要保留的已知边界

| 路线 | 仍需关注的事项 |
|---|---|
| V4 普通源码集成 | 同一 prompt 在初始请求和后续贪心请求间存在已记录差异；稳定 token 相等不证明冷启动确定性 |
| V4 native decoder | 冻结生产参考 token / logprob 差异，及独立异步链诊断的退出时 heap failure |
| V4.1 默认 C1 | 相对前一候选存在输出变化；完整质量、prefill 数值一致性、长 replay / shutdown 尚待完成 |
| V4.1 DSpark | 接受 / 拒绝、回滚和状态复用需单独验证；完整请求性能还有已记录回退 |
| FlashInfer native 候选 | 某算子或某 shape 通过，不代表整个模型已切换或质量门槛已通过 |

依据：[V4 源码集成](docs/features/deepseek_v4_flash.md) · [V4 native](docs/features/deepseek_v4_native_decode.md) · [V4.1](docs/features/deepseek_v41.md) · [DSpark #25](https://github.com/1CatAI/1cat-vllm-gaudi/pull/25) · [FlashInfer native](docs/features/flashinfer_native_backend.md)。

**已合并、入口默认启用、原生库加载成功、生产质量通过，是不同状态。**

---

# 📐 基准测试规范

本项目分别报告：

```text
kernel / operator latency
完整算子链成本
prefill / TTFT
target-only decode
speculative transaction
实际输出 token 吞吐
完整请求延迟
任务质量、token / logprob 与状态一致性
```

比较前固定插件与引擎 SHA、模型 revision、原生库指纹、硬件数量、TP / PP、输入输出长度、实际并发、精度、bucket、CPU / NUMA 和采样配置。

**SSE 事件不等于单个 token。** speculative 路径必须记录实际接受与输出数量，不能把 draft 提议数计为吞吐。

**Profiler 请求与性能计分请求分开。** 多 rank 模型需要完整 trace；只查看一个 rank 或一个算子，不能推断整个流水线。

<details>
<summary><strong>展开：仓库内的部分测试与基准入口</strong></summary>

以下需要对应的软件、硬件和原生构建条件，执行前先保留设备并检查工具的 `--help`：

```bash
# 默认入口行为，不是模型质量测试。
python -m pytest -q tests/unit_tests/test_deepseek_v41_entrypoint.py

# GDN prefill graph tactic。
python tools/benchmark_flashinfer_gaudi_gdn_prefill.py \
  --tokens 2048,4096,16384 --iterations 10 --waves 7

# Direct-state decode 的参考布局比较。
python tools/benchmark_flashinfer_gaudi_gdn.py \
  --batches 1,8,16,32 --timing-mode both \
  --reference-layout legacy --backend pytorch

# DFlash2 MTP 候选，不是完整模型吞吐。
python tools/benchmark_flashinfer_gaudi_dflash2.py \
  --batches 1,2,4,8,16 --warmups 20 --wave-iterations 100 --waves 7

# 原生 norm / quant 的硬件契约测试。
FLASHINFER_GAUDI_RUN_HARDWARE_TESTS=1 \
python -m pytest -q tests/unit_tests/ops/test_flashinfer_norm_quant_hardware.py
```

这些入口不应在其他任务正在使用的设备上随意运行。更多说明见 [工程手册：验证](docs/1cat_gaudi_guide.md#validation)。

</details>

---

# 🧱 Runtime，不只是 Kernels

原生执行的可复现单位，不是单独一个 `.so`：

```text
插件与引擎源码
    +
模型 / prepared / sidecar 身份
    +
Bridge / Synapse / HCL
    +
TPC / PT2 / host gather 原生库
    +
ABI manifest 与运行配置
```

部分 replay runtime 基于旧公开源码快照及兼容性补丁，**不等于已安装 SDK 的等价源码发行版**。通过 ABI 指纹检查，不能代替数值与编译器等价性验证。

只在独立环境中使用实验库，不覆盖共享服务的系统库；更换二进制或权重布局后，重新准备对应 recipe 和状态绑定。

具体构建依据：[native runtime 指南](tools/communication/patches/native-runtime/README.md)。

---

<a id="source-map"></a>
# 🗂️ 源码与文档导航

| 模块 | 入口 |
|---|---|
| 完整安装、配置、排障与 FAQ | [工程手册](docs/1cat_gaudi_guide.md) |
| HPU 平台与插件注册 | [vllm_gaudi/__init__.py](vllm_gaudi/__init__.py) · [platform.py](vllm_gaudi/platform.py) |
| 模型与 prepared loader | [vllm_gaudi/models/](vllm_gaudi/models/) |
| DeepSeek 专用入口 | [vllm_gaudi/entrypoints/](vllm_gaudi/entrypoints/) |
| HPU 算子与编译优化 | [vllm_gaudi/ops/](vllm_gaudi/ops/) · [compilation/](vllm_gaudi/compilation/) |
| Worker 与 runner | [vllm_gaudi/v1/](vllm_gaudi/v1/) |
| FlashInfer 兼容层与状态报告 | [flashinfer_gaudi/](flashinfer_gaudi/) |
| DeepSeek TPC / Bridge 内核 | [csrc/deepseek_v4/](csrc/deepseek_v4/) |
| Engram host gather | [csrc/deepseek_v41/](csrc/deepseek_v41/) |
| FlashInfer 原生内核 | [csrc/flashinfer_gaudi/](csrc/flashinfer_gaudi/) |
| 多卡通信与 native runtime | [vllm_gaudi/distributed/](vllm_gaudi/distributed/) · [tools/communication/](tools/communication/) |
| 环境变量与 profile | [envs.py](vllm_gaudi/envs.py) · [变量说明](docs/configuration/env_variables.md) |
| 数值、状态与模型测试 | [tests/](tests/) |

第一次阅读 V4.1，建议沿着 **入口 → 模型注册 / loader → prepared stage → state / replay → native runtime → 测试** 的顺序，而不是只从某个 TPC kernel 开始。

---

# 🛠️ 常见问题

<details>
<summary><strong>为什么装好了 Python 包，却提示先构建 native libraries？</strong></summary>

Python 插件安装与 DeepSeek kernel、PT2 扩展、host gather、replay runtime 是不同步骤。检查对应入口要求的库目录及 manifest，不要只复制一份 `.so`。

</details>

<details>
<summary><strong>为什么 strict native 模式报错，auto 却能运行？</strong></summary>

`auto` 可以使用适用的编译图策略；严格模式要求整个操作满足原生契约。当前部分 GDN 原型尚不满足，明确拒绝比静默改变后端或在状态写入后失败更安全。

</details>

<details>
<summary><strong>为什么关闭 V4.1 总开关后反而无法启动？</strong></summary>

总开关只阻止默认值注入，不会恢复完整参考配置。prepared shards、Engram、runtime 等基础条件仍需明确配置。请使用归档的完整对照环境，而不是只改一个变量。

</details>

<details>
<summary><strong>这是 1Cat-vLLM 的 V100 后端吗？</strong></summary>

不是。本仓库面向 Intel Gaudi / HPU。[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) 面向 NVIDIA Volta / SM70；两者共享 1CatAI 的工程方向，但代码、驱动、算子与性能记录不能互换。

</details>

更详细的 ABI、sidecar、Engram、设备可见性、HTTP 500 与退出排查见 [工程手册：故障排查](docs/1cat_gaudi_guide.md#troubleshooting)。

---

# 🧭 项目方向

我们关注的问题很直接：

> **如果围绕 Gaudi 的真实执行方式重做模型数据流，而不止于兼容模型接口，还能释放多少实际推理性能？**

结合当前实现与尚未完成的验证，重点工程方向包括降低普通 decode 热路径成本、缩小 prefill 兼容开销、完善 speculative 状态与质量验证、提高原生算子覆盖，并让源码、运行时和权重制品更容易成套复现。

这些是工程方向，不是对某个未来版本的交付承诺。

---

# 🤝 参与贡献

欢迎模型适配、TPC / MME、编译、通信、数值验证、系统工程和文档方面的贡献。

一个有价值的性能 PR，应能说明 **基线是什么、改变了什么、适用哪些形状、完整模型是否受益，以及哪些验证仍未完成**。

通过 feature branch 和 Pull Request 提交，保留 `Signed-off-by`；具体规范见 [AGENTS.md](AGENTS.md)。提交前移除凭据、客户请求、私有地址和不必要的内部路径。

[提交 Issue](https://github.com/1CatAI/1cat-vllm-gaudi/issues) · [查看 Pull Requests](https://github.com/1CatAI/1cat-vllm-gaudi/pulls)

---

# 💬 1CatAI 开源社区

关注 [1Cat-vLLM 主项目](https://github.com/1CatAI/1Cat-vLLM) 的 **WeChat Community** 部分获取更新的社区邀请。群二维码会过期，本页不复制另一项目的限时二维码。

Gaudi 相关复现、安装和功能问题，请优先在本仓库保留可搜索的 Issue 记录。

---

# ❤️ 致谢

本项目建立在 [vLLM](https://github.com/vllm-project/vllm)、[vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi)、Intel Gaudi 软件生态，以及相关模型和工具链贡献者的工作之上。部分可移植 API 语义参考 [FlashInfer](https://github.com/flashinfer-ai/flashinfer)。

感谢参与本分支实现、测试与复现的贡献者，也感谢 [@yangzhuxinyzx](https://github.com/yangzhuxinyzx) 在上述 DeepSeek 和原生执行 PR 中的贡献。

1CatAI 的项目品牌不替代原有版权；本项目不表示 Intel、vLLM 或 FlashInfer 对这些实验路径提供额外认证或背书。

---

# License

本仓库采用 [Apache License 2.0](LICENSE)。模型权重、第三方运行时与其他依赖仍适用各自许可证；分发时请保留相应声明。

---

**源码基准：** `ac14567637ca1d8f3d4434e62dbab3317ebe9fbe` · **2026-09-14**。性能采用对应 PR 留存的测量口径；安装命令和实验状态以这份源码快照为准。升级后请重新核对默认值、运行时、模型制品与质量结果。

<p align="center"><strong>1CatAI · 不止让模型启动，让优化落到真实推理。</strong></p>
