<!-- markdownlint-disable MD041 -->

<p align="center">
  <img src="assets/1cat-gaudi-logo.png" alt="1Cat vLLM for Intel Gaudi" width="420">
</p>

# 1Cat-vLLM-Gaudi

<!-- pyml disable-next-line no-trailing-punctuation -->
## Let's Gaudi it.

### 面向 Intel® Gaudi® 的现代大模型推理与原生执行优化

<strong>Qwen · DeepSeek · MiniMax H3 · FlashInfer-Gaudi · TPC / MME · Native Replay</strong>

> 让现代大模型，在 Gaudi 上真正跑快。
>
> 从权重进入内存，到第一个 token 返回，再到每一轮 decode——每一段执行成本，都值得认真优化。

**1Cat-vLLM-Gaudi** 是 1CatAI 维护的 [vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi) 硬件插件工程分支，需要配套的 [vLLM](https://github.com/vllm-project/vllm) 推理引擎。我们将 **Gaudi2** 作为重点优化目标，把模型适配、权重布局、TPC / MME 算子、多卡通信、`torch.compile` 与 Native Replay 放在同一条推理路径里打磨。

从单卡响应，到双卡协同，再到四卡大模型与单卡音视频生成——让性能结果有对应的实现，让每一轮优化能够被验证、被复现。

> ☁️ **免费体验 Gaudi2**
>
> 我们自建的 **1Cat-vLLM-Gaudi2 服务器集群**已开放免费体验。[进入 1CatDL 算力工作台 →](http://dx.1catai.com:53852/rental)

在已留存的 **Qwen3.8-27B-FP8 · Gaudi2 · TP1** 单请求记录中，输入 2,048 tokens：

### 0.4036 s 首字 · 57.17 tokens/s Decode

对应 A800 单卡记录为 **0.7739 s / 48.64 tokens/s**；Gaudi2 首字延迟缩短 **47.8%**，Decode 提升 **17.5%**。

这里展示已有实测记录，配置与统计说明随数据保留。[查看完整对照](#performance) · [数据来源与口径](docs/benchmarks/qwen38-a800-gaudi2.md)

[免费体验](#free-trial) · [性能记录](#performance) · [核心能力](#engineering) · [双卡与四卡](#tp2) · [Graph + Compile](#graph-compile)

[FlashInfer-Gaudi](#flashinfer-gaudi) · [音视频生成](#h3) · [模型支持](#models) · [构建与启动](#quick-start) · [参与贡献](#contributing)

### 从模型适配，到完整执行路径

| 能力 | 项目中的工作 |
|---|---|
| **模型与多卡** | Qwen GDN / DFlash2、DeepSeek V4 / V4.1、MiniMax M3 / H3；TP1、TP2 与 TP2 × PP2 各有适用范围 |
| **TPC / MME 与 FlashInfer-Gaudi** | 原生算子、Host Glue、CustomOp、编译图与按形状选择的执行路线 |
| **Graph + Compile** | 区域编译、固定 bucket、recipe 持有、计算与通信重放、状态生命周期 |
| **Prepared Weights** | rank-local 分片、MXFP4 / FP8 布局、sidecar、manifest 与指纹校验 |
| **性能与质量工具** | 微基准、完整请求、逐 kernel trace、正确性检查与候选对照 |

当前源码包含 **100 余个 DeepSeek 相关 TPC kernel 源文件**，并配套布局、模型接入、构建与验证工具。[查看原生实现](https://github.com/1CatAI/1Cat-vLLM-Gaudi/tree/5fa3e08302716d458a82aa7d95d107e201b51d8c/csrc/deepseek_v4)

---

<a id="performance"></a>

## 📊 Performance First

### Qwen3.8-27B-FP8 · A800 / Gaudi2

**从单请求响应，到 32 并发吞吐。**

每个请求输入 **2,048 tokens**，并发请求数为 **1 / 8 / 16 / 32**。Gaudi2 的 **TP1 与模型名称**已由记录提供者确认；A800 单卡与模型信息来自原始截图。

![Qwen3.8-27B-FP8 在 A800 与 Gaudi2 上的 TTFT、推算 Prefill 和 Decode 总吞吐](assets/benchmarks/qwen38-a800-gaudi2.svg)

| 并发请求数 | TTFT：A800 → Gaudi2（秒） | 首字延迟缩短 | Decode 总吞吐：A800 → Gaudi2（tokens/s） | Decode 变化 |
|---:|---:|---:|---:|---:|
| **1** | 0.7739 → **0.4036** | **47.8%** | 48.64 → **57.17** | **+17.5%** |
| **8** | 5.7206 → **2.9354** | **48.7%** | 342.11 → **380.25** | **+11.1%** |
| **16** | 9.8716 → **5.8028** | **41.2%** | 614.62 → **644.21** | **+4.8%** |
| **32** | 16.5182 → **11.6282** | **29.6%** | 991.44 → **958.44** | **−3.3%** |

这组记录里，Gaudi2 在四档并发下都有更短的首字延迟。单请求 Decode 为 **57.17 tokens/s**，32 并发 Decode 总吞吐达到 **958.44 tokens/s**。

吞吐曲线也保留了值得继续优化的一段：在 32 并发时，Gaudi2 的 Decode 总吞吐比 A800 低 **3.3%**。让响应更快、让并发扩展更好，是两项需要分别检验的工作。

> **测量口径：**
>
> 图中“推算 Prefill”按 `2,048 × 并发数 ÷ TTFT` 计算，是输入速率估算；TTFT 包含其他开销，也未必等于整批请求的 prefill 完成时间。Decode 沿用原记录的纯 Decode 总吞吐。完整软件配置、输出长度、Decode 计时窗口与一致的 TTFT 统计口径尚待补齐；本表描述已有记录，不将差异归因于某一项优化。

[完整数值与计算公式](docs/benchmarks/qwen38-a800-gaudi2.md) · [原始截图](assets/benchmarks/a800-gaudi2-source.png) · [结构化数据](docs/benchmarks/qwen38-a800-gaudi2.json)

### DeepSeek V4.1 · 从更快的 Decode，到更长的生成

<strong>4× Gaudi2 · TP2 × PP2 · 普通 C1 · V2 异步执行</strong>

#### 73.50 tokens/s · 研究候选记录

| 输入 / 输出 | 三轮平均稳态 ITL | 由 ITL 换算的 Decode 速度 | 总上下文容量配置 |
|---:|---:|---:|---:|
| **2,052 / 256 tokens** | **13.606 ms/token** | **73.50 tokens/s** | **1,048,576 tokens** |

该归档候选以 `5fa3e083` 为源码基线，使用 4× Gaudi2、TP2 × PP2，DSpark 关闭。三轮 ITL 为 **13.6086 / 13.6044 / 13.6045 ms/token**；候选配置记录 `max_num_seqs=32`。

> **记录范围：** 数值来自项目提供的 README 草稿；精确补丁、运行时指纹与原始计时文件尚未随附。`max_num_seqs=32` 是调度容量配置，实际请求并发未注明，不能读作 32 并发实测；1M 同样表示容量。该成绩按研究候选记录保留，独立于下面各 PR 的验证结果。

[候选配置、计算与待补证据](docs/benchmarks/deepseek-v41-candidate-20260921.md)

#### 已合入的优化进展

让下一轮 decode 尽早接上，也让长生成中的每一份状态都对得上。

普通 C1 路线已从融合算子，继续推进到 **V2 设备续跑、分组 Prefill、分页状态与分桶 Replay**。[PR #34](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/34) 将已验证的数值加速组合收进专用入口的默认配置，并修复 SWA 环形缓存回绕与 Reindex 前缀发布问题；修复后的长流式生成未出现此前的重复坍塌。

| 路线 | 测量范围 | 相对对应基线的变化 | 来源 |
|---|---|---:|---|
| **普通 C1 · 融合组合** | 三次完整模型运行的稳态 decode；该 PR 的 trace-driven 起点 | 延迟降低约 **1/3** | [#26](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/26) |
| **普通 C1 · V2 设备续跑** | 三次 × 192-token 固定参考验证；复用的可比基线 | decode 延迟中位数降低约 **10.4%** | [#30](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/30) |
| **普通 C1 · 长上下文候选** | 三轮请求计时；前一长上下文候选 | decode 延迟降低约 **22%** | [#32](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/32) |
| **DSpark C6** | 完整 C6 transaction；对应 DSpark 候选基线 | 延迟降低 **29.7%** | [#25](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/25) |
| 同一 DSpark 工作负载 | post-TTFT decode | 延迟降低 **28.5%** | [#25](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/25) |
| 同一 DSpark 工作负载 | 包含 prompt / prefill 的完整请求 | 延迟增加约 **0.8%** | [#25](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/25) |

**这些结果来自不同配置与基线，收益不相加。** #34 的状态修复没有增加 kernel 或 copy，延迟与其已验证父版本相差约 1% 以内；这是一项回归检查，不是额外的加速成绩。DSpark 的历史完整请求仍保留 N256 prefill 兼容路径的成本。

V4.1 分页路线现可配置 **1,048,576-token 总上下文容量**，prefill 使用最多 **8,192 tokens** 的调度分块。已归档的端到端检查覆盖跨越 8,192-token 分块边界的 prompt、后续短请求复用和多轮聊天；完整 1M 窗口的质量与长期稳定性仍需继续验证。

**C1 表示普通单 token decode，C6 表示六 token 验证工作负载；它们与上表的并发请求数含义不同。** V4.1 当前服务契约仍是单请求、贪心采样，DSpark 默认关闭。

[V4.1 当前配置与验证范围](docs/features/deepseek_v41.md) · [早期 DeepSeek 测量记录](docs/benchmarks/deepseek-records.md)

---

<a id="engineering"></a>

## 🔥 Built for Gaudi

Gaudi 上的推理优化，需要同时看见 **内存里的权重、设备上的状态，以及每一轮实际提交的工作**。

```text
Checkpoint / Prepared weights
              ↓
       模型分片与输入准备
              ↓
    Attention / MoE / GDN / mHC
              ↓
   KV / Recurrent state / Engram
              ↓
   TPC + MME + 通信 + Native replay
              ↓
        采样、状态提交与输出
```

### 权重按执行方式准备

DeepSeek 的 prepared 路径先校验模型 revision 与文件身份，再按目标 TP / PP 拓扑生成分片，整理 **Q16 / S16 专家布局、scale 编码与 K 对齐**。

每个 rank 保留一份压缩专家权重的常驻分配，让后续算子直接消费需要的布局，减少长期重复的展开副本。V4.1 普通 C1 组合进一步使用 N256 FP8 expert 与融合准备路径；可选的 **N256 runtime 分片**把最终布局提前准备好，通过文件指纹与逆布局检查后直接加载，减少重复启动时的准备工作。

**把可以提前做的准备，移出每一轮 decode。**

存储精度与计算精度分别记录：例如 block-FP8 linear 可以经 TPC 解量化后进入 **BF16 MME**；`--dtype bfloat16` 也不能概括所有权重、缓存、scale 与累加精度。

### Attention，沿着数据依赖一起优化

Q / KV 投影、归一化、RoPE、稀疏选择、KV 写入和输出投影，串起了完整的 Attention 执行链。

本分支围绕这条链组织融合与布局：让 **Q scaling + RoPE** 连续执行，让 selected-row 路径只处理有效行，让 shared-KV MME 与指定投影的 FP8 sidecar 配合，并保留 KV producer 到 consumer 的依赖。

TPC 处理适合专用实现与融合的张量计算，MME 承担相应矩阵计算。二者交接时少一次冗余转换、少一个临时结果，都有机会降低完整算子链的成本；是否采用候选实现，仍由精度和整模型测量决定。

### Prefill：按专家分组，让权重服务更多输入行

V4.1 专用入口默认启用有界的 **expert-grouped prefill**：按专家负载组织输入行，让解码后的权重服务同组 token，同时保留 clamp、BF16 舍入边界、路由顺序与有序累加。调度器仍按最多 **8,192 tokens** 分块接收 prompt。

跨分块进入 decode 时，CPU prompt 完成事件与设备 token 各自保留明确的归属；不同 prompt 形状也有独立的编译缓存，避免连续请求耗尽共享重编译额度。

### Engram：设备续跑，主机按需供给

V4.1 的 Engram 源表按完整 hash heads 分片，主机路径通过源文件映射与 native gather 供给数据。默认 V2 路线进一步让 **layer 1 从采样后的设备 token 继续准备**，主机 C1 则准备后续的 **layer 14** 输入：

```text
只读 checkpoint → mmap / 页缓存 → native gather
               → HPU-pinned staging → DMA → 模型消费
```

generation 与完成事件保护主机路径的 staging buffer，确保上一轮消费完成后再复用。设备续跑同时检查请求身份、历史与晚到输入的依赖，让两条供给路径在各自的消费点接上。

**原始 Engram 源文件仍是运行依赖。** prepared 权重生成后需继续保留这些文件；主机 RAM、页缓存、磁盘与 NUMA 条件也属于部署配置。

[深入了解权重、Attention 与 Engram](docs/features/1cat_gaudi_execution.md)

---

<a id="tp2"></a>

## 🔗 TP2 / TP2 × PP2：让多张卡协同工作

双卡执行同时取决于权重如何切分、每个 rank 做了什么，以及通信结果在何时被消费。

### Qwen TP2 · 把归约接到正确的边界

Qwen TP2 将 row-parallel 与 vocabulary embedding 的归约延后到匹配的归一化边界，组织 **AllReduce → residual add → RMSNorm**，并保留 stock HCCL 传输。Qwen3.5 的文本 embedding 则留在语言模型内，避免多模态包装绕过这条路径。

| 已归档路线 | 对应工作负载与结果 | 来源 |
|---|---|---|
| **融合归约边界** | Qwen3.8 FP8 TP2 单请求，稳态 decode 延迟降低约 **30%** | [#5](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/5) |
| **融合边界 + 8 层区域图** | 相对同一原始 TP2 路线，延迟降低约 **45%** | [#5](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/5) |
| **TP2 紧凑 Q/K Prefill** | 正确性门控后的 warm-request A/B，完整请求吞吐提升约 **8%** | [#10](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/10) |

**30% 与 45% 是独立配置对同一基线的结果，不能相加。** #5 的数字来自其已归档工作负载；当时合并主线后的新一轮硬件复测受设备故障阻断，未计入结果。

[#10](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/10) 还修复了 GemmaRMSNorm 未消费 deferred-reduction 标记导致的 collective 缺失：先检查所有 norm consumer，再改变归约标记；Gemma 边界保留安全的 HCCL 路线，native fused Gemma decode 仍关闭。两个 rank 的必要归约均核对实际执行，早期缺失 collective 的运行不作为有效基线。

### DeepSeek · 从双卡 TP2，到四卡分阶段执行

**V4** 在两张 Gaudi2 上切分权重与执行；**V4.1** 进一步以两个 TP2 stage 组成四卡 TP2 × PP2：

```text
PP0 · TP rank 0 + TP rank 1
       target layers 0–19 · embedding · Engram
                    │
       hidden states + previous mHC mixing
                    │  生产者完成 → 发送 → 消费前等待
                    ▼
PP1 · TP rank 0 + TP rank 1
       target layers 20–39 · output norm / head
```

四份 rank-local 权重、持久 PP buffer、完成事件与状态代次一起组成执行计划。普通 C1 不加载 DSpark draft；开启 DSpark 时才进入独立的 draft / verify 状态路线。

分页、bucket 与请求 slot 为扩展提供了实现基础；已验证的普通 C1 服务范围仍按 [V4.1 专页](docs/features/deepseek_v41.md)记录，候选配置中的序列上限不替代并发验收。

---

<a id="graph-compile"></a>

## ⚡ Graph + Compile

`torch.compile`、区域编译与 Native Replay 分别覆盖计算融合、图的边界和重复提交成本，组合起来服务完整请求：

| 层次 | 主要作用 |
|---|---|
| **`torch.compile`** | 融合张量计算，交由 Gaudi 编译后端 lowering 与生成 recipe |
| **区域编译** | 在多层或 stage 内组织图，控制编译规模并减少重复 Python 边界 |
| **Native Replay** | 复用 recipe、固定输入地址、通信命令与完成事件 |
| **通信计划** | 按真实 producer / consumer 依赖安排 collective 的提交与等待 |

### Native Replay · 让每一轮 Decode 复用执行计划

固定形状 decode 会重复经过相同的模型层。native replay 预先准备权重、输入与状态地址，捕获计算和通信计划；每一轮更新真实输入、执行计划并提交状态。

```text
准备固定地址 → 编译 / 捕获 → 恢复 capture 状态
                                 ↓
             更新输入 → Replay → 消费输出与安全复用
                ↑__________________________|
```

V4 native decoder 的限定路线覆盖 **43 层、86 次 decoder reductions**，包括末尾的 **mHC、HC head 和 norm**。embedding reduction、最终输出投影和采样由外围路径执行。

V4.1 则以 **TP2 × PP2** 组织 stage replay、PP 状态交接与 mHC / TP 依赖。普通 C1 默认使用 V2 异步调度、设备 token 提前提交和分段 PP0 输入发布；长上下文按 search bucket 准备执行计划，确认下一桶就绪后才允许续跑。DSpark 保留独立的验证路线。

长生成还要求状态地址保持一致：**decoded SWA 与 packed KV 使用同一环形行编号**，hot Reindex 前缀在更大搜索桶首次读取前完成发布。算子足够快之后，状态交接同样决定整段输出能否稳定延续。

每轮仍会更新 token IDs、positions、KV slot 等真实输入，路由专家也由当前 token 计算。capture 中修改过的状态需要恢复；缓存重分配、权重重载、通信器变化或状态地址重绑时，旧计划必须失效。状态写入后发生异常，当前执行会终止，避免带着旧状态继续。

一次 replay 入口可以包含多段计算、通信与 DMA。性能检查分别记录这些工作，再用完整设备窗口核对实际收益。

[执行架构](docs/features/1cat_gaudi_execution.md#native-replay) · [V4.1 prepared execution](docs/features/deepseek_v41.md)

---

## 🧠 GDN：从状态布局到编译图

Gated Delta Rule 的成本来自矩阵计算，也来自 recurrent state 的读取、更新和搬运。

Qwen GDN 路线围绕 **紧凑 Q / K heads、KKT 与 causal-decay 复用、静态三角 mask、分块求解、fused direct-state decode** 组织计算。MME 执行矩阵工作，外围张量操作由编译图融合。

当前重点 prefill 形状：

| 配置 | Q / K heads | V heads | K / V dimension |
|---|---:|---:|---:|
| **TP1** | 16 | 48 | 128 / 128 |
| **TP2，每 rank** | 8 | 24 | 128 / 128 |

该 prefill tactic 面向 **单条均匀序列、BF16 输入、chunk 128**；默认 recurrent state 保持 **FP32**。重点 fused decode buckets 为 **1 / 2 / 4 / 8 / 16 / 32**，具体状态布局与路径需要分别匹配。

在已配对的 Qwen 引擎与插件环境中启用：

```bash
export VLLM_HPU_FLASHINFER_GDN=1
export FLASHINFER_GAUDI_BACKEND=auto
```

TP2 另需 `VLLM_HPU_FLASHINFER_GDN_TP2=1`。归一化、状态精度与回滚语义均保留独立验证。

[PR #29](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/29) 进一步修复 **Qwen3.5 FP8 短 prompt 的数值漂移**，将缓存精度与 prefill 计算精度分开，并提供 `VLLM_GDN_PREFILL_STATE_FP32_MAX_TOKENS` 阈值开关：短桶可选择 FP32，长桶保留优化后的 BF16 / MME 路线；该开关默认值仍为 `0`，需要显式启用。已留存语义、自然 EOS 与 245,760-token 输入检索检查。这组验证与首页 **Qwen3.8-27B-FP8** 历史性能记录分别归档。

[形状与精度契约](docs/features/1cat_gaudi_execution.md#gdn-与-dflash2) · [Qwen 启动指南](docs/1cat_gaudi_guide.md#qwen-dflash2)

---

## 🚀 DFlash2：一次提出候选，整块验证

Qwen DFlash2 一次提出 **7 个 draft tokens**，经过 HPU top-16 selector 选出路径，由 target 验证 **8-token block**。

```text
Draft → HPU selector → Target block verification
                                   ↓
                    接受前缀 → 恢复对应状态 → 下一轮
```

每一步卷积与 GDN checkpoint 都参与接受和回滚，让 speculative decoding 的状态更新与 target 执行保持一致。

当前实验范围为 **贪心文本、TP1 / PP1 / DP1、最多 16 个序列、compact GDN state、关闭 prefix caching 与 LoRA**。DFlash2 默认关闭；DeepSeek V4.1 的 DSpark 采用另一套模型与执行集成。

首页 Qwen 历史数据没有附带 DFlash2 开关记录，因此不作为 DFlash2 的性能成绩。

[配置与完整启动命令](docs/1cat_gaudi_guide.md#qwen-dflash2)

---

<a id="h3"></a>

## 🎬 MiniMax H3：让画面与声音一起生成

**从文字，到首尾帧，再到图像、视频与音频参考。**

[PR #31](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/31) 将 H3 接入本插件与配套 vLLM-Omni：视频与音频 latent 在同一次 DiT 调用中联合预测，单张 **Gaudi2** 通过阶段卸载安排编码、去噪与解码的设备驻留。

| 生成路线 | 输入 | 四次 DiT 前向配置 |
|---|---|---|
| **T2VA** | 文本提示 | FastH3 / FlashGen，或匹配的 LightX2V |
| **FL2VA** | 首帧、尾帧或两者 | 匹配的 LightX2V FL2V |
| **Ref2VA** | 图像、视频、音频混合参考；至少含图像或视频 | 匹配的 LightX2V Ref2V |

默认 FastH3 路线采用 **BF16 基座 + 配套蒸馏适配器**。四次前向由适配器与采样日程共同决定；原始 Base 权重仅修改步数，不能获得同一套生成契约。在线 FP8 仍是显式 A/B 实验。

### 13.6 s · 视频 VAE 稳态解码

在已验证的 **124 帧、BF16** 形状上，FusedSDPA 与完整 TransformerBlock 编译将视频解码器稳态时间推进到约 **13.6 秒**；全帧质量检查记录为 **PSNR 50.86 dB / 最低 SSIM 0.99991**。首个请求另外包含建图成本。

当前单卡入口还默认启用相邻两个时间片成组与有界异步回传，让解码、搬运与帧顺序一起受到约束。

> **成绩属于对应阶段：** 13.6 秒只计视频 VAE 解码。已归档的 5 秒 T2VA 完整请求，在不同驻留与适配器配置下，后三次中位数仍为 **249.10 / 256.75 / 301.03 秒**；这些是历史基线，不能用来代表最新组合的完整请求耗时。文档中的 **25 秒**为后续验收目标。

[H3 环境、权重与单卡启动](docs/features/minimax_h3.md) · [完整请求与分阶段测量](docs/features/minimax_h3.md#reproduce-the-5-second-single-card-measurement)

---

<a id="flashinfer-gaudi"></a>

## 🧩 FlashInfer-Gaudi：让数据流变成实际收益

**熟悉的推理原语，面向 Gaudi 的执行选择。**

`flashinfer_gaudi` 使用独立命名空间，部分 API 语义对齐 **FlashInfer 0.6.18**，并提供 Gaudi 扩展：GDN prefill / decode、fused decode、MTP / rollback、activation / quant、residual / norm / quant，以及 block-FP8 linear。

### 从算子吞吐，到完整请求

在 **Qwen3.8-27B-FP8 · 单张 Gaudi2 · TP1** 上，[PR #4](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/4) 相对此前 vLLM-Gaudi GDN 路线归档了：

| 测量范围 | 对应已验证形状下的收益 |
|---|---:|
| **GDN Prefill 算子吞吐** | **2.3–2.8×** |
| **单请求 TTFT 加速比** | **1.34–1.47×** |
| **稳态 Decode 吞吐** | **1.25–1.92×** |
| **完整请求吞吐** | **1.27–1.55×** |
| Fused decode-step，相对已优化 direct-state 路线 | 进一步提升约 **0.4–0.8%** |

TTFT 加速比表示基线延迟除以优化后延迟；吞吐倍数表示优化后除以基线。各项对应自己的测量窗口和负载范围，也与首页 A800 / Gaudi2 历史对照分开。

独立的 [PR #2](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/2) 在同形状 exact-8K A/B 中获得约 **49%** 的输入吞吐提升；compact-KKT producer 微基准约 **2×**，完整 GDN core 进一步提升约 **3%**，生成输出与接受基线一致。

### 提速来自哪里

```text
紧凑 Q/K heads + grouped Q/K reuse
                ↓
FlashQLA Prefill + 静态 mask / KKT
                ↓
direct recurrent state + direct convolution state
                ↓
小 batch 动态 FP8 quant + fused decode-step
                ↓
区域编译图 + 按已验证形状选择的执行路线
```

这条路线减少 Q/K 的重复展开、状态搬运和逐 kernel 提交，同时保留 FP32 recurrent state 与已验证的更新顺序。**算子与完整请求共同决定是否采用优化**；未验证的 dtype、layout、状态索引或 bucket 保留既有路线。

### 原生算子，也按完整操作验收

[PR #11](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/11) 的 block-linear 在部分小 batch down-projection 形状上获得约 **6%** 的完整算子加速，并用真实 checkpoint 权重确认；激活仍为合成输入。这条路径是 **TPC 解量化 + BF16 MME GEMM**，其完整形状矩阵平均仍较慢，未自动替换模型路径。

[PR #15](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/15) 进一步提供 residual add、RMSNorm 与 quantization 的原生融合；默认关闭，针对新模型仍需独立验证，尚不据此声明 V4.1 整体收益。

```python
import json
from flashinfer_gaudi import get_capabilities

print(json.dumps(get_capabilities(), indent=2, ensure_ascii=False, default=str))
```

`auto` 保留已建立的服务编译图策略；`pytorch` 提供参考路线。`native`、`public`、`bridge` 要求操作满足对应完整原生契约，不满足时明确拒绝。部分 GDN TPC 原型仍带数值 prologue；**兼容接口、原生实现与自动启用资格需分别查询**。

[API 与后端策略](docs/features/1cat_gaudi_execution.md#flashinfer-gaudi) · [兼容与形状契约](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/5fa3e08302716d458a82aa7d95d107e201b51d8c/docs/features/flashinfer_gaudi.md)

---

<a id="models"></a>

## 🎯 模型与执行范围

| 路线 | 当前配置范围 | 启用方式与状态 |
|---|---|---|
| **Qwen GDN / TP2** | Qwen3.5 / Qwen3.8；重点形状与数值契约分别验证 | GDN、紧凑 Q/K 与归约融合各有独立开关及回退条件 |
| **Qwen DFlash2** | TP1 / PP1 / DP1；贪心文本；最多 16 个序列 | 默认关闭，实验路径 |
| **DeepSeek V4 Flash** | Gaudi2 × 2；TP2 / PP1；单请求；512 tokens 总上下文 | 专用入口使用普通 decode 与区域编译 |
| **DeepSeek V4 native decoder** | TP2 / PP1；单 token decode；512 tokens 总上下文 | 默认关闭，需独立 native runtime |
| **DeepSeek V4.1 Flash C1** | Gaudi2 × 4；TP2 × PP2；单请求；贪心；默认 512 tokens，可配置分页容量至 1,048,576 tokens | **专用入口默认启用 V2 与已验证的数值组合**；需配套 runtime 和两个 FP8 sidecar |
| **DeepSeek V4.1 DSpark** | 独立 prepared / draft / replay 配置 | 默认关闭 |
| **MiniMax M3** | 已接入 HPU 模型、视觉语言入口与 `minimax_m3_py` 工具解析器 | 当前 MSA 注意力按 dense causal 路线执行；此处仅列代码集成范围 |
| **MiniMax H3** | 单张 Gaudi2；T2VA / FL2VA / Ref2VA；分阶段卸载 | 配套 vLLM-Omni；默认 BF16，四次前向须匹配蒸馏适配器 |
| 通用 HPU 模型 | 由引擎、插件、模型及软件版本共同决定 | 参考[继承的功能范围](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/5fa3e08302716d458a82aa7d95d107e201b51d8c/docs/features/supported_features.md) |

MiniMax M3 的模型注册、当前注意力实现与工具解析可分别查看[模型源码](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/5fa3e08302716d458a82aa7d95d107e201b51d8c/vllm_gaudi/models/minimax_m3.py)和[解析器](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/5fa3e08302716d458a82aa7d95d107e201b51d8c/vllm_gaudi/entrypoints/openai/tool_parsers/minimax_m3.py)。

DeepSeek 的上述专用配置仍处于限定范围的工程验证阶段。总上下文预算由输入与输出共享；**V4 仍限定 512 tokens，V4.1 的 512 是入口默认值，已不代表容量上限。** V4.1 完整 1M 窗口与长期稳定性尚待验收，从零部署所需的引擎锁定信息也尚未完整归档，请从已验证的配套环境开始。当前开关与容量契约以 [V4.1 专页](docs/features/deepseek_v41.md)为准。

---

<a id="quick-start"></a>
<a id="getting-started"></a>
<a id="installation"></a>

## 📦 Build & Run

<a id="free-trial"></a>

### 先在真机上试一试

暂时没有 Gaudi2 设备？从我们的自建服务器集群开始。

**1Cat-vLLM-Gaudi2** 服务器集群提供 **Intel® Gaudi® 2 免费体验**，欢迎动手尝试模型推理、适配与开发，让 README 里的工程路线变成自己的实践。

**[立即免费体验 Gaudi2 →](http://dx.1catai.com:53852/rental)**

体验方式、可用资源与具体规则以 **1CatDL 算力工作台**页面说明为准。需要在自己的环境中部署，可继续阅读下方的准备与启动步骤。

### 准备配套环境

DeepSeek 源码集成的基线为 **Gaudi Software 1.24.1 + 匹配的 Gaudi PyTorch 2.11**。专用工具使用 Python 3.11+ 接口，建议选择配套的 Python 3.11 / 3.12 环境。

```bash
git clone https://github.com/1CatAI/1Cat-vLLM-Gaudi.git
cd 1Cat-vLLM-Gaudi
```

`vllm_gaudi` 是 Python 包名。按[安装指南](docs/1cat_gaudi_guide.md#installation)固定插件、引擎和 Bridge 版本，并完成对应构建；以下命令使用 Linux / Bash，从仓库根目录执行。

### 选择模型路线

<a id="serving"></a>
<a id="v4-serving"></a>

#### DeepSeek V4 Flash · 2×Gaudi2

完成[固定引擎与原生库构建](docs/1cat_gaudi_guide.md#deepseek-v4)后：

```bash
HABANA_VISIBLE_MODULES=0,1 \
python -m vllm_gaudi.entrypoints.deepseek_v4 \
  /path/to/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --host 127.0.0.1 \
  --port 8000
```

入口选择 TP2、单请求和 512-token 总上下文。整段 native decoder replay 需要另外启用。

<a id="v41-serving"></a>

#### DeepSeek V4.1 Flash · 4×Gaudi2

先完成[引擎、runtime、prepared 权重与 sidecar 准备](docs/features/deepseek_v41.md#preparation)，并用当前修订重新构建 native kernels 与 replay bridge。**两个 FP8 sidecar 必须同时就绪**：`sidecars/wo_a_fp8` 与 `sidecars/attention_dense_fp8`；专用入口会在模型加载前检查，缺失时直接报错。

使用显式模块选择和共享设备锁启动：

```bash
python tools/run_deepseek_v41.py \
  --lock-dir /path/to/shared-device-locks \
  --runtime-profile /path/to/verified-runtime-profile.json \
  --modules 0,1,2,3 \
  /path/to/new-run-evidence \
  -- \
  python -m vllm_gaudi.entrypoints.deepseek_v41 \
    /path/to/DeepSeek-V4.1-Flash-prepared \
    --checkpoint-audit /path/to/checkpoint-audit \
    --host 127.0.0.1 \
    --port 8000
```

路径与设备编号需要对应本机。启动器选项放在 evidence 位置参数之前。上例默认 **512-token 总上下文、普通 C1、V2 异步执行与已验证的数值加速组合**，DSpark 默认关闭；设备租赁入口会保留所选模块。配置优先级和 profile 结构见[运行配置](docs/1cat_gaudi_guide.md#runtime-profile)。

需要长上下文时，可先将最终 N256 布局准备成可复用分片：

```bash
python tools/prepare_deepseek_v41_n256.py \
  /path/to/DeepSeek-V4.1-Flash-prepared \
  /path/to/DeepSeek-V4.1-Flash-n256
```

随后在上面的 `deepseek_v41` 命令后追加下列参数；这些是**命令续接参数**，需与同一启动命令一起使用：

```text
--n256-prepared-dir /path/to/DeepSeek-V4.1-Flash-n256
--max-model-len 1048576
--max-num-batched-tokens 8192
--max-num-seqs 1
--block-size 128
--num-gpu-blocks-override 8193
```

这里给出 1M **容量配置**；完整窗口的质量验收需另行完成。N256 缓存可选，不传入时仍在加载阶段准备。分片指纹、页表与 runtime 要求见[长上下文配置](docs/features/deepseek_v41.md#prepared-runtime-weights-and-long-context-serving)。

通用 vLLM 入口仍保留显式启用的默认策略。诊断时可在启动前设置 `VLLM_HPU_DSV41_EXPERIMENTAL_NUMERIC_FASTPATHS=0` 选择结构兼容组合；`VLLM_HPU_DSV41_DEFAULT_FASTPATHS=0` 只停止专用入口注入默认值，不会清除环境中已有的开关。

<a id="dflash2-serving"></a>

#### Qwen GDN / DFlash2

在已配对的 Qwen 引擎与插件环境中，可启用 GDN：

```bash
export VLLM_HPU_FLASHINFER_GDN=1
export FLASHINFER_GAUDI_BACKEND=auto
```

TP2 另需 `VLLM_HPU_FLASHINFER_GDN_TP2=1`。DFlash2 的独立启动命令、状态与缓存条件见[Qwen 指南](docs/1cat_gaudi_guide.md#qwen-dflash2)。

#### MiniMax H3 · 单张 Gaudi2

先按 [H3 专页](docs/features/minimax_h3.md#qualified-dependency-set)配对固定版本的 vLLM、vLLM-Omni 与 Gaudi 软件环境，再安装音视频依赖并准备本地 BF16 模型和 FastH3 适配器：

```bash
python -m pip install -e '.[omni]'

VLLM_GAUDI_LOCK_DIR=/path/to/shared-device-locks \
python tools/minimax_h3/serve_single_hpu.py \
  /path/to/MiniMax-H3-FastH3-HPU \
  --partition FL2VA \
  --fasth3-4step-adapter /path/to/FastH3-4-step-Preview-v1-LoRA \
  --module 0 \
  --temp-dir /path/to/h3-tmp \
  --port 8097
```

上例加载 FL2VA 分区中的基座，配合 FastH3 提供 **T2VA** 四次前向服务。需要首尾帧或混合参考生成时，请使用匹配的 LightX2V 配置；[请求示例](docs/features/minimax_h3.md#requests)分别列出每种路线。

### 发起请求

以下请求对应 V4.1 的默认服务名：

```bash
curl --fail --show-error http://127.0.0.1:8000/health

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

V4.1 目前要求不附加采样修饰的贪心请求；`logprobs`、结构化输出及若干 token 约束暂不支持。完整参数范围见[V4.1 服务契约](docs/features/deepseek_v41.md)。示例监听 localhost，对外服务需配置认证与访问控制。

---

## ✅ Correctness & Quality

**速度、输出质量和执行稳定性，一起推进。**

从微基准到服务，每一层都保留自己的验收范围：

| 层次 | 主要检查 |
|---|---|
| **Kernel** | 编码、边界值、shape、layout、真实 dtype 与参考结果 |
| **执行链** | producer / consumer 依赖、搬运量、SRAM / HBM 放置与必要 collective |
| **模型** | token / logprob、KV / Engram / Router 状态与请求隔离 |
| **服务** | TTFT、ITL、吞吐、取消与复用、长运行及正常退出 |

微基准筛选候选，完整请求确认实际收益。源码、模型与运行时指纹、输入输出长度、实际并发、计时窗口、失败与退化结果，应与成绩一起保存。以下列出已归档进展：

| 工程进展 | 已记录的检查 | 验证边界 |
|---|---|---|
| **V4.1 默认组合与长生成状态 · [#34](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/34)** | 54 项针对性单元 / 契约测试通过、1 项跳过；Gaudi2 环形回绕检查与 packed cache 逐位一致；修复后的长流式生成未发生重复坍塌 | 限于该组合与归档生成；完整窗口、广泛任务质量及长期稳定性仍需验证 |
| **V4.1 分页与分组 Prefill · [#32](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/32)** | 129 项 CPU 契约测试通过；跨 8,192-token 边界、短请求复用、算术、流式 / 非流式 / 多轮聊天记录 | 1M 是容量契约；跨分块检查不等于完整 1M 窗口验收 |
| **V4.1 V2 设备续跑 · [#30](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/30)** | 132 项 CPU 测试通过、53 项设备依赖测试跳过；三轮各 192-token 生成与冻结参考一致 | 一致性结果属于当时固定组合，不外推至所有数值配置 |
| **Qwen3.5 FP8 短 prompt · [#29](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/29)** | 223 项针对性测试通过；语义与自然 EOS、有限 logprob、245,760-token 输入检索及状态槽复用记录 | 短桶 FP32 阈值须显式启用；与 Qwen3.8 历史性能记录区分 |
| **MiniMax H3 · [#31](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/31)** | PR 摘要记录 69 项针对性测试通过；单卡音视频完整解码、124 帧质量与确定性检查 | 不同阶段分别采用逐位或数值质量门槛；最新完整请求耗时需重新测量 |

V4.1 的检查已覆盖此前缺失的设备续跑、跨分块状态和环形缓存回绕，质量证据会继续随具体组合向前更新。[#33](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/33) 的 streaming bucket 实验仍是 Draft，未计入本页已合入的能力。

V4 普通路径仍保留冷启动与后续贪心输出差异记录；V4 native 的 frozen-reference 差异及独立异步链退出问题也尚无新的闭环证据。早期 V4 / DSpark 结果见[历史验证记录](docs/benchmarks/deepseek-records.md)，不能套用 V4.1 的修复结论。

这些是对应 PR 与文档留存的测试记录，**跳过项不计为通过**；性能、逐位一致、数值质量与完整请求分别保留自己的验证范围。

---

## 🧱 Runtime Matters

一条可复现的原生执行路径，需要 **插件、引擎、Bridge、Synapse、HCL、原生扩展和模型制品** 成套匹配。

本项目用 manifest、二进制与配置指纹记录这些关系。部分 replay runtime 基于旧公开源码快照及兼容性补丁，ABI 检查通过只证明所检查的接口与制品关系；数值和编译器等价性仍需单独验证。

这也是工程指南同时保存构建步骤、profile、设备选择、prepared 身份与运行证据的原因：让一次优化能够被定位、被复现，也能在后续升级中继续验证。

[环境与构建指南](docs/1cat_gaudi_guide.md) · [Native runtime 源码与补丁](tools/communication/patches/native-runtime/README.md)

---

<a id="source-map"></a>

## 🗂️ 文档导航

| 文档 | 从这里开始 |
|---|---|
| [工程指南](docs/1cat_gaudi_guide.md) | 环境、构建、prepared 权重、运行配置、API 与排障 |
| [执行架构](docs/features/1cat_gaudi_execution.md) | 权重、Attention、TPC / MME、Engram、replay、GDN 与 DFlash2 |
| [V4.1 专用路线](docs/features/deepseek_v41.md) | N256 分片、当前默认组合、V2 与分页长上下文 |
| [MiniMax H3](docs/features/minimax_h3.md) | 模型与适配器、单卡音视频生成、VAE 与完整请求测量 |
| [Qwen 数据记录](docs/benchmarks/qwen38-a800-gaudi2.md) | 实测数值、推算公式、来源与条件 |
| [V4.1 研究候选记录](docs/benchmarks/deepseek-v41-candidate-20260921.md) | 73.50 tokens/s 的来源、配置、ITL 换算与待补证据 |
| [DeepSeek 早期优化记录](docs/benchmarks/deepseek-records.md) | V4、DSpark 与早期 C1 的测量范围、测试结果 |
| [环境变量](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/5fa3e08302716d458a82aa7d95d107e201b51d8c/docs/configuration/env_variables.md) | 原始默认值、依赖与 profile 范围 |

## 🧭 项目方向

我们希望 Gaudi 用户可以持续获得现代模型支持，并且看得见每一轮优化带来的实际变化：

- **更快的响应**：缩短首字与稳态 decode 延迟。
- **更好的扩展**：减少状态搬运、通信与调度成本，推进长上下文质量验证。
- **更丰富的生成**：让文本与音视频路线拥有清晰的执行与测量契约。
- **更清楚的质量证据**：让性能成绩对应明确的模型输出和验证范围。
- **更容易复现的环境**：将运行时、权重与配置一起归档。

从单个算子，到整段执行，再到完整请求——让优化持续落到用户真正等待的时间上。

<a id="contributing"></a>

## 🤝 参与贡献与交流

欢迎模型适配、算子、编译、通信、数值验证和文档贡献。性能变更请附上父配置、可复现命令与制品指纹、微基准及完整请求结果、质量与状态检查，以及失败或尚未完成的验证。

提交请使用 feature branch 和 Pull Request，保留 `Signed-off-by`，遵循 [AGENTS.md](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/5fa3e08302716d458a82aa7d95d107e201b51d8c/AGENTS.md)。

[提交 Issue](https://github.com/1CatAI/1Cat-vLLM-Gaudi/issues) · [查看 Pull Requests](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pulls) · [1CatAI 主项目与社区](https://github.com/1CatAI/1Cat-vLLM)

## ❤️ 致谢

感谢 [vLLM](https://github.com/vllm-project/vllm)、[vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 与 Intel Gaudi 软件生态、DeepSeek / Qwen / MiniMax 等模型团队，也感谢包括 [@yangzhuxinyzx](https://github.com/yangzhuxinyzx) 在内的实现、测试与复现贡献者。

## License

代码采用 [Apache License 2.0](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/5fa3e08302716d458a82aa7d95d107e201b51d8c/LICENSE)。模型权重与第三方依赖适用各自许可。

<sub>合并整理：2026-09-21。执行源码基准：5fa3e08302716d458a82aa7d95d107e201b51d8c（main / PR #34）；本次重新核对主分支与引用 PR。性能与质量数字沿用各自 PR、文档或注明来源的研究候选记录，本次未运行硬件测试。Qwen 数据整理与条件补充：2026-09-15；测量日期未提供。</sub>

<p align="center"><strong>1CatAI · Let's Gaudi it.</strong></p>
