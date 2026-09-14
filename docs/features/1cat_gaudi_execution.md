# Gaudi 执行架构

本页说明 1Cat-vLLM-Gaudi 新增路线的数据流与适用范围。安装见[工程指南](../1cat_gaudi_guide.md)，测量结果见[性能记录](../../README.md#performance)。

```text
checkpoint / prepared weights
              ↓
模型分片与输入准备
              ↓
Attention / MoE / GDN / mHC
              ↓
KV、recurrent state 与 Engram 数据访问
              ↓
图编译、TPC / MME、多卡通信
              ↓
采样、状态提交与 API 输出
```

## 权重布局与精度

prepared 路径先校验模型 revision 和文件身份，再按目标 TP / PP 拓扑生成分片、整理专家布局及 scale 编码。V4 / V4.1 的专家路径包含 Q16 / S16 prepared 布局和明确的 K 对齐；V4.1 普通 C1 组合使用 N256 FP8 expert 及融合准备路径。

每个 rank 保留一份压缩专家权重的常驻分配，避免在 decode 中维持重复的展开权重。临时张量和 prefill 兼容计算仍有各自的成本。

| 路线 | 存储与计算含义 |
|---|---|
| MXFP4 prepared | 保留压缩专家权重；具体算子决定实际计算路线 |
| Block-FP8 linear | 可由 TPC 解量化后接 BF16 MME |
| V4.1 FP8 sidecar | 为指定投影生成精度、布局与身份绑定的派生制品 |
| `--dtype bfloat16` | 不是对全部权重存储、缓存、scale、累加和 logits 的统一精度声明 |

## Attention 与状态

Attention 路径包括投影、归一化、RoPE、稀疏选择、物理缓存定位和状态写入。

| 环节 | 本分支中的工作 |
|---|---|
| 输入 | Q / KV 投影融合、Attention norm、Q scaling + RoPE |
| KV producer | packed KV / FP4 cache write、SWA 写入与固定位置元数据 |
| 稀疏读取 | selected-row valid-only、decoded KV state、CSA2 / MLA |
| 矩阵计算 | shared-KV MME Attention、投影 FP8 sidecar |
| 输出 | 输出投影、状态依赖与 replay 交接 |

选择条件由模型、形状、精度和 profile 共同决定。原生状态写入必须保留 producer / consumer 依赖。

## TPC 与 MME

TPC 处理适合专用实现与融合的张量计算，MME 执行相应矩阵计算。压缩权重可以经 TPC 解码后交由 MME；残差、RMSNorm、scale 与 FP8 packing 可以沿消费者路径组织；expert 输出可以融合 scale 与 finalize reduction。

一个 Synapse recipe 可以包含多个 device kernels。是否采用原生候选，以完整算子链、精度及整模型结果决定。已有 mixed-engine、quantization 候选的未通过性能门槛也保留在[原生后端说明](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/docs/features/flashinfer_native_backend.md)中。

## Engram 主机表

V4.1 按完整 hash heads 对主机表分片，manifest 保存源文件及字节偏移。执行过程为：

```text
只读 checkpoint → mmap / 页缓存 → native gather
               → 小型 HPU-pinned staging → DMA → 模型消费
```

generation 与完成事件保护 staging buffer，前一轮仍在消费时不能覆写。默认不强制把完整主机表锁进物理内存。原始 Engram 文件仍是运行依赖，减少 HBM 占用不会消除 RAM、页缓存和磁盘成本。

## Native replay

固定形状 decode 可以预先准备权重、输入和状态地址，捕获计算与通信计划，再在各轮更新输入并 replay。capture 改动的状态必须恢复，消费者完成后才能安全复用缓冲区。

V4 限定路线保留 **43 层、86 次 decoder reductions**，包括最后的 mHC、**HC head** 和 norm。embedding reduction、最终输出投影与采样在外围路径执行。

V4.1 使用 TP2×PP2 stage replay，并维护 PP 状态交接、mHC / TP 依赖及独立的 DSpark 验证路线。缓存重分配、权重重载、通信器变化或状态地址重绑都会使旧计划失效。

## FlashInfer-Gaudi

`flashinfer_gaudi` 是独立命名空间，部分 API 语义对齐 FlashInfer 0.6.18，并提供 Gaudi 扩展。

| 领域 | 接口举例 |
|---|---|
| GDN prefill | `chunk_gated_delta_rule` |
| GDN decode | `gated_delta_rule_decode`、`gated_delta_rule_decode_pretranspose` |
| Fused decode | `gdn_fused_decode_step` |
| MTP / rollback | `gated_delta_rule_mtp_packed`、`gated_delta_rule_mtp_rollback` |
| Activation / quant | `silu_and_mul`、`silu_and_mul_quant` |
| Residual / norm / quant | `fused_add_rmsnorm_quant` |
| Block-FP8 | `block_fp8_dequant`、`block_fp8_linear` |
| 后端与能力 | `get_capabilities`、`set_backend_policy` |

```python
import json
from flashinfer_gaudi import get_capabilities
print(json.dumps(get_capabilities(), indent=2, ensure_ascii=False, default=str))
```

`auto` 保留已建立的服务编译图策略；`pytorch` 使用参考路线；`native`、`public`、`bridge` 要求对应完整原生契约。部分 GDN TPC 原型仍带数值 prologue，严格策略会明确拒绝不完整实现。

接口、形状、dtype 与原生资格分别查询。完整说明见[兼容算子文档](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/docs/features/flashinfer_gaudi.md)。

## GDN 与 DFlash2

Qwen GDN 使用紧凑 Q/K heads、KKT 与 causal-decay 复用、静态三角 mask、分块求解和 fused direct-state decode。矩阵计算保留 MME 路线，外围计算由编译图组织；默认 recurrent state 为 FP32。

重点 prefill 形状为 TP1 的 16 个 Q/K heads、48 个 V heads，或 TP2 每 rank 的 8 / 24；K / V dimension 为 128，chunk 128，单条均匀序列。重点 fused decode buckets 为 1 / 2 / 4 / 8 / 16 / 32；TP2 仍需单独启用与验证。

GDN decode 的 Q/K 归一化保留 `x * rsqrt(sum(x*x) + 1e-6)` 的加 epsilon 契约。数值基线改变后，需要重新建立质量与性能证据。

DFlash2 一次提出七个 draft tokens，通过 HPU top-16 selector 选出路径，target 验证八 token block；每一步卷积与 GDN checkpoint 用于接受前缀和回滚。当前初始范围是贪心文本、TP1/PP1/DP1、最多 16 序列、compact GDN state、关闭 prefix caching 与 LoRA。

DeepSeek V4.1 DSpark 和 Qwen DFlash2 各有模型与执行集成。开关、状态与测量应按对应路线配置。

## 源码导航

| 模块 | 入口 |
|---|---|
| 插件与平台 | [vllm_gaudi](https://github.com/1CatAI/1Cat-vLLM-Gaudi/tree/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/vllm_gaudi) |
| 模型与 loader | [models](https://github.com/1CatAI/1Cat-vLLM-Gaudi/tree/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/vllm_gaudi/models) |
| 算子与编译 | [ops](https://github.com/1CatAI/1Cat-vLLM-Gaudi/tree/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/vllm_gaudi/ops) · [compilation](https://github.com/1CatAI/1Cat-vLLM-Gaudi/tree/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/vllm_gaudi/compilation) |
| 原生内核 | [csrc](https://github.com/1CatAI/1Cat-vLLM-Gaudi/tree/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/csrc) |
| 多卡与 runtime | [distributed](https://github.com/1CatAI/1Cat-vLLM-Gaudi/tree/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/vllm_gaudi/distributed) · [communication](https://github.com/1CatAI/1Cat-vLLM-Gaudi/tree/ac14567637ca1d8f3d4434e62dbab3317ebe9fbe/tools/communication) |
