# DeepSeek V4.1 Flash · 研究候选记录

本页保存合并版 README 中 **73.50 tokens/s** 的数据来源与配置范围。它是一条项目提供的研究候选记录，独立于主分支 PR 的归档验证结果。

## 来源与状态

- 来源：用户提供的 `1cat-vllm-gaudi-README-DRAFT.md`。
- 文档整理日期：2026-09-21；原始测量日期未注明。
- 草稿声明的源码基线：`5fa3e083`，对应主分支完整 SHA `5fa3e08302716d458a82aa7d95d107e201b51d8c`。
- 硬件与拓扑：4× Gaudi2，TP2 × PP2；DSpark 关闭。
- 精确候选补丁、引擎版本、runtime / 模型指纹、原始日志、计时窗口及实际请求并发：未随草稿提供。
- 本次只做来源整理、配置对照与计算核对，未复跑模型。

## 候选记录

| 项目 | 原始记录 |
|---|---:|
| 输入 tokens | 2,052 |
| 输出 tokens | 256 |
| 第 1 轮稳态 ITL | 13.6086 ms/token |
| 第 2 轮稳态 ITL | 13.6044 ms/token |
| 第 3 轮稳态 ITL | 13.6045 ms/token |
| 三轮平均稳态 ITL | 13.606 ms/token（四舍五入） |
| 由平均 ITL 换算的 Decode 速度 | 73.50 tokens/s（四舍五入） |

计算：

```text
平均 ITL = (13.6086 + 13.6044 + 13.6045) / 3
         = 13.605833... ms/token

Decode 速度 = 1000 / 平均 ITL
            ≈ 73.50 tokens/s
```

该速度是 ITL 的倒数换算，不是独立测得的完整请求吞吐；计算本身不验证测量方法。草稿未说明 ITL 的起止事件、warmup 与各轮内部汇总方法。

## 候选容量配置

| 参数 | 记录值 |
|---|---:|
| `max_model_len` | 1,048,576 |
| `max_num_batched_tokens` | 8,192 |
| `max_num_seqs` | 32 |
| `block_size` | 128 |
| KV blocks | 8,193 |

`max_num_seqs=32` 是配置的序列上限，不证明测试实际并发为 32，也不证明 32 并发已通过质量或吞吐验收。1,048,576 是容量配置，不等于完整 1M 窗口质量验收。

在本次基线的 [V4.1 专页](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/5fa3e08302716d458a82aa7d95d107e201b51d8c/docs/features/deepseek_v41.md)中，已验证的服务契约仍是单请求、贪心采样；入口默认 `max_num_seqs=1`。因此合并版 README 的启动示例保留该值，并把候选容量单独记录于此。

## 完整复现还需提供

精确源码补丁及引擎/runtime/模型指纹、完整启动参数与环境覆盖、实际请求并发和输入输出长度、原始计时文件与统计方法、匹配的输出质量与状态复用记录。

这条候选记录不与 PR #26 的约 1/3、#30 的约 10.4% 或 #32 的约 22% 相对收益相加，也不视为已经发布的 release 成绩。

[结构化记录](deepseek-v41-candidate-20260921.json) · [返回 README](../../README.md#performance)
