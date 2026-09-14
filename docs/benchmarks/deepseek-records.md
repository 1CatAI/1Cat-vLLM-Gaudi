# DeepSeek 优化与验证记录

本页汇总仓库 PR 作者保留的测量与测试摘要。性能基线按每个 PR 自身的定义读取；文档修订没有重新执行硬件测试。

## 性能口径

| 记录 | 配置 / 范围 | PR 中保留的基线描述 | 结果 |
|---|---|---|---|
| [PR #26](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/26) | V4.1、4×Gaudi2、TP2×PP2、DSpark off；三次完整模型运行的稳态 decode | trace-driven starting point | 延迟降低约 1/3 |
| [PR #25](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/25) | DSpark、TP2×PP2、完整 C6 transaction | retained candidate reference | 延迟降低 29.7% |
| 同一 PR #25 工作负载 | post-TTFT decode | 同上 | 延迟降低 28.5% |
| 同一 PR #25 工作负载 | expert 组件 | 同上 | 延迟降低 40.5% |
| 同一 PR #25 工作负载 | 包含 prompt / prefill 的完整请求 | 同上 | 延迟增加约 0.8% |

C1 是普通单 token decode；C6 是 DSpark 六 token 验证工作负载。C6 不表示并发 6，也不表示每轮一定接受六个输出 token。

PR #25 的完整请求结果保留了 N256 prefill 兼容路径的开销。不同测量范围的改善不能相加。

## 复现信息

PR 摘要已明确相对基线与测量范围，但没有提供完整的可独立复现配置包。后续补齐时，应保存：

- 基线与候选的插件、引擎 commit 和实际补丁；
- 模型 revision、prepared / sidecar 身份、native runtime 指纹；
- prompt、输入输出长度、采样、实际并发、warmup 和计时边界；
- CPU/NUMA、设备拓扑、统计方式、波动及失败记录。

PR 的代码 head/merge commit 只标识代码版本。性能基线需要由对应归档运行记录识别，不能由 PR 的 base commit 自动推定。

## 测试记录

| 来源 | 作者记录的检查 | 保留的限制 |
|---|---|---|
| [PR #26](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/26) | 61 项单元测试通过，14 项硬件依赖测试在 CPU 运行中跳过；另有 mHC gates、expert-finalize HPU 检查 | 完整质量、prefill 一致性和长 replay/shutdown 未完成 |
| [PR #25](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/25) | 494 项相关单元测试通过、58 跳过；记录覆盖 28 项针对性 Gaudi 硬件检查 | DSpark 状态与完整请求性能需继续验证 |
| [PR #21](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/21) | 96 项 CPU/meta 测试通过、16 项硬件门控测试跳过；native topology 回归通过 | 生产参考 token/logprob 一致性及独立异步链退出未完成 |

这些测试集合各有范围，不能相加作为全仓库覆盖率。跳过项不计为通过。

## 当前状态

V4.1 专用入口默认启用实验 C1 组合，通用入口仍使用显式开关，DSpark 默认关闭。V4 native decoder、prepared MXFP4 与 Sinkhorn 仍需单独启用。

V4 普通路径已记录同一 prompt 在初始与后续贪心请求间的输出差异。V4 native 的 frozen-reference 差异和独立异步链退出时 heap failure 仍未解决。V4.1 C1 相对前一候选存在输出变化，其完整质量、prefill 数值一致性和生命周期验证仍在进行。

检查命令与运行记录字段见[工程指南](../1cat_gaudi_guide.md#validation)。
