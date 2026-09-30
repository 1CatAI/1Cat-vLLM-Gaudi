# TP4 decode 长期开发台账

更新：2026-09-30。只记录具有可比完整消费链微基准收益、通过适用正确性检查的改动。
未验证方向、失败或变慢试验留在实验INDEX，不计累计。端到端通过后关账，不再次计入待验收收益。

## 当前正式测量与待验收累计

2026-09-30：共享C1计算、N卡点对点交换与完整stage原生重放已正式满足≤12.3 ms。
修复concat轴的shape-agnostic缓存键后，恢复全部预热形状，并完成一次无profiler正式16K→EOS复测：
**11.083109 ms/token，90.227389 tokens/s，2579输出token**。
输出token与此前11.084135 ms那次精确一致；两次均通过固定资料事实检查。
该差值不算新增优化收益，只确认完整预热保持已验收速度。

| 项目 | 当前值 |
|---|---:|
| 最新正式测量 | shared-stage-W3-serving-11：**11.083109 ms/token** |
| 历史正式基线serving11 | 14.622559 ms/token；16K→2989自然EOS |
| 本轮正式请求 | **1次**；16K→2579自然EOS；无profiler |
| 完整启动预热 | 两种起始位置×8192/4096/2048/1024/512/256/128×四卡，**56/56通过** |
| 新增质量检查 | **4个不同固定16K样本，4/4自然EOS与语义通过** |
| 服务容量 | 上下文1,048,576；prefill8192；32请求槽 |
| 共享重放16层门槛结果 | chain42：5.673781→shared-stage-W3-real16-06：**5.260995 ms** |
| 已接受组合覆盖的待验收项 | P05与共享C1 W1/W2/W3关账，不再累计 |
| 当前待验收微基准收益 / 端到端预估 | **0 / 0 ms** |
| TP2×PP2同代码回归 | 用户明确取消；未运行，不计验收结果 |
| 草稿PR | [#52](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/52)已创建；未合并，与最新main的改动仍需解决集成冲突 |

concat修复让轴参与eager缓存键，同一轴仍保留跨形状复用；全局shape-agnostic缓存开启。
Bridge源补丁已包含修复；当前安装前端使用锁定ABI的私有适配库，系统共享库不变。
临时仅预热C8192的启动绕行及`_TP4_FORCE_DISABLED`均已移除。
四卡退役计数确认本轮所有请求无热期prepare/capture，无逐分组fallback。
全局关闭缓存导致real16-07=6.953042 ms的方案已拒绝，未计收益。

新增固定样本分别检查中文预算修订、订单去重汇总、英文批准政策和并行依赖。
完成响应已逐一阅读，金额/日期/关键路径正确，未知联系方式及税率均为null。
这些有限样本的通过不等于覆盖全部模型任务或长请求边界。

[正式复测与完整预热证据](../../evidence/20260928_tp4-decode-gap-1p5/shared-stage-W3-serving-11/QUALIFICATION.json)，
[质量检查](../../evidence/20260928_tp4-decode-gap-1p5/shared-stage-W3-serving-11/quality/QUALITY_REPORT.json)。
此前临时启动绕行的11.084135结果及预测关账历史保留在shared-stage-W3-serving-09。

已完成其decode trace离线重建：63完整token，每token179计算调用、99个peer点，四卡模板一致，因果违规0。
生产端完成错位中位4.150µs、P95 6.136µs；首个采集token的embedding点最大1185.601µs保留。
rank0原生stage包络10.792221 ms，原生TPC/MME活动并集7.534278 ms，内部peer依赖交接区间
合并去重后1.550628 ms/token。完整四卡计算外区间1.177626 ms；两种区间会相互重叠，不能相加。
这些可见生产/消费边界不等于NIC持续时间；逐点逐卡证据见
[NATIVE_POINT_SKEW.md](../../evidence/20260928_tp4-decode-gap-1p5/shared-stage-W3-serving-09/NATIVE_POINT_SKEW.md)。

以下保留P05微基准/预测与历史报告，不再次计入当前待验收累计。

## P05微基准与预测历史（已由最新组合覆盖关账）

主KV复用的TPC搬运由每行8次半宽BF16向量操作改为4次完整128元素向量操作，
保留FP32 V的两个64元素分段。SWA解码、mask、QK/softmax/PV精度和组内生命周期不变。
没有增加kernel、通信或中间/常驻张量；通过原有TP4路径自动选用新构建，不增加性能开关。

| 项目 | 父组件chain30 | 新组件chain42 | 实测节省 |
|---|---:|---:|---:|
| 同步host完整16层反馈链 | 5.747971 ms | **5.673781 ms** | **0.074190 ms，1.29%** |
| 设备事件区间，含暴露的提交空档 | 5.602245 ms | **5.533052 ms** | **0.069193 ms** |

复用已有父组件时间；新候选仅一个连续32步计时窗口，前置32步预热，包含最终设备排空。
四卡协同执行不是四次独立重复，没有跨运行置信区间。
四卡各65个反馈token及全部状态精确一致；16个原生消费用例/卡、5个C2/C6和槽位/边界用例/卡通过，
无热编译；57份共有计划通信顺序一致；43个计算片段、32次AllReduce、8次AllGather保持原合同。
K仍在SRAM，V为原有FP32 DRAM张量；172个其他TPC代码段逐字节一致。

端到端增量预估采用受影响kernel调用数：完整40层27次 / 组件16层9次 = 3，
得到**约0.222569 ms/token**。这是按相同形状和每次暴露成本外推的低置信度预测，
没有证明所有调用都在相同关键路径上，不能当作正式整模收益，也不套用历史P01–P04组合倍率。
37–41的失败、变慢或无法分辨的尝试不计收益；P01–P04已关账，不重复累计。

组件四卡峰值约37.85–38.18 GiB，仅作该组件资源记录；未测新整模峰值，不声称显存节省。
P05组件42阶段新增端到端请求和trace均为0。后续serving87已测试P05＋81版选择组合，正式15.599333 ms回退，不能归因或验收P05单项；当时P05保持待验收、14.622559 ms基线不变。2026-09-30的shared-stage-W3-serving-09已覆盖P05使用的主KV发布/复用核并正式通过，组合关账，不归因P05单项实际收益。

- [P05数值、状态和计时证据](../../evidence/20260928_tp4-decode-gap-1p5/continuation-chain-42/QUALIFICATION.json)
- [编译后数据流](../../evidence/20260928_tp4-decode-gap-1p5/main-reuse-vector-42/DATAFLOW_REVIEW.json)
- [调用次数与预测依据](../../evidence/20260928_tp4-decode-gap-1p5/main-reuse-vector-42/CALL_COUNT_MAPPING.json)
- [待验收服务配置](../../evidence/20260928_tp4-decode-gap-1p5/runtime-profile-main-reuse-vector-serving-v1.json)
- [本轮失败尝试与决策记录](../../evidence/20260928_tp4-decode-gap-1p5/DECODE_OPTIMIZATION_REVIEW_20260929.md)

## 2026-09-29 最新组合验收未通过

serving87一次正式16K→3891自然EOS，15.599333 ms/token；预热/正式/trace三次token一致，单工作负载语义通过，但与历史输出不同。选择四卡并集1.708151→0.544479 ms，整体kernel并集仅减少0.548473 ms，无可见设备活动增加0.959067 ms、仅DMA/调度增加0.170855 ms。未增加收益条目，未更新正式基线。用户随后要求修复收益转化，目标约13 ms；先定点完整链验证，再一次正式验收。

[本次完整报告](../../evidence/20260928_tp4-decode-gap-1p5/serving-fullaccel-87/REPORT.md)。

## 已验收改动

| ID | 改动与父组件 | 实测组件增量 | 验收前端到端预估增量 | 状态 |
|---|---|---:|---:|---|
| P01 | logical MLA及紧凑offsets；chain14→18 | −0.384701 ms | 约−0.46 ms | serving11组合验收 |
| P02 | 32K有界index Key镜像及contract21缓存/生命周期修复；chain18→20 | −0.241424 ms | 约−0.29 ms | serving11组合验收 |
| P03 | 四层组内主KV复用；chain20+contract21→24 | −0.040967 ms | 约−0.05 ms | serving11组合验收 |
| P04 | feature并行attention post/FFN collapse融合及组内metadata共用；chain24→30 | −0.162561 ms | 约−0.19 ms | serving11组合验收 |
| 整体 | chain14→30 | **−0.829652 ms，12.61%** | **−0.992860 ms，5.99%** | 正式端到端实际**−1.941169 ms，11.72%** |

各行端到端增量仍是验收前估计；未做逐项整模消融，实际1.941169 ms仅属于组合。
所有组件计时复用可比父版本，不重复测量已存在的正式基线。

## 统一验收与预测对照

| 项目 | ms/token |
|---|---:|
| serving10正式基线 | 16.563729 |
| 验收前组合预测 | 15.570868 |
| serving11正式结果 | **14.622559** |
| 预测节省 / 实际节省 | 0.992860 / **1.941169** |
| 预测误差：实际比预测更低 | **0.948309** |

原预测使用此前一次组件→整模绝对节省倍率1.196719。本次观测的组合倍率为2.339739，
不能自动视作未来优化的通用倍率。保留预测时点，后续按受影响链路与真实消费者估算。
单次正式请求不提供跨运行置信区间；小幅组件收益的稳定性限制仍保留在原始记录。

## 当前decode trace与显存

31个完整周期，四卡同一时钟，15.363824 ms/token；与无profiler14.622559 ms分别报告。

| 互斥分配项 | ms/token |
|---|---:|
| Attention/CSA2 | 4.598474 |
| MoE | 2.997506 |
| mHC及其他辅助 | 3.558122 |
| 跨功能组重叠，只计一次 | 0.814826 |
| 仅DMA/设备调度 | 1.126809 |
| 无可见设备活动空档 | 2.268088 |
| 合计 | **15.363824** |

空档分为输入/step衔接0.149736、compiled入口1.189101、消费提交0.929251 ms。
这些是位置，不能直接等同CPU计算、网络等待或可消除时间。
历史同口径trace的Attention独占5.829707→4.598474 ms；总空档2.127356→2.268088 ms未下降。
每周期11次主KV发布、27次复用；融合实际执行，无热分组编译。
NIC时长未暴露；部分kernel总调用次数未知，保留完整样本均值与活动并集。

正式请求3秒间隔采样的最高HBM占用较基线低291–793 MiB/卡，不是连续峰值。
已知常驻缓冲静态净差约619.2 MiB/卡，额外publication workspace和分配器生命周期仍需分别评估；
不能将静态差值当成整模峰值节省。

## 证据与可复现版本

- [正式结果及trace报告](../../evidence/20260928_tp4-decode-gap-1p5/serving-continuation-11/REPORT.md)
- [可搜索kernel明细](../../evidence/20260928_tp4-decode-gap-1p5/serving-continuation-11/kernel-decode/index.html)
- [验收前完整台账快照](../../evidence/20260928_tp4-decode-gap-1p5/DEVELOPMENT_LEDGER_PRE_SERVING11.md)
- [验收前预测快照](../../evidence/20260928_tp4-decode-gap-1p5/E2E_ESTIMATE_20260929_PRE_SERVING11.json)
- [当前机读台账](../../evidence/20260928_tp4-decode-gap-1p5/E2E_ESTIMATE_20260929.json)
- [组件30数值/生命周期门禁](../../evidence/20260928_tp4-decode-gap-1p5/continuation-chain-30/QUALIFICATION.json)
- [实际执行路径及热编译核查](../../evidence/20260928_tp4-decode-gap-1p5/serving-continuation-11/DECODE_PATH_AUDIT.json)

服务Python源码与engine指纹以serving11/process.json冻结快照为准；
native使用mhc-post-collapse-native-02/source/csrc，配置runtime-profile-mhc-post-collapse-serving-v2.json。
未通过候选不纳入合格native构建与正常启用路径。
当前结论限定于本次单请求16K文本工作负载；更大并发不是本轮端到端验收范围。

## 长期维护规则

1. 新增条目须有实际完整消费者链收益、正确性/生命周期和编译后数据流证据，写明父版本与指纹。
2. 失败、变慢、未验证、无法分辨的小差值只进实验INDEX，不进收益累计。
3. 连续兼容组合按组件起点减终点计算，不相加不同分母百分比，不重复计算重叠链或已验收收益。
4. 保留每轮端到端预测快照；微基准、预测、正式实测分开。修复合同不单独领取性能收益。
5. 累积兼容优化后统一验收：必要预热、一次正式自然EOS、输出语义及token检查；复用已测正式基线。
6. 通过后更新正式基线并清零已接受项的待验收累计；失败时保留原基线，先用已有数据和组件定位。
7. 任何后续整模或trace动作遵循用户最新范围；本轮不再继续prefill分析，不因分析元数据问题补测。
