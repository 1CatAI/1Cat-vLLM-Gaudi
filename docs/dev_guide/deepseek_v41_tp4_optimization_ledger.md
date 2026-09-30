# TP4 decode 长期开发台账

更新：2026-10-01。只记录具有可比完整消费链微基准收益、通过适用正确性检查的改动。
未验证方向、失败或变慢试验留在实验INDEX，不计累计。端到端通过后关账，不再次计入待验收收益。

## 当前正式测量与待验收累计

当前工作线为 rebase 后 TP4×PP1 C1 decode ≤10.0 ms。PR #52 保持草稿；用户已取消合并、prefill 验收与 TP2 冒烟。
正式父版本14afd2dc：完整56/56预热，一次无profiler的16K→2532token→自然EOS，
**11.105891782 ms/token**；固定样本14项事实/约束通过。与rebase前输出不逐位一致，语义检查与token身份分开记录。
历史同卡0/1/4/5、CPU10/15/38/43的真实16层父值为**4.875365906 ms/步**。
新测量采用常驻同进程ABABAB，每段≥200步；主统计是四卡最晚token回读间隔，
中位数差值必须超过基线IQR的2倍。跨进程旧值不再用于新候选判定。
本轮10ms目标的5个固定质量样本已在最新组合中重新验收，语义全部通过。

最新组合正式测量：**11.021114593 ms/token**，16K→2784token→自然EOS，完整56/56预热，
一次无profiler请求，5个固定样本语义全部通过。**未达到≤10.0 ms目标**。
相对11.105891782 ms父版本的单次观测差值为0.084777 ms，不能将预估1.307400 ms写成实际收益。
P06＋P07＋组合P08已进行端到端测试，移出“未端到端”累计；保留各微基准与预测快照，
最新未端到端累计为**0 ms**。本次不是新的10ms验收基线；下一步由同代码trace定位收益未兑现。

最新独立decode trace已完成四卡逐点和算子调用审计：63完整周期，生产完成错位中位3.414µs、
P95 5.777µs；单卡stage无计算区间2.866–2.954ms。rank0物理模板3064个计算节点/token，
76.6个/层，历史模板3384；还未达到≤40个/层。原生尾部结束到下一stage的空档0.060336ms。
不把该profiler周期或kernel数量下降替代正式TPOT。

本轮三项筛选均未进入收益累计：索引镜像取行融合7→5节点但消费链14.894→28.068µs；
专家取消分片17→7节点但原生消费链55.102→88.704µs；inverse-RoPE/wo_a融合9→8节点、
小链28.965→27.121µs，但两次16层未过门槛，第二次4.236337→4.220668ms，
差值0.015669ms低于2×IQR 0.114214ms。停止这三个候选，生产原型已恢复，不再重复计时。
首轮16层基线漂移和后续稳态原始数据均保留。工作卡0/1/4/5，计时前校正并记录全部worker
主/辅助线程亲和到CPU10/15/38/43；另一组卡为idle、768MiB，没有加载权重。

[完整trace报告](../../evidence/20260928_tp4-decode-gap-1p5/kernel-fusion-serving-02/REPORT.md)，
[每层/每卡调用与依赖](../../evidence/20260928_tp4-decode-gap-1p5/kernel-switch-audit-04/SUMMARY.json)，
[未达门槛的16层复测](../../evidence/20260928_tp4-decode-gap-1p5/resident-ab-control-10/1790795434586616290-0ec13302/result.json)。

单算子改善、失败或回退不入收益台账，见[实验INDEX](../../evidence/20260928_tp4-decode-gap-1p5/INDEX.md)。

- [当前正式基线](../../evidence/20260928_tp4-decode-gap-1p5/pr52-rebased-decode-01/SEMANTIC_REFERENCE_CHECK.json)
- [同卡16层基线](../../evidence/20260928_tp4-decode-gap-1p5/pr52-rebased-real16-05/continuation-rank0.json)

以下P06–P08的数据是验收前微基准与预测快照；最新正式结果覆盖它们的待验收状态，不再纳入未端到端累计。

### P06：输入投影与共享专家FP8（组合正式已测，10ms未达标）

已恢复正式14afd2dc父臂的BF16/v1精度与重放外FP32 head HCCL；两个臂使用同一native库。
候选仅替换wq_a/wkv/shared_w1/shared_w3/shared_w2，TP4切分保持32×32对齐，
沿用离线UE8M0感知FP8 sidecar，不增加运行时反量化常驻权重。
两臂各有独立完整stage计划，均为72计算调用、41原生通信点、1个重放外HCCL head点。

| 常驻ABABAB，A/B各600个回读间隔 | A | B |
|---|---:|---:|
| 中位数（ms/16层步） | 4.766596 | 4.600908 |
| IQR（ms） | 0.03914175 | 0.04155375 |

实测中位数差值**0.165688 ms >2×A IQR=0.0782835 ms**；A段漂移0.002206 ms。
首次候选准备与六段计时合计88.353秒，含其他组加载等待；每段至少200步。
四卡token一致、每个臂重复token序列稳定、无热编译。
FP8与BF16反馈token序列不同；独立单层数值与C1–C6契约已通过，正式5样本语义仍待测。
只有这一完整链条目计入累计，之前单算子估计、跨进程回退、以及中间父臂结果均不叠加。

尾部单独对正式父臂为4.777644→4.763707ms，差值0.013937低于门槛0.151900；
A的第一段5.193459随后回到4.767128/4.766735，先记录基线漂移，不归因候选变慢。
此历史16层结果不新增收益条目；按用户最新安排，尾部不再由16层工具判定，留到正式服务路径验收。

FP8＋尾部的稳态组合为4.766884→4.593715ms，差值0.173169高于门槛0.082394，
A段漂移0.000690，已准备计划的复测仅31.732秒。不能从与纯FP8的差值给尾部单独归因，
不重复计入P06累计。组合仍低于整模预估1.2ms（16层0.48ms）的正式门槛，未运行正式请求。

首次resident01/02父臂已包含未正式验收的FP32 peer head传输，保留为不同父臂的消融记录；
不得将这些结果当作相对正式11.106ms版本的组合收益。原生FP32 peer默认关闭，仅尾部候选选择。
resident03最后稳态复测前另做十秒负载检查，记录另一组卡重启/加载；新工具已强化为十秒稳定检查。

[正式父臂纯FP8结果](../../evidence/20260928_tp4-decode-gap-1p5/resident-ab-control-03/1790760366569493903-0806fbf7/result.json)，
[逐段逐卡时间与token](../../evidence/20260928_tp4-decode-gap-1p5/resident-ab-control-03/1790760366569493903-0806fbf7/periods.json)，
[组合、尾部、负载与判定汇总](../../evidence/20260928_tp4-decode-gap-1p5/RESIDENT_AB_FORMAL_PARENT_SUMMARY.json)。

### P07：SWA-only packed MLA（组合正式已测，10ms未达标）

父臂是P06输入/共享专家FP8，使用相同源、库、0/1/4/5卡位、10/15/38/43CPU和重放外HCCL head。
只对没有压缩KV的第0/1层复用现有paged packed MLA；避免每token用通用浮点/位操作展开整块256行SWA。
TP按head几何参数化，不新增TP4专用实现；不改prefill或C2–C6服务默认。
18项硬件检查包含KV pack/ring index_copy生产者、TP4H16/TP2H32、C1/C2/C6、位置0/255/16384，结果精确一致。
CPU1181通过、58设备条件跳过。参考和候选的attention模块旗标独立，缓存/不可变权重存储复用。

| ABABAB各600步，ms/16层 | FP8父臂A | FP8＋packed SWA B |
|---|---:|---:|
| 中位数 | 4.599719 | 4.489538 |
| IQR | 0.04488575 | 0.04046400 |

差值**0.110181 ms >2×A IQR=0.0897715 ms**，A段漂移0.013044 ms。
四卡和两臂的完整反馈token精确一致，无热编译；72计算调用、41原生通信点、1重放外head点保持相同。
第0层Attention冷recipe的实际TPC/MME节点**77→49（−28）**；未声称每层已到40。
另一组2/6/7/3卡在前置检查期间加载/预热，六段计时前的负载观察完整保留。

按用户×2.5换算，P07预估0.275453 ms；它只作用两层，因此另列按适用层数的保守整模预估0.110181 ms。
P06＋P07按连续父臂累计0.275869 ms/16层，机械预估0.689673 ms，保守预估0.524401 ms。
不叠加未过噪声门槛的I32试验；两者都删除SWA通用解码中的部分工作。
未达到正式触发门槛1.2 ms，未跑正式请求/5个语义样本；SWA候选默认关闭。

[实际A/B](../../evidence/20260928_tp4-decode-gap-1p5/resident-ab-control-07/swa-packed-01/result.json)，
[逐段/逐卡/负载](../../evidence/20260928_tp4-decode-gap-1p5/resident-ab-control-07/swa-packed-01/periods.json)，
[层数与外推汇总](../../evidence/20260928_tp4-decode-gap-1p5/SWA_PACKED_QUALIFICATION.json)。

### P08：Attention norm/quant＋BF16跨层handoff＋Engram update（组合正式已测，10ms未达标）

本组合替换原纯norm条目0.165042 ms，不能与它叠加。父臂仍为P06＋P07。
resident09同进程ABABAB，各臂600步，同卡0/1/4/5、CPU10/15/38/43：
A **4.4937795 / IQR0.084842 ms**，B **4.2466885 / IQR0.07248025 ms**；
差值 **0.247091 >2×A IQR0.169684 ms**，A段漂移0.0021015 ms。
72计算调用、41原生peer点、1个重放外head点不变；无热编译，四卡与各臂反馈稳定。
两臂反馈不逐位一致，按用户要求采用TP2原生数值参考，不据通用TP4反馈拒绝候选。
实际epsilon=1e-20的融合norm/quant、BF16跨层handoff均已通过C1/C2及适用C6、重排/零/小值的完整消费数值检查；
Engram更新及后续control/collapse在C1逐位一致，C2/C6误差保留在报告。CPU1186通过、58设备条件跳过。

首次同resident计时A首段5.113224、后两段约4.493 ms，差值未超过2IQR；该噪声记录完整保留。
上述稳态复测有明确基线漂移依据，未覆盖首次原始结果。计时期间另一组2/6/7/3卡空闲、每卡768MiB。
完整norm→FP8→MME小链物理节点9→3；Engram更新→control/collapse小链13→4。
这些局部计数不能当作整模每层已≤40个kernel；组合完整服务trace仍待验收后统计。

本项×2.5预估0.617728 ms；连续兼容P06＋P07＋P08累计 **0.522960 ms/16层，整模预估1.307400 ms**。
P07仅两层适用的保守外推合计1.142128 ms。预估达到用户1.2 ms正式触发线，开始准备一次正式请求和5个样本。
正式基线仍为11.105891782 ms，尚无新增正式收益。尾部重放只在此次服务中验收，不叠加16层估计。

[组合原始A/B](../../evidence/20260928_tp4-decode-gap-1p5/resident-ab-control-09/1790783707003021473-1609d95a/result.json)，
[组合数值、噪声与累计判定](../../evidence/20260928_tp4-decode-gap-1p5/NORM_HANDOFF_QUALIFICATION.json)，
[原纯norm参考](../../evidence/20260928_tp4-decode-gap-1p5/ATTENTION_NORM_QUANT_QUALIFICATION.json)。

### 2026-10-01组合正式结果（预测未兑现，保留记录）

P06/P07/P08兼容真实16层预估1.307400 ms，尾部不计入组件累计。
默认服务一次正式实测 **11.021114593 ms/token（90.734924 tokens/s）**；自然EOS2784token，
5个固定样本全部语义通过。实际单次观测差值仅0.084777 ms；预估比实测多省1.222623 ms。
不能给三个组件分别归因整模收益。计时前另一组四卡空闲、每卡768MiB，卡位/CPU保持一致。
正式服务保留上下文1048576、prefill8192、32槽；shape-agnostic缓存启用，concat完整预热通过。
同版64步decode采集四卡prepare120→120、capture11→11，无热编译；硬件时间线正在解析。
已测组合不再属于“未端到端”台账，微基准数字作为历史保存，后续必须以trace解决剩余差距。

[正式测量](../../evidence/20260928_tp4-decode-gap-1p5/kernel-fusion-serving-01/speed-01/result.json)，
[五样本检查](../../evidence/20260928_tp4-decode-gap-1p5/kernel-fusion-serving-01/quality/QUALITY_REPORT.json)，
[验收状态](../../evidence/20260928_tp4-decode-gap-1p5/kernel-fusion-serving-01/QUALIFICATION.json)。

### Rebase前已验收历史（已关账）

2026-09-30：共享C1计算、N卡点对点交换与完整stage原生重放已正式满足≤12.3 ms。
修复concat轴的shape-agnostic缓存键后，恢复全部预热形状，并完成一次无profiler正式16K→EOS复测：
**11.083109 ms/token，90.227389 tokens/s，2579输出token**。
输出token与此前11.084135 ms那次精确一致；两次均通过固定资料事实检查。
该差值不算新增优化收益，只确认完整预热保持已验收速度。

| 项目 | 当前值 |
|---|---:|
| Rebase前历史正式测量 | shared-stage-W3-serving-11：**11.083109 ms/token** |
| 历史正式基线serving11 | 14.622559 ms/token；16K→2989自然EOS |
| 该历史版本正式请求 | **1次**；16K→2579自然EOS；无profiler |
| 完整启动预热 | 两种起始位置×8192/4096/2048/1024/512/256/128×四卡，**56/56通过** |
| 新增质量检查 | **4个不同固定16K样本，4/4自然EOS与语义通过** |
| 服务容量 | 上下文1,048,576；prefill8192；32请求槽 |
| 共享重放16层门槛结果 | chain42：5.673781→shared-stage-W3-real16-06：**5.260995 ms** |
| 已接受组合覆盖的待验收项 | P05与共享C1 W1/W2/W3关账，不再累计 |
| 当前待验收微基准收益 / 端到端预估 | **0 / 0 ms** |
| TP2×PP2同代码回归 | 用户明确取消；未运行，不计验收结果 |
| 草稿PR | [#52](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/52)保持草稿；已rebase到含PR51的main并保留14afd2dc，集成验收与合并由用户另行安排 |

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
