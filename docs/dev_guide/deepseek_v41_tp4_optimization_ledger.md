# TP4 decode 长期开发台账

更新：2026-10-05。只记录具有可比完整消费链微基准收益、通过适用正确性检查的改动。
未验证方向、失败或变慢试验留在实验INDEX，不计累计。端到端通过后关账，不再次计入待验收收益。

## 当前正式测量与待验收累计（2026-10-05）

官方采样 T=1.0、top_p=0.95、seed=42，512K、C16384、缓存开、完整预热：
`decode-physical-fusion-serving-03` 独立安装服务，16K未命中→1858 tokens自然EOS，
**9.564282 ms/token**；14项事实/约束通过，采样回退0。输入token与父版本相同，
输出从第32个token起不同，未声称逐token相同。前一正式基线为9.997343 ms/token。

本批组件预估1.123126ms，正式减少 **0.433061ms**。用户于2026-10-05明确接受该正式结果，
撤销“低于预估一半暂不提升默认”的限制：**9.564282 ms/token 为新的正式基线**。
TP4 正常启动默认开启 CSA publish、W2 三路、Q/KV prologue、候选坐标、peer/post collapse；
STATIC_COORDINATES 原已默认开启，共六项。显式诊断覆盖仍可关闭，prefill/C2–C6保留既有条件。

原始 trace 已足够指导后续：计算约6.97ms、非计算约2.43ms、<2us碎空隙约1.30ms。
recipe编号冲突停止追查，不再作逐项归因。输出/post-norm/WOa/发布链小融合归档，
默认关闭；共享-main输出原型提交 `0dd878fb`。旧项目中的组件预测不能继续叠加到新基线。

**当前本轮微基准待验收累计：0.458715133 ms/token（估计，尚未端到端）。**
MoE 合并候选0.206373750ms、mHC0.039834531ms、Attention主体向量化0.176384352ms，
量化整行amax去重0.036122500ms。
各项5组输入四卡通过适用正确性检查、原生重放3轮A/B方向一致，默认均关闭。
mHC 新项按官方方程/DeepGEMM归一化误差容差验收，不声称逐位一致；其余保留原逐位检查。详见文末。
新MoE替代旧的两片候选，mHC替代旧0.031500469ms，均不重复相加。
正式验收触发线仍为累计≥1ms。按逻辑阶段计数，GEMM内部TPC/MME流水切片单列，
保留默认切片策略。两指令FP4字典解码停止：ISA访存槽位预算高于算术发射预算，收益记0。


## 2026-10-03及以前的测量记录

当前官方采样口径：temperature=1.0、top_p=0.95、seed=42，512K容量、C16384、缓存开启、完整预热。
最新安装服务测量为 `long-prefix-serving-01`：源码31e519fd，16384→2128自然EOS，
prefill8028.980763tokens/s、decode10.568225ms/token、客户端TTFT2.126094s，14项事实/约束通过，
全部输出token与visible-prefix-serving-05一致。9.5ktokens/s和10ms目标尚未通过。
同配置归档父版本b7553683为prefill8851.110164tokens/s、decode10.565244ms/token；
本次prefill降低9.287%，decode差0.002981ms，原因未归因，不记作正式收益。
用户此前接受发布的回退版本为8914.621782tokens/s、10.599712ms/token。
长文组件已组合安装到公网服务；用户于2026-10-03明确要求通过PR57合并当前实现到main。
此次源码合并不表示速度或长文语义验收通过。长文短诊断实测见文末，
组件数据不折算为16K官方采样收益；当前未端到端累计16K收益仍0ms。
下述9.933990731ms属于历史greedy验收，不能作为当前官方采样的发布验收数字。

2026-10-01本轮≤10ms目标已完成：最新正式基线 **9.933990731 ms/token（100.6645 tokens/s）**，
完整56/56预热，16K→2784token→自然EOS，无profiler正式请求；5个固定语义样本全部通过。
可辨2780个ITL间隔：中位9.911872、P90 10.314221、P99 12.199491ms；>15ms0.071942%，
正向尾部超额0.176862ms/token。初始合并隐藏3个间隔，不人为拆分。计时前PSI<1%，计时内峰2.06%如实记录。
共用PagedStageState保留页表追加的旧映射与索引镜像，允许同owner在回读前启动V2prefix；
真正remap/owner/geometry切换保留保护。补齐桌面、状态采集器和WebKit的worker辅助物理核隔离。
正式观测包含已合格计算路径及OS隔离，不将全部差值归给单页发布。待验收微基准累计仍 **0 ms**。
第5样本首次因磁盘写满没有有效结果；保持原失败并无损归档旧图缓存后，以同份冻结源码重启，
重放前三样本并补测第5样本全部通过。截断的process.json不冒充完整原始记录；源码、runtime、
engine身份、startup库指纹和正式输出保留，见恢复说明。PR52保持草稿，不合并。

[本轮最终报告](../../evidence/20260928_tp4-decode-gap-1p5/kernel-fusion-serving-06/REPORT.md)，
[验收机器记录](../../evidence/20260928_tp4-decode-gap-1p5/kernel-fusion-serving-06/QUALIFICATION.json)，
[5样本记录](../../evidence/20260928_tp4-decode-gap-1p5/kernel-fusion-serving-06/quality/QUALITY_REPORT.json)。

以下11.021及后续未达标记录保留历史语义，不作为当前正式基线。

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

2026-10-01 host复测：模型/原生库不变，第一次正式10.936414ms、2784自然EOS与原token一致，
decode内PSI最高2.67%，尾部超额0.835612ms，未成为安静基线。修正EngineCore/辅助核竞争与
桌面SMT隔离后，正式10.030447ms、同样2784EOS精确一致，仍未≤10ms。开头一次4-token
合并使完整ITL不可辨；2778个单token相邻间隔的条件分布为中位9.913294、P90 10.739519、
P99 14.745713ms，>15ms占0.971922%，正向超额0.314448ms，未过0.2ms门槛，不新设基线。
GC冻结候选正式10.734059ms，2784自然EOS且token一致；2777个可辨单token间隔的正向超额0.721990ms。
相对隔离后10.030447ms变慢，已撤回freeze调用，CPU934通过/112跳过，不计收益。
分页边界新证据：隔离后27个>15ms尖峰中21个位于每128token的新页位置；定向trace在P16640
四卡均跳过提前prefix，target提交耗时5.000–5.138ms，普通邻步约1.1–1.3ms。
该周期下一target间隔15.438–15.769ms；短请求perf的258/386号尖峰无>0.2ms关键线程排队。
首次BatchStageState候选未命中生产：BATCH_DECODE未开启，生产为PagedStageState；
正式10.552049ms、2784自然EOS且token精确一致，2779个可辨间隔正向超额0.629729ms，21个分页尖峰仍在。
该无效原型已撤回并归档，不计收益。
正确候选改为PagedStageState：同owner追加仅发布未读页表项，保留已有索引镜像，remap仍同步及失效。
CPU945通过/112跳过。HPU单页发布ABABAB，每段200步，旧76µs、新66–68µs，数值映射通过；
没有包含真实mirror rebuild或模型stage，不计台账收益。生产服务验收待测。
gnome-shell在上一正式请求中峰值20.6核；候选服务将桌面移离worker主/辅助物理核和控制核，
亲和变化单独保存；后续不能将跨进程TPOT差值全部归因于分页算法。

2026-10-01用户修正：后续16层有效差值按**×1.5**外推，整模1.2ms触发线对应16层0.8ms。
历史×2.5预测保留原值。11.021版本先进行一次安静机器复测，检验ITL正向尾部超额
`mean(max(ITL−median(ITL),0))`能否降至0.2ms以内；未完成前不改变正式基线。
安静条件采用CPU PSI some avg10<1%、主核无其他用户任务竞争、其他卡未加载权重；不使用load1门槛。

最新独立decode trace已完成四卡逐点和算子调用审计：63完整周期，生产完成错位中位3.414µs、
P95 5.777µs；单卡stage无计算区间2.866–2.954ms。rank0物理模板3064个计算节点/token，
76.6个/层，历史模板3384；还未达到≤40个/层。原生尾部结束到下一stage的空档0.060336ms。
不把该profiler周期或kernel数量下降替代正式TPOT。

mHC线性加载也停止：稳态16层4.239385→4.192107ms，差值0.047278低于2×IQR0.091217ms；
两次未过门槛，原型已归档并恢复生产源码，不记收益。

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


### Independent default installation acceptance (2026-10-01)

The ordinary installed C1 path passed one unprofiled 16K→natural-EOS formal
request: **9.992734 ms/token (100.0727 tokens/s)**, 2,784 output tokens. All five
fixed semantic samples exactly match the accepted reference tokens and final
text. Complete warmup, 1M context and 32 request slots are retained. This
acceptance validates relocation/default dispatch; it is not a new component gain.
Pending microbenchmark gain remains **0 ms**.

The runtime installation materializes Python dependencies, engine/plugin code,
selected native libraries and relocated ABI records outside the editable
workspace. It discovers dense precision from the installed sidecar. A machine
allocation template and supervisor replace manual worker/control thread binding;
model assets and the system SDK remain prerequisites. Native manifests match
all maintained source fingerprints, and workers map no old-workspace libraries.
CPU regression: 969 passed /114 skipped; three installation/API checks passed.

The preflight initially rejected the SDK's automatic `profile_api_light` mode
and sent no formal request. The corrected audit confirms no configured or active
profiler. Observable ITL median9.915, P90 10.457, P99 12.271ms; >15ms0.252%,
positive-tail excess0.235ms. One formal request is retained; no repeat cohort.
CPU/card conditions and the continuous-foreign-card telemetry limitation are
recorded in `evidence/20260928_tp4-decode-gap-1p5/portable-serving-acceptance-02`.
The separate public streaming adapter restricts exposure to inference routes.

### Ordinary sampling qualification (2026-10-01)

This is a functional correction, not a gain-ledger item. Ordinary decode
incorrectly shared DSpark's greedy-only validator and argmax tail. It now
supports temperature, top-p, top-k and seed while retaining shared stage replay
and reusing local head logits. DSpark still requires greedy verification.

Independent installed source, full warmup and unprofiled 16K→natural EOS:
greedy 9.975 ms/token, 2,784 output tokens, exact match to the accepted reference;
model-recommended temperature1/top-p0.95/seed42: 10.426 ms/token, 2,130 output
tokens, 95.91 tokens/s. Five fixed official-sampling quality samples passed
semantic checks; categorical frequencies, TP seed ownership and replayed
changing controls passed CPU/HPU tests. CPU suite: 976 passed/114 skipped.
The requests share their input but generate different output histories; their
0.451 ms average difference is not a token-matched causal ablation. Existing
greedy results do not qualify stochastic sampling, which remains above the
10 ms target. No speculative decoding or performance profiler was active.
Evidence: `evidence/20260928_tp4-decode-gap-1p5/sampling-validation.json`,
`sampling-greedy-acceptance-01`, `sampling-official-acceptance-01`,
`sampling-quality-report.json`; host/card snapshots accompany both formal runs.
No pending component gains are added by this change.

### First-token cache retention correction (2026-10-01)

This is a serving reliability correction, not a decode gain-ledger item.
On parent e58f21a0, a 16K request followed by the same warmed 273-token
prompt reproduced first-token latency of **16.812 s**, while its immediate
repeat took **0.594 s**. A diagnostic stack sample attributed 18.20 of
19.16 request seconds to Dynamo compilation, predominantly the per-layer
SWA workspace. The profiled request is diagnostic only, not a latency result.
The global 64-contract pure-function cache mixed independent functions,
lengths and forty SWA layer-offset contracts, evicting warmed short shapes.

Retain executors by function and primary token geometry, with bounded
geometry and per-geometry contract counts. All tensor/scalar/precision guards,
model-generation invalidation and native expert workspace limits remain.
The independently installed candidate 5ff4ac31 completed full warmup:
after a 16K request, the same short prompt took **0.597 s**, then **0.585 s**
on repeat; first short request after startup took **1.892 s**. A warmed public
temperature1/top-p0.95 streaming diagnostic returned its first text in
**0.679 s**. These short probes generate one token and are not natural-EOS
decode qualification. KV prefix caching remains disabled.

One unprofiled full-warmup 16K→natural-EOS formal regression took
**9.973275 ms/token**, **100.267962 tokens/s**, with all **2,784 tokens**
exactly matching the accepted greedy reference. The code changes no sampling
math; official-sampling decode performance retains its prior qualification,
and no new stochastic performance claim is made. CPU regression:
**979 passed /114 skipped**. Pending component gains remain **0 ms**.
Evidence: `evidence/20260928_tp4-decode-gap-1p5/prefill-cache-conditioning.json`,
`prefill-cold-stack-summary.json`, `prefill-compile-caller-summary.json`,
`ttft-cache-serving-results.json`, `ttft-cache-public-probe.json`,
`ttft-cache-formal-01`, and its host/environment records. The other four
cards held a small idle allocation at the formal preflight; continuous
foreign-card load telemetry was not collected.

### Ordinary C1 prefix checkpoint activation (2026-10-01)

Functional state/TTFT correction; no component gain and no new pending savings.
The independent public installation enables prefix caching, retains full warmup,
1M context and 32 request slots, and includes the pinned engine's bounded
all-rank auxiliary-checkpoint ABI. Request-slot ownership no longer requires
the optional batch executor. Native replay/communication interfaces are unchanged.

Failed candidates are retained in `prefix-cache-01` through `prefix-cache-03`.
An intervening long request exposed stale derived index mirrors at a scalar
2049-token prefill tail; owner/page-remap invalidation fixes cached/fresh token
equivalence. New page-boundary region contracts use the existing eager bodies
after startup instead of evicting warmed guarded executors. Scalar C1 tails
publish the canonical slot before replay. The legacy 1024-position replay
bound still sent long scalar tails through synchronous group compilation: a
cached 2048→2064 continuation took 39.717 s before its first token. CPU tests
reproduced false replay admission at positions2048 and16384. Removing that
bound for slot-owned C1 tails reduces the same continuation to **0.904 s**,
with all32 output tokens exact against its forced-miss counterpart.

Fresh candidate65f0c231, independently installed and fully warmed:

| State/TTFT probe | First uncached request | Cached after slot reuse | Reused tokens |
| --- | ---: | ---: | ---: |
| 273-token greedy prompt | 4.216 s | 0.385 s | 256 |
| 2048-token recommended-sampling prompt | 5.064 s | 0.446 s | 1920 |
| 2049-token greedy prompt, intervening4K/decode request | 1.952 s | 0.343 s | 2048 |

All cached/fresh state and appended-suffix comparisons pass token equality.
The public recommended-sampling2K cached probe takes **0.609 s**, with1920
reported cached tokens. A16K cached64-token state probe reuses16256 tokens
and matches its uncached formal output prefix. These length-limited probes
qualify state/TTFT only; they do not qualify natural-EOS decode performance.

One unprofiled full-warmup16K→natural-EOS request: **10.069162 ms/token**,
**99.313134 tokens/s**,2904 output tokens; uncached TTFT11.633 s. All2904
tokens exactly match prefix-cache-03. They differ from the prefix-disabled
2784-token reference at token37; factual semantics pass, and no reference-exact
claim is made for enabling checkpoint splits. Four additional frozen tasks
at temperature1/top-p0.95/seed42 reach natural EOS and pass every expected
semantic field: five tasks total including the greedy formal factual task.
No new stochastic decode performance claim; the≤10ms target remains unmet
for this prefix-enabled formal result. CPU967 passed/111 skipped;257 plugin
Python files and18 locked engine files match maintained candidate source.

Formal preflight CPU PSI some avg10=0.95%; per-second CPU/core/context-switch
and desktop-process telemetry is retained. The other four modules2/3/6/7
hold768MiB each at0% utilization at preflight and throughout periodic startup
checks; no foreign weight load was observed. Continuous foreign-card telemetry
was not collected during the formal request. Candidate and parent use modules
0/1/4/5 and CPUs10/15/38/43.

Cache hits require the same token prefix plus retained pages and all-rank
SWA/compressor/Engram checkpoint acknowledgment. Current checkpoints stop at
the prompt boundary; generated tokens do not extend them automatically for
the next chat turn. Cache capacity/eviction and restart can cause misses;
uncached prompt processing remains necessary. Evidence: existing
`evidence/20260928_tp4-decode-gap-1p5/prefix-cache-04`, including
`transitions.json`, `formal`, `formal-host.jsonl`, `environment.json`,
`reference-comparison.json`, `parent-reference-comparison.json` and
`quality-results.json`. Pending microbenchmark gains remain **0 ms**.


### Complete-prefill and prefix integration — serving qualification pending

PR51 qualified complete prompt execution at **9857.756805 tokens/s** from
16384 input tokens. The installed prefix-enabled release processed smaller
scheduler tiles and split the checkpoint boundary; PR52 also reset the existing
`w13_single_bucket` grouped W13 FP8 prefill default to BF16. The maintained
candidate restores the scheduler budget and W13 mode, captures interior rings
from live full/decoder-halo producers, retile-checkpoints shorter transactions,
and stages bounded request rings through fixed warmed working addresses.
After checkpoint publication the continuing prefill owner is rebound, so later
prompt writes publish independently of the immutable saved boundary.

Candidate816ff598, independently installed with complete warmup, modules
0/1/4/5 and CPUs10/15/38/43: one uncached16384→natural-EOS formal request
returns3024 tokens at **10.137485 ms/token**; prefill **8290.594339 tokens/s**,
engine prefill1.976215 s and client TTFT2.100969 s. Decode remains near the
prefix-enabled formal parent10.069162 ms, but complete-prefill throughput does
**not** restore PR51. Four additional recommended-sampling tasks pass every
frozen semantic field, making five samples including the formal factual task.
Their prefill rates are3535,8364,8246 and8404 tokens/s; these are different
prompts, not repeat formal benchmark measurements. Their sampling decode rates
are separately recorded and do not qualify the greedy target.

Cache probes: full16K repeat after another request reuses16256 tokens with
client TTFT0.702 s; appended16K continuation reuses16256 with TTFT0.906 s
versus4.650 s for its forced miss. Recommended-sampling2K repeats reuse1920
with TTFT0.371 s locally and0.493 s publicly. Short cached/fresh sampled tokens
match. Long cached/fresh token sequences differ; restored grouped FP8 and
changed prompt geometry do not promise token equality. A cached long chat and
valid appended-user chat both reach natural EOS and pass all factual checks.
The original raw-completion semantic probe appended arbitrary archive tokens
after the assistant opening, continued the archive to the length limit, and
is retained as an invalid chat-semantic test; the valid chat probe replaces it.

CPU994 passed/111 skipped. HPU checkpoint→tail publication→native unpack/MME
handoff passes changing inputs and slot reorder/reuse, and real W13 FP8 through
routed W2 consumes outputs with finite values (BF16 relative L2 **0.035350**;
this component tolerance is not the model quality gate). C1 before/after that
prefill component is exact. No new component gain is admitted; pending gains
remain **0 ms**. PR56 remains draft pending prefill performance qualification.

Interrupted/failed attempts are retained:01 near-full unowned residual OOM;
03 stopped before readiness after finding tail ownership;04 stopped during
loading after finding the precision-default reset. In05 every rank completed
prefill but CPU/Gloo acknowledgment attempted an obsolete non-local public IPv6
and timed out. The single-host supervisor now selects loopback by default;
the unchanged all-rank protocol passes36 transactions per rank in a four-rank
HPU-producer→acknowledgment component.06 completes real serving qualification
for state and quality.07 captures diagnostic prefill spans to locate the
remaining throughput difference; diagnostic timers do not advance the baseline.

Evidence: `evidence/20260928_tp4-decode-gap-1p5/unified-prefill-decode-01`
through`unified-prefill-decode-07`;06 includes formal, cache probes, corrected
chat continuation, fixed quality outputs, host telemetry and source manifests.
Other modules2/3/6/7 remain768MiB and0% utilization at sampled preflights;
CPU PSI and bound-core/context-switch data accompany the formal result.
### 2026-10-03 prefix transition component (formal pending)

Parent33ef72c8. Preserve inline checkpoint tensors and the live working-ring
owner across capture; promote the same owner to B1 without export/import.
Compile the existing SWA reader once instead of eagerly expanding its
bit/scale operations on every restored layer. Warm the reader and owner
transition before admitting requests. No new state format or native ABI.

`prefix-owner-codec-02`: module0,CPU10–14; ABABAB200. The measured
producer/checkpoint/promotion/SWA-score chain covers40layers,46rings and
three changing packed inputs. Baseline wall median54.309955ms; candidate
8.748552ms; difference45.561403ms exceeds2×baselineIQR8.100912ms.
Device-event timing agrees. Checkpoint boundary histories, latest live
histories, BF16 score outputs and B2 publication agree. Compressed-cache
mirror reconstruction is unchanged and excluded from this fixture.

This is a per-request transition gain, not per-token decode gain. It does
not enter the decode cumulative total or establish the9.5k prefill gate.
Formal serving remains pending. CPU1005passed112skipped. The earlier
owner-only fixture was noisy:60.285510→54.907627ms, below its7.209410ms
threshold; it is not a separately counted gain.


### 2026-10-03 release rollback, formal release gate not passed

User withdrew the native bounded-sampling integration and its diagnostic variants. Restore d85ebce5 generic official sampler; retain efbbd824/prefill/cache repairs and warm official sampling graphs before readiness. The withdrawn sampling component is outside all pending gain totals; its failed serving and diagnostic evidence remains in the local experiment index. The current release gate is one official-seed uncached16K natural-EOS request at512K/C16384/prefix enabled, with approximately10.7ms decode and at least9000tps prefill. The10ms device-RNG/once-per-request-controls sampler will use a separate branch and PR after release.

`release-rollback-01`, source `5a787dbe`: complete warmup, official sampling
temperature=1.0/top_p=0.95/seed=42, uncached16384 input, 2128 natural-EOS tokens.
Measured prefill **8914.621782 tokens/s**, decode **10.599712 ms/token**;
14/14 fixed facts/constraints passed. CPU977 passed/112 skipped. Decode recovered,
but prefill missed the >=9000 release gate by0.95%. PR56 remains draft and main
has not been changed. This failed combined release acceptance does not advance
the accepted baseline or add any pending component savings. See the local
`release-rollback-01/OUTCOME.md` and raw `formal/result.json` for the result.

Release decision update: the user explicitly accepted the measured rollback
version for publication and authorized merging PR56. Publish the measured
8914.621782 tokens/s and10.599712 ms/token without claiming the original
>=9000 prefill gate passed or the later10ms sampling target was achieved.
No additional formal request is required for this unchanged runtime.


### 2026-10-03 long-context index path — real-stage qualified, serving pending

Parent main ea017d60; maintained source b51026b5 (measurement checks9056543d).
The published C1 path falls back from decoded-key MME to serial packed TPC
dot products past32K. Reuse the existing paged-key/MME producer and ordered
selection/reindex consumer in the shared runtime; no sampling or native ABI change.
Capacity clipping alone was rejected and is not credited.

`visible-index-real16-05`: actual41984 prompt,512Kcapacity, modules2/3/6/7,
CPU64/69/92/97; separate captures and ABABAB200. Four-rank first device
positions are41984. Latest-token-delivery medians10.0259465→5.0995845ms,
measured saving4.926362ms;2×baselineIQR0.1233685ms. Baseline drift0.002920ms.
All200feedbacktokens agree across six periods and four ranks; no hot compile.
Full/Reindex/packed-row consumer tests at32K/42K/62K/64K boundaries are exact.
The owned other-group loader was temporarily frozen and automatically thawed;
interrupted and wrong-position fixtures01–04 remain outside gain totals.

This is a42Kstage comparison. It is excluded from the16Kbaseline cumulative
savings; the1.5×heuristic from the16Kfusion campaign is not used to predict
this different workload. End-to-end long-request saving is not yet known.
One fixed official16K EOS request and short arbitrary-length/cache diagnostics
are pending on the ordinary installed service. No API improvement is claimed yet.


### 2026-10-03 shared 64K prefill index bound — component qualified

Parent e9b8b541. Extend existing shared Full/Reindex query partition and native candidate-gather validation from 32768 to 65536 source rows; TPC arithmetic and runtime-sized loading are unchanged. No new communication or native ABI.

`prefill64k-index-chain-06`: modules0/1/4/5, CPU10/15/38/43 with four helpers each, production512Kpage capacity, 1024queries and62464visible rows. Full→Reindex→actual FlashInferMLA consumer, three changing head-shard inputs, ABABAB200. Per-step slowest-rank wall median1356.454539→110.669224ms; device1356.218496→110.542200ms. Difference1245.785315ms exceeds2×baselineIQR16.777981ms. All48outputs exact. First/last A medians1354.849702/1354.690389ms. Other-group startup/CPUpressure overlapped later periods and is archived; absolute times require serving qualification.

This is a long-prefill chain saving. Estimated direction: remove repeated serial Reindex work beyond32K; whole-request magnitude remains unknown because tile/layer scheduling changes. No16Kdecode gain or cumulative ms/token credit. End-to-end and MLA exchange admission are pending; keep this item outside decode totals.


### 2026-10-03 existing MLA query partition at long search — component qualified

Parent97bd46b1. Preserve existing TP query-owner exchange, FlashInferMLA and reverse exchange, admitting4K/16Kquery tiles at32K/64Ksearch. Native ABI and communication interfaces unchanged; shapes beyond64K retain ordinary fallback.

`prefill64k-mla-chain-01/02`, normal0/1/4/5 and10/15/38/43, ABABAB200 with three changing inputs. Four-rank maximum wall:4Kqueries7.564369→6.009738ms (saving1.554632ms,2×IQR1.144491ms);16Kqueries23.228257→16.706033ms (saving6.522224ms,2×IQR1.227978ms). Device timing agrees,24comparisons exact. Other-group warmup is recorded. Query/head/cache tensor contracts match actual MLA production consumer.

This is a long-prefill component improvement, separate from the preceding Full/Reindex chain whose MLA consumer was unchanged. No16Kdecode savings credited; request-level estimate is not quantified before combined serving. Official16KEOS and arbitrary-length/cache diagnostics are the next gates.


### 2026-10-03 combined arbitrary-prefix service result — partial acceptance

`visible-prefix-serving-05`, installedb7553683,512K/C16384/prefixON, default paths, officialtemperature1/top_p.95/seed42. Complete warmup, one uncached16384→2128naturalEOS request:8851.110164tps,10.565244ms/token,TTFT1.939030s;14facts/constraints pass and token IDs exactly match prior04. The9.5ktps/10ms goals are NOT achieved; no claim of completing the global target.

Changed-boundary80token diagnostics:41983prefill9.104279s (4611.348tps),decode12.539508ms;62463prefill11.803032s (5292.115tps),decode12.738289ms. Prior04 same-input engine times52.733681/75.946818s were substantially higher; competing-load conditions differed, so retain both records without a pure isolated end-to-end attribution. These close the64Kprefill component entries as service-observed improvements, outside16Kdecode cumulative totals. Larger-than64Kshared prefill still falls back and is unqualified for speed.

Short1024cold official diagnostic:prefill1.728926s,decode10.527486ms,TTFT1.765307s. Cachehit896rows:prefill1.125010s,decode10.257645ms,TTFT1.148419s. Initial short decode regression is removed after startup handoff-reader warmup. Cachetrajectories are coherent but token-exactness remains unproven;80token diagnostics are not naturalEOS quality acceptance. Public adapter restored to independently installed branch source;main unchanged andPR57draft. Remaining~2mslongdecode gap and short-request fixed prefill latency stay open. Static mirror/direct-score32Kbound is recorded outside gain totals pending measurement.


### 2026-10-03 bounded source windows beyond the former prefill cutoff — component qualified

Parent8e92e25d, source patch archived in `prefill-windowed-index-chain-01`. Full query partition no longer stops at65536source rows; Reindex uses transaction-local65536-row key windows and bounded1024-query candidate workspaces. Restore scores into original candidate slots before sequential TopK merges; duplicate IDs, cutoff ties and causal masks remain unchanged.

Modules2/3/6/7, CPU20/25/48/53 plus disjoint helpers, production512Kpage capacity,1024queries,131072source rows,82944visible rows. ABABAB200, three changing inputs, actual Full→Reindex→MLA consumer. Slowest-rank wall median1490.352746→229.049167ms, saving1261.303579ms;2×baselineIQR42.908518ms. All48outputs exact. Early periods overlap existing public prefill on0/1/4/5; HBM oscillates rather than monotonically loading, PSI and protected affinities archived.

Long-prefill component only, no16Kdecode cumulative credit or whole-request estimate. Paged selected-KV query-owner MLA and configured-capacity decode mirrors are still separate pending component gates. Public inference remainsb7553683 until combined normal-service qualification. Final CPU suite1044passed111skipped; native ABI unchanged.


### 2026-10-03 query-owner MLA with bounded paged KV loading — component qualified

`prefill-paged-mla-chain-01`, same2/3/6/7 and20/25/48/53,4096queries,131072source rows,512Kpage capacity,ratio1. The previous flat-cache budget switched to replicated selected-row gather/generic FP4 decode before MLA. Feed logical selected IDs into the existing sequence exchange instead, decode only the owner’s selected rows inside each128-query MLA recipe, and preserve complete640-column softmax/PV. No new communication or native ABI.

ABABAB200, three changing head shards, actual paged load/decode/MLA consumer in both arms. Maximum-rank wall1129.703831→15.171890ms, saving1114.531941ms exceeds2×IQR5.465538ms;12consumer outputs exact. Source and competing load recorded. Largest query stride/ratio2 also passed in `prefill-paged-mla-contract-02`: C16384, ratio2, 12 exact consumer outputs, correctness-only; ordinary combined serving remains pending. This is a long-prefill gain, independent of the preceding index chain’s unchanged MLA consumer; do not sum either into16Kdecode or an estimated request-level number.


### 2026-10-03 configured-capacity decode index mirror — component qualified, real16 numerical diagnosis pending

Native32Kcap is a per-score workspace bound, not a context cutoff. Keep canonical packed pages authoritative, derive index keys for the configured capacity, restore reachable source windows at owner/prefill transitions, maintain only newly written rows during decode, and stream Full scores through the existing native MME epilogue. Prefix pruning and its finite warmup geometries now cover long searches. Extra resident index-key storage for512K is approximately300MiB/card versus the old20MiB; this is static accounting, not serving peak-memory acceptance.

`long-index-mirror-chain-06`: actual native model-entry replay of peer query/weight exchange, canonical/main/mirror writes,Full/Reindex and publish/reuse MLA.2/3/6/7,20/25/48/53,source82944/search131072/capacity512K,ABABAB200,three immutable changing banks. Ratio1 wall2.415673→1.233708ms,delta1.181965,gate0.178182; ratio2 wall1.664854→1.084340ms,delta0.580514,gate0.279144. All96outputs exact.01/03 host-entry measurements were not representative and ratio2 stayed below their IQR gates;02/04/05 failed cold harness capability/binding contracts and collected no timings. They remain outside all credit.

Real16-01 failed the saved-feedback oracle before qualification. Current source changes prefill query ownership as well as decoder state; archived prefill tensors were not retained, so the old token sequence cannot isolate mirror correctness. No gain credit. Real16-02 compares canonical paged MME and mirror readers from the same current prefill snapshot, using the resident tool and complete native replay; this missing stateful reference is the only new baseline. No16Kdecode cumulative credit and no whole-model prediction for the different long-context layer mix. Public inference unchanged.


### 2026-10-03 long-mirror real16 qualification withheld — experiment, no credit

`long-index-mirror-real16-02`: same current prefill snapshot, canonical paged MME versus mirror reader, native full16 chain, same weights/modules/CPU, ABABAB200. Observed medians4.886498/4.290870ms, threshold0.140881ms, but cross-arm feedback differs first at warm step11. Within-arm feedback is stable and all four ranks agree. No confirmed gain, no ledger/cumulative credit, and no serving deployment. Retain complete periods and mark this numerical gate unresolved rather than attributing it to host noise.

`long-index-mirror-sequence-01/02`: untimed actual native full producer/consumer replay with advancing positions, changing logical/physical addresses, positive gains and a forced newest key.32steps×2ratios×4ranks×4outputs=1024 exact comparisons per geometry, at82944/search131072 and41984/search65536. This does not reproduce a general dynamic-state failure. A same-forced-token real16 numerical diagnostic now records hidden, selectedIDs and historical/newly written canonical/mirror key probes, without another baseline speed run. Public source remainsb7553683 andPR57draft.


### 2026-10-03 corrected long real16 physical-page contract — experiment correction

The long real16 fixture retained256requestpages despitecontext41984. Later block_table entries were0, so all visible tokens beyond32768 aliased nullpage. Canonical writes changed many older aliased logicalrows; mirror maintenance updated one newly written logicalrow. This explains why the valid isolated changing-position chain passed while the invalid full fixture diverged. `visible-index-real16-05` and `long-index-mirror-real16-01/02` are superseded for long-context qualification, including their timing and feedback oracles. Their originals remain archived; historical16Kfixtures stay within their pool and are unaffected. No long real16 gain is credited.

Allocate distinct pages for context plus the reserved continuation and warm steps, check every reachable mapping is positive/unique/inbounds before timing, and establish the missing valid reference once in corrected`long-index-mirror-real16-03`. Two numerical diagnostic attempts also failed tool interfaces before collecting data: nativeI32 token contract and missing fixture stop attribute; their records are invalid and preserved. These are tool failures, not production model evidence. Corrected token and observer CPU contracts plus page uniqueness checks pass21tests. Public source unchanged; no deployment until the corrected complete-chain gate.


### 2026-10-03 corrected actual long real16 mirror gate — component qualified

`long-index-mirror-real16-03`, parent0da018c8: actual context41984,logicalcapacity512K,distinct333requestpages plus null,512reservedsteps,modules2/3/6/7,CPU64/69/92/97. Canonical paged MME and mirror share the same valid prefill snapshot/weights/native replay. ABABAB200 with32continuouswarmsteps: medians4.890051/4.297652ms, delta0.592399ms exceeds2×baselineIQR0.165892ms. Allsixperiods have exact cross-arm and four-rank feedback and stable within-arm tokens; no hot compilation/profiler. Retain firstA coldhosttail and its0.234926msperiod drift, no period discarded. Othergrouppublicservice wasidle, no otherweightloading. This resolves the apparent numerical divergence as an invalid fixture ownership contract.

This is a valid long-context whole16-layer component saving, not a16Kdecode or formal end-to-end gain. No×1.5/×2.5 extrapolation or cumulative16Kcredit because the source-range/layer mix differs. All compatible long source-window/MLA/mirror changes now advance together to one normal independently installed serving qualification, officialtemperature1/top_p.95/seed42,512K/C16384/cacheon/diagnostics off. Public remainsoldsource until frozen update/restart; formal result stillpending.


### 2026-10-03 combined long-prefix serving — measured, strict release goals not met

`long-prefix-serving-01`, ordinary independently installed31e519fd, all277inferencePython files match;
512K/C16384/cacheon/defaultacceleration/fullwarmup, diagnostics/profileroff, officialT1/.95/seed42.
No model injection, workspace imports, native ABI or sampling change. One formal uncached16K→2128naturalEOS
request: prefill8028.980763tps,decode10.568225ms,TTFT2.126094s,14facts pass,
all2128tokens exact to archivedb755. CPU1044passed111skipped; diagnostics/fixture21passed.
The existing formal reference8851.110164tps/10.565244ms is reused, not rerun.
Prefill regression is unresolved, not explained by context size or assigned to the algorithm without evidence.

Official-sampling80token changed-boundary diagnostics (length-limited, not naturalEOS quality qualification):
82945:prefill13.168563s/6298.713tps,decode11.423877ms;
144385:21.310411s/6775.327tps,12.329330ms;
300001:64.860913s/4625.297tps,14.503627ms.
The82945/144385/300001reasoning refers to the actual input-end instruction; these short outputs are not a quality cohort.
1024cold/hit896:TTFT2.142596/.600957s,decode10.570936/10.228068ms;
cache accounting hits correctly, same-seed tokens differ as in earlier records, exact state equivalence remains unproven.
No new broad baseline, trace or repeat formal request. Near-capacity524161boundary completed:prefill136.826331s/3830.849tps,decode17.318101ms,80tokens,noOOM/restart;endingCPU PSIavg10was9.39%,sharedhosttiming only. Its reasoning does not accurately restate the final instruction and has no final answer before the80tokenlimit; long quality remainsunqualified. The500003terminal-instructionnaturalEOScheck FAILED:419EOS; requiredLONG-PREFIX-END-500003, actual askswhatuserwants. Same requestwith499968cachedtokens reproducesall419tokens exactly. Returned inputIDsdetokenizetoactualtailincludingrequestedmarker, soHTTPpromptlossandcache-onlyperturbationdonotexplain thispair. Rootunresolved; no arbitrary-long semanticqualification.

Formal host monitor retained per-core utilization/scheduling and PSI; other group had stable loaded weights,
then released before the request, with no recorded ongoing growth. Shared-machine conditions are retained.
Startup reports5,271,424KVtokens, mirror335544320bytes/card, profile peak91990273536bytes/card,
resident devices approximately91259–91264MiB at formal readiness. Long300001completed withoutOOM;
this does not qualify unrestricted concurrency or every length. No16Kgain or release-goal success is claimed.

All long-window/selected-paged-MLA/mirror items have now been exercised together in normal serving,
so move them out of the untested-component queue without adding or summing overlapping request gains.
Public API is online on this candidate; PR57 remainsdraft andmain unchanged while strict goals remain open.


### 2026-10-03 user-authorized integration of current implementation

User explicitly requested merging currentPR57into main after the recorded qualification results.
This supersedes the earlier draft-only integration instruction. Retain the failed speed and long-terminal-instruction
results; do not advance the formally accepted performance baseline, award component gain credit, or create a qualified
release tag. Ordinary installed API remains online on measuredsource31e519fd; no new model request, profiling,
service restart or unsupported deployment claim is introduced by this merge. Existing CPU/component/formal evidence
is reused because the latest changes only clarify documentation and startup audit text. Unrelated localAGENTS.md
and reproducer artifacts are excluded. Follow-up work must resolve long semantic correctness and prefill regression
with the accepted prompt tail and exact cold/cache feedback evidence already archived.


### 2026-10-03 merged-main single-request decode campaign

User switches scope to officialT1/top_p.95 TP4decode:16Kfirst≤9ms then≤8ms;128K≤11ms.
Parentmain e6f0a36c (PR57merged), modules0/1/4/5 andmainCPU10/15/38/43.
Publicgateway/backend stopped; private frozenmain baseline service nowwarming for one16Kdecode trace.
Reuse archived unprofiled10.568225ms; do not turn theprofilerrequest into a formal baseline or repeat distributions.
500Ksemantic failure stays inPR57anditsrecord, explicitly deferred rather than being treated as passed.

Work order:one maintrace→sharedI32/coordinates→all-sixexpert decode/MME→mHC→attentionTPCfusion→exactboundedofficialsampling→128K/512Kdecode traces and source submission reduction. Each candidate must pass exact/reference checks, then same-processreal16ABABAB. One formal16KnaturalEOS plus one128K80token diagnostic only after compatible real16savings total≥1ms. Continue all items or attain the requested targets, then merge qualified changes andrestore512K/prefixonpublicAPI. Current confirmed cumulative real16savings0ms; no newgainledgerentry.

Own retiredgraphs were losslessly archived withtarcompare/SHA256 before removing original folders; newtrace/postgraphs/logs remain onSSD. Disk headroom restored beyond30GB. Historical result/trace/source records retained. New I32armuses the qualified norm/SWA/Engram/mHC handoff parent, not the older dense-only parent whose archivedIQR exceeded its measured difference. Its numerical/real16result is pending.

2026-10-03：合并main后的常驻ABABAB200：I32+只读静态工厂4.250790→4.136407 ms/16层，差值0.114383 >2IQR0.071210，跨卡/跨候选token一致，无热编译。按×1.5仅预估整模0.171575 ms，尚未端到端；累计确认16层0.114383 ms。I32单项、FP4双输出、I32镜像gather均未过噪声门槛，不相加。详见本地decode-kernel-resident-01/INITIAL_AB_REPORT.json。

### 2026-10-04：当前微基准与批量验收规则

生产形状、真实权重、直接生产者与消费者，使用生产原生重放。正确性检查 3–5 组真实输入；同进程 ABABAB，三轮设备时间差方向一致即可，不用 IQR 门槛。候选默认关闭；仅通过完整链微基准的候选记一行“名称／每 token 或每轮节省／开关”。累计预估 ≥1 ms 或 3–5 项后，统一进行一次官方采样正式请求（seed 42、自然 EOS）及随带 trace。端到端兑现 ≥一半预估才整批默认开启。未测、失败、重复方向不计收益。

固定启动工具：`tools/launch_deepseek_v41_decode_micro.py`，CPU 10/15/38/43，SSD 临时目录、编译缓存、诊断默认关闭和桥接接口前置检查。现有微基准收益不改记为端到端收益。

写完候选到微基准结论以 30 分钟为目标；超过先完善复用脚本。预期不足 0.1 ms/token（或 0.3 ms/轮）不做，同方向失败两次先读已有 trace／编译图。归因精度 ±0.5 ms 即可指导下一步，只随批量验收采集；编译／服务运行期间继续写下一项。固定启动器已加入设备异步退出等待，常驻工具可直接构建已验收父候选作为 A/B 基线，避免重测父优化。短诊断明确标记，不通过自然 EOS 正式验收。

2026-10-04：设备闭环（采样帧→双 Engram→下一次原生重放）／真实16层三轮设备中位差 1.012987 ms/token（1.012987、1.015639、0.879438）／`VLLM_HPU_DSV41_DEVICE_CLOSED_LOOP=0`。五输入 token、hidden、33 份可变状态逐位一致；计数为一次交接，不乘层数。证据 `decode-micro-resident-06`；相对含有界采样的父路径，端到端待批量验收，不与已含的采样/静态坐标收益重复相加。

2026-10-04 批量验收：设备闭环完整预热、缓存开启，官方采样 seed42，16K 未命中→2382 tokens 自然EOS，正式 TPOT **9.997343 ms**。token 与正文均与 `decode-position-serving-01` 完全一致（14事实参考已通过）。相对该父版本改善 1.395094 ms，超过微基准预估一半；相对最初10.568225基线改善 0.570882 ms。关闭此项待验收余额，单 stage 原生默认启用采样／设备输入／闭环／已验收静态坐标，PP 阶段及显式禁用不变。目标7ms未完成。随带 trace 启动缺必需写出参数，未生成设备trace；正式数字保留，只修复采集入口。

2026-10-04 资源规则更新：按最新用户指令，不持有显卡锁。启动器检查 HBM、设备利用率及打开的计算设备句柄；优先空闲的 0/1/4/5，否则选其他空闲四卡，不足则等待。记录实际映射；可用 `--modules` 固定某组卡。08 的锁包装进程已结束，原测量进程保留，不影响其他会话。资源调整本身不计收益。

2026-10-04 待整体验收：peer 求和→mHC post/collapse 融合／原生短链每 Attention 边界节省 **0.008950 ms**（三轮0.008950、0.012500、0.005773）／`VLLM_HPU_DSV41_PEER_POST_COLLAPSE=0`。父正式基线9.997343 ms。生产形状与真实 checkpoint 权重、5个 checkpoint embedding 派生输入，投影→四卡交换→mHC→FFN norm/量化全部逐位一致，无热编译。同进程 ABABAB200，取四卡最慢每步设备中位数。原计时完成后的退出异常已修复，另做无计时的资源退出检查，原测量保留。仅按40个 Attention 边界预估 **0.358000 ms/token**；MoE边界未验证不加算，实际 Attention 激活与real16状态接入仍待验证，未跑端到端，不称为正式收益。证据 `decode-peer-post-collapse-01/OUTCOME.json`。当前兼容微基准候选预估余额0.358000 ms；累计不足1ms，暂不重启正式服务。

2026-10-04 最新任务调整：优先手写多输入、多输出 TPC 融合，按 MoE → Attention/CSA2 → mHC → 非通信段合并推进。每项同时验收编译后物理节点数和生产形状原生微基准；三模块累加后一次官方16K→EOS及trace。预期不足0.3ms/token的零碎候选暂停。有序peer/post候选保留历史0.358ms预估及证据，但从当前待验收队列移出，开关保持0；当前这批有效待验收余额为0ms。正式基线仍为9.997343ms，7ms目标未完成。

| CSA2 norm/RoPE/FP4 publication | parent formal 9.997343 ms/token | native repeated16-group production static compiler, actual index-mirror MME→MLA→WO→TP sum; 5 checkpoint-derived inputs, all outputs/states exact/4ranks; physical consumer 44→17 | AB [0.0468718125, 0.0508414375, 0.053027531249999996] ms/source, median 0.050841 | ratio2 sources2/8/14 only: **0.152524 ms/token estimated**; ratio1 source20 unmeasured | `VLLM_HPU_DSV41_COMPRESSOR_FUSED_PUBLISH=0` | decode-csa-publish-07 on SSD | pending combined official-sampling acceptance |

Current compatible pending component estimate: **0.152524 ms/token**. No formal gain; superseded05 and accepted baseline gains are excluded. Weight/input fingerprints and reduced component-native topology are recorded with the evidence.

### 2026-10-05 累加规则更新

按用户最新要求取消小项收益门槛。最终物理节点减少、5 组输入逐位一致、同进程三轮设备 A/B 全部为正即可进入待验收；累计预估达到 0.6 ms/token 或减少 300 个物理节点后，统一官方采样 16K→EOS 加 trace。约 2 µs/节点仅作为节点数量估计，不能替代实测组件或正式收益。方向不一致的候选保留到同模块融合更多后再测。

| 名称 | 完整生产者/消费者微基准节省 | 节点 | 开关 | 状态 |
| --- | --- | --- | --- | --- |
| KV norm/RoPE/发布＋main reuse gather，行优先映射 | 三轮每层 0.002396 / 0.002216 / 0.002347 ms；27 个复用层预估 **0.063372 ms/token** | 23→19，27 层少108个 | `VLLM_HPU_DSV41_KV_REUSE_FUSION=0` | 5输入4卡输出及SWA状态逐位一致；decode-kv-reuse-fusion-03；正式待验收 |

CSA publish07 继续计 **0.152524 ms/token**，3 个 ratio2 源少81个节点；不重复计 superseded05 或基线已含收益。当前兼容组件预估合计 **0.215897 ms/token**、少189个节点，尚未正式测量。

KV 发布20→14按用户要求列入整模块组合候选：此前三轮方向不一致，暂不把节点估计当作已确认的微基准节省。它与 reuse 分支存在覆盖关系，组合验收时按实际生效层数计数，不能将两项按40层同时相加。原生分段合并常驻任务在父候选冷准备期间遇到严重主机换页，之后主机重新启动；无有效数值/计时数据，不入账。下一版保留 W13 六路，W2 两组三路，先验收最终编译图及完整链。

在最新累加规则下恢复 mHC 模块中的 peer→post/collapse 融合待验收项：原生完整生产者/消费者5输入精确、三轮每边界节省0.008950/0.012500/0.005773 ms；最终 compiler JSON 确认消费链3→2个物理节点。仅40个Attention边界预估 **0.358000 ms/token**，MoE不计；`VLLM_HPU_DSV41_PEER_POST_COLLAPSE=0`。该项是手写消费融合，未将单独有序求和的未验证变体恢复。当前兼容待验收预估 **0.573897 ms/token**、少229个物理节点；real16状态门槛与整模正式验收仍待验证，不称为端到端收益。

MoE 两组三路试验04：物理24→24，未进入数值/原生计时门槛，不入账。源码确认Gaudi2流水拒绝同一TPC同时供给MME两个输入；把W2权重恢复单输出以维持SRAM，保留W13六路和W2两组三路后重测。

Q/KV联合准备与KV发布／完整输入FP8投影→Q norm/quant＋KV norm/RoPE/SWA→Q MME/RoPE→MLA→WO→TP→mHC→FFN norm/quant，5组输入/4卡结果与SWA字节逐位一致；三轮每层节省 **0.008666 / 0.008529 / 0.008466 ms**，中位0.008529 ms／`VLLM_HPU_DSV41_QKV_FUSED_PROLOGUE=0`。最终物理生产链23→19。按40层预估 **0.341141 ms/token**，实际服务覆盖待核实；证据decode-qkv-prologue-fusion-03。

该项覆盖此前KV发布/reuse生产者，组合中关闭KV_REUSE_FUSION和ATTN_FUSED_PROLOGUE，移出其0.063372ms估计。当前三项兼容余额：CSA07 **0.152524**＋Q/KV **0.341141**＋Attention peer/post **0.358000**＝**0.851666 ms/token预估**；节点估计少281个。已达到最新0.6ms触发线，进入一次组合正式16K→EOS及随带trace。目标7ms、全部逐token物理节点≤1000均未达标，不称为正式收益。

2026-10-05 实验索引（不入收益余额）：WO scale/roundtrip＋dense quant 的手写核，最终生产链19→18，5输入4卡逐位一致；三轮每层差值0.000874、0.076331、-0.074175 ms，方向不一致。按最新规则保留至更大 Attention/mHC 组合，不单独放弃，也不计已确认收益。证据decode-woa-handoff-fusion-01/OUTCOME.json。下一组合已加入 peer/post/collapse＋FFN norm/quant 的手写核，并带更新残差的控制/Gates和归一化输出的router两个实际消费分支；编译与68项CPU接口检查通过，硬件验证等待空卡。

三项0.851666ms预估的正式验收尚未产生数字：02:29模型进程收到终止信号，启动时间未变，不能当作编译器崩溃或主机重启。SSD inode耗尽后迁移本任务安装依赖与recipe缓存至Optane，日志/trace继续SSD；发现并修复ABI校验中预期路径未resolve的问题，保持二进制和运行库SHA256严格校验，三项路径身份/篡改CPU检查通过。正式基线仍9.997343ms，目标7ms未完成。

2026-10-05 启动问题修复（不计收益）：DUMP_PRE_GRAPHS/DUMP_POST_GRAPHS 的字符串0被Bridge当成输出目录，完整预热生成约5.5万个调试文件后耗尽SSD inode；本次预热失败未进入正式请求。移除变量及继承值、加入启动前检查，清理本次未完成预热的导出；14项安装/环境/运行库身份CPU检查通过。正式服务依旧仅三项已确认组件候选，不包含尚未测量的WO和peer/norm新核。

MoE 下一组实验（未测，不入账）：针对batch GEMM物理拆分根因，W13保留六路水平SAT；每组三路W2水平权重配合一个普通GEMM，手写finalizer只读匹配对角块，保留每路BF16边界与有序共享输出归约。额外交叉乘积是否被MME最小M粒度吸收仍需生产编译图与微基准判断，不能仅凭一次GEMM判为收益。实现按route_pack参数化，共用TP路径。


### 2026-10-05: candidate-block coordinate screen — not yet in gain ledger

The 16K mirror / 256-block screen produced 26→20 physical nodes, five exact
fixtures and three positive native replay pairs (2.317578 / 2.317531 / 2.315703 µs).
However, the current 512K service reserves 262144 ratio-2 mirror rows and consumes
2048 candidate blocks in eight score tiles. The screen did not use that full
production shape. Its earlier 0.060256 ms/token extrapolation is withdrawn pending
that validation; it contributes **zero** to the cumulative forecast.

Qualified pending forecast remains **0.8516655625 ms/token** (CSA publication,
Q/KV joint prologue, attention peer/post collapse), with no new formal result.
The small-shape whole-output variant reduced 20→19 nodes but all three pairs
slowed by 0.106–0.121 µs; it remains off and contributes zero.
See SSD `decode-candidate-coordinate-chain-01` and
`decode-candidate-coordinate-whole-chain-01` for preserved raw observations.

### 2026-10-05: Reindex coordinate fusion — qualified production-shape micro

| Name | Savings / Reindex layer | Physical nodes | Flag | Pending token estimate |
|---|---|---|---|---|
| Candidate expand + sentinel mask + safe address, stock gather/MME | 0.008844797 ms | 69→49 | `VLLM_HPU_DSV41_CANDIDATE_COORDINATES` (default 0) | 0.035379188 ms |

Evidence: SSD `decode-candidate-coordinate-production-02/DECISION.json`.
Checkpoint identifies four Reindex layers 24/28/32/36, **ratio 1**; their mirror is
524288×128, candidate pool 2048 blocks, eight 2048-row scoring tiles. Five
checkpoint-derived projection fixtures are bit-exact. Native ABABAB savings:
0.008841797 / 0.008844797 / 0.008850953 ms. Producer is one physical TPC, 13 clamp nodes and three coordinate
transposes removed; existing 13 gather / 15 MME nodes remain. No whole-output
variant is needed. Previous smaller / ratio-2 screens remain non-credited.

Qualified pending forecast: **0.887044750 ms/token**. This remains an estimate;
formal baseline is unchanged at 9.997343230 ms/token. Token-entry shared metadata
is still unqualified and not included.

### 2026-10-05: six-route W13 / three-route W2 native chain

| Name | Micro saving | Physical nodes | Flag | Pending token estimate |
|---|---|---|---|---|
| W13 horizontal SAT, W2 groups3 + diagonal ordered shared finalization | 0.005902031 ms/layer | 24→21 per producer | `VLLM_HPU_DSV41_EXPERT_W2_THREE_ROUTES` (default 0) | 0.236081250 ms |

SSD `decode-sat-horizontal-w2-fusion-04/DECISION.json`: real layers 0/4/14/19,
12 checkpoint experts, five inputs on all four ranks bit-exact including peer and
mHC/FFN consumers. Three native four-layer savings: [0.02360812500000009, 0.023651500000000047, 0.02360193750000006].
W2 FP8 remains in SRAM; W13 still has four physical slices, so the ≤12/≤9 MoE
node target is not achieved. The previous result-lost run receives no separate
credit. Qualified pending forecast is **1.123126000 ms/token**;
formal baseline remains 9.997343230 ms/token until combined serving acceptance.

### 2026-10-05: exact deferred mHC gates/post — next batch only

| Name | Micro saving | Physical nodes | Flag | Pending token estimate |
|---|---|---|---|---|
| Original FP32 control/RRMS → fused gates + peer/post/collapse | 0.000787512 ms/boundary | 9→8 complete WO/TP/FFN/router chain | `VLLM_HPU_DSV41_MHC_DEFERRED_GATES` (default 0) | 0.031500469 ms (40 boundaries only) |

Five checkpoint-derived inputs on four ranks are bit-exact. Three paired native
savings: [0.0007875117187499947, 0.0008048632812499984, 0.0007771210937499995] ms. Evidence: SSD
`decode-mhc-exact-post-01/DECISION.json`. Reference already contains fused
peer/post; none of batch03's reduction saving is counted again. Serving prototype
also covers the MoE post boundary, but that second boundary receives **zero**
additional credit until its producer chain is measured. This item is **not** in
the frozen batch03 formal request (forecast 1.123126 ms). Total pending forecast
including this next-batch item is 1.154626469 ms; no new formal result yet.

### 2026-10-05: shared-main PV / inverse RoPE / WO projection

Reuse producer physical nodes **19→18**, unchanged downstream **3**. Five checkpoint-derived inputs × four ranks are bit-exact, including SWA writes and post/norm outputs. Three native-chain savings **0.001368859 / 0.001823461 / 0.001617090 ms per reuse layer**. Flag `VLLM_HPU_DSV41_MAIN_MLA_PROJECTION=0`. Source topology with four-layer groups has 27 reuse and 11 publish occurrences at the 16K logical path; only reuse is qualified so far, yielding **0.043661426 ms/token estimated**, not formal. The first publish test failed before timing due to tensor input indexing after a scalar ratio; it receives zero credit. Evidence: SSD `decode-main-mla-projection-01/DECISION.json` and `decode-main-mla-publish-projection-01/DECISION.json`.

Shared-main publish follow-up (`decode-main-mla-publish-projection-02`): five inputs × four ranks exact, 19→18 producer nodes, paired savings 0.000173684/0.000192953/−0.000049105 ms. Mixed direction: retain for the combined module but **0 pending gain**. The nonsliceable producer experiment increases nodes to20 due to two DMA copies and a split scale kernel; reverted, **0 credit**. Qualified pending balance remains **0.075161895 ms/token** (mHC40-boundary estimate plus 27 reuse projections).

### 2026-10-05 — pending MoE decoder schedule (default off)

| Candidate | Parent | Complete-chain saving | 40-layer forecast | Switch / status |
|---|---|---:|---:|---|
| Unrolled SAT decode + two W13 slices | batch03 groups3 / 9.564 formal baseline | 0.000310086 ms/layer | 0.0124034 ms/token | `unrolled_sat_shared` experimental schema + scoped slice policy 2; not wired into serving |

Evidence: `/opt/ssd960/1cat-vllm-decode-archives/decode-moe-unroll-chain-01/DECISION.json`.
Five checkpoint-derived inputs × four ranks exact; native replay ABABAB direction
consistent; complete producer 21→17 physical nodes, decoded weights in SRAM.
This entry is micro-qualified only. New accumulated forecast: **0.0124034 ms/token**;
formal baseline remains **9.564282 ms/token**. Integration into the normal cold
compile policy is required before any combined serving acceptance.

The entry above is superseded (not added) by `decode-moe-streamed-01`: same
expanded decoder and two slices, with the exact one-route SiLU kernel. Three
four-layer differences: 0.00303965625/0.00308975/0.003012375 ms. Five inputs
× four ranks exact, 21→17 producer nodes and decoded weights in SRAM. Saving
0.000759914 ms/layer; active forecast **0.0303966 ms/token**. Experimental
`streamed_sat_shared` schema plus scoped slice policy 2; serving remains off.
The activation pipeline is **not** in SRAM yet; this forecast does not claim
completion of the MegaMoE design. Formal baseline is unchanged.

### 2026-10-05 — retain GEMM slicing; revise active micro balance

The two-slice unrolled/streamed experiments above are archived, inactive and
excluded from the cumulative balance under the revised four-slice policy.
Their fixed-route fixtures also omitted the production router dependency.
Active qualified forecast for this MoE batch: **0 ms/token**. The accepted
formal baseline remains **9.564282 ms/token**. Future reports distinguish
independent logical stages from physical TPC/MME fragments of one pipeline.

### 2026-10-05 — MoE producer with production routing, slicing retained

| Name | Native micro saving | Forecast | Switch |
|---|---:|---:|---|
| SAT load schedule + scalar route scale + per-route SiLU | 0.003281063 ms/layer | 0.1312425 ms/token (40 layers) | `VLLM_HPU_DSV41_EXPERT_STREAMED_SAT=0` |

`decode-moe-router-four-slice-01`: BF16 router GEMM/top6, all 384 checkpoint
experts in four layers, shared expert, native peer exchange and mHC/FFN consumer.
Five embedding-derived input rows × four ranks exact; ABABAB four-layer
savings 0.013155594 / 0.013124250 / 0.013116688 ms. Policy restored to 4;
compiler chooses three W13 fragments in both arms for this producer graph.
Independent logical stages 17→16 (scale broadcast removed), physical nodes
23→22. W13 products and decoded weights reside in SRAM. This supersedes
all two-slice variants; no addition of their earlier estimates. Active forecast
**0.1312425 ms/token**, formal baseline unchanged at **9.564282 ms/token**.

The same-baseline combined producer in `decode-moe-dual-quant-01` supersedes
that estimate: FFN norm emits the routed and shared FP8 operands with one
amax and distinct rounding. Three four-layer savings: **0.020614031 /
0.020665156 / 0.020637375 ms**, five inputs × four ranks exact; W13 activation
remains in SRAM. Combined estimate **0.20637375 ms/token**, not 0.1312425 +
0.20637375. Add `VLLM_HPU_DSV41_FFN_DUAL_QUANT=0` alongside the streamed SAT
flag. Both remain off; no formal request is authorized by the ≥1 ms gate yet.

### 2026-10-05 — mHC linear load plus exact deferred gates

Five inputs × four ranks exact in WO/control → native peer → gates/post →
FFN/router. Native A/B savings **0.000647535 / 0.000645180 / 0.000649762 ms
per boundary**; physical and logical nodes 9→8. Count only the 40 measured
Attention post boundaries: **0.025901406 ms/token forecast**. This replaces
the older 0.031500469 mHC estimate; they must not be added. The other 40 FFN
boundaries receive no unmeasured credit. Flags `VLLM_HPU_DSV41_MHC_LINEAR_LOAD=0`
and `VLLM_HPU_DSV41_MHC_DEFERRED_GATES=0`. Evidence: `decode-mhc-linear-post-01`.

Current active forecast with the combined MoE candidate: **0.232275156 ms/token**.
Official baseline remains **9.564282 ms/token**; the ≥1 ms trigger is not met.

### 2026-10-05 — MLA gather/decode vector utilization

| Scope | Native micro saving | Qualified occurrences | Token forecast |
|---|---:|---:|---:|
| Reuse: SWA codec 64→128 values | 0.004391113 ms/layer | 27 | 0.118560059 ms |
| Publish: SWA + FP4 main codec 64→128 values | 0.005256754 ms/layer | 11 | 0.057824293 ms |

Five checkpoint-derived fixtures × four ranks exact, including modified SWA
and exported main rows/masks. Both native ABABAB runs are consistently positive.
The Q/KV producer, QK/softmax/PV, original WO path and peer/mHC consumers are
retained. Physical/logical node counts do not decrease; this is instruction
and vector utilization improvement, not fusion credit. Cases:
`decode-mla-reuse-vector-01`, `decode-mla-publish-vector-01`.
Flag `VLLM_HPU_DSV41_MLA_VECTOR_CODEC=0`. These distinct reuse/publish
occurrences total **0.176384352 ms/token estimated**.

Active micro-qualified cumulative forecast: **0.408659508 ms/token**.
No new formal request: the ≥1 ms trigger is not reached. The two-instruction
expert decoder remains unqualified and receives zero credit. Official baseline
remains **9.564282 ms/token**.

### 2026-10-05 — remove repeated whole-row amax from existing fused quantizer

ISA audit found the disabled WOa scale/dense-quant candidate recalculated the
whole row for each output tile (16 or 32 copies). The corrected kernel caches
the rounded BF16 row, computes one amax, loads each group scale once and uses
frontend loop expansion (the old backend unroll pragma was ignored). No GEMM
slicing change. Five inputs/four ranks exact; native three-round savings
**0.000903062 / 0.001213043 / 0.000779516 ms per layer**.
`decode-woa-single-amax-02`, `VLLM_HPU_DSV41_WOA_DENSE_HANDOFF=0`.
40-layer estimate **0.036122500 ms/token**. This is a correction to the
previously inefficient hand-written kernel, not credit for its archived trial.

Active cumulative forecast **0.444782008 ms/token**; formal baseline unchanged.
No end-to-end request until cumulative forecast reaches at least 1 ms.


### 2026-10-05 — roofline reprioritization and tolerance-qualified controller

`decode-mhc-parallel-k-01`: checkpoint WO/control→native peer→post→FFN/router,
five inputs×four ranks, three savings **0.000988996 / 0.000995863 / 0.001007996 ms/boundary**.
TPC controller keeps24 row owners and uses eight independent K accumulators; it changes
FP32 summation order. CPU oracle implements DeepSeek dba1be0 hc_mixes/Sinkhorn/post equations;
DeepGEMM 057ca596 normalized squared-error limit5e-5, observed maximum **7.679e-9**.
The upstream CUDA kernel itself was not executed. Router IDs remain exact on all fixtures.
Physical/logical producer+consumer **9→8**, no GEMM pipeline slices removed.
Switches `VLLM_HPU_DSV41_MHC_PARALLEL_CONTROL=0` plus `VLLM_HPU_DSV41_MHC_DEFERRED_GATES=0`.
Conservative credit only40 measured boundaries: **0.039834531 ms/token forecast**,
replaces0.025901406 linear-post estimate (not additive); other40 not credited.
Active total **0.458715133 ms/token**, official baseline **9.564282** unchanged.
Planning realization~40%; ≥1ms remains the batch-formal trigger.

Communication pruning and both MME-controller boundaries had no gain; they are in INDEX,
not this gain total. Architectural bounds and unresolved communication/SRAM lifetime work:
[ROOFLINE](../../evidence/ROOFLINE.md).

### 2026-10-05 — vector BF16 multiply at the routed quantization boundary

`decode-ffn-bf16-quant-03`: same-process, same2/3/6/7 and CPU affinity;
both arms use the already-qualified streamed MoE and dual norm/quant parent.
Five checkpoint-derived fixtures×four ranks exact. Three native four-layer savings
**0.000047688 / 0.000263937 / 0.000114344 ms**; median per-layer0.000028586ms.
40-layer forecast **0.001143438 ms/token**, incremental to the MoE parent.
TPC simulation3791→3439 VLIWs, same482loads/363stores, all20488 output bytes exact
for each of five fixtures. Physical nodes stay21→21; no fusion credit.
Switch `VLLM_HPU_DSV41_FFN_BF16_QUANT=0`, requires dual quantization and C1.
Active cumulative forecast **0.459858570 ms/token**; formal baseline9.564282 unchanged.
The small result is recorded without rounding it into a material latency claim.

### 2026-10-05 — cold index gain replica

`decode-index-gain-replica-01`: query projection, gain projection, native peer
exchange, and a production 2048-row scoring consumer. Five inputs x four ranks
query/gain/score bytes exact; three native A/B savings **0.003837906 /
0.003863922 / 0.003833695 ms per index layer**. Eight index layers forecast
**0.030703250 ms/token**. Only the small [32,5120] BF16 gain projection is
replicated at load time, adding 2.5 MiB per rank across eight layers while retaining
local checkpoint shards. Query projection and scoring remain unchanged. Shared
TP-parameterized C1 source uses `VLLM_HPU_DSV41_INDEX_GAIN_REPLICA=0`; prefill,
C2–C6 and request-batch keep the original path. 64 relevant CPU checks passed.
This is a component-qualified forecast, not a formal end-to-end result.

Active cumulative forecast **0.490561820 ms/token**.
Official baseline **9.564282 ms/token**, next formal trigger **1 ms**.

### 2026-10-05 — mHC K128/head weight layout

`decode-mhc-swizzled-control-01`: original [24,20480] weights prepared once as
[160,24,128], same parallel controller MAC and reduction order. Five fixtures x
four ranks exact through peer/post, norm/quant and router; three native boundary
savings **.000147008 / .000138254 / .000138160 ms**. Only 40 measured boundaries
receive credit: **0.005530156 ms/token**, incremental to the parallel controller.
No logical-node reduction; source-layout experiment, not measured HBM utilization.
Default-off `VLLM_HPU_DSV41_MHC_SWIZZLED_CONTROL`, requires PARALLEL_CONTROL and
DEFERRED_GATES. Prefill and C2–C6 retain checkpoint layouts. Active forecast
**0.496091977 ms/token**, still below the 1ms formal trigger.

### 2026-10-05 — direct shared-RRMS post layout

`decode-mhc-rrms-post-01`: five x four exact, native A/B differences
.000002531 / .000007297 / .000007254ms per boundary. Specializing the known
25-value input removes unused RMS code and six static VLM reload sites, but
complete-chain time barely changes. Literal 40-boundary forecast
**0.000290156 ms/token**, flag `VLLM_HPU_DSV41_MHC_RRMS_POST=0`. This tiny
component value does not establish a material end-to-end improvement.
Active forecast **0.496382133 ms/token**; formal unchanged.

### 2026-10-05 — BF16 controller weight alternative (not additive)

`decode-mhc-bf16-weight-03`: five checkpoint fixtures x four ranks, upstream
equation errors at most3.20124e-6 (limit5e-5), router IDs exact. Native full
WO/control→peer→post→FFN/router savings .000157316/.000175648/.000163961ms
per boundary. Forty-boundary forecast **.006558438ms/token**.
Default-off `VLLM_HPU_DSV41_MHC_BF16_CONTROL_WEIGHT`, requires parallel and
deferred control. Mutually exclusive with swizzled weights: this replaces
.005530156ms, never adds to it. Retain swizzle in the active batch for now;
active forecast **.496382133ms/token** and formal **9.564282ms/token** unchanged.
The candidate retains original FP32 weights for wider batches and adds75MiB/rank.

Simulation correction: FFN BF16 quantization was rechecked with typed FP32
descriptors in `decode-ffn-bf16-quant-01/correct-descriptors.log`; all five
20488-byte outputs exact, instruction counts unchanged. Earlier malformed
simulator outputs are superseded; hardware evidence and ledger credit unchanged.

### 2026-10-05 — vector shared-main reuse mask producer

`decode-mla-vector-mask-01`: parent vector KV codec held fixed in both arms.
Five checkpoint-derived inputs x four ranks bit-exact through QKV, MLA, WO,
native peer and post/FFN. Three native savings **.005064145/.005103352/.005115961ms
per reuse layer**. Ten vector stores replace640 scalar mask stores; all-row
access is declared to the compiler, with the original shared-row ownership.
Five-position simulator outputs exact, including early SWA history and200K
position. Physical producer count stays23→23; this is ISA/traffic efficiency,
not fusion credit. 53 relevant CPU checks passed.
Default-off `VLLM_HPU_DSV41_MLA_VECTOR_MASK`, requires `MLA_VECTOR_CODEC`.
27 qualified reuse layers forecast **.137790492ms/token**, additive to the
vector-codec parent. Active total **.634172625ms/token**; formal baseline
**9.564282ms/token** unchanged. Combined formal trigger remains1ms.

### 2026-10-05 — corrected hardware E4M3 reuse codec

`decode-kv-hardware-codec-02/reuse-native-03`: vector codec and vector mask
parents held fixed. Five checkpoint-derived fixtures×four ranks alloutputs
bit-exact through QKV→MLA→WO→nativepeer→post/FFN. Native paired savings
**.000899145/.000845145/.000916508ms per reuse layer**. Executable producer
count stays23→23; the simulator reduces75268→69124 executed instructions.
Hardware E4M3 finite-upper-exponent, NaN and zero corrections were checked
for all256 codes; complete producer fixtures are also exact.
Default-off `VLLM_HPU_DSV41_MLA_REUSE_HW_CODEC` requires both vector parents.
Only27 reuse layers receive **0.024276902ms/token forecast**; no publish credit.
Active cumulative forecast **0.658449527ms/token**; formal **9.564282ms/token**
unchanged and combined formal trigger remains1ms.

The earlier reuse-native-02 attempt failed before timing because its nested
TMPDIR exceeded the Unix-domain socket limit. The shared component launcher
now uses a short profile scratch root and checks the UUID path before leasing
devices; no correctness/performance number is taken from that failed attempt.
