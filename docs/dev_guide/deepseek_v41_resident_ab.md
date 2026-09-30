# 常驻 TP4 decode A/B

16层固定权重只加载一次。A使用已验收的BF16/v1 stage及重放外FP32 head HCCL，候选各自捕获独立stage原生计划；
未改动的权重共享同一张量分配，每段恢复相同16K后缓存与Engram历史。生产算子来自共用模型实现。

入口为`check_deepseek_v41_tp4_continuation_parent.py --shared-stage-replay --speed-probe --resident-ab`，
另传`--resident-control-dir`、`--ab-dense-sidecar`和`--ab-dense-config`。
使用常规受锁launcher，固定模块0/1/4/5、CPU10/15/38/43，并锁定同一native库。
首次权重加载与各臂冷准备不计时；相邻候选不重启模型。

控制工具`deepseek_v41_resident_ab.py --control-dir <目录> --candidate dense_fp8|tail|dense_fp8_tail --steps 200`
提交一个原子JSON请求。`--stop`在完成当前请求后释放本工具的计划和设备租约。
`ready.json`表示基线已准备；每个请求目录保留逐段逐卡时间、token、其他四卡负载和结果。

每次ABABAB，每段32步状态连续预热后计时至少200步。主统计为四卡最晚token回读时刻的相邻间隔，
A/B各600个样本的中位数与inclusive IQR；另记排空后的整段host均值和设备事件均值。
前置设备/分布式同步、状态恢复、加载检查均在计时之外。所有原生prepare/capture计数必须不变。
只有A中位数−B中位数严格大于2×A IQR才通过速度门槛；保留A三段中位数漂移用于区分环境变化。
慢或不可区分的结果不自动触发profiler。

每段前采另一组模块2/3/6/7的11次显存/利用率快照，间隔1秒；十秒内显存变化≥16MiB时等待，
增长和驱动内存池重置下降都必须稳定，避免启动阶段的短暂停顿被误判为加载完成。
并将全部检查追加到`competing-load.jsonl`。记录已允许的其他组常驻/运行负载，不改它们的进程或锁。
每个候选最长45分钟；保留失败或部分段数据，再换方向。

尾部候选只把LM head和argmax并入最后一组，不重复下一步embedding或引入额外embedding AllReduce。
16层A为41原生点＋1重放外head点；尾部B为42原生点＋0重放外点，完整链通信总数相同。
Python/C++拓扑允许终端head增加单个点；捕获仍验证所有生产、消费和晚输入切分边界。
原生FP32 peer默认关闭，只由PreparedGreedyTail选择；不能让父臂提前包含候选head传输。
共用StageReplay公共返回仍为三项；greedy_tail_token按隐藏张量归属与generation取已计算token，
C2–C6/DSpark入口保持原有采样合同。未验收的尾部默认关闭。

累计门槛采用兼容组合实际A/B差值，不能把不同父臂的局部差值相加。
组合16层节省达到0.4ms后才运行一次正式16K→自然EOS、完整预热、无profiler及5个固定语义样本。
组件有效不代表正式TPOT≤10ms；正式父值保留至端到端通过。
