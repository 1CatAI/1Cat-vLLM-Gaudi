# TP4 官方采样 API 发布测试

模型：`DeepSeek-V4.1-Flash`。OpenAI 兼容地址：`http://dx.1catai.com:51499/v1`。
上下文上限：524288 tokens；prefill chunk：16384；前缀缓存开启。
推荐参数：`temperature=1.0`、`top_p=0.95`。复现验收时设 `seed=42`。

```bash
curl http://dx.1catai.com:51499/v1/chat/completions \
  -H "Authorization: Bearer $ONECAT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"请介绍一下你自己"}],"temperature":1.0,"top_p":0.95,"seed":42,"stream":true}'
```

发布验收：同一固定16K输入，未命中缓存，完整预热后一次无profiler请求自然EOS。
记录prompt吞吐、decode TPOT、首字时间、输出长度及固定事实/约束检查。
历史main发布实测：prefill **8,914.6 tokens/s**、decode **10.600 ms/token**、首字 **1.924 s**，
用户接受该可用版本；并未达到后续9.5k/10ms目标。

2026-10-03当前公网运行PR57的独立安装候选31e519fd，main尚未合入。
当前同一配置、一次官方采样未命中缓存16K→自然EOS：prefill **8,029.0 tokens/s**、
decode **10.568 ms/token**、首个流式token **2.126 s**，输出 **2,128 tokens**，
14项固定事实/约束检查通过，全部输出token与归档父版本一致。
这次prefill回退尚未归因，严格9.5k/10ms目标未达标，因此候选PR保持草稿，不作为已达标发布标签。

共享索引、查询归属MLA和派生索引镜像使用有界源窗口，总上下文长度不作为退出这些快路径的条件。
临时工作空间仍受控，支持容量仍为512K，不代表无限上下文或所有长度都已有相同速度。
长文官方采样80token诊断（未到自然EOS，不作正式质量验收）：

| 实际输入token | 未命中缓存prefill tokens/s | decode ms/token |
|---:|---:|---:|
| 82,945 | 6,298.7 | 11.424 |
| 144,385 | 6,775.3 | 12.329 |
| 300,001 | 4,625.3 | 14.504 |

接近容量上限的检查另行记录在开发台账；其完成前不声称所有边界均已经服务验收。
前缀缓存短诊断命中896token，首字约0.601s；同seed的输出不同，精确状态等价尚未证明。

这些数字来自同一次请求的引擎计时；首字为客户端 SSE 到达时间。思考模式的最终正文开始时间不等于首个流式 token 时间，公网客户端还会受到网络延迟影响。

可复现安装由main中的 `tools/install_deepseek_v41_runtime.py` 完成；检查点、系统Gaudi SDK、量化侧车和匹配的引擎/原生库为部署前提。服务从独立安装目录运行，配置不得依赖实验目录或通过PYTHONPATH引用工作区。诊断与trace采集默认关闭。
