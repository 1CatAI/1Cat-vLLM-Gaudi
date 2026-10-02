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
本次官方采样实测：prefill **8,914.6 tokens/s**；decode **10.600 ms/token**；首个流式 token **1.924 s**；输出 **2,128 tokens**，自然 EOS，14 项固定事实与约束检查全部通过。此次发布按用户确认接受该实测版本；原 prefill 9,000 门槛尚未达到，decode 10 ms 目标留待后续独立采样优化。

这些数字来自同一次请求的引擎计时；首字为客户端 SSE 到达时间。思考模式的最终正文开始时间不等于首个流式 token 时间，公网客户端还会受到网络延迟影响。

可复现安装由main中的 `tools/install_deepseek_v41_runtime.py` 完成；检查点、系统Gaudi SDK、量化侧车和匹配的引擎/原生库为部署前提。服务从独立安装目录运行，配置不得依赖实验目录或通过PYTHONPATH引用工作区。诊断与trace采集默认关闭。
