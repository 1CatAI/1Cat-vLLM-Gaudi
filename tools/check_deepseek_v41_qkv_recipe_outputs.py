# SPDX-License-Identifier: Apache-2.0
"""Expose all Q/KV compound outputs across two native replay recipes."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.prepared=args.prepared.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    os.chdir(args.output)
    if any('DUMP' in k for k in os.environ):raise RuntimeError('DUMP settings forbidden')
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from safetensors import safe_open
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_math import rotary_table
    from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
    from tools.deepseek_v41_micro_replay import RecipeRecorder
    torch.set_num_threads(1)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    recorder=RecipeRecorder(args.output)
    shard=PreparedV41Shard(args.prepared,0,0)
    sidecar=DenseFP8Sidecar(args.prepared/'sidecars/attention_dense_fp8',shard)
    norm=shard.tensor('layers.3.attn.q_norm.weight','hpu')
    kvnorm=shard.tensor('layers.3.attn.kv_norm.weight','hpu')
    weight=sidecar.tensor('layers.3.attn.wq_b.weight','hpu')
    scale=sidecar.tensor('layers.3.attn.wq_b.channel_scale','hpu')
    cfg=json.loads((args.prepared/'config.json').read_text())['text_config'];r=cfg['rope_scaling']
    phase=rotary_table(64,524288,cfg['compress_rope_theta'],r['original_max_position_embeddings'],
                        r['factor'],r['beta_fast'],r['beta_slow'])
    phase=torch.cat((phase[...,0],phase[...,1]),-1).contiguous().to('hpu')
    with safe_open(args.prepared/'pp0-tp0.safetensors',framework='pt',device='cpu') as cp:
        fixtures=[cp.get_slice('embed.weight')[t:t+1].clone() for t in (17,41,128,512,1024)]
    q=fixtures[0][:,:1280].contiguous().to('hpu');kv=fixtures[0][:,:512].contiguous().to('hpu')
    pos=torch.tensor([16384],dtype=torch.int32,device='hpu')
    cache=torch.zeros((256,528),dtype=torch.uint8,device='hpu')
    decoded=torch.zeros((512,512),dtype=torch.bfloat16,device='hpu')
    op=torch.ops.custom_op.custom_deepseek_v41_qkv_projection_publish_ordered_gaudi2
    def produce(q,n,kv,kn,pos,phase,cache,decoded,w,s):
        query,rotated,completion,normalized=op(q,n,kv,kn,pos,phase,cache,decoded,w,s,1e-20,-1)
        # Different public output order exposes each internal result to Bridge.
        return completion,query,normalized,rotated
    producer=torch.compile(produce,backend=make_backend(),fullgraph=True,dynamic=False)
    consumer=torch.compile(lambda marker:(marker+1,),backend=make_backend(),fullgraph=True,dynamic=False)
    operands=(norm,kv,kvnorm,pos,phase,cache,decoded,weight,scale)
    def chain(q,*args):
        out=producer(q,*args)
        return (*out,*consumer(out[0]))
    chain(q,*operands);torch.hpu.synchronize()
    plan=recorder.prepare(chain,[q],[operands])
    checks=[]
    for i,(fixture,position) in enumerate(zip(fixtures,(0,127,255,16384,131071),strict=True)):
        q.copy_(fixture[:,:1280].contiguous().to('hpu'))
        kv.copy_(fixture[:,:512].contiguous().to('hpu'));pos.fill_(position)
        expected=produce(q,*operands)
        expected=tuple(v.cpu() for v in expected)
        plan();torch.hpu.synchronize()
        actual=[v.cpu() for v in plan.outputs]
        exact=[torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(expected,actual[:4],strict=True)]
        exact.append(torch.equal(actual[4],expected[0]+1))
        checks.append(dict(input=i,exact=exact))
        if not all(exact):raise RuntimeError(f'Native output binding failed: {checks[-1]}')
    (args.output/'result.json').write_text(json.dumps(dict(status='passed',checks=checks,recipes=plan.recipes,
        scope='Cached recipe output bindings; not a performance measurement'),indent=2)+'\n')
    plan.close()


if __name__=='__main__':
    import torch
    with torch.inference_mode():main()
