# SPDX-License-Identifier: Apache-2.0
"""Keep raw shared-MME output and scales together until the routed finalizer."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def shared_expert():
    path=Path(__file__).resolve().parents[3]/'vllm_gaudi/models/deepseek_v41_program.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PreparedMoE')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='shared_expert')
    namespace={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['shared_expert']


@pytest.mark.parametrize('rows',[1,2,6])
@pytest.mark.parametrize('deferred',[False,True])
def test_shared_output_keeps_rounding_ownership(monkeypatch,rows,deferred):
    calls=[]
    def t(shape,dtype=torch.float8_e4m3fn):return torch.empty(shape,dtype=dtype,device='meta')
    scale=t((rows,1,1),torch.float32);channel=t((1,5120),torch.float32)
    def gemm(*args):
        calls.append(args)
        return t((rows,args[2].shape[0]),args[5])
    monkeypatch.setattr(torch.ops.hpu,'fp8_gemm_v2',gemm,raising=False)
    monkeypatch.setattr(torch.ops.custom_op,'custom_deepseek_v41_shared_silu_quant_gaudi2',
        lambda *args:(t((rows,1,640)),scale),raising=False)
    moe=SimpleNamespace(weights=SimpleNamespace(shared_experts=SimpleNamespace(w2=SimpleNamespace(channel_scale=channel))),
        shared_gate_up_channel=t((1,5,256),torch.bfloat16),shared_gate_up_weight=t((1280,5120)),
        shared_down_weight=t((5120,640)))
    out=shared_expert()(moe,t((rows,5120),torch.bfloat16),(t((rows,5120)),t((rows,1),torch.float32)),deferred_scale=deferred)
    assert calls[-1][5]==torch.bfloat16
    if deferred:
        assert isinstance(out,tuple) and len(out)==3 and out[2] is channel
        assert out[0].dtype==torch.bfloat16 and out[1].shape==(rows,1)
        assert calls[-1][6] is None and calls[-1][7] is None
    else:
        assert out.shape==(rows,5120) and out.dtype==torch.bfloat16
        assert calls[-1][6].shape==(rows,1) and calls[-1][7] is channel
