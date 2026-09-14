# SPDX-License-Identifier: Apache-2.0
"""C1 RoPE and visibility preparation: changed inputs and exact BF16/I32 outputs."""
import json
import os
from pathlib import Path

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("CSA2 native tests require an explicit HPU lease", allow_module_level=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
prepare_environment()

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_math import apply_rope, rotary_table  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
ROPE = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2
INDICES = torch.ops.custom_op.custom_deepseek_v41_c1_indices_i32_gaudi2
EVIDENCE = Path(os.environ["DSV41_RUN_EVIDENCE"])


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("heads,width", ((1,128),(1,512),(32,512)))
def test_c1_rope_matches_original(heads, width):
    torch.manual_seed(4120 + heads + width)
    table_cpu = rotary_table(64,512,10000,4096,16,32,1)
    table = table_cpu.to("hpu")
    prepared = [torch.cat((table_cpu[...,0], sign * table_cpu[...,1]),dim=-1).contiguous().to("hpu")
                for sign in (1,-1)]
    compiled = torch.compile(ROPE,backend="hpu_backend",fullgraph=True,dynamic=False)

    def original(value, position, inverse):
        return apply_rope(value,position,table,inverse=inverse)

    reference = torch.compile(original,backend="hpu_backend",fullgraph=True,dynamic=False)
    results = []
    for inverse in (False,True):
        for position in (0,1,2,63,64,127,128,255,256,511,0):
            value = torch.randn(1,heads,width).bfloat16()
            value[...,::16], value[...,1::16] = 0., -0.
            if position == 256:
                value *= torch.exp2(torch.linspace(-64,64,width)).bfloat16()
            value = value.to("hpu")
            pos = torch.tensor([position],dtype=torch.int32,device="hpu")
            expected = reference(value,pos,inverse).cpu()
            actual = compiled(value,pos,prepared[inverse]).cpu()
            eager = ROPE(value,pos,prepared[inverse]).cpu()
            mismatches = (actual.view(torch.int16) != expected.view(torch.int16)).nonzero()
            item = {"position":position,"inverse":inverse,"bf16_mismatches":len(mismatches),
                    "ordinary_compiled_exact":torch.equal(actual.view(torch.int16),eager.view(torch.int16))}
            if len(mismatches):
                torch.save({"input":value.cpu(),"actual":actual,"expected":expected},
                           EVIDENCE / f"rope-{heads}-{width}-{inverse}-{position}.pt")
                item["first_mismatch"] = mismatches[:8].tolist()
            results.append(item)
    (EVIDENCE / f"rope-{heads}-{width}.json").write_text(json.dumps(results,indent=2)+"\n")
    assert all(r["bf16_mismatches"] == 0 and r["ordinary_compiled_exact"] for r in results), results


@pytest.mark.parametrize("ratio",(0,1,2))
def test_c1_indices_consume_current_publication(ratio):
    offsets = torch.arange(128,dtype=torch.int32,device="hpu").unsqueeze(0)

    def program(position, published):
        return INDICES(position,published,ratio)

    def original(position, published):
        window = position.unsqueeze(-1) - 127 + offsets
        ids = torch.where(window >= 0,window,-1)
        if ratio:
            ids = torch.cat((ids,torch.where(published >= 0,published + 512,-1)),-1)
        lengths = 128 + (position + 1) // ratio if ratio else torch.full_like(position,128)
        return ids,lengths

    compiled = torch.compile(program,backend="hpu_backend",fullgraph=True,dynamic=False)
    reference = torch.compile(original,backend="hpu_backend",fullgraph=True,dynamic=False)
    results = []
    for generation, position in enumerate((0,1,63,127,128,255,256,510,511,0)):
        pos = torch.tensor([position],dtype=torch.int32,device="hpu")
        count = (position + 1) // ratio if ratio else 0
        published = torch.arange(512,dtype=torch.int32)
        published = torch.where(published < count,published,-1).unsqueeze(0)
        if generation % 2:
            published = published.flip(-1).contiguous()
            published[0,:4] = torch.tensor([0,0,-1,3],dtype=torch.int32)
        published = published.to("hpu")
        expected = tuple(t.cpu() for t in reference(pos,published))
        actual = tuple(t.cpu() for t in compiled(pos,published))
        ordinary = tuple(t.cpu() for t in program(pos,published))
        results.append({"position":position,"indices_exact":torch.equal(actual[0],expected[0]),
                        "lengths_exact":torch.equal(actual[1],expected[1]),
                        "ordinary_compiled_exact":all(torch.equal(a,b) for a,b in zip(actual,ordinary))})
    (EVIDENCE / f"indices-ratio{ratio}.json").write_text(json.dumps(results,indent=2)+"\n")
    assert all(r["indices_exact"] and r["lengths_exact"] and r["ordinary_compiled_exact"] for r in results), results
