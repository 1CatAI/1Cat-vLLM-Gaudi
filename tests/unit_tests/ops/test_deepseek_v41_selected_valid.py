# SPDX-License-Identifier: Apache-2.0
"""Unused selected rows stay private; only ordered attention consumes them."""
import json
import os
from pathlib import Path

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Selected-row tests require an explicit HPU lease", allow_module_level=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
prepare_environment()

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
SWA = torch.ops.custom_op.custom_deepseek_v41_swa_pack_write_bf16_gaudi2
FP4 = torch.ops.custom_op.custom_deepseek_v41_fp4_cache_write_bf16_gaudi2
ATTN = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_ordered_lengths_gaudi2
CACHE_ATTN = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_ordered_cache_lengths_gaudi2
EVIDENCE = Path(os.environ["DSV41_RUN_EVIDENCE"])


@pytest.mark.parametrize("main_rows,write_compressed", ((0,False),(256,False),(256,True),(512,True)))
def test_selected_valid_only_matches_zero_filled_consumer(main_rows, write_compressed):
    torch._dynamo.reset()
    torch.manual_seed(4121 + main_rows)
    swa = pack_swa(torch.randn(512,512).bfloat16()).to("hpu")
    main = pack_fp4(torch.randn(main_rows,512).bfloat16(),16).to("hpu") if main_rows else swa
    index = pack_fp4(torch.randn(max(main_rows,1),128).bfloat16(),32).to("hpu")
    sink = torch.randn(32,dtype=torch.float32,device="hpu")
    scale = torch.tensor([512**-.5],device="hpu")

    def program(q,kv,main_value,index_value,position,slot,ids,lengths):
        swa_done = SWA(swa,kv,position)
        if write_compressed:
            fp4_done = FP4(main,index,main_value,index_value,slot)
            expected = CACHE_ATTN(q,swa,main,ids,sink,scale,lengths,swa_done,fp4_done,False)
            actual = CACHE_ATTN(q,swa,main,ids,sink,scale,lengths,swa_done,fp4_done,True)
        else:
            expected = ATTN(q,swa,main,ids,sink,scale,lengths,swa_done,False)
            actual = ATTN(q,swa,main,ids,sink,scale,lengths,swa_done,True)
        return expected,actual

    compiled = torch.compile(program,backend="hpu_backend",fullgraph=True,dynamic=False)
    results = []
    for generation,pos in enumerate((0,1,127,128,255,256,510,511,0)):
        window = torch.arange(pos-127,pos+1,dtype=torch.int32)
        ids = torch.where(window >= 0,window,-1)
        visible = min(pos+1,main_rows)
        if main_rows:
            rows = torch.arange(512,dtype=torch.int32)
            ids = torch.cat((ids,torch.where(rows < visible,rows+512,-1)))
        if generation == 0:
            ids.fill_(-1)
        elif generation % 3 == 0:
            ids[:8] = torch.tensor([0,0,-1,0,511,-1,511,0],dtype=torch.int32)
        position = torch.tensor([pos],dtype=torch.int32,device="hpu")
        slot = torch.tensor([min(pos,max(main_rows,1)-1)],dtype=torch.int32,device="hpu")
        lengths = torch.tensor([128+visible if main_rows else 128],dtype=torch.int32,device="hpu")
        values = [torch.randn(1,w).bfloat16().to("hpu") for w in (512,512,128)]
        query = torch.randn(1,32,512).bfloat16().to("hpu")
        expected,actual = compiled(query,*values,position,slot,ids.unsqueeze(0).to("hpu"),lengths)
        expected,actual = expected.cpu(),actual.cpu()
        mismatch = int((expected.view(torch.int16) != actual.view(torch.int16)).sum())
        results.append({"generation":generation,"position":pos,"mismatches":mismatch})
        if mismatch:
            torch.save({"expected":expected,"actual":actual,"indices":ids,"swa":swa.cpu()},
                       EVIDENCE/f"selected-{main_rows}-{write_compressed}-{generation}.pt")
    (EVIDENCE/f"selected-{main_rows}-{write_compressed}.json").write_text(json.dumps(results,indent=2)+"\n")
    assert all(r["mismatches"] == 0 for r in results),results
    torch._dynamo.reset()
