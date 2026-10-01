# SPDX-License-Identifier: Apache-2.0
"""Observe the already materialized FP32 mix to establish its reduction order."""
import json
import os
from pathlib import Path

os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment  # noqa: E402
prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers  # noqa: E402

bind_worker_cpu(0)
torch.hpu.set_device(0)
bind_worker_helpers(0)
torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
root = Path(os.environ['DSV41_RUN_EVIDENCE'])
source = Path(os.environ['DSV41_MHC_ROUNDING_FIXTURE'])
fixture = torch.load(source, weights_only=True)


def observe(value, residual, post, comb, pre, weight):
    mixed = (comb.unsqueeze(-1) * residual.float().unsqueeze(2)).sum(1)
    updated = (value.float().unsqueeze(1) * post.unsqueeze(-1) + mixed).bfloat16()
    collapsed = (updated.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
    norm, quant, scale = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(collapsed, weight, 1e-6)
    return updated, collapsed, norm, quant.view(torch.uint8), scale, mixed


with torch.inference_mode():
    function = torch.compile(observe, backend='hpu_backend', fullgraph=True, dynamic=False)
    values = [x.cpu() for x in function(*(x.to('hpu') for x in fixture['args']))]
    exact = [torch.equal(a, b) for a, b in zip(values[:5], fixture['expected'], strict=True)]
    torch.save(dict(original_outputs_exact=exact, mixed=values[-1], args=fixture['args']), root/'observed.pt')
    (root/'result.json').write_text(json.dumps(dict(original_outputs_exact=exact,
                                                   scope='FP32 intermediate observation only; no timing'), indent=2)+'\n')
    assert all(exact), exact
    print('Original five outputs unchanged; captured FP32 mix', flush=True)
