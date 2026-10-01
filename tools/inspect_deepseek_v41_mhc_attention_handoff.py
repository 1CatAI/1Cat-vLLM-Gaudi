# SPDX-License-Identifier: Apache-2.0
"""Observe the existing FP32 attention collapse, preserving original outputs."""
import json
import os
from pathlib import Path

os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment  # noqa: E402
prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import hc_post, rms_norm, quantize_activation  # noqa: E402

bind_worker_cpu(0)
torch.hpu.set_device(0)
bind_worker_helpers(0)
torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
root = Path(os.environ['DSV41_RUN_EVIDENCE'])
fixture = torch.load(os.environ['DSV41_MHC_ROUNDING_FIXTURE'], weights_only=True)


def observe(value, residual, post, comb, pre, weight, projection):
    updated = hc_post(value, residual, post, comb)
    updated = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(updated.reshape(1, -1)).reshape_as(updated)
    collapsed = (updated.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
    norm = rms_norm(collapsed, weight, 1e-6)
    quant = quantize_activation(norm)
    projected = F.linear(quant, projection)
    return updated, projected, collapsed.float(), norm, quant


with torch.inference_mode():
    function = torch.compile(observe, backend='hpu_backend', fullgraph=True, dynamic=False)
    values = [x.cpu() for x in function(*(x.to('hpu') for x in fixture['args']))]
    exact = [torch.equal(a, b) for a, b in zip(values[:2], fixture['expected'], strict=True)]
    torch.save(dict(original_outputs_exact=exact, collapsed=values[2], norm=values[3], quant=values[4],
                    updated=values[0], args=fixture['args']), root/'observed.pt')
    result = dict(original_outputs_exact=exact, scope='FP32 intermediate observation only; no timing')
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    assert all(exact), exact
    print('Original residual and QKV outputs unchanged; captured attention FP32 collapse', flush=True)
