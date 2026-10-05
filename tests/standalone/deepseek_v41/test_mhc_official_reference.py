# SPDX-License-Identifier: Apache-2.0
import torch
from tools.deepseek_v41_mhc_reference import normalized_error, gate_reference, post_reference


def test_metric_matches_upstream_and_rejects_nonfinite():
    a = torch.tensor([1., -2., 3.])
    b = torch.tensor([1.001, -2.002, 3.001])
    expected = 1 - 2*(a.double()*b.double()).sum()/(a.double().square()+b.double().square()).sum()
    assert abs(normalized_error(a, b)-float(expected)) < 1e-15
    assert normalized_error(torch.zeros(2), torch.zeros(2)) == 0
    assert normalized_error(torch.tensor([float('nan')]), torch.ones(1)) == float('inf')


def test_zero_control_has_symmetric_mixing_and_bf16_boundaries():
    residual = torch.arange(32).reshape(1,4,8).bfloat16()/32
    weights = torch.zeros(24,32)
    scale, base = torch.ones(3), torch.zeros(24)
    gates = gate_reference(residual, weights, scale, base, 1e-6)
    assert torch.equal(gates[:, :4], torch.full((1,4), .500001))
    assert torch.equal(gates[:, 4:8], torch.ones(1,4))
    assert torch.allclose(gates[:, 8:], torch.full((1,16), .25), atol=1e-6)
    result = post_reference(torch.zeros(4,1,8).bfloat16(),residual,weights,scale,base,torch.ones(8),1e-6)
    assert result[0].dtype == result[1].dtype == result[3].dtype == torch.bfloat16
    assert torch.equal(result[0][:,0],result[0][:,3])
