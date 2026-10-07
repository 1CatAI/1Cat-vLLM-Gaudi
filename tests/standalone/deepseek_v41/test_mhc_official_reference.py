# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tools.deepseek_v41_mhc_reference import normalized_error, gate_reference, post_reference, routing_error_by_expert


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


def test_routing_weight_comparison_aligns_each_token_by_expert():
    ids = torch.tensor([[263, 175, 159, 132, 124, 286], [3, 4, 5, 6, 7, 8]], dtype=torch.int32)
    weights = torch.tensor([[.32, .25, .20, .10, .07, .06], [.20, .19, .18, .17, .14, .12]])
    order = torch.tensor([[0, 1, 2, 3, 5, 4], [5, 3, 1, 4, 2, 0]])
    result = routing_error_by_expert(ids, weights, ids.gather(1, order), weights.gather(1, order))
    assert result == dict(same_expert_set=True, order_equal=False, error=0)


def test_same_experts_does_not_hide_incorrect_routing_weights():
    ids = torch.tensor([[12, 24, 48]])
    weights = torch.tensor([[.2, .3, .5]])
    result = routing_error_by_expert(ids, weights, ids.flip(1), weights)
    assert result['same_expert_set'] and not result['order_equal']
    assert result['error'] > 5e-5


@pytest.mark.parametrize('actual', ([[12, 24, 49]], [[12, 24, 24]]))
def test_changed_or_duplicate_expert_set_is_rejected(actual):
    ids, weights = torch.tensor([[12, 24, 48]]), torch.tensor([[.2, .3, .5]])
    result = routing_error_by_expert(ids, weights, torch.tensor(actual), weights)
    assert not result['same_expert_set'] and result['error'] == float('inf')


def test_routing_comparison_rejects_invalid_shapes_and_noninteger_ids():
    ids, weights = torch.tensor([[1, 2]]), torch.tensor([[.4, .6]])
    with pytest.raises(ValueError):
        routing_error_by_expert(ids, weights, ids, weights[:, :1])
    with pytest.raises(ValueError):
        routing_error_by_expert(ids.float(), weights, ids, weights)
