# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy

import pytest

from tools.deepseek_v41_dma_prefetch_audit import dense_dma_proof


def graph():
    def tensor(name, location):
        return f'{name}  | Sizes = [2048,5120], sizeInBytes = 10485760, location = in {location}'

    return dict(graph='compiled', sha256='hash', physical_nodes=2, nodes=[
        dict(name='prefetch', op='DmaMemcpy', tensors={
            'inputTensor:0': tensor('checkpoint', 'DRAM'), 'outputTensor:0': tensor('weight', 'SRAM')}),
        dict(name='projection', op='GEMM', tensors={'inputTensor:1': tensor('weight', 'SRAM')})])


def test_real_dma_sram_consumer_is_required():
    proof = dense_dma_proof(graph(), 10485760)
    assert proof['qualified'] and proof['edges'][0]['operand'] == 1


@pytest.mark.parametrize('change', ['tpc', 'dram', 'no_consumer', 'wrong_size'])
def test_rejected_structures_cannot_reach_timing(change):
    value = deepcopy(graph())
    if change == 'tpc':
        value['nodes'][0]['op'] = 'custom_weight_copy'
    elif change == 'dram':
        key = 'outputTensor:0'
        value['nodes'][0]['tensors'][key] = value['nodes'][0]['tensors'][key].replace('SRAM', 'DRAM')
    elif change == 'no_consumer':
        value['nodes'].pop()
    else:
        assert not dense_dma_proof(value, 1024)['qualified']
        return
    assert not dense_dma_proof(value, 10485760)['qualified']
