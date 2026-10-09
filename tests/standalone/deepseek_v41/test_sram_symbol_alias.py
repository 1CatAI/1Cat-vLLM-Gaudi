# SPDX-License-Identifier: Apache-2.0
"""A reshape alias preserves a decoded operand; an offset slice does not."""
from tools.audit_deepseek_v41_sram import audit


def symbol(name, op, attributes):
    attrs = ''.join(f'  attr {{ key: "{k}" value {{ s: "{v}" }} }}\n' for k, v in attributes.items())
    return f'\nnode {{\n  name: "{name}"\n  op: "{op}"\n{attrs}}}\n'


def tensor(name, alias=''):
    return f'{name}  |  Sizes = [128,256]  |  sizeInBytes = 32768  |  {alias}location = in SRAM  |  '


def test_zero_offset_decoder_reshape_has_one_real_consumer(tmp_path):
    graph = symbol('decode', 'custom_deepseek_v41_expert_n256_sat_fp8_gaudi2', {
        'outputTensor:0': tensor('decoded3d', 'isAliased = matrix2d, type = alias, offset: 0  |  ')})
    graph += symbol('consume', 'GEMM', {'inputTensor:0': tensor('x'),
                                       'inputTensor:1': tensor('matrix2d'), 'outputTensor:0': tensor('y')})
    path = tmp_path / 'PostGraph.pbtxt'
    path.write_text(graph)
    result = audit(path)
    assert result['expert_mme_node_count'] == 1
    assert result['all_decoded_weights_consumed_once']
    assert result['all_mme_weights_in_sram']


def test_nonzero_alias_does_not_match_another_slice(tmp_path):
    graph = symbol('decode', 'custom_deepseek_v41_expert_n256_sat_fp8_gaudi2', {
        'outputTensor:0': tensor('decoded', 'isAliased = base, type = slice, offset: 32768  |  ')})
    graph += symbol('consume', 'GEMM', {'inputTensor:0': tensor('x'),
                                       'inputTensor:1': tensor('base'), 'outputTensor:0': tensor('y')})
    path = tmp_path / 'PostGraph.pbtxt'
    path.write_text(graph)
    result = audit(path)
    assert result['expert_mme_node_count'] == 0
    assert not result['all_decoded_weights_consumed_once']
