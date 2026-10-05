# SPDX-License-Identifier: Apache-2.0
from tools.audit_deepseek_v41_physical_nodes import logical_stages


def test_only_fragments_of_same_producer_collapse():
    names = ['moe/batch_gemm/24_bundle_0/op_2_gemm',
             'moe/batch_gemm/24_bundle_0/op_6_gemm',
             'moe/batch_gemm/24_bundle_0/op_9_gemm',
             'moe/gemm/33_bundle_1/op_2', 'moe/gemm/39_bundle_2/op_2']
    nodes = [dict(name=name, op='GEMM', execution_index=i) for i, name in enumerate(names)]
    result = logical_stages(nodes)
    assert result['logical_stage_count'] == 3
    assert result['pipeline_fragment_excess'] == 2


def test_independent_small_kernels_not_merged_by_guid():
    nodes = [dict(name=f'cast/{i}', op='cast_i64_to_i32', execution_index=i) for i in range(4)]
    assert logical_stages(nodes)['logical_stage_count'] == 4


def test_unsliced_names_are_retained():
    nodes = [dict(name='fusedTPCNode_1_0', op='fused_kernel_123', execution_index=7)]
    assert logical_stages(nodes)['stages'][0]['source'] == 'fusedTPCNode_1_0'
