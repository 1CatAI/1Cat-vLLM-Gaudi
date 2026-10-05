# SPDX-License-Identifier: Apache-2.0
"""Require a compiled DMA -> SRAM weight -> MME edge before timing."""
from tools.audit_deepseek_v41_sram import tensor_info


def dense_dma_proof(graph, weight_bytes):
    copies, matrices = [], []
    for node in graph['nodes']:
        tensors = {key: tensor_info(value) for key, value in node['tensors'].items()}
        output = tensors.get('outputTensor:0', {})
        source = tensors.get('inputTensor:0', {})
        if (node['op'] == 'DmaMemcpy' and output.get('bytes') == weight_bytes
                and source.get('bytes') == weight_bytes and source.get('location') == 'DRAM'
                and output.get('location') == 'SRAM'):
            copies.append((node, output))
        if 'gemm' in node['op'].lower():
            for index in (0, 1):
                operand = tensors.get(f'inputTensor:{index}', {})
                if operand.get('location') == 'SRAM' and operand.get('bytes') == weight_bytes:
                    matrices.append((node, operand, index))
    edges = [dict(dma=copy['name'], consumer=matrix['name'], operand=index, bytes=weight_bytes)
             for copy, output in copies for matrix, operand, index in matrices
             if output['name'] in (operand['name'], operand.get('alias'))]
    return dict(graph=graph['graph'], sha256=graph['sha256'], edges=edges,
                qualified=bool(edges), physical_nodes=graph['physical_nodes'])
