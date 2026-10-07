# SPDX-License-Identifier: Apache-2.0
from tools.audit_deepseek_v41_physical_nodes import audit


def test_logical_strided_alias_is_excluded_but_materialization_is_counted(tmp_path):
    graph = tmp_path / 'PostGraph-symbol.pbtxt'
    graph.write_text('''node {
  name: "decoder"
  op: "custom_decoder"
}
node {
  name: "weight_view"
  op: "StridedView"
}
node {
  name: "copy_sparse_view"
  op: "DMA"
}
node {
  name: "matrix_consumer"
  op: "BatchGemm"
}
''')
    result = audit(graph)
    assert result['physical_nodes'] == 3
    assert result['logical_nodes_excluded'] == {'StridedView': 1}
    assert result['operations']['DMA'] == 1
