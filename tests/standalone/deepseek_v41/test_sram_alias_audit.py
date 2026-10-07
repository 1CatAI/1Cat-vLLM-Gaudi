# SPDX-License-Identifier: Apache-2.0
"""Only same-size, zero-offset logical views may connect an SRAM producer."""
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location('sram_audit', Path(__file__).parents[3] / 'tools/audit_deepseek_v41_sram.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def graph(tmp_path, *, offset=0, consumer_size=256, consumer_location='SRAM'):
    text = '''
node {
  name: "decode"
  op: "custom_deepseek_v41_expert_token_wide3_sat_fp8_gaudi2"
  attr { key: "outputTensor:0" value { s: "weight3d  |  Sizes = [1,16,16]  |  sizeInBytes = 256  |  isAliased = weight2d, type = alias, offset: OFFSET  |  location = in SRAM  |  " } }
}
node {
  name: "mme"
  op: "GEMM"
  attr { key: "inputTensor:0" value { s: "x  |  Sizes = [1,16]  |  sizeInBytes = 16  |  location = in SRAM  |  " } }
  attr { key: "inputTensor:1" value { s: "weight2d  |  Sizes = [16,16]  |  sizeInBytes = CSIZE  |  location = in CLOC  |  " } }
}
'''.replace('OFFSET', str(offset)).replace('CSIZE', str(consumer_size)).replace('CLOC', consumer_location)
    path = tmp_path / 'graph.pbtxt'
    path.write_text(text)
    return audit_module.audit(path)


def test_same_size_alias_proves_one_consumer(tmp_path):
    result = graph(tmp_path)
    assert result['all_decoded_weights_in_sram']
    assert result['all_mme_weights_in_sram']
    assert result['all_decoded_weights_consumed_once']


def test_partial_alias_is_not_full_consumption_proof(tmp_path):
    assert not graph(tmp_path, offset=16)['all_decoded_weights_consumed_once']
    assert not graph(tmp_path, consumer_size=128)['all_decoded_weights_consumed_once']


def test_dram_view_is_not_sram_proof(tmp_path):
    assert not graph(tmp_path, consumer_location='DRAM')['all_mme_weights_in_sram']
