# SPDX-License-Identifier: Apache-2.0
"""Do not mistake symbolic shape metadata for physical SRAM placement."""
import pytest

from tools.analyze_deepseek_v41_expert_placement import audit_graph


def graph(allocation):
    return dict(name="expert-chain", recipe_debug_id=7,
                tensors=[dict(name="decoded", dtype_bit_size=8, max_shape=[256, 128, 2],
                              allocation=allocation, persistent=False)],
                nodes=[dict(name="decode", guid="custom_expert_sat_fp8", engine="TPC",
                            input_tensors=[], output_tensors=["decoded"], bundle_index=3),
                       dict(name="consume", guid="batch_gemm", engine="MME",
                            input_tensors=["decoded"], output_tensors=[], bundle_index=3)])


@pytest.mark.parametrize("allocation", ["SRAM", "DRAM"])
def test_physical_handoff_preserves_consumer_and_allocation(allocation):
    result = audit_graph(graph(allocation))
    assert result["all_decoded_operands_in_sram"] == (allocation == "SRAM")
    assert result["allocation_bytes"][allocation] == 65536
    assert result["decoded_operands"][0]["consumers"][0]["engine"] == "MME"
    assert not result["traffic_measured"]


def test_missing_physical_metadata_is_not_sram_proof():
    value = graph("SRAM")
    del value["tensors"][0]["allocation"]
    with pytest.raises(ValueError, match="Physical allocation"):
        audit_graph(value)
