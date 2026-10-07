# SPDX-License-Identifier: Apache-2.0
"""CPU scale qualification: SAT never consumes unqualified nonzero padding."""
import numpy as np
import pytest
from vllm_gaudi.ops.deepseek_v41_expert_n256 import saturated_decode_eligible

@pytest.mark.parametrize('compact', [False, True])
def test_qualified_range_and_padded_suffix(compact):
    codes = np.full((1, 4, 256), 127, np.uint8)
    if compact:
        codes[:, -1] = 0
        planes = np.concatenate((codes, np.full((1, 1, 256), 127, np.uint8)), axis=1).view('<i2').reshape(1,640)
    else:
        offsets = np.full_like(codes, 0)
        offsets[:, -1] = 56
        planes = np.concatenate((codes, offsets), axis=-1).view('<i2').reshape(1,1024)
    assert not saturated_decode_eligible(planes)
    assert saturated_decode_eligible(planes, active_k=96)
    with pytest.raises(ValueError):
        saturated_decode_eligible(planes, active_k=95)
