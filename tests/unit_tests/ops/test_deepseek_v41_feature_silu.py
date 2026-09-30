# SPDX-License-Identifier: Apache-2.0
"""TP4 intermediate-width and row/scale ownership contracts without HPU allocation."""
import os

import pytest
import torch


@pytest.fixture(scope='module')
def operation():
    if os.getenv('DSV41_TEST_NATIVE_META') != '1':
        pytest.skip('Requires explicit built native artifact; no device allocation')
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    return torch.ops.custom_op.custom_deepseek_v41_silu_quant_feature_gaudi2


def args(rows=2, scale_rows=1):
    return [torch.empty(rows, 1, 1280, dtype=torch.float32, device='meta'),
            torch.empty(1, rows, dtype=torch.int32, device='meta'),
            torch.empty(scale_rows, 1, dtype=torch.float32, device='meta'),
            torch.empty(384, 5, 256, dtype=torch.bfloat16, device='meta'),
            torch.empty(1, rows, dtype=torch.float32, device='meta')]


@pytest.mark.parametrize('rows', [1, 2, 6, 12, 36])
@pytest.mark.parametrize('broadcast', [True, False])
def test_quant_scale_shapes(operation, rows, broadcast):
    q, scale = operation(*args(rows, 1 if broadcast else rows))
    assert q.shape == (rows, 1, 640) and q.dtype == torch.float8_e4m3fn
    assert scale.shape == (rows, 1, 1) and scale.dtype == torch.float32


@pytest.mark.parametrize('rows,scale_rows', [(0, 1), (37, 1), (2, 3)])
def test_reject_rows_and_scale_mismatch(operation, rows, scale_rows):
    with pytest.raises(RuntimeError):
        operation(*args(rows, scale_rows))


@pytest.mark.parametrize('index', range(5))
def test_reject_wrong_operand_type(operation, index):
    values = args()
    values[index] = values[index].to(torch.float64)
    with pytest.raises(RuntimeError):
        operation(*values)


def test_rejects_non_tp4_width(operation):
    values = args()
    values[0] = torch.empty(2, 1, 2304, device='meta')
    with pytest.raises(RuntimeError, match='1280'):
        operation(*values)


def moe_args(tokens=1, intermediate=640):
    def t(shape, dtype):
        return torch.empty(shape, dtype=dtype, device='meta')
    return [t((tokens, 5120), torch.bfloat16), t((tokens, 6), torch.int32), t((tokens, 6), torch.float32),
            t((17, intermediate * 2 // 256, 5120 * 64), torch.int16),
            t((17, 20, intermediate * 64), torch.int16),
            t((17, intermediate * 2 // 256, 5120 * 8), torch.int16),
            t((17, 20, intermediate * 8), torch.int16), t((128,), torch.bfloat16),
            t((17, intermediate * 2 // 256, 256), torch.bfloat16), t((17, 20, 256), torch.bfloat16),
            t((tokens, 5120), torch.float8_e4m3fn), t((tokens, 1), torch.float32),
            t((tokens, 5120), torch.bfloat16), True]


@pytest.mark.parametrize('tokens,width,valid', [(1, 640, True), (2, 640, False), (1, 1152, False)])
def test_full_moe_meta_preserves_tp4_c1_scope(operation, tokens, width, valid):
    full = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_prequant_shared_feature_silu_fp8_gaudi2
    if valid:
        output = full(*moe_args(tokens, width))
        assert output.shape == (1, 5120) and output.dtype == torch.bfloat16
    else:
        with pytest.raises(RuntimeError):
            full(*moe_args(tokens, width))
