# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported shared-main operands without acquiring an HPU."""
import os

import pytest
import torch


@pytest.fixture(scope='module')
def native_meta():
    if os.getenv('DSV41_TEST_NATIVE_META') != '1':
        pytest.skip('Requires the built native library; all tensors stay on Meta')
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands(ratio=1):
    def tensor(shape, dtype):
        return torch.empty(shape, device='meta', dtype=dtype)

    return [tensor((1, 16, 512), torch.bfloat16), tensor((256, 528), torch.uint8),
            tensor((1024, 288), torch.uint8), tensor((1, 512), torch.int32),
            tensor((1,), torch.int32), tensor((8192,), torch.int32), tensor((16,), torch.float32),
            tensor((1,), torch.float32), tensor((1,), torch.int32), ratio]


@pytest.mark.parametrize('ratio', [1, 2])
def test_shared_main_meta_outputs_remain_distinct(native_meta, ratio):
    args = operands(ratio)
    output, rows, mask = torch.ops.custom_op.custom_deepseek_v41_main_publish_mla_gaudi2(*args)
    assert output.shape == (1, 16, 512) and output.dtype == torch.bfloat16
    assert rows.shape == (1, 640, 512) and rows.dtype == torch.bfloat16
    assert mask.shape == (1, 640) and mask.dtype == torch.float32
    reused = torch.ops.custom_op.custom_deepseek_v41_main_reuse_mla_gaudi2(
        args[0], args[1], rows, mask, args[4], args[6], args[7], args[8])
    assert reused.shape == output.shape and reused.dtype == output.dtype


@pytest.mark.parametrize('index,shape,dtype', [
    (0, (2, 16, 512), torch.bfloat16), (0, (1, 16, 512), torch.float32),
    (1, (128, 528), torch.uint8), (2, (0, 288), torch.uint8),
    (3, (1, 511), torch.int32), (4, (1,), torch.int64),
    (5, (0,), torch.int32), (5, (8193,), torch.int32), (6, (32,), torch.float32),
])
def test_publisher_rejects_invalid_contract(native_meta, index, shape, dtype):
    args = operands()
    args[index] = torch.empty(shape, device='meta', dtype=dtype)
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_main_publish_mla_gaudi2(*args)


@pytest.mark.parametrize('ratio', [0, 3])
def test_publisher_rejects_unsupported_ratio(native_meta, ratio):
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_main_publish_mla_gaudi2(*operands(ratio))


@pytest.mark.parametrize('bad_rows', [True, False])
def test_reuse_rejects_short_rows_or_nonfloat_mask(native_meta, bad_rows):
    args = operands()
    rows = torch.empty((1, 512 if bad_rows else 640, 512), device='meta', dtype=torch.bfloat16)
    mask = torch.empty((1, 640), device='meta', dtype=torch.float32 if bad_rows else torch.bfloat16)
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_main_reuse_mla_gaudi2(
            args[0], args[1], rows, mask, args[4], args[6], args[7], args[8])
