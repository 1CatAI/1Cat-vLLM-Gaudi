# SPDX-License-Identifier: Apache-2.0
"""Paged producer mutation/output contract for changing C1-C6 row buckets."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_PAGED_DECODED_LIBRARY")
    if not path:
        pytest.skip("Set the independently built decoded publication registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def inputs(rows=6, physical_rows=1048704, logical_rows=32768):
    def tensor(shape, dtype):
        return torch.empty(shape, device="meta", dtype=dtype)
    return [tensor((physical_rows, 288), torch.uint8), tensor((physical_rows, 68), torch.uint8),
            tensor((rows, 512), torch.bfloat16), tensor((rows, 128), torch.bfloat16),
            tensor((rows,), torch.int32), tensor((rows,), torch.int32),
            tensor((logical_rows, 512), torch.bfloat16)]


@pytest.mark.parametrize("rows", (1, 2, 6))
def test_mutation_and_completion_geometry(rows):
    op = torch.ops.custom_op.custom_deepseek_v41_fp4_paged_decoded_rows_gaudi2
    values = inputs(rows)
    output = op(*values)
    assert output.shape == (rows, 36) and output.dtype == torch.int32
    assert output.device.type == "meta"
    aliases = [arg.alias_info.is_write for arg in op.default._schema.arguments if arg.alias_info]
    assert aliases == [True, True, True]
    assert op.default._schema.returns[0].alias_info is None


@pytest.mark.parametrize("kind", ("tokens", "coordinates", "mirror"))
def test_reject_incompatible_state_geometry(kind):
    values = inputs(7 if kind == "tokens" else 6)
    if kind == "coordinates":
        values[5] = torch.empty((1,), device="meta", dtype=torch.int32)
    if kind == "mirror":
        values[6] = torch.empty((32769, 512), device="meta", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="Paged decoded publication"):
        torch.ops.custom_op.custom_deepseek_v41_fp4_paged_decoded_rows_gaudi2(*values)


def test_functionalized_fake_inputs_use_meta_dispatch():
    # Directly invoking the HPU implementation here would try to execute
    # fake/meta storage as a CPU tensor. The ordered dispatcher must resolve
    # Meta before a real device implementation is reached.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        values = [torch.empty(v.shape, dtype=v.dtype) for v in inputs()]
        fn = torch.func.functionalize(
            lambda *args: torch.ops.custom_op.custom_deepseek_v41_fp4_paged_decoded_rows_gaudi2(*args))
        completion = fn(*values)
        assert completion.shape == (6, 36) and completion.dtype == torch.int32
