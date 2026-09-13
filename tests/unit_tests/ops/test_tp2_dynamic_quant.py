# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os

import pytest
import torch


@pytest.fixture(scope="module")
def quant():
    path = os.environ.get("TP2_QUANT_TEST_BRIDGE")
    if not path:
        pytest.skip("requires the isolated TP2 native adapter")
    import habana_frameworks.torch  # noqa: F401

    spec = importlib.util.spec_from_file_location("tp2_fused_ar_norm_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return torch.ops.custom_op.tp2_dynamic_quant


@pytest.mark.parametrize("width", [3072, 5120, 8704, 17408])
def test_native_quant_meta_preserves_shape_and_scale_type(quant, width):
    x = torch.empty((1, width), dtype=torch.bfloat16, device="meta")
    values, scales = quant(x)
    assert values.shape == x.shape and values.dtype == torch.float8_e4m3fn
    assert scales.shape == (1, 1) and scales.dtype == torch.float32


@pytest.mark.parametrize("kind", ["batch", "width", "dtype", "stride", "gradient"])
def test_native_quant_rejects_unsupported_contract_before_submission(quant, kind):
    shape = (2, 5120) if kind == "batch" else (1, 5130) if kind == "width" else (1, 5120)
    x = torch.empty(shape,
                    device="meta",
                    dtype=torch.float32 if kind == "dtype" else torch.bfloat16,
                    requires_grad=kind == "gradient")
    if kind == "stride":
        x = torch.empty((1, 10240), device="meta", dtype=torch.bfloat16)[:, ::2]
    with pytest.raises(RuntimeError, match="native TP2 quant requires"):
        quant(x)
