# SPDX-License-Identifier: Apache-2.0
"""Shape and numerical-contract guards of the opt-in projection graph adapter."""

import operator
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_gaudi.ops import flashinfer_projection_fusion as fusion


def _fake_op(*args):
    return args


def _graph(violation=None):
    graph = torch.fx.Graph()

    def metadata(node, shape, dtype):
        node.meta.update(output_shapes=[torch.Size(shape)],
                         output_dtypes=[dtype],
                         output_device=torch.device("hpu"),
                         output_contiguous=[True],
                         output_offset=[0],
                         val=torch.empty((), dtype=dtype))
        return node

    batch, width, projection = (1 if violation == "batch" else 8), 5120, 34816
    x = metadata(graph.placeholder("x"), (batch, width), torch.bfloat16)
    residual = metadata(graph.placeholder("residual"), (batch, width), torch.bfloat16)
    gamma = metadata(graph.placeholder("gamma"), (width, ), torch.bfloat16)
    weight = metadata(graph.placeholder("weight"), (projection, width), torch.float8_e4m3fn)
    weight_scale = metadata(graph.placeholder("weight_scale"), (projection, ), torch.float32)
    summed = metadata(graph.call_function(torch.ops.aten.add.Tensor, (x, residual)), (batch, width), torch.bfloat16)
    norm = graph.call_function(_fake_op, (summed, gamma, 1e-6))
    norm.meta["op_name"] = "hpu.rms_norm"
    norm.meta["output_offset"] = [0, 0]
    normed = metadata(graph.call_function(operator.getitem, (norm, 0)), (batch, width), torch.bfloat16)
    if violation in ("identity_view", "reshape_rank"):
        shape = (batch, width) if violation == "identity_view" else (1, batch, width)
        normed = metadata(graph.call_function(torch.ops.aten.view.default, (normed, shape)), shape, torch.bfloat16)
    cguid = graph.call_function(_fake_op, (normed, 2, 0, -1, True, 240.))
    cguid.meta["op_name"] = "hpu.calculate_scale_for_cast"
    scale_add = graph.call_function(torch.ops.aten.add.Tensor, (cguid, 1e-8 / 240.))
    reciprocal = graph.call_function(torch.ops.aten.reciprocal.default, (scale_add, ))
    inverse = graph.call_function(torch.ops.aten.mul.Tensor, (reciprocal, 1.0))
    cast = graph.call_function(_fake_op, (normed, inverse, False, False, torch.float8_e4m3fn))
    cast.meta["op_name"] = "hpu.cast_to_fp8_v2"
    q = metadata(graph.call_function(operator.getitem, (cast, 0)), (batch, width), torch.float8_e4m3fn)
    scale = metadata(graph.call_function(torch.ops.aten._to_copy.default, (scale_add, ), {"dtype": torch.float32}),
                     (batch, 1), torch.float32)
    gemm = metadata(graph.call_function(_fake_op, (q, False, weight, True, None, torch.bfloat16, scale, weight_scale)),
                    (batch, projection), torch.bfloat16)
    gemm.meta["op_name"] = "hpu.fp8_gemm_v2"
    output = (gemm, summed, normed) if violation == "extra_user" else (gemm, summed)
    graph.output(output)
    if violation == "gamma_dtype":
        gamma.meta["output_dtypes"] = [torch.float32]
    elif violation == "cpu":
        residual.meta["output_device"] = torch.device("cpu")
    elif violation == "strided":
        x.meta["output_contiguous"] = [False]
    elif violation == "metadata":
        x.meta.clear()
    elif violation == "grad":
        gamma.meta["val"] = torch.empty((), requires_grad=True)
    elif violation == "epsilon":
        norm.args = (summed, gamma, 1e-5)
    elif violation == "range":
        cguid.args = (normed, 2, 0, -1, True, 448.)
    elif violation == "stochastic":
        cast.args = (normed, inverse, False, True, torch.float8_e4m3fn)
    elif violation == "ordinary":
        cguid.meta["op_name"] = "aten.amax"
    elif violation == "bias":
        gemm.args = (*gemm.args, gamma)
    return torch.fx.GraphModule({}, graph)


@pytest.mark.parametrize("violation", [
    None, "identity_view", "reshape_rank", "batch", "extra_user", "gamma_dtype", "cpu", "strided", "metadata", "grad",
    "epsilon", "range", "stochastic", "ordinary", "bias"
])
def test_only_exclusive_qualified_cguid_projection_rewrites(violation):
    module = _graph(violation)
    before = module.code
    original_target = fusion._target
    with mock.patch.object(
            fusion,
            "_target",
            side_effect=lambda node, name:
        (node.meta.get("op_name") == name
         if isinstance(node, torch.fx.Node) and "op_name" in node.meta else original_target(node, name))), mock.patch(
             "flashinfer_gaudi._native.add_rmsnorm_quant_op", return_value=SimpleNamespace(default=_fake_op)):
        count = fusion.fuse_projection_graph(module, {(8, 5120, 34816)})
        expected = violation in (None, "identity_view")
        assert count == int(expected)
        module.graph.lint()
        if not expected:
            assert module.code == before
        else:
            assert "reciprocal" not in module.code and "_to_copy" not in module.code
            assert len([node for node in module.graph.nodes if node.target is _fake_op]) == 2
            fused = next(node for node in module.graph.nodes if node.target is _fake_op)
            assert fused.meta["output_offset"] == [0, 0, 0, 0]
            assert fusion.fuse_projection_graph(module, {(8, 5120, 34816)}) == 0


def test_no_default_promotion_and_unqualified_shape_rejected():
    module = _graph()
    before = module.code
    assert fusion.fuse_projection_graph(module, ()) == 0
    fusion.register_projection_fusion_pass()
    assert module.code == before
    with pytest.raises(ValueError, match="Unqualified"):
        fusion.fuse_projection_graph(module, {(32, 5120, 34816)})
    assert not fusion.projection_fusion_stats()["production_default"]
