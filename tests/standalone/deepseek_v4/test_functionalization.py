# SPDX-License-Identifier: Apache-2.0
"""Alias and mutation tests for the optional HPU functionalization lowering."""

import pytest
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2
from torch.fx.experimental.proxy_tensor import make_fx

from vllm_gaudi.compilation.functionalization import lower_functionalized


@torch.library.custom_op("dsv4_af_test::write", mutates_args={"out"})
def write(x: torch.Tensor, out: torch.Tensor) -> None:
    out.copy_(x + 1)


@write.register_fake
def write_fake(x, out):
    return None


@torch.library.custom_op("dsv4_af_test::partial", mutates_args={"out"})
def partial(x: torch.Tensor, out: torch.Tensor) -> None:
    out[0].copy_(x[0] + 1)


@partial.register_fake
def partial_fake(x, out):
    return None


WRITE = torch.ops.dsv4_af_test.write.default
PARTIAL = torch.ops.dsv4_af_test.partial.default


def call(op, x, out):
    return auto_functionalized_v2(op, x=x, _out_base_index=0, _all_bases=[out])[1]


@pytest.mark.parametrize("reinplace", [False, True])
@pytest.mark.parametrize("case", ["fresh", "live", "input", "alias", "partial", "strided", "view"])
def test_alias_and_output_semantics(case, reinplace):
    def function(x):
        base = x if case in ("input", "alias") else torch.zeros_like(x)
        if case in ("alias", "strided"):
            base = base.t()
        if case == "view":
            return auto_functionalized_v2(
                WRITE, x=x[:1], _out_base_index=0, _out_size=[1, 4],
                _out_stride=[4, 1], _out_storage_offset=0, _all_bases=[base],
            )[1]
        source = x.t() if case in ("alias", "strided") else x
        result = call(PARTIAL if case == "partial" else WRITE, source, base)
        return (result, base) if case == "live" else result

    value = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    reference = make_fx(function, tracing_mode="fake")(value)
    code = reference.code
    candidate, audit = lower_functionalized(reference, (WRITE, PARTIAL), reinplace=reinplace)
    before = value.clone()
    expected = reference(value)
    actual = candidate(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(value, before), "Caller input was mutated"
    assert reference.code == code, "Original FX module was modified"
    assert "higher_order.auto_functionalized_v2" not in candidate.code
    must_clone = not reinplace or case in ("input", "alias", "live")
    assert audit["nodes"][0]["cloned_bases"] == ([0] if must_clone else [])


def test_unknown_operator_and_missing_metadata():
    reference = make_fx(lambda x: call(WRITE, x, torch.zeros_like(x)), tracing_mode="fake")(torch.ones(3, 4))
    untouched, audit = lower_functionalized(reference, ())
    assert untouched.code == reference.code
    assert audit["nodes"] == []
    next(iter(reference.graph.nodes)).meta.clear()
    with pytest.raises(ValueError, match="Missing fake/alias metadata"):
        lower_functionalized(reference, (WRITE,), reinplace=True)


def test_repeated_calls_do_not_drop_mutations():
    def function(x):
        for _ in range(43):
            x = call(WRITE, x, torch.empty_like(x))
        return x

    value = torch.ones(1, 64, 512, dtype=torch.bfloat16)
    reference = make_fx(function, tracing_mode="fake")(value)
    candidate, audit = lower_functionalized(reference, (WRITE,), reinplace=True)
    assert len(audit["nodes"]) == 43
    assert all(not node["cloned_bases"] for node in audit["nodes"])
    for _ in range(3):
        torch.testing.assert_close(candidate(value), reference(value), rtol=0, atol=0)


def test_input_copyback_preserves_mutation_contract():
    def function(x):
        result = call(WRITE, torch.ones_like(x), x)
        x.copy_(result)
        return x

    value = torch.zeros(3, 4)
    reference = make_fx(function, tracing_mode="fake")(value)
    candidate, audit = lower_functionalized(reference, (WRITE,), reinplace=True)
    expected_input, actual_input = value.clone(), value.clone()
    torch.testing.assert_close(candidate(actual_input), reference(expected_input), rtol=0, atol=0)
    assert torch.equal(actual_input, expected_input)
    assert audit["nodes"][0]["cloned_bases"] == []


def test_unselected_boundaries_and_fp8_scale_bits_are_preserved():
    def function(x, scale):
        first = call(WRITE, x, torch.zeros_like(x))
        second = call(PARTIAL, first, torch.zeros_like(x))
        return second, scale

    x = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    scale = torch.arange(16, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    reference = make_fx(function, tracing_mode="fake")(x, scale)
    candidate, audit = lower_functionalized(reference, (WRITE,), reinplace=True)
    expected, expected_scale = reference(x, scale)
    actual, actual_scale = candidate(x, scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_scale is scale and expected_scale is scale
    assert torch.equal(actual_scale.view(torch.uint8), torch.arange(16, dtype=torch.uint8))
    assert len(audit["nodes"]) == 1
    remaining = [node for node in candidate.graph.nodes if node.target is auto_functionalized_v2]
    assert len(remaining) == 1 and remaining[0].args[0] is PARTIAL


def test_boxed_graph_can_lower_but_not_reinplace_without_alias_metadata():
    value = torch.ones(3, 4)
    reference = make_fx(lambda x: call(WRITE, x, x), tracing_mode="fake")(value)
    graph = reference.graph
    original = next(iter(graph.nodes))
    with graph.inserting_before(original):
        boxed = graph.placeholder("input_list")
    original.op = "call_function"
    import operator

    original.target = operator.getitem
    original.args = (boxed, 0)
    reference.recompile()
    candidate, audit = lower_functionalized(reference, (WRITE,))
    before = value.clone()
    torch.testing.assert_close(candidate([value]), reference([value]), rtol=0, atol=0)
    assert torch.equal(value, before)
    assert audit["nodes"][0]["cloned_bases"] == [0]
    with pytest.raises(ValueError, match="Missing fake/alias metadata"):
        lower_functionalized(reference, (WRITE,), reinplace=True)
