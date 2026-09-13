# SPDX-License-Identifier: Apache-2.0
"""Opt-in native state contracts; no device is acquired by the CPU cases."""
import importlib.util
import os

import pytest
import torch
from torch._functorch.aot_autograd import aot_function
from functorch.compile import make_boxed_func

from vllm_gaudi.ops.gdn_state_update import lower_gdn_state_updates


@pytest.fixture(scope="module", autouse=True)
def native():
    path = os.environ.get("GDN_STATE_TEST_LIBRARY")
    if not path:
        pytest.skip("Set GDN_STATE_TEST_LIBRARY to the built TP2 candidate")
    spec = importlib.util.spec_from_file_location("tp2_fused_ar_norm_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    rng = torch.Generator().manual_seed(17)
    return tuple(
        torch.randn(shape, generator=rng) for shape in ((1, 8, 3, 128, 128), (1, 8, 3, 128, 1), (1, 8, 1, 1, 128)))


def test_destination_and_unused_slots():
    bank = torch.full((5, 24, 128, 128), 137.0)
    dst = bank[2:3]
    decayed, delta, key = inputs()
    ptr = dst.data_ptr()
    out = torch.ops.custom_op.gdn_state_update_out(decayed, delta, key, dst)
    assert ptr == out.data_ptr()
    torch.testing.assert_close(out, torch.addcmul(decayed, delta, key).reshape_as(dst), atol=0, rtol=0)
    assert (bank[[0, 1, 3, 4]] == 137).all()


def test_overlap_rejected():
    decayed, delta, key = inputs()
    with pytest.raises(RuntimeError, match="overlap"):
        torch.ops.custom_op.gdn_state_update_out(decayed, delta, key, decayed.reshape(1, 24, 128, 128))


def _function(state, delta, key):
    decayed = state.reshape(1, 8, 3, 128, 128) * 0.5
    result = torch.ops.custom_op.gdn_state_update(decayed, delta, key, state)
    output = result.sum(-1)
    state.copy_(result)
    return output


def test_continuous_aot_state_and_cache_rebind(tmp_path):
    graphs = []

    def compiler(graph, _):
        (tmp_path / f"before-{len(graphs)}.py").write_text(graph.code)
        assert lower_gdn_state_updates(graph)
        (tmp_path / f"after-{len(graphs)}.py").write_text(graph.code)
        assert "gdn_state_update_out" in graph.code
        assert "aten.copy_" not in graph.code
        assert "aten.clone" not in graph.code
        graphs.append(graph)
        return make_boxed_func(graph.forward)

    compiled = aot_function(_function, fw_compiler=compiler, keep_inference_input_mutations=True)
    _, delta, key = inputs()
    with torch.no_grad():
        for generation in range(2):
            pool = torch.full((5, 24, 128, 128), float(generation + 1))
            state = pool[1:2]
            reference = state.clone()
            for step in range(12):
                d = delta * (0, 0.01, 0.1, 1)[step % 4]
                expected = _function(reference, d, key)
                actual = compiled(state, d, key)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(state, reference, atol=0, rtol=0)
            assert (pool[[0, 2, 3, 4]] == generation + 1).all()
    assert len(graphs) == 1


def test_live_old_state_rejected():

    def live(state, delta, key):
        decayed = state.reshape(1, 8, 3, 128, 128) * 0.5
        result = torch.ops.custom_op.gdn_state_update(decayed, delta, key, state)
        old = state * 2
        state.copy_(result)
        return old

    def compiler(graph, _):
        lower_gdn_state_updates(graph)
        return make_boxed_func(graph.forward)

    _, delta, key = inputs()
    with torch.no_grad(), pytest.raises(RuntimeError, match="old state remains live"):
        aot_function(live, fw_compiler=compiler, keep_inference_input_mutations=True)(torch.ones(1, 24, 128, 128),
                                                                                      delta, key)
