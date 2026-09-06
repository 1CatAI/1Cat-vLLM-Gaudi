from types import SimpleNamespace

from habana_frameworks.torch.dynamo._fx_to_jit_lowering import FxToJitLowering

from vllm_gaudi import patches


def test_hpu_fx_stack_trace_patch_accepts_resume_trace_without_code(monkeypatch):
    original = FxToJitLowering._get_stack_trace
    monkeypatch.setattr(FxToJitLowering, "_get_stack_trace", original)

    patches._patch_hpu_fx_stack_trace_parser()

    node = SimpleNamespace(
        stack_trace=(
            '  File "/tmp/qwen3_5.py", line 302, '
            "in torch_dynamo_resume_in_forward_at_238\n"
        )
    )
    assert FxToJitLowering._get_stack_trace(None, node) is None


def test_hpu_fx_stack_trace_patch_preserves_valid_trace(monkeypatch):
    original = FxToJitLowering._get_stack_trace
    monkeypatch.setattr(FxToJitLowering, "_get_stack_trace", original)

    patches._patch_hpu_fx_stack_trace_parser()

    node = SimpleNamespace(
        stack_trace=(
            '  File "/tmp/qwen3_5.py", line 302, in forward\n'
            "    value = x[:, 1:]\n"
        )
    )
    stack_trace = FxToJitLowering._get_stack_trace(None, node)
    assert stack_trace == (
        "File: qwen3_5.py:302 in forward, code: value = x[:, 1:]"
    )
