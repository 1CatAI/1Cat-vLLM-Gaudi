# SPDX-License-Identifier: Apache-2.0
"""Preserve buffered content at the V4.1 reasoning-to-content handoff."""

from functools import wraps


def install():
    try:
        from vllm.parser.deepseek_v41 import DeepSeekV41Parser
    except ImportError:
        return
    original = DeepSeekV41Parser.extract_reasoning_streaming
    if getattr(original, "_dsv41_flush_content", False):
        return

    @wraps(original)
    def extract(self, *args, **kwargs):
        delta = original(self, *args, **kwargs)
        if self.reasoning_ended:
            # The parser lexer can hold a quote or '<' after </think> as a
            # possible tool delimiter. The delegating parser drains that
            # buffer at the handoff, but its reasoning-only engine path does
            # not append the drained content to the returned delta. Publish
            # it here, before that adapter stops consuming reasoning deltas.
            pending = self.finish_streaming()
            if pending is not None:
                if delta is None:
                    return pending
                if pending.content:
                    delta.content = (delta.content or "") + pending.content
                if pending.reasoning:
                    delta.reasoning = (delta.reasoning or "") + pending.reasoning
                if pending.tool_calls:
                    delta.tool_calls = (delta.tool_calls or []) + pending.tool_calls
        return delta

    extract._dsv41_flush_content = True
    DeepSeekV41Parser.extract_reasoning_streaming = extract
