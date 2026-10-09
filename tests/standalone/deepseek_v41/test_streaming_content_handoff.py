# SPDX-License-Identifier: Apache-2.0
"""Content must survive different speculative streaming chunk boundaries."""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.parser.deepseek_v41 import DeepSeekV41Parser

from vllm_gaudi.entrypoints.openai.deepseek_v41_streaming import install


class Tokenizer:
    all_special_tokens = []
    all_special_ids = []

    def __init__(self, pieces):
        self.pieces = pieces

    def get_vocab(self):
        return {value: index for index, value in enumerate(self.pieces)}

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[index] for index in ids)


@pytest.mark.parametrize("width", range(1, 7))
def test_content_conserved_at_reasoning_handoff(monkeypatch, width):
    original = DeepSeekV41Parser.__mro__[1].extract_reasoning_streaming
    monkeypatch.setattr(DeepSeekV41Parser, "extract_reasoning_streaming", original)
    install()
    pieces = ["<think>", "reason", "</think>", "{\n", " ", ' "', "approved_retention_days", '": 45}', "<literal>"]
    tokenizer = Tokenizer(pieces)
    parser_cls = ParserManager.get_parser(reasoning_parser_name="deepseek_v41", tool_parser_name=None)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "json"}], stream=True)
    output = []
    ids = list(range(1, len(pieces)))
    for start in range(0, len(ids), width):
        batch = ids[start:start + width]
        delta = parser.parse_delta(delta_text=tokenizer.decode(batch),
                                   delta_token_ids=batch,
                                   request=request,
                                   prompt_token_ids=[0],
                                   finished=start + width >= len(ids))
        if delta is not None:
            output.append(delta.content or "")
    assert "".join(output) == '{\n  "approved_retention_days": 45}<literal>'


def test_install_idempotent(monkeypatch):
    original = DeepSeekV41Parser.__mro__[1].extract_reasoning_streaming
    monkeypatch.setattr(DeepSeekV41Parser, "extract_reasoning_streaming", original)
    install()
    installed = DeepSeekV41Parser.extract_reasoning_streaming
    install()
    assert DeepSeekV41Parser.extract_reasoning_streaming is installed
