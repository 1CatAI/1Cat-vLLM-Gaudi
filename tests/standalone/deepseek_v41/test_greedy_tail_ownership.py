# SPDX-License-Identifier: Apache-2.0
"""Native greedy output preserves public decoder arity and rejects stale hidden owners."""
from types import SimpleNamespace
import weakref

import torch

from vllm_gaudi.ops.deepseek_v41_replay import StageReplay, StageVariant


def test_tail_keeps_three_public_outputs_and_generation_guard():
    variant = object.__new__(StageVariant)
    torch.nn.Module.__init__(variant)
    variant.tail_enabled = True
    hidden = torch.ones(1, 8, dtype=torch.bfloat16)
    token = torch.tensor([[27]], dtype=torch.int32)
    values = hidden, None, None, token
    outputs = variant.publish_outputs(values)
    assert len(outputs) == 3 and variant.tail_values[0] is token
    program = torch.nn.Module()
    program.generation = 1
    replay = object.__new__(StageReplay)
    replay.program = weakref.ref(program)
    replay.latest_tail = None
    replay._publish_tail(variant, outputs)
    assert replay.greedy_tail_token(hidden[:]) is token
    assert replay.greedy_tail_token(hidden.clone()) is None
    assert replay.greedy_tail_token(hidden.float()) is None
    program.generation += 1
    assert replay.greedy_tail_token(hidden) is None


def test_non_tail_decode_clears_prior_sample():
    variant = SimpleNamespace(tail_values=None)
    replay = object.__new__(StageReplay)
    replay.latest_tail = ('old',)
    outputs = torch.zeros(2, 8), None, torch.zeros(2, 3)
    assert replay._publish_tail(variant, outputs) is outputs
    assert replay.latest_tail is None


def test_direct_stage_entry_publishes_its_completed_tail():
    program = torch.nn.Module()
    program.generation = 1
    program.dspark = False
    program.search_length = 512
    hidden = torch.ones(1, 8, dtype=torch.bfloat16)
    token = torch.tensor([[17]], dtype=torch.int32)
    outputs = hidden, None, None

    def variant(*args):
        return outputs

    variant.tail_values = (token,)
    replay = object.__new__(StageReplay)
    replay.program = weakref.ref(program)
    replay.variants = {1: variant}
    replay.latest_tail = None
    replay._with_index_mirror = lambda key, search: key
    assert replay(hidden, None, None, torch.tensor([1]), ()) is outputs
    assert replay.greedy_tail_token(hidden) is token
