# SPDX-License-Identifier: Apache-2.0
"""The discrepancy row must consume the preceding token and identical state."""
import json
from types import SimpleNamespace

import torch

from tools.check_deepseek_v41_prefix_logits import compare_worker
from vllm_gaudi.ops.deepseek_v41_state import PagedStageState


def test_teacher_forced_anchor_and_state_rollback(monkeypatch, tmp_path):
    from vllm_gaudi.ops import deepseek_v41_replay as replay
    monkeypatch.setattr(torch.hpu, 'synchronize', lambda: None)
    main = torch.zeros(256, 288, dtype=torch.uint8)
    cache = SimpleNamespace(main=main, index=torch.zeros(256, 68, dtype=torch.uint8), ratio=1)
    position = torch.zeros(1, dtype=torch.int64)
    monkeypatch.setattr(replay, 'stage_state_tensors', lambda program: [main, cache.index, position])
    state = object.__new__(PagedStageState)
    state.active, state.blocks = None, 8193
    state.release = lambda request: None
    pending, consumed, forwards = [], [], []
    program = SimpleNamespace(shared=SimpleNamespace(sources={'0': cache}), runtime_precision={},
                              logits=lambda value: value)

    def forward(request_id, tokens, start, *, decode, reset=False, request=None):
        assert start == int(position[0])
        pending[:] = tokens
        before = position.item()
        position.add_(len(tokens))
        main[0, 0].add_(1)
        forwards.append((tuple(tokens), start, decode))
        return torch.tensor([[float(before), float(before) - .25, -100.]] * len(tokens))

    def complete(count):
        consumed.extend(pending[:count])
        pending.clear()

    runner = SimpleNamespace(requests={}, device_round_queue=None,
                             pp=SimpleNamespace(group=SimpleNamespace(is_first_rank=True, is_last_rank=True)),
                             state=state, request_slots_enabled=False, prefill_capacity=8192,
                             _bind_request=lambda request: None, _forward=forward,
                             model=SimpleNamespace(tp_rank=0, program=program, complete_step=complete,
                                                   engram_host=SimpleNamespace(release_request=lambda request: None)))
    prompt, reference = list(range(128)), [0, 1, 0, 3, 4, 5, 6, 7]
    result = compare_worker(SimpleNamespace(model_runner=runner), prompt, reference, 2, str(tmp_path))
    assert forwards[0] == (tuple(prompt), 0, False)
    assert forwards[1] == ((0,), 128, True)
    assert forwards[2:5] == [((1,), 129, True), ((1, 0, 3, 4, 5, 6), 129, True), ((1,), 129, True)]
    assert consumed == prompt + reference[:1]
    assert result['c1_reference_top1_reproduced']
    assert result['max_abs_diff'] == result['c6_future_sensitivity_max_abs'] == 0
    assert int(position[0]) == 129
    report = json.loads((tmp_path / 'logits-rank0.json').read_text())
    assert report['c1_repeated_byte_exact'] and not report['formal_qualified']
