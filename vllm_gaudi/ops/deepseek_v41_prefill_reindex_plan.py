# SPDX-License-Identifier: Apache-2.0
"""Bound score intermediates and publish each replay into its current output."""
from collections import OrderedDict
from functools import lru_cache
from types import FunctionType

import torch

from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import INDEX_KEY_TILE, _candidate_slot_topk

SOURCE_WINDOW = 8192
QUERY_TILE = 128
_families = OrderedDict()


def source_scores(query, weights, keys, positions, rows, ratio):
    pieces = []
    for start in range(0, keys.shape[0], INDEX_KEY_TILE):
        current = keys[start:start + INDEX_KEY_TILE]
        ids = rows[start:start + INDEX_KEY_TILE]
        count = current.shape[0]
        if count % 128:
            current = torch.cat((current, current.new_zeros((128 - count % 128, 128))), 0)
            ids = torch.nn.functional.pad(ids, (0, 128 - count % 128), value=-1)
        score = torch.ops.custom_op.custom_deepseek_v41_prefill_index_scores_gaudi2(query, weights, current, positions,
                                                                                    ids, ratio)
        pieces.append(score[:, :count])
    return torch.cat(pieces, -1)


def select_scores(positions, blocks, ratio, *scores):
    return _candidate_slot_topk(torch.cat(scores, -1), positions, blocks, ratio, True)


def layout(value):
    return tuple(value.shape), value.stride(), value.dtype, value.device


@lru_cache(maxsize=64)
def compiled_scores(signature):
    entry = FunctionType(source_scores.__code__.replace(co_name=f"reindex_scores_{signature}"),
                         source_scores.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@lru_cache(maxsize=64)
def compiled_select(signature):
    entry = FunctionType(select_scores.__code__.replace(co_name=f"reindex_choices_{signature}"),
                         select_scores.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


class _Step:

    def __init__(self, arguments, ratio):
        from vllm_gaudi.ops.deepseek_v41_prefill_plan import PrefillExpertPlan

        self.rows = []
        scorers = []
        start = 0
        for keys in arguments[4:]:
            self.rows.append(torch.arange(start, start + keys.shape[0], device=keys.device, dtype=torch.int32))
            scorers.append(compiled_scores((layout(arguments[0]), layout(keys), ratio)))
            start += keys.shape[0]
        selector = compiled_select((tuple(layout(v) for v in arguments), ratio))

        def body(query, weights, positions, blocks, *sources):
            # Every key window is a root argument, not an unbound captured view.
            parts = [
                scorer(query, weights, keys, positions, rows, ratio)
                for scorer, keys, rows in zip(scorers, sources, self.rows, strict=True)
            ]
            self.output = selector(positions, blocks, ratio, *parts)
            return self.output

        self.execution = PrefillExpertPlan(body, arguments, None, require_prefix=True)

    def replay(self, arguments):
        self.execution.replay(arguments, 1)


def _close(family):
    if family:
        torch.hpu.synchronize()
        for step in family.values():
            step.execution.plan.invalidate()
    family.clear()


def prepared_reindex_selection(query, weights, keys, positions, blocks, ratio):
    if torch.hpu.current_stream().hpu_stream != torch.hpu.default_stream().hpu_stream:
        raise RuntimeError("Reindex plans belong to the default model compute stream")
    signature = (tuple(layout(v) for v in (query, weights, keys, positions, blocks)), ratio)
    family = _families.get(signature)
    if family is None:
        if len(_families) >= 2:
            _, expired = _families.popitem(last=False)
            _close(expired)
        family = OrderedDict()
        _families[signature] = family
    else:
        _families.move_to_end(signature)
    outputs = []
    sources = tuple(keys[start:start + SOURCE_WINDOW] for start in range(0, keys.shape[0], SOURCE_WINDOW))
    for start in range(0, query.shape[0], QUERY_TILE):
        stop = min(start + QUERY_TILE, query.shape[0])
        arguments = (query[start:stop].clone(), weights[start:stop].clone(), positions[start:stop].clone(),
                     blocks[start:stop].clone(), *sources)
        binding = tuple(layout(v) for v in arguments)
        step = family.get(binding)
        if step is None:
            step = _Step(arguments, ratio)
            family[binding] = step
        else:
            step.replay(arguments)
        # Replay retains its internal results. Preserve this transaction before
        # the next query tile writes the same plan-owned result buffer.
        outputs.append(step.output.clone())
    return torch.cat(outputs, 0)


def invalidate_reindex_plans():
    for family in _families.values():
        _close(family)
    _families.clear()
    compiled_scores.cache_clear()
    compiled_select.cache_clear()
