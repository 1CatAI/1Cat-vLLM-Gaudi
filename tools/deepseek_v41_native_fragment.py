# SPDX-License-Identifier: Apache-2.0
"""Real four-layer qualification through the production joined-recipe entry."""
from types import SimpleNamespace

import torch


class NativeBatchFragment(torch.nn.Module):

    def __init__(self, group, compiled, states, restore, extra_collectives, host_positions):
        super().__init__()
        self.group = group
        self.compiled = compiled
        self.states, self.restore = tuple(states), restore
        self.host_positions = host_positions
        self.reindex_ratios = tuple(layer.attention.ratio for layer in group.layers
                                    if layer.attention.batch_reindex_mme and layer.attention.bounded_reindex
                                    and layer.attention.owns_index and layer.layer > layer.attention.candidate_source)
        self.metadata = SimpleNamespace(native_completion=None)
        self.fixed = None
        from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
        self.adapter = DecoderTopology("deepseek_v41_fragment", (len(group.layers), ), 2, False, extra_collectives)

    @staticmethod
    def flatten(value):
        return tuple(item for part in value for item in (part if isinstance(part, (tuple, list)) else (part, )))

    def forward(self, hidden, pre, positions, ids, engram, slots, pages, selections, pool, main_ready, index_ready):
        from vllm_gaudi.ops.tp2_prepared_plan import (
            collect_prepared_group_replays,
            record_native_decoder_outputs,
            replay_native_decoder,
        )
        from vllm_gaudi.ops.deepseek_v41_reindex_compact import bounded_reindex_bucket, unique_pool_tile_bound
        roots = dict(hidden_states=hidden,
                     pre_mix=pre,
                     positions=positions,
                     input_ids=ids,
                     attention_inputs=(*engram, slots, pages, *selections, pool, *main_ready, *index_ready),
                     metadata=self.metadata,
                     state_generation=(1, ),
                     state_tensors=self.states)
        if bounded_reindex_bucket(positions.numel()) and self.reindex_ratios:
            roots["reindex_tile_bound"] = max(
                unique_pool_tile_bound(self.host_positions, ratio) for ratio in self.reindex_ratios)
        actual = replay_native_decoder(self, **roots)
        if actual is not None:
            return actual
        # Match BatchStageVariant's cold inputs. PP hidden/pre are differently
        # typed views of one wire allocation; lowering them directly can make
        # the Bridge infer an invalid reinterpret during eager preparation.
        from torch.utils._pytree import tree_flatten, tree_map
        inputs = (hidden, pre, positions, ids, engram, slots, pages, selections, pool, main_ready, index_ready)
        if self.fixed is None:
            self.fixed = tree_map(lambda value: value.clone(), inputs)
        else:
            for destination, source in zip(tree_flatten(self.fixed)[0], tree_flatten(inputs)[0], strict=True):
                destination.copy_(source)
        h, p, pos, token, eg, slot, page, selection, candidates, main, index = self.fixed
        roots.update(hidden_states=h,
                     pre_mix=p,
                     positions=pos,
                     input_ids=token,
                     attention_inputs=(*eg, slot, page, *selection, candidates, *main, *index))
        with collect_prepared_group_replays(owner=self,
                                            adapter=self.adapter,
                                            snapshot=lambda: SimpleNamespace(restore=self.restore),
                                            **roots) as context:
            context["group_index"] = 0
            result = self.flatten(self.compiled(*self.fixed))
            record_native_decoder_outputs(*result)
        return result

    def require_ready(self):
        from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
        if self not in _native_entries:
            raise RuntimeError("Real fragment did not prepare its complete native plan")

    def close(self):
        from vllm_gaudi.ops.tp2_prepared_plan import invalidate_prepared_group_plans
        invalidate_prepared_group_plans(owner=self, reason="fragment_done")
