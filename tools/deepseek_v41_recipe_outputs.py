# SPDX-License-Identifier: Apache-2.0
"""Read existing recipe outputs without exporting additional graph tensors.

Diagnostic only: synchronization changes timing. A run is valid only when its
final outputs equal the previously saved unobserved outputs for both arms.
"""
import torch


class RecipeOutputs:

    def __init__(self, maximum_elements):
        from habana_frameworks.torch.dynamo.compile_backend.passes import (OptimizationPassPlacement,
                                                                           register_pass_at_optimization_pass)
        self.maximum_elements = maximum_elements
        self.records = None
        owner = self

        class Observe(torch.nn.Module):

            def __init__(self, name, recipe):
                super().__init__()
                self.name, self.recipe = name, recipe

            def forward(self, *inputs):
                saved_inputs = None
                if owner.records is not None:
                    saved_inputs = {
                        index: value.detach().cpu().clone()
                        for index, value in enumerate(inputs)
                        if isinstance(value, torch.Tensor) and value.numel() <= owner.maximum_elements
                    }
                result = self.recipe(*inputs)
                if owner.records is not None:
                    outputs = list(result) if isinstance(result, (tuple, list)) else [result]
                    values = {
                        index: value.detach().cpu().clone()
                        for index, value in enumerate(outputs)
                        if isinstance(value, torch.Tensor) and value.numel() <= owner.maximum_elements
                    }
                    owner.records.append(
                        dict(name=self.name,
                             recipe=self.recipe._recipe_id,
                             shapes=[list(v.shape) if isinstance(v, torch.Tensor) else str(type(v)) for v in outputs],
                             values=values,
                             inputs=saved_inputs,
                             fx_code=self.recipe._fx_module.code,
                             jit_ir=str(self.recipe._jit_ir)))
                return result

        def observe(context):
            changed = False
            for name, child in list(context.graph_module.named_children()):
                if hasattr(child, "_recipe_id"):
                    context.graph_module.add_module(name, Observe(name, child))
                    changed = True
            return changed

        register_pass_at_optimization_pass(observe, OptimizationPassPlacement.POST_PARTITIONER)
