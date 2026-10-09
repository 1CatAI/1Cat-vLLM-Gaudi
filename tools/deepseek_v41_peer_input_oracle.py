# SPDX-License-Identifier: Apache-2.0
"""Save the first real projection and peer consumer, outside timed graphs."""
import types

import torch


def install(attention, rows):
    width = attention.weights.wo_b.weight.shape[1]
    attention.register_buffer('peer_oracle_projection_input',
                              torch.empty((rows, width), dtype=torch.float32, device='hpu'), False)
    attention.register_buffer('peer_oracle_partial',
                              torch.empty((rows, 5120), dtype=torch.bfloat16, device='hpu'), False)
    attention.register_buffer('peer_oracle_reduced', torch.empty_like(attention.peer_oracle_partial), False)
    projection = attention.project_output_consumer
    install_reduction(attention, rows)

    def project(self, value):
        self.peer_oracle_projection_input.copy_(value)
        self.peer_oracle_projection_dtype = str(value.dtype)
        return projection(value)

    attention.project_output_consumer = types.MethodType(project, attention)


def install_reduction(module, rows):
    if 'peer_oracle_partial' not in module._buffers:
        module.register_buffer('peer_oracle_partial',
                               torch.empty((rows, 5120), dtype=torch.bfloat16, device='hpu'), False)
        module.register_buffer('peer_oracle_reduced', torch.empty_like(module.peer_oracle_partial), False)
    reduction = module.reduce

    def reduce(self, value, *, ready_outputs=()):
        self.peer_oracle_partial.copy_(value)
        result = reduction(value, ready_outputs=ready_outputs)
        self.peer_oracle_reduced.copy_(result)
        return result

    module.reduce = types.MethodType(reduce, module)


def operands(attention):
    return dict(projection_input=attention.peer_oracle_projection_input.cpu(),
                projection_input_dtype=attention.peer_oracle_projection_dtype,
                partial=attention.peer_oracle_partial.cpu(), reduced=attention.peer_oracle_reduced.cpu())


def reduction_operands(module):
    return dict(partial=module.peer_oracle_partial.cpu(), reduced=module.peer_oracle_reduced.cpu())
