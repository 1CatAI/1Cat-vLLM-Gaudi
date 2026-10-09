# SPDX-License-Identifier: Apache-2.0
"""CPU checks for payload identity, signal ancestry and unchanged recipe count."""
import json
import os

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')

import torch

from vllm_gaudi.compilation.deepseek_v41_mhc_producer import merge_mhc_producers
from vllm_gaudi.compilation.deepseek_v41_overlap import _partition_metadata
from vllm_gaudi.ops.tp2_prepared_plan import _tensor_ready_output_indices


def main():
    definitions = torch.library.Library('custom_op', 'FRAGMENT')
    definitions.define('custom_deepseek_v41_peer_ready_identity_gaudi2(Tensor x, bool external) -> Tensor')
    peers = torch.library.Library('vllm_gaudi', 'FRAGMENT')
    peers.define('tp_peer_allgather(Tensor x, int count) -> Tensor')
    marker = torch.ops.custom_op.custom_deepseek_v41_peer_ready_identity_gaudi2.default
    exchange = torch.ops.vllm_gaudi.tp_peer_allgather.default

    def signal_output(kind):
        graph = torch.fx.Graph()
        x = graph.placeholder('x')
        ready = graph.call_function(marker, (x, True))
        if kind == 'view':
            ready = graph.call_function(torch.ops.aten.view.default, (ready, [6, 5120]))
        elif kind == 'arithmetic':
            ready = graph.call_function(torch.ops.aten.add.Tensor, (ready, x))
        elif kind == 'second':
            ready = graph.call_function(marker, (ready, True))
        graph.output((ready,))
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    assert _tensor_ready_output_indices(signal_output('view')) == {0}
    for kind in ('arithmetic', 'second'):
        try:
            _tensor_ready_output_indices(signal_output(kind))
        except RuntimeError:
            pass
        else:
            raise AssertionError('Signal must not pass further arithmetic or an ambiguous second signal')

    def projection(columns, width):
        graph = torch.fx.Graph()
        x, w = graph.placeholder('x'), graph.placeholder('w')
        output = graph.call_function(torch.ops.aten.mm.default, (x, w))
        output.meta['val'] = torch.empty((6, columns), dtype=torch.bfloat16)
        x.meta['val'] = torch.empty((6, width), dtype=torch.bfloat16)
        w.meta['val'] = torch.empty((width, columns), dtype=torch.bfloat16)
        graph.output((output,))
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    owner = torch.nn.Module()
    owner.producer = projection(5120, 2048)
    owner.consumer_mhc_submod_0 = projection(28, 20480)
    root = torch.fx.Graph()
    x, w, residual, cw = [root.placeholder(name) for name in ('x', 'w', 'residual', 'cw')]
    producer = root.call_module('producer', (x, w))
    producer.meta = _partition_metadata(owner.producer)
    import operator

    packet = root.call_function(operator.getitem, (producer, 0))
    packet.meta['val'] = torch.empty((6, 5120), dtype=torch.bfloat16)
    flat = root.call_function(torch.ops.aten.view.default, (packet, [1, 30720]))
    peer = root.call_function(exchange, (flat, 4))
    control = root.call_module('consumer_mhc_submod_0', (residual, cw))
    control_value = root.call_function(operator.getitem, (control, 0))
    root.output((peer, control_value))
    module = torch.fx.GraphModule(owner, root)
    partitions = [dict(independent_partition='consumer_mhc_submod_0')]
    audit = merge_mhc_producers(module, partitions, (exchange,), mark_tensor_ready=True)
    assert len(audit) == 1 and audit[0]['tensor_ready_output'] == 0
    calls = [node for node in module.graph.nodes if node.op == 'call_module']
    assert len(calls) == 1
    joined = module.get_submodule(calls[0].target)
    assert _tensor_ready_output_indices(joined) == {0}
    ready = next(node for node in joined.graph.nodes if node.op == 'call_function' and node.target == marker)
    assert not torch._C._is_alias_of(ready.meta['val'], ready.args[0].meta['val'])
    result = next(node for node in joined.graph.nodes if node.op == 'output').args[0][0]
    assert torch._C._is_alias_of(result.meta['val'], ready.meta['val'])
    assert sum(node.op == 'call_function' and node.target == torch.ops.aten.mm.default
               for node in joined.graph.nodes) == 2
    print(json.dumps(dict(status='passed', alias_signal=True, arithmetic_rejected=True,
                          ambiguous_signal_rejected=True, producer_control_recipe_count=1,
                          unchanged_gemms=2)))


if __name__ == '__main__':
    main()
