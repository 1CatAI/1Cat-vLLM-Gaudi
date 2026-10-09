# SPDX-License-Identifier: Apache-2.0
"""CPU/Meta compiler contract: retain the owned flag storage without cloning."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from torch.fx.experimental.proxy_tensor import make_fx

    torch.ops.load_library(str(args.library.resolve()))
    report = dict(status='running', device_acquired=False, tests=[])
    try:
        for rows in (2, 6):
            value = torch.empty((rows, 2048), dtype=torch.bfloat16, device='meta')
            flags = torch.empty((1 << 24,), dtype=torch.int32, device='meta')

            def body(x, state):
                copied = torch.ops.custom_op.custom_deepseek_v41_future_epoch_gaudi2(x, state)
                return copied

            graph = make_fx(torch.func.functionalize(body), tracing_mode='fake')(value, flags)
            calls = [str(node.target) for node in graph.graph.nodes if node.op == 'call_function']
            clones = [node for node in calls if 'clone' in node or 'copy_' in node]
            ordered = [node for node in calls if 'custom_deepseek_v41_future_epoch_ordered_gaudi2' in node]
            report['tests'].append(dict(rows=rows, calls=calls, flags_clone_or_copy=clones,
                                        ordered_count=len(ordered)))
            if clones or len(ordered) != 1:
                raise AssertionError('Epoch state lost the C1 owned ordered-write contract')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
