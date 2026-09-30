# SPDX-License-Identifier: Apache-2.0
"""Reproduce eager concat graph reuse across axis and shape changes."""
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch
from habana_frameworks.torch.internal import bridge_config

root = Path(os.environ['DSV41_RUN_EVIDENCE'])
report = dict(status='running', cases=[], shape_agnostic_environment=os.environ.get('PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH'),
              shape_agnostic_effective=bridge_config.get_pt_hpu_eager_shape_agnostic_graph())
try:
    with torch.inference_mode():
        for rows, axis in [(2048, 1), (1024, 1), (512, 1), (128, 0), (256, 0), (128, 1), (128, 0)]:
            first = torch.full((rows, 512), 1.0, device='hpu')
            second = torch.full((rows, 512), 2.0, device='hpu')
            actual = torch.cat((first, second), dim=axis).cpu()
            expected = torch.cat((torch.ones(rows, 512), torch.full((rows, 512), 2.0)), dim=axis)
            assert torch.equal(actual, expected), (rows, axis, actual.shape)
            report['cases'].append(dict(rows=rows, axis=axis, status='passed'))
            (root/'concat-reuse.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status'] = 'passed'
except BaseException as error:
    report.update(status='failed',error=repr(error))
    raise
finally:
    (root/'concat-reuse.json').write_text(json.dumps(report,indent=2)+'\n')
