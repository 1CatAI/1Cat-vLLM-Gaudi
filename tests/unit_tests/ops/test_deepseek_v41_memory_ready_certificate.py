# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual group entry without loading the HPU backend on CPU."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def entry():
    path = Path(__file__).resolve().parents[3] / "vllm_gaudi/models/deepseek_v41_program.py"
    module = ast.parse(path.read_text())
    group = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "PreparedLayerGroup")
    function = next(n for n in group.body if isinstance(n, ast.FunctionDef) and n.name == "memory_ready_forward")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope[function.name]


@pytest.mark.parametrize("case", range(5))
def test_faults_cross_groups_and_use_existing_certificate(case):
    statuses = [torch.ones(40, dtype=torch.int32) for _ in range(4)]
    if case in (1, 2):
        statuses[case][17] = -1
    payload = (torch.zeros(1, 5120), torch.ones(1, 4), None,
               torch.tensor([[123]], dtype=torch.int32), torch.ones(1), torch.ones(1),
               torch.tensor([[61]], dtype=torch.int32), torch.ones(1, dtype=torch.int32))

    def forward(*args, memory_ready, **kwargs):
        memory_ready[1].extend(statuses)
        return payload

    owner = SimpleNamespace(layers=[None] * 4, native_input=None, fp8_decode=True,
                            greedy_tail=object(), _forward=forward)
    prior = torch.tensor(case != 3)
    output, root, observed, valid = entry()(
        owner, None, None, torch.zeros(1, dtype=torch.int32), None, None,
        torch.zeros(40, 32, dtype=torch.int32), torch.ones(32, dtype=torch.int32), prior)
    good = case in (0, 4)
    assert bool(valid) == good
    assert output[3].item() == (123 if good else -1)
    assert all(output[i] is payload[i] for i in (0, 1, 4, 5, 6, 7))
    assert torch.equal(observed, torch.stack(statuses))
