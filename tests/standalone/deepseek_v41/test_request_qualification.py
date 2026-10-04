# SPDX-License-Identifier: Apache-2.0
"""Limited trace requests must not weaken the natural-EOS qualification gate."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_path = Path(__file__).resolve().parents[3] / 'tools/qualify_deepseek_v41_request.py'
_spec = importlib.util.spec_from_file_location('request_qualification', _path)
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)


class Response:
    def __init__(self, text='', lines=()):
        self.text, self.lines = text, lines

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_lines(self, **kwargs):
        return iter(self.lines)


class Session:
    def __init__(self):
        self.sent = False
        self.headers = {}

    def close(self):
        pass

    def get(self, *args, **kwargs):
        count = int(self.sent)
        return Response('\n'.join(f'vllm:{metric}_{kind} {count}'
                                  for metric in client.METRICS for kind in ('count', 'sum')))

    def post(self, *args, **kwargs):
        self.sent = True
        payload = dict(usage=dict(prompt_tokens=4, completion_tokens=2),
                       choices=[dict(delta={}, token_ids=[3, 5], finish_reason='length')])
        return Response(lines=[b'data: ' + json.dumps(payload).encode(), b'data: [DONE]'])


@pytest.mark.parametrize('diagnostic', [False, True])
def test_length_termination_only_passes_as_an_explicit_diagnostic(monkeypatch, tmp_path, diagnostic):
    request, output = tmp_path/'request.json', tmp_path/'output'
    request.write_text(json.dumps(dict(stream=True, return_token_ids=True, max_tokens=2)))
    args = ['qualify', str(request), str(output), '--expected-prompt-tokens', '4']
    if diagnostic:
        args.append('--diagnostic-only')
    monkeypatch.setattr(sys, 'argv', args)
    monkeypatch.setattr(client.requests, 'Session', Session)
    if diagnostic:
        client.main()
    else:
        with pytest.raises(AssertionError, match='did not finish naturally'):
            client.main()
    result = json.loads((output/'result.json').read_text())
    assert not result['formal_measurement_passed']
    assert result['status'] == ('diagnostic_passed' if diagnostic else 'failed')
    if diagnostic:
        assert result['eos_proof'] is None
