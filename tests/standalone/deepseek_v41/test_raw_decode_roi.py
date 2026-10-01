# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'raw_decode_roi',
    Path(__file__).resolve().parents[3] / 'tools/normalize_deepseek_v41_raw_trace.py')
raw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw)


def test_decode_roi_keeps_padding_and_clips_to_calibration():
    metadata = {'scope_clock_domain': 'CLOCK_MONOTONIC_RAW'}
    cpu = {
        'clockDomain':
        'CLOCK_MONOTONIC_RAW',
        'baseTimeNanoseconds':
        1_000_000_000,
        'traceEvents': [
            {
                'ph': 'X',
                'name': 'v41::target::PP0::decode::C1',
                'ts': 100_000,
                'dur': 1000
            },
            {
                'ph': 'X',
                'name': 'v41::target::PP0::decode::C1',
                'ts': 110_000,
                'dur': 2000
            },
            {
                'ph': 'X',
                'name': 'v41::target::PP0::prefill::C8192',
                'ts': 0,
                'dur': 50_000
            },
        ]
    }
    assert raw.decode_parser_window(cpu, metadata, 1_000_000_000, 1_200_000_000, 50) == (1_050_000_000, 1_162_000_000)
    assert raw.decode_parser_window(cpu, metadata, 1_090_000_000, 1_120_000_000, 50) == (1_090_000_000, 1_120_000_000)


@pytest.mark.parametrize('padding', [-1, float('nan'), float('inf')])
def test_bad_padding_is_rejected(padding):
    with pytest.raises(ValueError, match='padding'):
        raw.decode_parser_window({}, {}, 0, 100, padding)


def test_uncalibrated_clock_is_rejected():
    with pytest.raises(ValueError, match='raw-clock'):
        raw.decode_parser_window({'clockDomain': 'wall'}, {}, 0, 100, 50)


def test_missing_decode_scopes_is_rejected():
    with pytest.raises(ValueError, match='no captured decode'):
        raw.decode_parser_window({
            'clockDomain': 'CLOCK_MONOTONIC_RAW',
            'traceEvents': []
        }, {'scope_clock_domain': 'CLOCK_MONOTONIC_RAW'}, 0, 100, 50)
