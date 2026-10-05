# SPDX-License-Identifier: Apache-2.0
"""Resident A/B table mappings have one owner per host/device/token bucket."""
from types import SimpleNamespace

import pytest
import torch

from tools import deepseek_v41_device_chain as chain


@pytest.fixture
def producers(monkeypatch):
    created, closed = [], []

    def create(host, ids, history):
        value = SimpleNamespace(owner=None, host=host, count=ids.numel())
        value.close = lambda: closed.append(value)
        created.append(value)
        return value

    monkeypatch.setattr(chain, '_inputs', {})
    monkeypatch.setattr(chain, '_frames', {})
    monkeypatch.setattr(chain, 'DeviceEngramRounds', create)
    return created, closed


def test_serial_arms_reuse_mapping_after_retirement(producers):
    created, _ = producers
    host = object()
    history = torch.tensor([3, 2, 1], dtype=torch.int32)
    first = chain.shared_device_inputs(host, torch.tensor([7]), history)
    first.owner = 'arm-a'
    with pytest.raises(RuntimeError, match='preceding consumer'):
        chain.shared_device_inputs(host, torch.tensor([8]), history)
    first.owner = None  # run_device_chain retires only after its final drain.
    assert chain.shared_device_inputs(host, torch.tensor([8]), history) is first
    assert len(created) == 1


def test_hosts_and_c1_c2_c6_buckets_have_separate_roots(producers):
    created, closed = producers
    history = torch.zeros(3, dtype=torch.int32)
    hosts = [object(), object()]
    for host in hosts:
        for count in (1, 2, 6):
            a = chain.shared_device_inputs(host, torch.zeros(count, dtype=torch.int32), history)
            assert chain.shared_device_inputs(host, torch.ones(count, dtype=torch.int32), history) is a
    assert len(created) == 6
    chain.close_device_chains()
    chain.close_device_chains()
    assert [id(value) for value in closed] == [id(value) for value in created]
    assert not chain._inputs


def test_failed_mapping_is_not_cached(producers, monkeypatch):
    def fail(*args):
        raise RuntimeError('mapping failed')

    monkeypatch.setattr(chain, 'DeviceEngramRounds', fail)
    with pytest.raises(RuntimeError, match='mapping failed'):
        chain.shared_device_inputs(object(), torch.tensor([7]), torch.zeros(3, dtype=torch.int32))
    assert not chain._inputs
