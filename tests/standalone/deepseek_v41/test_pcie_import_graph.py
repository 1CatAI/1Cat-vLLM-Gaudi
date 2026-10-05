# SPDX-License-Identifier: Apache-2.0
import random

import pytest

from tools.deepseek_v41_pcie_import_graph import epoch_reached, importer_first_order, star_imports


@pytest.mark.parametrize('size', [2, 3, 4])
def test_star_has_no_returning_device_reference(size):
    imports, order = star_imports(size)
    assert imports == [(rank, 0) for rank in range(1, size)]
    assert order[-1] == 0
    for importer, exporter in imports:
        assert order.index(importer) < order.index(exporter)


@pytest.mark.parametrize('edges', [[(0, 1), (1, 0)], [(0, 1), (1, 2), (2, 0)],
                                   [(a, b) for a in range(4) for b in range(4) if a != b]])
def test_old_cyclic_imports_are_rejected(edges):
    with pytest.raises(ValueError, match='Cyclic DMA-BUF'):
        importer_first_order(4, edges)


def test_epoch_wrap_does_not_accept_the_previous_payload():
    assert not epoch_reached(0xffffffff, 0)
    assert epoch_reached(0, 0)
    assert epoch_reached(0, 0xffffffff)
    assert not epoch_reached(0, 1)


@pytest.mark.parametrize('size', [2, 4])
@pytest.mark.parametrize('seed', [1, 7, 42])
def test_double_buffered_star_survives_a_slow_partial_reader(size, seed):
    # A leaf cannot announce epoch E+1 until it finishes consuming E. Thus
    # the hub cannot publish E+2 over E's result bank while any reader of E
    # remains. Model individual result-vector reads to test that lifetime.
    rng = random.Random(seed)
    epochs, phases, read_at = [1]*size, ['push']*size, [0]*size
    inbox, ready, result, publication = {}, [0]*size, {}, [0, 0]
    finished = [0]*size
    for _ in range(100000):
        rank = rng.randrange(size)
        epoch = epochs[rank]
        if epoch > 15:
            if min(finished) == 15:
                break
            continue
        bank = epoch % 2
        if phases[rank] == 'push':
            inbox[bank, rank] = epoch
            ready[rank] = epoch
            phases[rank] = 'gather' if rank == 0 else 'wait'
        elif phases[rank] == 'gather':
            if min(ready) < epoch:
                continue
            assert all(inbox[bank, peer] == epoch for peer in range(size))
            result[bank] = [epoch]*40
            publication[bank] = epoch
            phases[rank] = 'read'
        elif phases[rank] == 'wait':
            if publication[bank] < epoch:
                continue
            assert publication[bank] == epoch
            phases[rank] = 'read'
        else:
            assert result[bank][read_at[rank]] == epoch
            read_at[rank] += 1
            if read_at[rank] == 40:
                finished[rank] = epoch
                epochs[rank] += 1
                read_at[rank] = 0
                phases[rank] = 'push'
    assert finished == [15]*size


@pytest.mark.parametrize('size', [2, 4])
@pytest.mark.parametrize('seed', [1, 7, 42])
def test_hub_inbox_remains_live_until_each_local_reduction_finishes(size, seed):
    rng = random.Random(seed)
    epochs, phases, read_at = [1]*size, ['push']*size, [0]*size
    inbox, ready, finished = {}, [0]*size, [0]*size
    for _ in range(100000):
        rank = rng.randrange(size)
        epoch = epochs[rank]
        if epoch > 15:
            if min(finished) == 15:
                break
            continue
        bank = epoch % 2
        if phases[rank] == 'push':
            inbox[bank, rank] = [epoch]*40
            ready[rank] = epoch
            phases[rank] = 'wait'
        elif phases[rank] == 'wait':
            if min(ready) >= epoch:
                phases[rank] = 'read'
        else:
            for peer in range(size):
                assert inbox[bank, peer][read_at[rank]] == epoch
            read_at[rank] += 1
            if read_at[rank] == 40:
                finished[rank] = epoch
                epochs[rank] += 1
                read_at[rank] = 0
                phases[rank] = 'push'
    assert finished == [15]*size
