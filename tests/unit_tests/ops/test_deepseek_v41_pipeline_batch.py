# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_gaudi.v1.worker.deepseek_v41_pipeline_batch import split_requests


@pytest.mark.parametrize("count", [32, 33, 63, 64])
def test_split_keeps_absolute_order_and_distinct_owners(count):
    requests = tuple(
        SimpleNamespace(req_id=f"request-{i}", num_computed_tokens=i * 997) for i in reversed(range(count)))
    first, second = split_requests(requests)
    assert first + second == requests
    assert 0 <= len(first) - len(second) <= 1
    assert not {id(request) for request in first} & {id(request) for request in second}


@pytest.mark.parametrize("count", [0, 1, 31, 65])
def test_split_rejects_unqualified_bucket(count):
    with pytest.raises(ValueError):
        split_requests(tuple(SimpleNamespace(req_id=str(i)) for i in range(count)))


def test_split_rejects_two_writers_for_one_request():
    requests = tuple(SimpleNamespace(req_id=str(i % 31)) for i in range(32))
    with pytest.raises(ValueError):
        split_requests(requests)
