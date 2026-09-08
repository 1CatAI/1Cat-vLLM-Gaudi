# SPDX-License-Identifier: Apache-2.0
"""State intervals remain exact when an allocation facade hides view offsets."""
import pytest
import torch

from vllm_gaudi.ops.gdn_state_update import validate_direct_state_views


class AllocationFacade:

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def data_ptr(self):
        return self.tensor.untyped_storage().data_ptr()


def test_shared_pool_disjoint_views_with_allocation_facade():
    pool = torch.empty((3, 24, 128, 128), dtype=torch.float32)
    views = [AllocationFacade(pool[index:index + 1]) for index in range(3)]
    assert len({view.data_ptr() for view in views}) == 1
    assert validate_direct_state_views(views) == 3


@pytest.mark.parametrize("offset", [0, 17])
def test_shared_pool_true_overlap_is_rejected(offset):
    pool = torch.empty((2, 24, 128, 128), dtype=torch.float32)
    row = pool[:1]
    overlapping = pool.flatten()[offset:offset + row.numel()].reshape_as(row)
    with pytest.raises(RuntimeError, match="overlap"):
        validate_direct_state_views([AllocationFacade(row), AllocationFacade(overlapping)])
