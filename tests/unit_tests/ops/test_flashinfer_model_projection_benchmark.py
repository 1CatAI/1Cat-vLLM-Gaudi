# SPDX-License-Identifier: Apache-2.0
"""Keep cold rounds and incomplete model evidence out of valid screens."""

from unittest import mock

import pytest

from tools.benchmark_flashinfer_model_projection import warm_model, warm_timings_stable


def test_symmetric_warmup_requires_at_least_two_complete_rounds():
    llm = mock.Mock()
    with pytest.raises(ValueError, match="two"):
        warm_model(llm, ["prompt"], "sampling", 1)
    llm.generate.assert_not_called()
    warm_model(llm, ["prompt"], "sampling", 2)
    assert llm.generate.call_args_list == [mock.call(["prompt"], "sampling", use_tqdm=False)] * 2


@pytest.mark.parametrize("rates,valid", [([1., 1., 1.], True), ([.2, 1., 1.], False), ([1., 1., 2.], False),
                                         ([1., 1.], False), ([1., 0., 1.], False), ([1., float("nan"), 1.], False)])
def test_every_model_round_must_be_stable(rates, valid):
    workers = [{
        "rounds": [{
            "output_tokens_per_s": rate
        } for rate in values]
    } for values in ([1., 1., 1.], rates, [1., 1., 1.])]
    assert warm_timings_stable(workers) is valid
    assert not warm_timings_stable(workers[:2])
