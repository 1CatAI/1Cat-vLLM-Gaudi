# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops import tp2_prepared_plan as replay


class Plan:

    def __init__(self, key, output):
        self.key, self.output, self.valid = key, output, True

    def matches(self, inputs):
        return self.valid and inputs[0] == self.key

    def outputs(self):
        return [self.output]

    def invalidate(self):
        self.valid = False


@pytest.fixture(autouse=True)
def clear_native_graphs():
    replay._native_graphs.clear()
    yield
    replay._native_graphs.clear()


@pytest.fixture
def runtime(monkeypatch):
    events = []

    class Bridge:

        def replay_prepared_groups(self, plans, inputs):
            events.append((tuple(p.key for p in plans), inputs))

    monkeypatch.setattr(replay, "_runtime", lambda: (Bridge(), None))
    return events


def module(key):
    instance = replay.PreparedGroupModule(torch.nn.Identity())
    instance.plans.append(Plan(key, torch.tensor(key)))
    return instance


def test_cold_consumer_submits_prior_producers(runtime, monkeypatch):
    first, cold = module(1), module(9)

    def prepare(inputs):
        assert runtime == [((1, ), [[1]])]
        assert inputs == [2]
        return (torch.tensor(2), )

    monkeypatch.setattr(cold, "_prepare", prepare)
    with replay.collect_prepared_group_replays():
        assert first([1])[0].item() == 1
        assert runtime == []
        assert cold([2])[0].item() == 2
    assert len(runtime) == 1


def test_consecutive_groups_share_one_submission(runtime):
    first, second = module(1), module(2)
    with replay.collect_prepared_group_replays():
        first([1])
        with replay.collect_prepared_group_replays():
            second([2])
        assert runtime == []
    assert runtime == [((1, 2), [[1], [2]])]


def test_state_mutation_failure_is_never_retried(runtime, monkeypatch):
    first, cold = module(1), module(9)
    state = torch.tensor(0)

    def prepare(_):
        state.add_(1)
        raise RuntimeError("failure after state mutation")

    monkeypatch.setattr(cold, "_prepare", prepare)
    with pytest.raises(RuntimeError, match="after state mutation"), replay.collect_prepared_group_replays():
        first([1])
        cold([2])
    assert state.item() == 1
    assert runtime == [((1, ), [[1]])]


def test_invalidation_submits_pending_work_then_releases_generation(runtime):
    first = module(1)
    plan = first.plans[0]
    with replay.collect_prepared_group_replays():
        first([1])
        replay.invalidate_prepared_group_plans()
        assert runtime == [((1, ), [[1]])]
    assert not plan.valid and not first.plans


@pytest.mark.parametrize("groups", [1, 8])
def test_native_full_decoder_captures_then_updates_and_replays(monkeypatch, groups):
    events = []
    graphs = []

    class NativeGraph:

        def __init__(self):
            self.replays = 0
            graphs.append(self)

        def capture(self, plans, inputs):
            events.append(("capture", tuple(p.key for p in plans), inputs))

        def instantiate(self):
            events.append(("instantiate", ))

        def update_inputs(self, plans, inputs):
            events.append(("update", tuple(p.key for p in plans), inputs))

        def replay(self):
            self.replays += 1
            events.append(("replay", ))

        def replay_count(self):
            return self.replays

        def segment_count(self):
            return 258

        def collective_count(self):
            return groups * 16

        def external_collective_count(self):
            return int(groups == 8)

        def captured_command_count(self):
            return 256

        def captured_relocation_count(self):
            return 1024

        def global_program_bytes(self):
            return 4096

        def arc_program_bytes(self):
            return 2048

        def hcl_command_bytes_per_replay(self):
            return 8192

        def hcl_stream_ccb_bytes(self):
            return 262144

        def hcl_replay_bytes(self):
            return 16384 if self.replays else 0

        def hcl_ccb_wrap_count(self):
            return 0

        def hcl_submission_count(self):
            return 4 if self.replays else 0

    class Bridge:
        NativeDecodeGraph = NativeGraph

        @staticmethod
        def native_decode_graph_available():
            return True

        @staticmethod
        def replay_prepared_groups(plans, inputs):
            events.append(("host", tuple(p.key for p in plans), inputs))

    monkeypatch.setattr(replay, "_runtime", lambda: (Bridge(), None))
    monkeypatch.setenv("VLLM_HPU_NATIVE_DECODE_GRAPH", "1")
    monkeypatch.setenv("VLLM_HPU_NATIVE_DECODE_GRAPH_QUALIFICATION_GROUPS", str(groups))
    modules = [module(index) for index in range(8)]

    for _ in range(2):
        with replay.collect_prepared_group_replays():
            for index, prepared in enumerate(modules):
                prepared([index])

    assert len(graphs) == 1
    if groups == 8:
        assert [event[0] for event in events] == ["capture", "instantiate", "update", "replay"]
    else:
        assert [event[0]
                for event in events] == ["host", "capture", "instantiate", "host", "host", "update", "replay", "host"]
        assert [event[1] for event in events if event[0] in ("capture", "update")] == [(1, ), (1, )]
        assert [event[1] for event in events if event[0] == "host"] == [(0, ), (2, 3, 4, 5, 6, 7)] * 2
    assert replay.prepared_group_stats()["native_replays"] == 1


def test_native_request_fails_closed_when_runtime_symbols_are_missing(monkeypatch):

    class Bridge:
        NativeDecodeGraph = object

        @staticmethod
        def native_decode_graph_available():
            return False

        @staticmethod
        def replay_prepared_groups(*_):
            raise AssertionError("native request must not run host replay")

    monkeypatch.setattr(replay, "_runtime", lambda: (Bridge(), None))
    monkeypatch.setenv("VLLM_HPU_NATIVE_DECODE_GRAPH", "1")
    modules = [module(index) for index in range(8)]
    with pytest.raises(RuntimeError, match="no fallback was executed"), replay.collect_prepared_group_replays():
        for index, prepared in enumerate(modules):
            prepared([index])


def test_identity_boundary_view_and_inferred_dimension():
    value = torch.zeros(1, 5120)
    replay._verify_identity_view(value, value.view(1, -1))
    replay._verify_identity_view(value, value.as_strided((1, 5120), (5120, 1), 0))


def test_norm_singleton_alias_does_not_allow_rebinding_or_state_aliases():
    pool = torch.zeros(2, 5120, dtype=torch.bfloat16)
    value = pool[:1]
    assert replay._is_norm_singleton_view(value, value.view(1, 1, 5120))
    assert not replay._is_norm_singleton_view(value, pool[1:].view(1, 1, 5120))
    assert not replay._is_norm_singleton_view(value, value.clone().view(1, 1, 5120))
    assert not replay._is_norm_singleton_view(value, value.view(5120))
    state = value.float()
    assert not replay._is_norm_singleton_view(state, state.view(1, 1, 5120))


def test_consumer_fusion_coverage_requires_all_new_compute_programs():
    assert replay.native_compute_coverage_matches(16, 16, 1, True)
    assert replay.native_compute_coverage_matches(129, 128, 8, True)
    assert not replay.native_compute_coverage_matches(15, 16, 1, True)
    assert not replay.native_compute_coverage_matches(128, 128, 8, True)
    assert not replay.native_compute_coverage_matches(16, 16, 1, False)
    assert replay.native_compute_coverage_matches(33, 16, 1, False)


@pytest.mark.parametrize("change", ["shape", "offset", "storage"])
def test_nonidentity_view_is_rejected(change):
    pool = torch.zeros(2, 5120)
    value = pool[0:1]
    result = value.view(5120) if change == "shape" else pool[1:2] if change == "offset" else value.clone()
    with pytest.raises(RuntimeError, match="changes layout"):
        replay._verify_identity_view(value, result)
