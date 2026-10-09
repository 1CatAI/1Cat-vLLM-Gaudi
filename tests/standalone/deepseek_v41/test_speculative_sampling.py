# SPDX-License-Identifier: Apache-2.0
"""Check the output-distribution identity and longest-prefix semantics."""
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
import pytest

from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
    bounded_proposal_distribution,
    filtered_distribution,
    sample_speculative_prefix,
    sample_speculative_prefix_sharded,
    sample_proposal_sharded,
    SpeculativeRequestSampling,
    sample_full_distribution,
)


@pytest.mark.parametrize("rows", (1, 5, 6))
def test_native_full_repair_preserves_rank_order_and_wire_capacity(monkeypatch, rows):
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import full_sampling_peer_logits
    import vllm_gaudi.ops.deepseek_v41_replay as replay

    shards = [torch.arange(rows * 32320, dtype=torch.float32).reshape(rows, 32320) + rank * 1000000
              for rank in range(4)]
    # The final partial transfer must retain its columns within each rank.
    for rank in range(4):
        offset = 0
        calls = []

        def stage_collectives(tp_rank, native, tp_size, *, native_fp32_gather, rank=rank, calls=calls):
            assert (tp_rank, native, tp_size, native_fp32_gather) == (rank, True, 4, True)

            def gather(value, dim):
                nonlocal offset
                assert dim == -1 and value.dtype == torch.float32
                assert value.is_contiguous() and value.numel() * 2 <= 32768
                width = value.shape[-1]
                assert torch.equal(value, shards[rank][:, offset:offset + width])
                result = torch.cat([shard[:, offset:offset + width] for shard in shards], -1)
                calls.append(width)
                offset += width
                return result

            return None, gather

        monkeypatch.setattr(replay, "stage_collectives", stage_collectives)
        actual = full_sampling_peer_logits(shards[rank], rank, 4)
        assert torch.equal(actual, torch.cat(shards, -1))
        assert offset == 32320 and len(calls) == (13 if rows == 6 else 10 if rows == 5 else 2)


@pytest.mark.parametrize("kind", ("narrow", "wide", "overflow"))
def test_threshold_packet_certifies_exact_nucleus_or_rejects(monkeypatch, kind):
    from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet, sample_nucleus_packet

    def compact(scores, cutoff, width):
        values = torch.full((scores.shape[0], width), -float("inf"))
        ids = torch.zeros(values.shape, dtype=torch.int32)
        counts = torch.zeros((scores.shape[0], 1), dtype=torch.int32)
        for row in range(scores.shape[0]):
            selected = (scores[row] > cutoff[row // 8]).nonzero().flatten()
            counts[row] = min(selected.numel(), width + 1)
            selected = selected[:width]
            values[row, :selected.numel()] = scores[row, selected]
            ids[row, :selected.numel()] = selected.int()
        return values, ids, counts

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_vocab_filter_gaudi2", compact, raising=False)
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK_STREAM_FILTER_SAMPLING", "1")
    logits = torch.arange(8192).float().repeat(6, 1) * .00001
    if kind == "narrow":
        logits -= 20
        logits[:, 17:20] = torch.tensor([4., 3., 2.])
    elif kind == "overflow":
        logits -= 20
        logits[:, :65] = torch.arange(65).float() * .001 + 10
    controls = torch.tensor([[1., .95, .42, -1.]]).repeat(6, 1)
    packet = torch.cat([local_nucleus_packet(logits[:, rank * 2048:(rank + 1) * 2048], controls, rank, 64)
                        for rank in range(4)], -1)
    token, covered = sample_nucleus_packet(packet, controls, tp_size=4, width=64)
    shards = [bounded_proposal_distribution(packet, controls, tp_rank=rank, tp_size=4, width=64, local_vocab=2048)
              for rank in range(4)]
    probability = torch.cat([value[0] for value in shards], -1)
    assert torch.isfinite(probability).all()
    if kind == "narrow":
        expected_token, expected_probability = sample_full_distribution(logits, controls)
        assert covered.all() and all(value[1].all() for value in shards)
        assert torch.equal(token.reshape(-1), expected_token)
        torch.testing.assert_close(probability, expected_probability, rtol=2e-5, atol=2e-7)
    else:
        assert not covered.any() and not any(value[1].any() for value in shards)


def test_single_sort_probability_and_token_match_ordinary_sampling():
    from vllm_gaudi.ops.deepseek_v41_sampling import sample_probabilities

    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(6, 4096, generator=generator)
    logits[4].zero_()  # Preserve the ordinary sort's equal-score order.
    controls = torch.tensor([[1., .95, .42, -1.], [.7, .9, .77, 17.],
                             [1., 1., .15, -1.], [0., .95, .7, -1.],
                             [1., .95, .99, -1.], [1., .01, .2, 1.]])
    token, probability = sample_full_distribution(logits, controls)
    expected = sample_probabilities(logits, controls, filtered=True).reshape(-1)
    reference = filtered_distribution(logits, controls[:, 0], controls[:, 1], controls[:, 3])
    assert torch.equal(token, expected)
    assert torch.equal(probability, reference)


def test_request_owned_sampling_draws_survive_interleaving_and_reuse():
    first = SpeculativeRequestSampling((1., .95, -1.), 42, 32, "cpu")
    second = SpeculativeRequestSampling((1., .95, -1.), 43, 32, "cpu")
    reference = SpeculativeRequestSampling((1., .95, -1.), 42, 32, "cpu")
    first.proposal.fill_(.03125)
    assert second.proposal.count_nonzero() == 0
    assert first.proposal.data_ptr() != second.proposal.data_ptr()
    for _ in range(4):
        actual = first.next_draws()
        second.next_draws()
        expected = reference.next_draws()
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        assert bool(((actual[1] > 0) & (actual[1] < 1)).all())
    fresh = SpeculativeRequestSampling((1., .95, -1.), 42, 32, "cpu")
    assert fresh.counter.item() == 0
    assert not fresh.proposal_valid.item()
    assert first.counter.item() == 44


def test_actual_bounded_proposal_probability_matches_its_sampler():
    from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet, sample_nucleus_packet

    shards = [torch.tensor([[4., 3., 2., 1.], [3., 2., 1., 0.]]),
              torch.tensor([[.2, .1, 0., -.1], [1.5, .5, -.5, -1.5]])]
    controls = torch.tensor([[1., .95, .42, -1.], [0., .95, .87, -1.]])
    packet = torch.cat([local_nucleus_packet(x, controls, r, 2) for r, x in enumerate(shards)], -1)
    expected, certificate = sample_nucleus_packet(packet, controls, tp_size=2, width=2)
    results = [sample_proposal_sharded(x, controls, tp_rank=r, tp_size=2, width=2,
                                      all_gather=lambda value, dim: packet) for r, x in enumerate(shards)]
    for selected, _, covered in results:
        assert torch.equal(selected, expected.reshape(-1))
        assert torch.equal(covered, certificate.reshape(-1))
    actual = torch.cat([result[1] for result in results], -1)
    torch.testing.assert_close(actual.sum(-1), torch.ones(2))
    reference, _ = bounded_proposal_distribution(packet, controls, tp_rank=0, tp_size=2, width=2, local_vocab=8)
    torch.testing.assert_close(actual[:1], reference[:1])
    assert actual[1, expected[1].item()] == 1
    assert int((actual[1] != 0).sum()) == 1


def test_sampled_commit_preserves_rejection_even_when_correction_matches_proposal():
    from vllm_gaudi.ops.deepseek_v41_verify import verify_control_from_sampled

    # The first draft was rejected but correction sampling returned its ID.
    # ID equality must not turn this into an extra accepted position.
    metadata = torch.tensor([1, 6, 5, 100, 16384, 1048576, 1])
    result = verify_control_from_sampled(torch.tensor([7, -1, -1, -1, -1, -1]),
                                         torch.tensor([1]), torch.tensor([True]), metadata)
    output, committed, count, anchor, enabled, status = result
    assert output.tolist() == [7, -1, -1, -1, -1, -1]
    assert (committed.item(), count.item(), anchor.item(), enabled.item(), status.item()) == (1, 1, 7, True, 0)
    bad = verify_control_from_sampled(output, committed, torch.tensor([False]), metadata)
    assert bad[-1].item() == 1
    exhausted = metadata.clone()
    exhausted[3] = 1
    limited = verify_control_from_sampled(output, committed, torch.tensor([True]), exhausted)
    assert not limited[-2].item()


def test_sampled_prefix_rejects_uncertified_target_before_context_commit():
    from types import SimpleNamespace
    from vllm_gaudi.models.deepseek_v41_program import PreparedDraft

    controls = torch.tensor([[1., .95, .37, -1.]]).repeat(6, 1)
    metadata = torch.tensor([1, 6, 5, 100, 16384, 1048576, 1])
    for certified in (True, False):
        # Wide, nearly uniform targets cannot certify a two-candidate packet.
        logits = torch.full((6, 8), -6.) if certified else torch.arange(8).float().repeat(6, 1) * .01
        if certified:
            logits[:, 0] = 6.
        p = filtered_distribution(logits, controls[:, 0], controls[:, 1], controls[:, 3])
        q, proposed = p[:5], torch.zeros(5, dtype=torch.int64)
        barrier = threading.Barrier(2)
        buffers = [None] * 2

        def rank_case(rank, barrier=barrier, buffers=buffers, logits=logits, q=q, proposed=proposed):
            def gather(value, dim):
                buffers[rank] = value
                barrier.wait(timeout=10)
                result = torch.cat(buffers, dim)
                barrier.wait(timeout=10)
                return result

            commits = []
            owner = SimpleNamespace(tp_rank=rank, tensor_parallel_size=2, all_gather=gather,
                                    _head_projection=lambda x: x,
                                    insert_context=lambda states, positions, count: commits.append(count.clone()))
            result = PreparedDraft.verify_sampled_prefix(
                owner, logits[:, rank * 4:(rank + 1) * 4].contiguous(), proposed,
                q[:, rank * 4:(rank + 1) * 4].contiguous(), metadata,
                torch.zeros(6, 2), torch.arange(6), controls, torch.zeros(5), torch.tensor(.17), width=2)
            return result, commits

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(rank_case, range(2)))
        for result, commits in results:
            if certified:
                assert result[5].item() == 0
                assert result[1].item() == 6
                assert commits[0].item() == 6
            else:
                assert result[5].item() == 2
                assert result[1].item() == result[2].item() == commits[0].item() == 0
                assert result[0].tolist() == [-1] * 6
                assert not result[4].item()


def test_sharded_rejection_matches_global_control_with_zero_mass_shards():
    p = torch.tensor([[.1, .2, .3, .4, 0., 0., 0., 0.]] * 5 +
                     [[0., 0., 0., 0., .1, .2, .3, .4]], dtype=torch.float64)
    q = torch.tensor([[.4, .3, .2, .1, 0., 0., 0., 0.]] * 5, dtype=torch.float64)
    proposed = torch.tensor([0, 1, 2, 3, 0])
    for draws in (torch.zeros(5), torch.ones(5) * .9, torch.tensor([0., 0., .99, 0., 0.])):
        for correction in (.13, .77):
            expected = sample_speculative_prefix(p, q, proposed, draws, torch.tensor(correction))
            barrier = threading.Barrier(4)
            buffers = [None] * 4

            def rank_case(rank, draws=draws, correction=correction, barrier=barrier, buffers=buffers):
                def gather(value, dim):
                    buffers[rank] = value
                    barrier.wait(timeout=10)
                    result = torch.cat(buffers, dim)
                    barrier.wait(timeout=10)
                    return result

                return sample_speculative_prefix_sharded(
                    p[:, rank * 2:(rank + 1) * 2], q[:, rank * 2:(rank + 1) * 2], proposed, draws,
                    torch.tensor(correction), rank, 4, gather)

            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(rank_case, range(4)))
            for result in results:
                for actual, reference in zip(result, expected, strict=True):
                    assert torch.equal(actual, reference)


def test_certified_bounded_probabilities_match_full_official_nucleus():
    from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet, sample_nucleus_packet

    torch.manual_seed(42)
    logits = torch.randn(6, 256) * 6
    controls = torch.tensor([[1., .95, .37, -1.]]).repeat(6, 1)
    packet = torch.cat([local_nucleus_packet(logits[:, r * 64:(r + 1) * 64], controls, r, 32)
                        for r in range(4)], -1)
    parts = [bounded_proposal_distribution(packet, controls, tp_rank=r, tp_size=4, width=32, local_vocab=64)
             for r in range(4)]
    assert all(bool(cert.all()) for _, cert in parts)
    probability = torch.cat([p for p, _ in parts], -1)
    reference = filtered_distribution(logits, controls[:, 0], controls[:, 1], controls[:, 3])
    torch.testing.assert_close(probability, reference, rtol=2e-6, atol=1e-7)
    _, covered = sample_nucleus_packet(packet, controls, tp_size=4, width=32)
    assert bool(covered.all())


def test_uncertified_tied_packet_is_only_a_normalized_actual_proposal():
    from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet

    controls = torch.tensor([[1., .95, .4, -1.]])
    logits = torch.zeros(1, 256)
    packet = torch.cat([local_nucleus_packet(logits[:, r * 64:(r + 1) * 64], controls, r, 8)
                        for r in range(4)], -1)
    parts = [bounded_proposal_distribution(packet, controls, tp_rank=r, tp_size=4, width=8, local_vocab=64)
             for r in range(4)]
    assert all(not bool(cert.any()) for _, cert in parts)
    probability = torch.cat([p for p, _ in parts], -1)
    torch.testing.assert_close(probability.sum(-1), torch.ones(1))
    assert int((probability > 0).sum()) == 32
    assert not torch.equal(probability, filtered_distribution(logits, controls[:, 0], controls[:, 1], controls[:, 3]))


def test_foreign_packet_candidates_do_not_overwrite_local_token_zero():
    from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet

    logits = torch.tensor([[9., 0., -1., -2., 8., -3., -4., -5.]])
    controls = torch.tensor([[1., .95, .5, -1.]])
    packet = torch.cat([local_nucleus_packet(logits[:, r * 4:(r + 1) * 4], controls, r, 4)
                        for r in range(2)], -1)
    p, cert = bounded_proposal_distribution(packet, controls, tp_rank=0, tp_size=2, width=4, local_vocab=4)
    reference = filtered_distribution(logits, controls[:, 0], controls[:, 1], controls[:, 3])[:, :4]
    assert bool(cert.all()) and p[0, 0] > 0
    torch.testing.assert_close(p, reference)


def test_nucleus_and_top_k_preserve_original_vocabulary_and_normalize():
    logits = torch.tensor([[.1, .4, .2, .3]]).log()
    probability = filtered_distribution(logits, torch.ones(1), torch.tensor([.65]), torch.zeros(1))
    torch.testing.assert_close(probability, torch.tensor([[0., 4 / 7, 0., 3 / 7]]))
    capped = filtered_distribution(logits, torch.ones(1), torch.ones(1), torch.tensor([1]))
    assert capped.tolist() == [[0., 1., 0., 0.]]


def test_actual_proposal_probability_changes_with_markov_bias():
    logits = torch.tensor([[0., 1., -1.], [0., 1., -1.]])
    biased = logits + torch.tensor([[0., 0., 0.], [3., 0., 0.]])
    probability = filtered_distribution(biased, torch.ones(2), torch.ones(2), torch.zeros(2))
    torch.testing.assert_close(probability, biased.softmax(-1))
    assert probability.argmax(-1).tolist() == [1, 0]


def test_rejection_correction_recovers_target_distribution_exactly():
    # Analytically enumerate proposal draws; no noisy statistical assertion.
    p = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    q = torch.tensor([.4, .3, .2, .1], dtype=torch.float64)
    accepted_mass = q * (p / q).clamp_max(1)
    rejection_probability = 1 - accepted_mass.sum()
    residual = (p - q).clamp_min(0)
    result = accepted_mass + rejection_probability * residual / residual.sum()
    torch.testing.assert_close(result, p, rtol=0, atol=1e-15)
    # Exercise the actual control function too. These midpoint grids exactly
    # integrate the rational acceptance and correction boundaries above.
    measured = torch.zeros_like(p)
    for proposed_id in range(4):
        for acceptance_draw in range(12):
            for correction_draw in range(4):
                output, _, valid = sample_speculative_prefix(
                    p.repeat(6, 1), q.repeat(5, 1), torch.full((5,), proposed_id),
                    torch.full((5,), (acceptance_draw + .5) / 12, dtype=torch.float64),
                    torch.tensor((correction_draw + .5) / 4, dtype=torch.float64))
                assert valid.tolist() == [True]
                measured[output[0]] += q[proposed_id] / 48
    torch.testing.assert_close(measured, p, rtol=0, atol=1e-15)


def test_first_rejection_discards_later_acceptances_and_excludes_rejected_token():
    target = torch.tensor([[.2, .8]] * 6)
    proposal = torch.tensor([[.8, .2]] * 5)
    proposed = torch.zeros(5, dtype=torch.int64)
    output, committed, valid = sample_speculative_prefix(
        target, proposal, proposed, torch.tensor([0., .9, 0., 0., 0.]), torch.tensor(.5))
    assert output.tolist() == [0, 1, -1, -1, -1, -1]
    assert committed.tolist() == [2] and valid.tolist() == [True]


def test_all_accepted_uses_bonus_target_instead_of_proposal_residual():
    target = torch.tensor([[1., 0.]] * 5 + [[0., 1.]])
    proposal = torch.tensor([[1., 0.]] * 5)
    output, committed, valid = sample_speculative_prefix(
        target, proposal, torch.zeros(5, dtype=torch.int64), torch.full((5,), .9), torch.tensor(.5))
    assert output.tolist() == [0, 0, 0, 0, 0, 1]
    assert committed.tolist() == [6] and valid.tolist() == [True]


def test_greedy_proposal_can_be_used_only_with_probability_rejection():
    target = torch.tensor([[.25, .75]] * 6)
    proposal = torch.tensor([[1., 0.]] * 5)
    output, committed, valid = sample_speculative_prefix(
        target, proposal, torch.zeros(5, dtype=torch.int64), torch.full((5,), .5), torch.tensor(.5))
    assert output.tolist() == [1, -1, -1, -1, -1, -1]
    assert committed.tolist() == [1] and valid.tolist() == [True]


def test_sampled_metadata_matches_greedy_wire_budget_and_context_boundary():
    from vllm_gaudi.ops.deepseek_v41_verify import verify_control_from_sampled, verify_control_from_target

    target = torch.arange(6)
    proposed = torch.arange(5)
    for sample, remaining, start, limit in ((1, 100, 16384, 1048576), (0, 100, 16384, 1048576),
                                           (1, 6, 16384, 1048576), (1, 100, 1048570, 1048576)):
        metadata = torch.tensor([9, 6, 5, remaining, start, limit, sample])
        reference = verify_control_from_target(target, proposed, metadata)[1:]
        actual = verify_control_from_sampled(target, torch.tensor([6]), torch.tensor([True]), metadata)
        assert all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))


def test_full_sampled_fallback_commits_seed_and_uncovered_c6_targets():
    from types import SimpleNamespace
    from vllm_gaudi.models.deepseek_v41_program import PreparedDraft
    from vllm_gaudi.ops.deepseek_v41_sampling import sample_probabilities

    logits = torch.arange(48).reshape(6, 8).float() * .01
    controls = torch.tensor([[1., .95, .37, -1.]]).repeat(6, 1)
    probability = filtered_distribution(logits, controls[:, 0], controls[:, 1], controls[:, 3])
    seed_tokens = sample_probabilities(logits, controls, filtered=True).reshape(-1).long()
    proposed = seed_tokens[:5]
    for count, proposal_count in ((1, 0), (4, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5)):
        metadata = torch.tensor([9, count, proposal_count, 100, 16384, 1048576, 1])
        barrier, buffers = threading.Barrier(2), [None] * 2

        def rank_case(rank, buffers=buffers, barrier=barrier, metadata=metadata):
            def gather(value, dim):
                buffers[rank] = value
                barrier.wait(timeout=10)
                result = torch.cat(buffers, dim)
                barrier.wait(timeout=10)
                return result

            commits = []
            owner = SimpleNamespace(tp_rank=rank, tensor_parallel_size=2, all_gather=gather,
                                    _head_projection=lambda value: value,
                                    insert_context=lambda states, positions, valid: commits.append(valid.clone()))
            result = PreparedDraft.verify_sampled_prefix(
                owner, logits[:, rank * 4:(rank + 1) * 4].contiguous(), proposed,
                probability[:5, rank * 4:(rank + 1) * 4].contiguous(), metadata,
                torch.zeros(6, 2), torch.arange(6), controls, torch.zeros(5), torch.tensor(.17), full=True)
            return result, commits

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(rank_case, range(2)))
        for result, commits in results:
            assert result[5].item() == 0
            assert result[1].item() == commits[0].item() == count
            if proposal_count == 0:
                assert result[2].item() == 1
                assert result[0][0] == seed_tokens[count - 1]
            else:
                expected = torch.full((6,), -1, dtype=torch.int64)
                expected[:proposal_count] = proposed[:proposal_count]
                expected[proposal_count] = ((probability[proposal_count].cumsum(-1) < .17).sum()
                                             .clamp_max(logits.shape[-1] - 1))
                assert torch.equal(result[0], expected)


def test_unconsumed_target_rows_cannot_change_a_verified_prefix():
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
        sample_speculative_prefix, target_coverage_for_commit)

    proposed = torch.tensor([0, 1, 2, 3, 4])
    q = torch.nn.functional.one_hot(proposed, num_classes=8).float()
    acceptance = torch.full((5,), .5)
    for rejected in (0, 2, 4, 5):
        p = torch.cat((q, torch.nn.functional.one_hot(torch.tensor([7]), num_classes=8).float()))
        if rejected < 5:
            p[rejected].zero_()
            p[rejected, 7] = 1
        reference, count, valid = sample_speculative_prefix(p, q, proposed, acceptance, torch.tensor([.4]))
        changed = p.clone()
        changed[rejected + 1:].zero_()
        changed[rejected + 1:, 6] = 1
        actual, actual_count, actual_valid = sample_speculative_prefix(
            changed, q, proposed, acceptance, torch.tensor([.4]))
        assert valid and actual_valid and torch.equal(reference, actual) and torch.equal(count, actual_count)
        covered = torch.arange(6) <= rejected
        assert target_coverage_for_commit(covered, actual_count)
        covered[rejected] = False
        assert not target_coverage_for_commit(covered, actual_count)
