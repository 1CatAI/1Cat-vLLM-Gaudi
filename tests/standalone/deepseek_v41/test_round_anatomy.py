# SPDX-License-Identifier: Apache-2.0
"""Mutually exclusive engine allocation, including foreign-card activity."""
import importlib.util
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[3] / 'tools'
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location('round_anatomy', TOOLS / 'report_deepseek_v41_round_anatomy.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_cross_card_overlap_is_additive():
    edges = {0: [((0, 'TPC'), 1)], 1000: [((0, 'MME'), 1)],
             2000: [((0, 'TPC'), -1)], 3000: [((0, 'MME'), -1), ((1, 'TPC'), 1)],
             4000: [((1, 'TPC'), -1)]}
    result = REPORT.intervals(edges, 0, 5000, lambda active: REPORT.state(active, 0))
    assert result == dict(TPC_only=1., overlap=1., MME_only=1.,
                          single_card_idle_other_cards_active=1., four_cards_idle=1.)
    assert sum(result.values()) == 5.


def test_null_descriptor_does_not_count_as_useful_tpc():
    edges = {0: [((0, 'TPC_null'), 1)], 1000: [((0, 'MME'), 1)],
             2000: [((0, 'TPC_null'), -1), ((0, 'MME'), -1)]}
    result = REPORT.intervals(edges, 0, 3000, lambda active: REPORT.state(active, 0))
    assert result == dict(HCL_DMA_descriptor_only=1., MME_only=1., four_cards_idle=1.)


def test_empty_window_is_idle():
    assert REPORT.intervals({}, 0, 5000, lambda active: REPORT.state(active, 0)) == dict(four_cards_idle=5.)


def test_category_overlap_is_not_double_credited():
    spans = {'routed_experts': [(0, 2000), (1000, 3000)], 'mHC': [(2000, 4000)]}
    result = REPORT.category_allocation(spans, 0, 5000)
    assert result == {'exclusive:routed_experts': 2., 'overlap:mHC+routed_experts': 1.,
                      'exclusive:mHC': 1., 'no_useful_compute': 1.}
    assert sum(result.values()) == 5.


def test_category_activity_does_not_define_engine_idle():
    assert REPORT.category_allocation({}, 0, 5000) == {'no_useful_compute': 5.}


def test_category_window_excludes_neighboring_rounds():
    spans = {'mHC': [(-2000, -1000), (-100, 1000), (6000, 7000)]}
    assert REPORT.category_allocation(spans, 0, 5000) == {'exclusive:mHC': 1., 'no_useful_compute': 4.}


def test_submission_snapshot_scope_includes_terminal_round():
    before = dict(native_program_generation=1, v41={'decode_steps': 100},
                  native_entry_replays=200, native_joint_compute_submissions=1000)
    after = dict(native_program_generation=1, v41={'decode_steps': 116},
                 native_entry_replays=232, native_joint_compute_submissions=4696)
    result = REPORT.native_submission_counts(before, after)
    assert result['rounds'] == 16
    assert result['per_round'] == {'native_entry_replays': 2., 'native_joint_compute_submissions': 231.}


def test_submission_generation_reset_is_not_a_gain():
    assert 'incomparable' in REPORT.native_submission_counts(
        {'native_program_generation': 1}, {'native_program_generation': 2})['status']


def test_submission_empty_acquisition_is_unknown():
    assert 'no completed' in REPORT.native_submission_counts({}, {})['status']


def test_debug_contract_lookup_preserves_engine_and_full_context():
    rows = [dict(recipe_id='71@cached', symbol=dict(device_type=kind, full_context_id=context,
                                                  node=f'{kind}/{context}'))
            for kind, context in ((0, 0), (1, 0), (1, 1))]
    index = REPORT.contract_index(rows)
    node = dict(recipe='71@cached:', engine='TPC', raw_context_id='1', raw_unique_node_id='')
    assert REPORT.node_contract(node, index)['symbol']['node'] == '1/1'
    node.update(engine='MME', raw_context_id='0')
    assert REPORT.node_contract(node, index)['symbol']['node'] == '0/0'
    node['recipe'] = '72@cached:'
    assert REPORT.node_contract(node, index) == {}


def test_split_expert_decoder_is_not_auxiliary_and_silu_is_separate():
    def category(kernel):
        return REPORT.classify(dict(kernel=kernel, engine='TPC'), {})
    assert category('custom_deepseek_v41_expert_w13_split_scale_gaudi2') == 'routed_experts'
    assert category('custom_deepseek_v41_expert_n256_silu_quant_gaudi2') == 'expert_activation_reduce'
    assert category('custom_deepseek_v41_expert_n256_scale_reduce_gaudi2') == 'expert_activation_reduce'
    assert category('custom_deepseek_v41_router_logits_top6_gaudi2') == 'router'
