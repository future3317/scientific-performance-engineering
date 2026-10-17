"""No provider, SDK, gamefile, or environment needed for public replay audits."""
import pytest

from experiments.skill_evolution.discovery_audit import (
    opportunity_summary, public_opportunities, research_closeout,
)


def episode():
    observation = ('You are in the middle of a room.\n'
                   'Your task is to: put a pan in diningtable.')
    actions = ['go to stoveburner 1', 'go to stoveburner 2', 'go to diningtable 1']
    at_stove = 'You arrive at stoveburner 1. On the stoveburner 1, you see a pot 1.'
    listed = ['examine stoveburner 1', 'examine stoveburner 2', 'go to diningtable 1']
    trace = [{'step': 0, 'observation': observation, 'admissible_actions': actions,
              'raw_output': 'Action: go to stoveburner 1', 'selected_action': 'go to stoveburner 1',
              'invalid_action': False, 'reward': 0, 'completed_subgoals': []}]
    for step in range(1, 4):
        trace.append({'step': step, 'observation': at_stove, 'admissible_actions': listed,
            'raw_output': 'Action: go to stoveburner 2', 'selected_action': None,
            'invalid_action': True, 'fallback_reason': 'action_not_admissible'})
    trace.append({'step': 4, 'observation': at_stove, 'admissible_actions': listed,
        'raw_output': 'Action: examine stoveburner 2', 'selected_action': 'examine stoveburner 2',
        'invalid_action': False, 'reward': 0, 'completed_subgoals': []})
    return {'trial_id': 'fixture:one', 'cluster_id': 'one', 'trajectory': trace,
            'state': {'last_observation': 'On the stoveburner 2, you see nothing.'}}


def test_three_rejections_are_one_source_and_zero_environment_actions():
    audit = public_opportunities(episode())
    summary = opportunity_summary([audit])
    assert summary['n_opportunity_clusters'] == summary['n_missed_opportunity_clusters'] == 1
    assert summary['n_rejected_navigation_decisions'] == 3
    assert summary['n_trigger_turns'] == 4
    assert summary['n_correct_handling_clusters'] == summary['n_direct_inspection_clusters'] == 1
    missed = [row for row in audit['rows'] if row['primary_opportunity'] and row['missed_opportunity']]
    assert all(row['actual_environment_action'] is None for row in missed)
    assert [row['replan_note_before_action'] for row in missed] == [None, 'action_not_admissible', 'action_not_admissible']


def test_trigger_membership_does_not_depend_on_future_model_output():
    changed = episode()
    changed['trajectory'][1]['raw_output'] = 'Action: something else'
    before = public_opportunities(episode())
    after = public_opportunities(changed)
    keys = lambda audit: [(row['turn_id'], row['fixture'], row['primary_opportunity']) for row in audit['rows']]
    assert keys(before) == keys(after)
    assert opportunity_summary([after])['n_rejected_navigation_decisions'] == 2


def test_literal_current_fixture_is_not_uninspected_search_opportunity():
    audit = public_opportunities(episode())
    assert any(row['fixture'] == 'stoveburner 1' for row in audit['rows'])
    assert not any(row['primary_opportunity'] for row in audit['rows'] if row['fixture'] == 'stoveburner 1')


def test_already_inspected_fixture_is_not_a_new_primary_opportunity():
    row = episode()
    row['trajectory'].append({'step': 5, 'observation': row['state']['last_observation'],
        'admissible_actions': ['examine stoveburner 2'], 'selected_action': 'examine stoveburner 2',
        'raw_output': 'Action: examine stoveburner 2', 'reward': 0, 'invalid_action': False})
    audit = public_opportunities(row)
    assert not any(item['primary_opportunity'] for item in audit['rows'] if item['turn_id'] == 5)


def test_missing_admissible_data_is_unknown_not_correct_handling():
    row = episode()
    del row['trajectory'][1]['admissible_actions']
    audit = public_opportunities(row)
    assert opportunity_summary([audit])['n_unresolved_clusters'] == 1
    assert opportunity_summary([audit])['n_correct_handling_clusters'] == 0


def test_present_navigation_prevents_fallback_trigger():
    row = episode()
    for turn in row['trajectory'][1:]:
        turn['admissible_actions'] = turn['admissible_actions'] + ['go to stoveburner 2']
    assert opportunity_summary([public_opportunities(row)])['n_opportunity_clusters'] == 0


def test_prompt_hash_mismatch_is_reported_not_silently_accepted():
    row = episode()
    row['trajectory'][0]['prompt_sha256'] = 'not-the-prompt'
    assert public_opportunities(row)['prompt_mismatches'] == [0]


@pytest.mark.parametrize('kind,status', [
    ('TRIGGER_PRIORITY', 'G2_PROPOSAL_APPROVAL_CANDIDATE'),
    ('TEXT_NEW_RULE', 'G2_PROPOSAL_APPROVAL_CANDIDATE'),
    ('HARNESS_FIX', 'HARNESS_FIX_CANDIDATE'),
    ('DUPLICATE', 'ALFWORLD_TEXT_SKILL_DISCOVERY_PAUSED'),
])
def test_research_object_controls_next_application_not_execution(kind, status):
    result = research_closeout(complete=True, opportunity_clusters=2,
        reviewed_hypotheses=[{'eligible': True, 'kind': kind, 'independent_support_clusters': ['a', 'b']}])
    assert result['status'] == status
    assert result['next_provider_stage_authorized'] is False
    assert result['treatment_effect'] == 'NOT_DEFINED'


def test_duplicate_turns_and_different_hypotheses_cannot_pool_to_two_sources():
    result = research_closeout(complete=True, opportunity_clusters=2, reviewed_hypotheses=[
        {'eligible': True, 'kind': 'TRIGGER_PRIORITY', 'independent_support_clusters': ['a', 'a']},
        {'eligible': True, 'kind': 'TRIGGER_PRIORITY', 'independent_support_clusters': ['b']},
    ])
    assert result['status'] == 'ALFWORLD_TEXT_SKILL_DISCOVERY_PAUSED'


def test_zero_opportunities_is_undertested_not_causal_rejection():
    result = research_closeout(complete=True, opportunity_clusters=0, reviewed_hypotheses=[])
    assert result['status'] == 'ALFWORLD_TEXT_SKILL_DISCOVERY_PAUSED'
    assert result['hypothesis_under_tested'] is True
    assert 'MECHANISM_REJECTED_CAUSALLY' not in str(result)


def test_incomplete_data_cannot_enter_scientific_stop_or_positive_branch():
    result = research_closeout(complete=False, opportunity_clusters=2, reviewed_hypotheses=[])
    assert result['status'] == 'INCOMPLETE_OR_IDENTITY_BLOCKED'
