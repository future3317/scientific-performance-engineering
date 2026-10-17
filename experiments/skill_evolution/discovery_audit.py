"""Read-only public-trace audits; never construct a provider or environment."""
from core.acre.resource_budget import manifest_digest
from experiments.external_real.alfworld_runner import extract_action
from experiments.external_real.alfworld_stateful_runner import (
    DeepSeekStatefulClient, EpisodeState, ExecutionGuard, HistoryCompressor, SubgoalPlanner,
)
from .attribution import LENGTH_CONTROL


class _CapturePrompt:
    def complete_text(self, messages):
        return messages


def public_opportunities(episode, max_steps=60):
    """Replay saved public prefixes, authenticating available prompt hashes.

    The literal denominator includes current fixtures. The narrower primary
    denominator requires unresolved search and a not-yet-inspected, noncurrent
    fixture from the initial public navigation list. Both definitions are
    fixed before D1b; no future model action determines trigger membership.
    """
    trace = episode.get('trajectory', [])
    if not trace:
        return {'cluster_id': episode['cluster_id'], 'rows': [], 'unknown_turns': 1,
                'prompt_matches': 0, 'prompt_mismatches': [], 'prompt_hash_unknown': 0}
    state = EpisodeState.start(trace[0]['observation'], max_steps)
    fixtures = {action[6:] for action in trace[0].get('admissible_actions', [])
                if action.startswith('go to ')}
    inspected, history, rows, mismatches = set(), [], [], []
    guard, compressor = ExecutionGuard(), HistoryCompressor()
    reason, matches, unknown, hash_unknown = None, 0, 0, 0
    for index, turn in enumerate(trace):
        state.remaining_steps = max_steps - turn['step']
        admissible = turn.get('admissible_actions')
        if not isinstance(admissible, list):
            unknown += 1
            break  # Cannot authenticate later state/action prefixes across a gap.
        if state.last_observation != turn['observation']:
            raise ValueError('public observation replay diverged')
        progress, _ = compressor.render(state, history)
        messages = DeepSeekStatefulClient.choose_action(
            _CapturePrompt(), state, admissible, LENGTH_CONTROL, progress, reason)
        expected_hash = turn.get('prompt_sha256')
        actual_hash = manifest_digest({'messages': messages})
        if expected_hash is None:
            hash_unknown += 1
        elif actual_hash == expected_hash:
            matches += 1
        else:
            mismatches.append(turn['step'])
        candidates = sorted(r for r in fixtures if f'examine {r}' in admissible
                            and f'go to {r}' not in admissible)
        raw = str(turn.get('raw_output', ''))
        requested = extract_action(raw, admissible + [f'go to {r}' for r in candidates])
        accepted = turn.get('selected_action')
        executed = accepted if 'reward' in turn else None
        next_observation = (trace[index+1]['observation'] if index+1 < len(trace)
                            else episode.get('state', {}).get('last_observation'))
        before = set(state.completed_subgoals)
        subgoal = SubgoalPlanner.next_subgoal(state, admissible)
        searching = subgoal.startswith(('locate the goal object', 'acquire the visible goal object'))
        for fixture in candidates:
            primary = searching and fixture != state.current_room_or_location and fixture not in inspected
            missed = bool(turn.get('invalid_action')) and requested == f'go to {fixture}'
            rows.append({
                'cluster_id': episode['cluster_id'], 'turn_id': turn['step'], 'fixture': fixture,
                'public_goal': state.goal, 'visible_location': state.current_room_or_location,
                'visible_goal_state': {'inventory': state.inventory,
                    'completed_subgoals': sorted(before), 'subgoal': subgoal},
                'visible_admissible_commands_hash': manifest_digest({'commands': admissible}),
                'go_to_R_admissible': False, 'examine_R_admissible': True,
                'primary_opportunity': primary, 'raw_model_action': requested,
                'raw_final_content': raw, 'validator_accepted': accepted is not None,
                'actual_environment_action': executed, 'next_visible_observation': next_observation,
                'progress_delta': sorted(set(turn.get('completed_subgoals') or []) - before),
                'reason_for_invalidation': turn.get('fallback_reason'),
                'missed_opportunity': missed,
                'direct_inspection': executed == f'examine {fixture}',
                'legal_alternative': executed is not None and not bool(turn.get('invalid_action')),
                'replan_note_before_action': reason,
                'rejected_actions_in_prompt_history': False,
                'runner_existing_rule_ref': 'alfworld_stateful_runner.py:389-411,681-696; exact-list prompt, generic rejection note, invalid turns absent from recent history',
                'length_prompt_rule_ref': 'attribution.py:LENGTH_CONTROL; concision, exact next valid action',
                'v1_rule_ref': 'Phase7/internal/SKILL_V1.contract.json:procedure[2,8],protected_semantics[1]; inspect admissible actions, do not repeat without new evidence',
                'source_ref': f"episodes.jsonl:{episode['trial_id']}:trajectory[{index}]",
            })
        if turn.get('invalid_action'):
            reason = 'action_not_admissible'
            continue
        if executed is None or next_observation is None:
            unknown += 1
            break
        previous = state.last_observation
        state.observe(next_observation, executed)
        reason = guard.observe_result(state, previous, next_observation, executed)
        history.append({'step': turn['step'], 'action': executed,
            'observation_changed': next_observation != previous,
            'subgoal': SubgoalPlanner.next_subgoal(state, admissible),
            'completed_subgoals': list(state.completed_subgoals), 'stuck_counter': state.stuck_counter})
        if executed.startswith('go to '):
            inspected.add(executed[6:])
        elif executed.startswith('examine '):
            inspected.add(executed[8:])
    return {'cluster_id': episode['cluster_id'], 'rows': rows, 'unknown_turns': unknown,
            'prompt_matches': matches, 'prompt_mismatches': mismatches,
            'prompt_hash_unknown': hash_unknown}


def opportunity_summary(audits, *, primary=True):
    rows = [row for audit in audits for row in audit['rows']
            if not primary or row['primary_opportunity']]
    clusters = lambda key: sorted({row['cluster_id'] for row in rows if row.get(key)})
    missed, direct, correct = clusters('missed_opportunity'), clusters('direct_inspection'), clusters('legal_alternative')
    return {'definition': 'PRIMARY_UNINSPECTED_NONCURRENT_SEARCH_FIXTURE' if primary else 'LITERAL_LISTED_INSPECT_ABSENT_NAVIGATION',
        'n_opportunity_clusters': len({row['cluster_id'] for row in rows}),
        'n_trigger_turns': len({(row['cluster_id'], row['turn_id']) for row in rows}),
        'n_fixture_turn_pairs': len(rows), 'n_missed_opportunity_clusters': len(missed),
        'n_rejected_navigation_decisions': len({(row['cluster_id'], row['turn_id']) for row in rows if row['missed_opportunity']}),
        'n_correct_handling_clusters': len(direct), 'n_direct_inspection_clusters': len(direct),
        'n_legal_alternative_clusters': len(correct),
        'n_direct_inspection_decisions': len({(row['cluster_id'], row['turn_id']) for row in rows if row['direct_inspection']}),
        'n_unresolved_clusters': sum(audit['unknown_turns'] > 0 for audit in audits),
        'missed_clusters': missed, 'direct_inspection_clusters': direct,
        'correct_handling_clusters': direct, 'legal_alternative_clusters': correct,
        'overlap_note': 'A cluster may contain both a missed opportunity and later correct handling; legal alternatives need not inspect R.'}


def research_closeout(*, complete, reviewed_hypotheses, opportunity_clusters):
    """An investigator-supplied semantic review is not inferred from costs."""
    if not complete:
        return {'status': 'INCOMPLETE_OR_IDENTITY_BLOCKED', 'treatment_effect': 'NOT_DEFINED'}
    eligible = [h for h in reviewed_hypotheses if h['eligible']]
    text = [h for h in eligible if h['kind'] in {'TEXT_NEW_RULE', 'TRIGGER_PRIORITY'}]
    harness = [h for h in eligible if h['kind'] == 'HARNESS_FIX']
    supports = lambda hypotheses: max((len(set(h['independent_support_clusters'])) for h in hypotheses), default=0)
    status = ('G2_PROPOSAL_APPROVAL_CANDIDATE' if supports(text) >= 2 else
              'HARNESS_FIX_CANDIDATE' if supports(harness) >= 2 else 'ALFWORLD_TEXT_SKILL_DISCOVERY_PAUSED')
    return {'status': status, 'same_text_delta_supports': supports(text),
        'same_harness_delta_supports': supports(harness),
        'hypothesis_under_tested': opportunity_clusters == 0,
        'treatment_effect': 'NOT_DEFINED', 'next_provider_stage_authorized': False}
