def check_episode(solution,fixtures):
    result = solution.run_episode_task(fixtures.get('task_workspace', ''), fixtures.get('skill_view', {}), fixtures.get('budget', {}))
    action = result.get('action') if isinstance(result, dict) else None
    passed = isinstance(action, dict) and str(action.get('condition', '')).upper() in {'C', 'C_STRESS', 'D'}
    return {'passed': passed, 'details': {'action': action}}
