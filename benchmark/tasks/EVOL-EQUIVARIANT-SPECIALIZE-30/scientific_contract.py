from __future__ import annotations

def state_transition_valid(result):
    score=result.get('episode_score'); ok=isinstance(score,(int,float)) and 0.0<=float(score)<=1.0+1e-9 and isinstance(result.get('episode_metrics'),dict); return (ok, {'episode_score':score,'condition_used':result.get('condition_used')})
def specialization_applied(metrics):
    neg=metrics.get('negative_transfer_rate'); precision=metrics.get('rule_precision'); ok=isinstance(neg,(int,float)) and isinstance(precision,(int,float)) and 0.0<=float(neg)<=1.0 and 0.0<=float(precision)<=1.0; return (ok, {'negative_transfer_rate':neg,'rule_precision':precision})
