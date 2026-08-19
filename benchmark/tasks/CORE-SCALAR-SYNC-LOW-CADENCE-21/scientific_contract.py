from __future__ import annotations

def metric_semantics_preserved(solution, fixtures):
    a=solution.run_training(fixtures,1); b=solution.run_training(fixtures,1)
    ok=len(a['metrics'])==4 and all(abs(float(x)-float(y))<1e-7 for x,y in zip(a['metrics'],b['metrics']))
    return (ok, {'metric_count':len(a['metrics'])})
