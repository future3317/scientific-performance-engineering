# Scientific Performance Engineering

Evidence-governed performance engineering for PyTorch scientific workloads.
This is a Codex Skill and a small, executable Rule OS for diagnosing end-to-end
training/inference bottlenecks while preserving numerical and scientific
contracts.

[English](#english) · [中文](#中文) · [Validation](#validation)

## Contents

- [English](#english)
- [Install and use](#install-and-use)
- [Repository map](#repository-map)
- [Validation](#validation)
- [License](#license)

## English

The project covers CPU-driven training, fragmented CUDA execution, PyG/e3nn
and equivariant operators, multitask/autograd overhead, diffusion/flow
sampling, compilation, distributed scaling, precompute/cache cost, and
multi-seed campaign topology.

The evolution layer is deliberately bounded:

`Experience → Candidate → Paired replay → bounded utility + mixture-CS gate → Governance API → Canonical/Retired`

Rules are represented by three typed objects:

- `RuleSpec`: immutable meaning, typed predicates, intervention, invariants,
  relations, and provenance policy.
- `EvidenceEvent`: append-only on/off assignment, context, outcomes, gates,
  artifact references, versions, source, and independence group.
- `RuleState`: materialized confidence, retrieval utility, override rate, and
  drift status (`stable`, `suspected_drift`, `stale`, `revalidating`).

Retrieval follows `TaskContext → predicate match → conflict-aware greedy
selection under token budget`. Similarity may narrow candidates, but structured
predicates and hard conflicts decide what is selected. Runtime use never edits
`SKILL.md` or silently promotes a rule.

ACRE is one method core rather than a collection of independent pilot
primitives. `core/acre/engine.py` is the public façade; it coordinates
EvidenceEvent v2, asymmetric representative/adversarial evidence, version-space
CEGIS, adaptive experiment acquisition, factorial relation inference, and
conservative routing over canonical `RuleSpec`/`RelationSpec` state. BoundaryBench
and InteractionBench construct environments and evaluate sealed outcomes only.

### Current maturity

The governed Rule OS is implemented, but the canonical rule library is still
deliberately small: `registry/rules.json` is empty and `rules/` contains no
promoted cards. This repository therefore provides a governed evolution
framework and executable benchmark, not a mature self-evolving rule
ecosystem. Rules enter the registry only after reviewed replay evidence.

## 中文

这是一个面向 PyTorch 科学训练与推理的证据治理型性能工程 Skill，同时提供一个
可执行的规则系统。它覆盖 CPU 驱动、CUDA 碎片化、PyG/e3nn 与等变算子、多任务
autograd、扩散/流匹配采样、编译、多卡扩展、预计算/缓存成本，以及多 seed 的完整
资源拓扑。

经验演化严格遵循：

`Experience → Candidate → 成对回放 → 有界 utility + mixture-CS 门槛 → Governance API → Canonical/Retired`

规则语义、证据事件和可变状态分离；运行时只能记录证据和使用遥测，不能直接修改
`SKILL.md`、canonical 规则或科学验收语义。

## Install and use

Clone or copy this directory into the Codex skills directory and invoke
`$scientific-performance-engineering` for a scientific performance task.

```powershell
$python = 'D:\Anaconda\envs\EGNN\python.exe'
$skill = 'C:\path\to\scientific-performance-engineering'
& $python "$skill\scripts\validate_skill.py" $skill
& $python "$skill\scripts\rule_engine_tests.py"
& $python "$skill\scripts\validate_benchmark.py" "$skill\assets\benchmark_record.json"
```

Use the project-specific environment; do not run these checks from Conda
`base`. GPU monitoring requires an explicit GPU selection:

```powershell
python scripts/run_with_gpu_monitor.py --output monitor.json --gpu 0 -- python train.py
```

## Repository map

- `SKILL.md`: compact router and lifecycle policy.
- `core/`: canonical typed models, predicate matcher, and budgeted retriever.
- `references/`: detailed performance and scientific-domain procedures.
- `scripts/`: validators, replay, confidence, retrieval, telemetry, and tests.
- `assets/`: benchmark templates and generated model schemas.
- `benchmark/`: SPE-EvoBench's pinned sources, schemas, harness, split policy,
  and runnable prototype tasks for evaluating performance-engineering and
  skill-evolution behavior. Start with [`benchmark/README.md`](benchmark/README.md).
- `experience/`, `evolution/`, `rules/`, `registry/`: evidence and rule lifecycle.
- `core/acre/`: the single ACRE method core (`AcreEngine`, evidence policy,
  CEGIS, acquisition, factorial inference, routing, and bundle certificates).

## Validation

```powershell
python scripts/generate_rule_schemas.py
python scripts/validate_skill.py .
python scripts/behavioral_contract_tests.py
python scripts/experience_contract_tests.py
python scripts/evolution_contract_tests.py
python scripts/evolution_utility_tests.py
python scripts/rule_engine_tests.py
python assets/materials_gnn_checks.py --self-test
python scripts/compare_benchmarks.py --self-test
python benchmark/tests/run_all.py
python -m benchmark.harness.cli check-leakage benchmark/split/sequential.yaml
```

Static review and microbenchmarks establish hypotheses, not campaign-level
speedups. A claim must identify its comparison class, evidence level, frozen
scientific contract, timing scope, gates, noise, and limitations.

## License

MIT. Do not commit datasets, checkpoints, runtime logs, private environment
snapshots, or credentials.
