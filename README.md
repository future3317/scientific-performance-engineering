<h1 align="center">Scientific GPU Training Optimizer</h1>

<hr />

<p align="center">
  <strong>面向科学计算训练与推理的端到端性能工程 Skill，<br />从运行时预检、数据管线到长期训练与性能验收形成闭环。</strong>
</p>

<p align="center">
  CPU/GPU 瓶颈诊断 · CUDA/PyG/e3nn 优化 · 科学正确性门槛 · logical-update 与 time-to-quality
</p>

<p align="center">
  <a href="https://github.com/future3317/scientific-gpu-training-optimizer/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f?style=for-the-badge" alt="MIT license" /></a>
  <a href="https://github.com/future3317/scientific-gpu-training-optimizer/actions/workflows/validate.yml"><img src="https://img.shields.io/github/actions/workflow/status/future3317/scientific-gpu-training-optimizer/validate.yml?branch=main&label=checks&style=for-the-badge" alt="Validation checks" /></a>
  <img src="https://img.shields.io/badge/benchmark-contract-v4-2563eb?style=for-the-badge" alt="Benchmark contract v4" />
  <img src="https://img.shields.io/badge/languages-English%20%7C%20中文-7c3aed?style=for-the-badge" alt="English and Chinese" />
</p>

<p align="center">
  <a href="#中文">项目介绍</a> ·
  <a href="#安装与使用">安装与使用</a> ·
  <a href="#核心工作流">核心工作流</a> ·
  <a href="#仓库结构">仓库结构</a> ·
  <a href="#质量与验证">质量与验证</a> ·
  <a href="#english">English</a>
</p>

<hr />

## Contents

- [中文](#中文)
- [核心工作流](#核心工作流)
- [仓库结构](#仓库结构)
- [安装与使用](#安装与使用)
- [质量与验证](#质量与验证)
- [English](#english)

## 中文

这是一个面向 PyTorch 科学计算训练与推理的 Codex skill，重点处理“GPU 利用率低但训练很慢”这类端到端性能问题。它覆盖 PyG/e3nn 等材料图网络、等变算子、压电/物性多任务、晶体 diffusion/flow matching、CUDA kernel、torch.compile/CUDA Graphs、DDP/FSDP、checkpoint 和混合精度。

### 它解决什么问题

- 识别 CPU 驱动、Python 循环、短 CUDA kernel、同步和错误计时分桶，而不是只看 GPU 利用率。
- 把预计算/缓存构建和多 seed campaign 的进程、worker、线程、NUMA、GPU 拓扑纳入性能预算，避免“前置步骤耗几十小时”或 worker fan-out 把服务器拖垮。
- 将多任务 step 拆成 batch、图构建、H2D、backbone、mechanism/property/physical、`autograd.grad`、backward、optimizer 和 DDP communication。
- 在保持模型、数据、数值、物理约束、采样律和恢复语义不变的前提下，优先复用 backbone、向量化 ragged work、合并 VJP、预取和测量真实 logical update。
- 对 stochastic thinning、activation checkpoint、DDP used-parameter set、主机争用和 objective-specific acceptance 提供可验证合同。

### 核心工作流

`Preflight → Contract Freeze → Lifecycle Census → Profile → Activation Proof → Logical Update → Amortized Job → Gates`

每个候选优化都需要：可比 baseline、hypothesis card、Amdahl 上限、Micro → Module → End-to-end 证据、数值/梯度/等变性/物理/质量门槛，以及明确的 accepted、rejected 或 inconclusive 结论。

### 仓库结构

- [`SKILL.md`](SKILL.md)：主入口和性能工程工作流。
- [`references/PERFORMANCE_PLAYBOOK.md`](references/PERFORMANCE_PLAYBOOK.md)：分层 profiling、计时分桶、CPU/编译/DDP/checkpoint 指南。
- [`references/GNN_PREDICTION_WORKLOADS.md`](references/GNN_PREDICTION_WORKLOADS.md)：材料图预测和 PyG/PBC 路由。
- [`references/CRYSTAL_GENERATION.md`](references/CRYSTAL_GENERATION.md)：晶体生成、采样和 campaign 成本路由。
- [`references/EQUIVARIANT_OPERATOR_DESIGN.md`](references/EQUIVARIANT_OPERATOR_DESIGN.md)：明确授权后的等变算子架构审查。
- [`references/MEASUREMENT_CONTRACT.md`](references/MEASUREMENT_CONTRACT.md)：实验合同模板。
- [`scripts/collect_env.py`](scripts/collect_env.py)：采集硬件、软件、CPU/内存/swap 环境。
- [`scripts/run_with_gpu_monitor.py`](scripts/run_with_gpu_monitor.py)：采集 NVIDIA GPU 和可选主机时间线；仅作 supporting telemetry。
- [`scripts/compare_benchmarks.py`](scripts/compare_benchmarks.py)：拒绝不可比 benchmark，并计算 before/after delta。
- [`scripts/validate_skill.py`](scripts/validate_skill.py)：校验 skill 结构、链接、资产和 Python 语法。

### 开源使用与贡献

本项目采用 [MIT License](LICENSE)。欢迎提交 issue、改进参考资料或补充可复现的验证脚本；请不要提交数据集、checkpoint、运行日志、环境快照或任何凭据。仓库中的 `.gitignore` 已覆盖常见 Python 缓存、实验输出、模型文件和本地配置。

### 安装与使用

将此目录放入 Codex 的 skills 目录后，在需要 GPU 性能诊断或优化时使用 `scientific-gpu-training-optimizer`。项目专用环境示例：

```powershell
$python = 'D:\Anaconda\envs\EGNN\python.exe'
$skill = 'C:\path\to\scientific-gpu-training-optimizer'
$env:PYTHONDONTWRITEBYTECODE = '1'

& $python "$skill\scripts\validate_skill.py" $skill
& $python "$skill\scripts\validate_benchmark.py" "$skill\assets\benchmark_record.json"
& $python "$skill\assets\materials_gnn_checks.py" --self-test
& $python "$skill\scripts\compare_benchmarks.py" --self-test
```

监控时显式选择 GPU，例如：`python scripts/run_with_gpu_monitor.py --output monitor.json --gpu 0 -- python train.py`。`collect_env.py` 默认会脱敏 hostname、绝对路径和 GPU UUID；只有明确需要时才使用 `--include-sensitive-host-metadata`。

### 质量与验证

不要把 GPU utilization、单次 timing window 或 peak memory 单独当作 speedup 证明；比较必须冻结代码、硬件、软件、数据、任务组成、logical update 和计时边界。

仓库自带 GitHub Actions 会在普通 runner 上执行结构、schema、行为和 Python 自测；它不会在共享 runner 上强制 5% GPU 性能门槛。若配置了带 `gpu` 标签的 self-hosted runner，可通过 workflow dispatch 显式运行 GPU contract checks。

## English

This Codex skill targets end-to-end performance problems in PyTorch scientific training and inference, especially cases where training is slow despite low GPU utilization. It covers PyG/e3nn material GNNs, equivariant operators, piezoelectric/property multitask workloads, crystal diffusion/flow matching, CUDA kernels, `torch.compile`/CUDA Graphs, DDP/FSDP, checkpointing, and mixed precision.

### What it addresses

- Distinguishes CPU-driven execution, Python loops, short CUDA kernels, synchronization, and misleading timing buckets from genuine GPU compute limits.
- Budgets precompute/cache construction and multi-seed process, worker, thread, NUMA, and GPU topology, preventing hour-scale hidden setup cost and worker fan-out that overloads the host.
- Splits a multitask step into batch preparation, graph construction, H2D, backbone, mechanism/property/physical work, `autograd.grad`, backward, optimizer, and DDP communication.
- Preserves model, data, numerical, physical, sampler, and resume semantics while prioritizing feature reuse, ragged-work vectorization, batched VJPs, prefetching, and logical-update accounting.
- Defines verifiable contracts for stochastic thinning, activation checkpointing, DDP used-parameter behavior, host contention, and objective-specific acceptance.

### Operating model

`Preflight → Contract Freeze → Lifecycle Census → Profile → Activation Proof → Logical Update → Amortized Job → Gates`

Every candidate requires a comparable baseline, hypothesis card, Amdahl ceiling, Micro → Module → End-to-end evidence, numerical/gradient/equivariance/physical/quality gates, and an explicit accepted, rejected, or inconclusive decision.

Benchmark records declare comparison class and evidence level. Micro/module evidence does not require campaign-level lifecycle fields; amortized-job and time-to-quality evidence does.

### Layout

- [`SKILL.md`](SKILL.md): main entry point and performance-engineering workflow.
- [`references/PERFORMANCE_PLAYBOOK.md`](references/PERFORMANCE_PLAYBOOK.md): layered profiling, timing buckets, CPU/compiler/DDP/checkpoint guidance.
- [`references/GNN_PREDICTION_WORKLOADS.md`](references/GNN_PREDICTION_WORKLOADS.md): materials-graph prediction and PyG/PBC routing.
- [`references/CRYSTAL_GENERATION.md`](references/CRYSTAL_GENERATION.md): crystal generation, sampling, and campaign-cost routing.
- [`references/EQUIVARIANT_OPERATOR_DESIGN.md`](references/EQUIVARIANT_OPERATOR_DESIGN.md): explicitly authorized equivariant-operator architecture reviews.
- [`references/MEASUREMENT_CONTRACT.md`](references/MEASUREMENT_CONTRACT.md): measurement-contract template.
- [`references/CODE_AND_RUNTIME_AUDIT.md`](references/CODE_AND_RUNTIME_AUDIT.md): runtime compatibility, custom-op, synchronization, and numerical-localization gates.
- [`references/DATA_AND_TRAINING_LIFECYCLE.md`](references/DATA_AND_TRAINING_LIFECYCLE.md): data/cache/H2D, logical-update DAG, optimizer-adjacent work, and amortized campaign cost.
- [`references/EXPERIENCE_EVOLUTION.md`](references/EXPERIENCE_EVOLUTION.md): auditable experience capture, candidate promotion, replay, and retirement boundaries.
- [`references/MEMORY_COMPILER_DISTRIBUTED.md`](references/MEMORY_COMPILER_DISTRIBUTED.md): memory forensics, compile/CUDA Graphs, distributed diagnostics, and resume routes.
- [`scripts/collect_env.py`](scripts/collect_env.py): captures hardware, software, CPU, memory, and swap context.
- [`scripts/run_with_gpu_monitor.py`](scripts/run_with_gpu_monitor.py): captures NVIDIA GPU and optional host timelines; supporting telemetry only.
- [`scripts/compare_benchmarks.py`](scripts/compare_benchmarks.py): rejects incomparable benchmark records and computes deltas.
- [`scripts/validate_skill.py`](scripts/validate_skill.py): validates structure, links, assets, and Python syntax.
- [`scripts/validate_benchmark.py`](scripts/validate_benchmark.py): validates the schema-driven lifecycle and evidence contract.
- [`scripts/validate_experience.py`](scripts/validate_experience.py): validates experience records without promoting them to rules.
- [`scripts/capture_experience.py`](scripts/capture_experience.py): validates and copies one inbox record without overwriting existing cases.
- [`scripts/validate_evolution.py`](scripts/validate_evolution.py): audits candidate/retired cards and canonical registry promotion gates.
- [`scripts/run_rule_replay.py`](scripts/run_rule_replay.py): runs held-out paired interventions and emits digest-attested CRU/Bayesian replay manifests.
- [`scripts/validate_rule_usage.py`](scripts/validate_rule_usage.py) / [`scripts/capture_rule_usage.py`](scripts/capture_rule_usage.py): records retrieved, triggered, followed, overridden rules and outcomes; retrieval is not utility.
- [`scripts/score_rule_library.py`](scripts/score_rule_library.py): scores description length, utility distortion, and conflict cost, then recommends maintenance actions without mutating rules.
- `experience/` and `evolution/`: audited inbox, cases, candidates, conflicts, maintenance reports, and retired cards.
- `rules/`: reviewed canonical rule cards only; it is intentionally separate from the reference prose.
- [`registry/rules.json`](registry/rules.json): canonical-rule index; it does not duplicate reference prose.

### Open-source use and contributions

This project is released under the [MIT License](LICENSE). Issues, documentation improvements, and reproducible validation scripts are welcome. Do not commit datasets, checkpoints, runtime logs, environment snapshots, or credentials; `.gitignore` covers common Python caches, experiment outputs, model artifacts, and local configuration.

### Use and validate

Place this directory in the Codex skills directory and use `scientific-gpu-training-optimizer` for GPU performance diagnosis or optimization. Example with a project-specific environment:

```powershell
$python = 'D:\Anaconda\envs\EGNN\python.exe'
$skill = 'C:\path\to\scientific-gpu-training-optimizer'
$env:PYTHONDONTWRITEBYTECODE = '1'

& $python "$skill\scripts\validate_skill.py" $skill
& $python "$skill\scripts\validate_benchmark.py" "$skill\assets\benchmark_record.json"
& $python "$skill\assets\materials_gnn_checks.py" --self-test
& $python "$skill\scripts\compare_benchmarks.py" --self-test
```

Select GPUs explicitly when monitoring, for example: `python scripts/run_with_gpu_monitor.py --output monitor.json --gpu 0 -- python train.py`. `collect_env.py` redacts hostnames, absolute paths, and GPU UUIDs by default; use `--include-sensitive-host-metadata` only when needed.

GPU utilization, a single timing window, or peak memory alone is not proof of a speedup. Comparisons must freeze code, hardware, software, data, task composition, logical update definition, and timing boundaries.

GitHub Actions runs structure, schema, behavioral, and Python checks on a standard runner; it never imposes a 5% GPU gate there. If a self-hosted runner labeled `gpu` is configured, the workflow can be dispatched explicitly for GPU contract checks.
