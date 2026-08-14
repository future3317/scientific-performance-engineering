# Scientific GPU Training Optimizer

<p align="center">
  <a href="#中文">🇨🇳 中文</a>
  &nbsp;|&nbsp;
  <a href="#english">🇬🇧 English</a>
</p>

## Contents

- [中文](#中文)
- [English](#english)

## 中文

这是一个面向 PyTorch 科学计算训练与推理的 Codex skill，重点处理“GPU 利用率低但训练很慢”这类端到端性能问题。它覆盖 PyG/e3nn 等材料图网络、等变算子、压电/物性多任务、晶体 diffusion/flow matching、CUDA kernel、torch.compile/CUDA Graphs、DDP/FSDP、checkpoint 和混合精度。

### 它解决什么问题

- 识别 CPU 驱动、Python 循环、短 CUDA kernel、同步和错误计时分桶，而不是只看 GPU 利用率。
- 将多任务 step 拆成 batch、图构建、H2D、backbone、mechanism/property/physical、`autograd.grad`、backward、optimizer 和 DDP communication。
- 在保持模型、数据、数值、物理约束、采样律和恢复语义不变的前提下，优先复用 backbone、向量化 ragged work、合并 VJP、预取和测量真实 logical update。
- 对 stochastic thinning、activation checkpoint、DDP used-parameter set、主机争用和 objective-specific acceptance 提供可验证合同。

### 工作方式

`Explore → Plan → Execute → Integrate → Review`

每个候选优化都需要：可比 baseline、hypothesis card、Amdahl 上限、Micro → Module → End-to-end 证据、数值/梯度/等变性/物理/质量门槛，以及明确的 accepted、rejected 或 inconclusive 结论。

### 目录

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

### 使用与验证

将此目录放入 Codex 的 skills 目录后，在需要 GPU 性能诊断或优化时使用 `scientific-gpu-training-optimizer`。项目专用环境示例：

```powershell
$python = 'D:\Anaconda\envs\EGNN\python.exe'
$skill = 'C:\path\to\scientific-gpu-training-optimizer'
$env:PYTHONDONTWRITEBYTECODE = '1'

& $python "$skill\scripts\validate_skill.py" $skill
& $python "$skill\assets\materials_gnn_checks.py" --self-test
& $python "$skill\scripts\compare_benchmarks.py" --self-test
```

不要把 GPU utilization、单次 timing window 或 peak memory 单独当作 speedup 证明；比较必须冻结代码、硬件、软件、数据、任务组成、logical update 和计时边界。

## English

This Codex skill targets end-to-end performance problems in PyTorch scientific training and inference, especially cases where training is slow despite low GPU utilization. It covers PyG/e3nn material GNNs, equivariant operators, piezoelectric/property multitask workloads, crystal diffusion/flow matching, CUDA kernels, `torch.compile`/CUDA Graphs, DDP/FSDP, checkpointing, and mixed precision.

### What it addresses

- Distinguishes CPU-driven execution, Python loops, short CUDA kernels, synchronization, and misleading timing buckets from genuine GPU compute limits.
- Splits a multitask step into batch preparation, graph construction, H2D, backbone, mechanism/property/physical work, `autograd.grad`, backward, optimizer, and DDP communication.
- Preserves model, data, numerical, physical, sampler, and resume semantics while prioritizing feature reuse, ragged-work vectorization, batched VJPs, prefetching, and logical-update accounting.
- Defines verifiable contracts for stochastic thinning, activation checkpointing, DDP used-parameter behavior, host contention, and objective-specific acceptance.

### Operating model

`Explore → Plan → Execute → Integrate → Review`

Every candidate requires a comparable baseline, hypothesis card, Amdahl ceiling, Micro → Module → End-to-end evidence, numerical/gradient/equivariance/physical/quality gates, and an explicit accepted, rejected, or inconclusive decision.

### Layout

- [`SKILL.md`](SKILL.md): main entry point and performance-engineering workflow.
- [`references/PERFORMANCE_PLAYBOOK.md`](references/PERFORMANCE_PLAYBOOK.md): layered profiling, timing buckets, CPU/compiler/DDP/checkpoint guidance.
- [`references/GNN_PREDICTION_WORKLOADS.md`](references/GNN_PREDICTION_WORKLOADS.md): materials-graph prediction and PyG/PBC routing.
- [`references/CRYSTAL_GENERATION.md`](references/CRYSTAL_GENERATION.md): crystal generation, sampling, and campaign-cost routing.
- [`references/EQUIVARIANT_OPERATOR_DESIGN.md`](references/EQUIVARIANT_OPERATOR_DESIGN.md): explicitly authorized equivariant-operator architecture reviews.
- [`references/MEASUREMENT_CONTRACT.md`](references/MEASUREMENT_CONTRACT.md): measurement-contract template.
- [`scripts/collect_env.py`](scripts/collect_env.py): captures hardware, software, CPU, memory, and swap context.
- [`scripts/run_with_gpu_monitor.py`](scripts/run_with_gpu_monitor.py): captures NVIDIA GPU and optional host timelines; supporting telemetry only.
- [`scripts/compare_benchmarks.py`](scripts/compare_benchmarks.py): rejects incomparable benchmark records and computes deltas.
- [`scripts/validate_skill.py`](scripts/validate_skill.py): validates structure, links, assets, and Python syntax.

### Open-source use and contributions

This project is released under the [MIT License](LICENSE). Issues, documentation improvements, and reproducible validation scripts are welcome. Do not commit datasets, checkpoints, runtime logs, environment snapshots, or credentials; `.gitignore` covers common Python caches, experiment outputs, model artifacts, and local configuration.

### Use and validate

Place this directory in the Codex skills directory and use `scientific-gpu-training-optimizer` for GPU performance diagnosis or optimization. Example with a project-specific environment:

```powershell
$python = 'D:\Anaconda\envs\EGNN\python.exe'
$skill = 'C:\path\to\scientific-gpu-training-optimizer'
$env:PYTHONDONTWRITEBYTECODE = '1'

& $python "$skill\scripts\validate_skill.py" $skill
& $python "$skill\assets\materials_gnn_checks.py" --self-test
& $python "$skill\scripts\compare_benchmarks.py" --self-test
```

GPU utilization, a single timing window, or peak memory alone is not proof of a speedup. Comparisons must freeze code, hardware, software, data, task composition, logical update definition, and timing boundaries.
