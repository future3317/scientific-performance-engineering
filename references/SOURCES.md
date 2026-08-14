# Primary sources

Last reviewed: 2026-08-04.

This list is an audit map, not a package lock. Release numbers are a point-in-time snapshot checked against official release pages. Always record the installed versions and feature probes from `scripts/collect_env.py`; do not upgrade a project just because a newer release exists.

## Contents

- [2026 release snapshot](#2026-release-snapshot)
- [PyTorch runtime, CPU, compiler, and precision](#pytorch-runtime-cpu-compiler-and-precision)
- [Distributed and checkpointing](#distributed-and-checkpointing)
- [PyTorch Geometric and scientific kernels](#pytorch-geometric-and-scientific-kernels)
- [Materials tensors and crystal generation](#materials-tensors-and-crystal-generation)
- [NVIDIA profiling and kernels](#nvidia-profiling-and-kernels)
- [Project sources](#project-sources)

## 2026 release snapshot

- PyTorch 2.13.0: https://github.com/pytorch/pytorch/releases/tag/v2.13.0
- torchao 0.18.0: https://github.com/pytorch/ao/releases/tag/v0.18.0
- PyTorch Geometric 2.8.0: https://github.com/pyg-team/pytorch_geometric/releases/tag/2.8.0
- cuEquivariance 0.10.0: https://github.com/NVIDIA/cuEquivariance/releases/tag/v0.10.0
- Transformer Engine 2.17: https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.17

## Skill format and installation

- OpenAI, Build skills: https://learn.chatgpt.com/docs/build-skills
- Anthropic, Agent Skills overview: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview

## PyTorch runtime, CPU, compiler, and precision

- Performance Tuning Guide: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- PyTorch Profiler recipe: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- `torch.utils.benchmark`: https://docs.pytorch.org/docs/stable/benchmark_utils.html
- Automatic mixed precision: https://docs.pytorch.org/docs/stable/amp.html
- AMP examples: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- DataLoader: https://docs.pytorch.org/docs/stable/data.html
- `torch.set_num_threads`: https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html
- `torch.set_num_interop_threads`: https://docs.pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html
- CPU threading note: https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_intra_op.html
- `torch.compile`: https://docs.pytorch.org/docs/stable/torch.compiler.html
- Dynamic shapes: https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html
- Regional compilation: https://docs.pytorch.org/tutorials/recipes/regional_compilation.html
- Compiler troubleshooting: https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html
- Compiler CUDA Graph Trees: https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
- Optimizer implementations: https://docs.pytorch.org/docs/stable/optim.html
- torchao documentation: https://docs.pytorch.org/ao/stable/
- Custom operators, FakeTensor, and `opcheck`: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html
- User-defined Triton kernels with `torch.compile`: https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
- `torch.func` transforms and compiler FAQ: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html
- DebugMode numerical localization: https://docs.pytorch.org/tutorials/recipes/debug_mode_tutorial.html
- Data loading optimization and batched `__getitems__`: https://docs.pytorch.org/docs/stable/data.html
- `pin_memory()` and non-blocking transfers: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
- Compile cache configuration: https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html
- CUDA memory snapshot and allocator analysis: https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
- CUDA environment variables (`PYTORCH_ALLOC_CONF`): https://docs.pytorch.org/docs/stable/cuda_environment_variables.html
- Optimizer-in-backward: https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
- Activation checkpointing: https://docs.pytorch.org/docs/stable/checkpoint

## Distributed and checkpointing

- DistributedDataParallel: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- DDP design note: https://docs.pytorch.org/docs/stable/notes/ddp.html
- FSDP2 `fully_shard`: https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- Distributed checkpoint: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
- Distributed overview: https://docs.pytorch.org/docs/stable/distributed.html
- ProcessGroupNCCL flight recorder and desync diagnostics: https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html
- Asynchronous DCP checkpointing: https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html
- TorchTitan training system: https://github.com/pytorch/torchtitan
- MACE training implementation: https://github.com/ACEsuit/mace
- TorchBench machine validation: https://github.com/pytorch/benchmark

## PyTorch Geometric and scientific kernels

- Loader API, including PrefetchLoader/DynamicBatchSampler: https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html
- CPU affinity: https://pytorch-geometric.readthedocs.io/en/latest/advanced/cpu_affinity.html
- Advanced mini-batching: https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
- Memory-efficient aggregations: https://pytorch-geometric.readthedocs.io/en/latest/advanced/sparse_tensor.html
- Compiled GNNs: https://pytorch-geometric.readthedocs.io/en/latest/advanced/compile.html
- Profiling API: https://pytorch-geometric.readthedocs.io/en/latest/modules/profile.html
- e3nn documentation: https://docs.e3nn.org/
- e3nn TensorProduct API: https://docs.e3nn.org/en/stable/api/o3/o3_tp.html
- cuEquivariance documentation: https://docs.nvidia.com/cuda/cuequivariance/
- cuEquivariance MACE/layout tutorial: https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/MACE.html
- OpenEquivariance acceleration in NequIP: https://nequip.readthedocs.io/en/latest/guide/accelerations/openequivariance.html
- cuEquivariance segmented polynomial kernels: https://docs.nvidia.com/cuda/cuequivariance/tutorials/poly.html

## Materials tensors and crystal generation

- MACE: higher-order equivariant message passing: https://arxiv.org/abs/2206.07697
- Equiformer: equivariant graph attention: https://arxiv.org/abs/2206.11990
- Equivariant GNNs for crystal tensor properties: https://arxiv.org/abs/2406.03563
- Crystal tensor canonicalization: https://arxiv.org/abs/2410.02372
- Cartesian atomic cluster expansion: https://arxiv.org/abs/2402.07472
- SO(3)-to-SO(2) equivariant convolutions: https://arxiv.org/abs/2302.03655
- Gaunt Tensor Products: https://arxiv.org/abs/2401.10216
- Chemistry foundation model data distribution and kernel optimization: https://arxiv.org/abs/2504.10700
- Gradient surgery for multi-task learning (PCGrad): https://arxiv.org/abs/2001.06782
- FlowMM: generating materials with Riemannian flow matching: https://arxiv.org/abs/2406.04713
- DiffCSP: joint equivariant diffusion for crystal structure prediction: https://arxiv.org/abs/2309.04475
- CrystalFlow: flow-based crystalline material generation: https://arxiv.org/abs/2412.11693
- SymmCD: symmetry-preserving crystal generation: https://arxiv.org/abs/2502.03638
- DPM-Solver: fast diffusion ODE sampling: https://arxiv.org/abs/2206.00927
- VSTP: complete high-degree Clebsch-Gordan Tensor Products: https://arxiv.org/abs/2602.21466
- SpinGTP: complete parity-sensitive Tensor Products: https://arxiv.org/abs/2607.01408
- Sobek streaming equivariant Tensor Product convolutions: https://arxiv.org/abs/2607.18074
- Equivariant low-bit quantization: https://arxiv.org/abs/2601.02213
- Riemannian MeanFlow: https://arxiv.org/abs/2603.10718
- MatterGen repository: https://github.com/microsoft/mattergen
- MatterGen default sampling configuration: https://github.com/microsoft/mattergen/blob/main/sampling_conf/default.yaml
- MatterGen CSP sampling configuration: https://github.com/microsoft/mattergen/blob/main/sampling_conf/csp.yaml

## NVIDIA profiling and kernels

- Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Nsight Compute Documentation: https://docs.nvidia.com/nsight-compute/
- CUDA Graph sync-free code guidance: https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/sync-free-code.html
- Transformer Engine User Guide: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## Project sources

- https://github.com/future3317/General-Equivariant-Covariance-Networks-for-Probabilistic-Structured-Prediction
- https://github.com/future3317/gaugeflow
- https://github.com/future3317/PiezoJet
