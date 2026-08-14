# Patch patterns

These are review patterns, not blind recipes. Adapt them to the project's scientific contract and benchmark every change.

## Contents

1. Prove autocast is on the real path
2. Guard DataLoader options by worker count
3. Schedule a profiler window
4. Avoid CUDA scalar extraction in ragged loops
5. Group per-parameter gradient scaling
6. Change DDP settings only after proof
7. Compile a stable region
8. Prove optimized-kernel activation
9. Add NVTX ranges for Nsight Systems
10. Audit misleading timing buckets
11. Batched data and transfer proof
12. Custom operator contract
13. Sync budget and amortized lifecycle
14. Periodic-geometry kernel boundary

## 1. Prove autocast is on the real path

```python
amp_enabled = device.type == "cuda" and config.use_bf16
with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
    prediction = model(batch)
    fast_terms = fast_loss_terms(prediction, batch)

# Keep sensitive algebra outside autocast.
with torch.autocast(device_type=device.type, enabled=False):
    stable_term = stable_loss_term(prediction["operator"].float(), batch)

loss = fast_terms.float() + stable_term
loss.backward()
```

Add a focused test or trace assertion that representative backbone outputs use the intended dtype and sensitive terms use FP32/FP64. Do not place backward inside autocast.

## 2. DataLoader options guarded by worker count

```python
loader_kwargs = {
    "batch_size": batch_size,
    "num_workers": num_workers,
    "pin_memory": device.type == "cuda",
    "collate_fn": collate_fn,
}
if num_workers > 0:
    loader_kwargs.update(
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
loader = DataLoader(dataset, **loader_kwargs)
```

Confirm that the returned custom/PyG batch is actually pinned. More workers/prefetch are not automatically faster.

## 3. Scheduled profiler window

```python
activities = [torch.profiler.ProfilerActivity.CPU]
if device.type == "cuda":
    activities.append(torch.profiler.ProfilerActivity.CUDA)

with torch.profiler.profile(
    activities=activities,
    schedule=torch.profiler.schedule(skip_first=10, wait=2, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    record_shapes=False,
    profile_memory=False,
) as prof:
    for step, batch in enumerate(loader):
        with torch.profiler.record_function("train_step"):
            train_step(batch)
        prof.step()
```

Run separate focused traces with shapes/stacks/memory; those options add overhead.

## 4. Avoid CUDA scalar extraction in ragged loops

Bad hot-path signal:

```python
for i in range(ptr.numel() - 1):
    start = int(ptr[i])
    stop = int(ptr[i + 1])
```

Possible approaches, in order of increasing invasiveness:

1. Keep graph sizes/offsets as CPU metadata before batch transfer.
2. Bucket graphs by size and run a batched padded kernel with a mask.
3. Use segmented/block-diagonal operations.
4. Compile a stable grouped subregion.

Do not replace a correct ragged formulation with excessive padding without measuring both memory and time.

## 5. Group per-parameter gradient scaling

```python
groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
for parameter in model.parameters():
    grad = parameter.grad
    if grad is not None:
        groups.setdefault((grad.device, grad.dtype), []).append(grad)
for grads in groups.values():
    torch._foreach_div_(grads, normalizer)
```

Check sparse gradients, bucket views, higher-order autograd, and numerical equivalence. Private/underscored APIs need version pinning or a public alternative when available.

## 6. DDP settings only after proof

```python
model = DistributedDataParallel(
    model,
    device_ids=[local_rank],
    find_unused_parameters=needs_unused_detection,
    static_graph=can_set_static_graph,
    gradient_as_bucket_view=True,
)
```

Obtain evidence from stage behavior and DDP logging. A wrong `static_graph`/unused-parameter assumption can hang or corrupt distributed training.

## 7. Compile a stable region, not the whole ragged trainer

```python
compiled_block = torch.compile(
    model.repeated_message_block,
    dynamic=None,
    fullgraph=False,
)
```

Benchmark eager and compiled with identical work. Inspect graph breaks and recompiles. Include cold-start cost when it recurs in the actual workflow.

## 8. Explicit optimized-kernel activation

```python
op = FastTensorProduct(..., use_fallback=False)
```

Add startup logging/tests that record library version, method, layout, dtype, and active kernel. Never claim native acceleration when a fallback executed.

## 9. NVTX ranges for Nsight Systems

```python
from torch.cuda import nvtx

nvtx.range_push("forward")
prediction = model(batch)
nvtx.range_pop()
```

Prefer context-manager wrappers in production code and keep ranges coarse enough to avoid instrumentation noise.

## 10. Audit misleading timing buckets

Place the boundary after the work it claims to measure, and keep auxiliary computation in its own range. For a CUDA phase, pair events on the same stream (or synchronize at the boundary); a CPU timestamp around asynchronous launches is not a GPU duration. `record_function` remains useful for attribution:

```python
import torch
from torch.profiler import record_function

with record_function("batch_fetch"):
    batch = next(loader)
with record_function("cpu_graph_build"):
    batch = build_graphs_if_needed(batch)
data_ready = True

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
with record_function("mechanism"):
    mechanism_loss = mechanism_task(batch)
end_event.record()
end_event.synchronize()
mechanism_ms = start_event.elapsed_time(end_event)

with record_function("property_heads"):
    property_loss = property_tasks(batch)
with record_function("physical_features_autograd"):
    physical_loss = physical_terms(batch)
```

Do not name the whole interval before `data_ready` as `data` when it contains model or `autograd.grad` work. Reconcile these ranges with the synchronized wall-clock step and report any unaccounted interval.

## 11. Batched data and transfer proof

```python
def __getitems__(self, indices):
    return [self._parse_one(index) for index in indices]

batch = batch.pin_memory()  # custom batch must implement this method
device_batch = batch.to(device, non_blocking=True)
```

Benchmark batched fetch against per-sample fetch, then prove pinned state, copy stream, source lifetime, consumer dependency, and actual overlap. Preserve ordering, augmentation RNG, and cache-key provenance.

## 12. Custom operator contract

Keep a reference implementation and exact signature. Register schema/mutation/aliasing and FakeTensor/meta behavior; run `torch.library.opcheck`, `assert_close`, `gradcheck`, and `gradgradcheck` where required. Add empty/non-contiguous/edge-shape/compile/forward-backward cases. `opcheck` is a dispatch contract, not a numerical proof.

## 13. Sync budget and amortized lifecycle

Record `.item()`, `.cpu()`, metric all-reduce, progress/logging, validation, checkpoint staging, barriers, and explicit synchronizations in the record's `sync_census`. Classify each event as required, removable, amortizable, or overlappable. Measure EMA/SWA/scheduler/clipping and checkpoint/logging/validation cadence in a separate amortized job metric.

## 14. Periodic-geometry kernel boundary

If profiling attributes material time to minimum-image, Cartesian/fractional transforms, radial basis, or scatter/reduction, first prove the same short-kernel pattern on representative cells and edge counts. Then compare one fused `torch.library.triton_op`/CUDA op (or the smallest compatible group) against the eager reference. Keep FP32 behavior and existing periodic, cell-basis, O(3), empty, edge-shape, forward/backward, and compile tests. Do not optimize AdamW or rewrite the scientific architecture while forward geometry still dominates; a kernel win is accepted only when the logical-update measurement moves.
