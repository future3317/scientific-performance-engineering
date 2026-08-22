Tempting wrong patch: replace the two explicit VJPs with one torch.autograd.grad call using is_grads_batched=True. It may be correct, but the task asks for evidence that the batching overhead is worthwhile at output_count=2.

