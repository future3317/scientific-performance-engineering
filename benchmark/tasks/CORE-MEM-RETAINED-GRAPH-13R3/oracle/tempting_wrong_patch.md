# Tempting but wrong patch

Do not remove the segmented checkpoint path or change the loss, batch, or
optimizer contract. The accepted intervention is limited to clearing retained
history and keeping detached activation diagnostics; the measured work units
must remain unchanged.
