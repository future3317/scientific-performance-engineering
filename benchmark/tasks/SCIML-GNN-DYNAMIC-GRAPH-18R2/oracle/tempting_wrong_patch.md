# Tempting but wrong patch

Do not cache a graph across changing geometry. The registered oracle only
batches the current ragged graph and keeps graph construction inside each
evaluation.
