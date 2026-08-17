# Tempting but wrong patch

Do not apply an unconditional `FIX-COMPILE-TINY-ABSTAIN` rewrite: compiling every tiny graph changes the cost regime and is not justified outside the registered short-lived workload.
