# Real-artifact feasibility lane

This directory contains two packaging prototypes only. They are not part of
the v1.0-20 population, do not enter the formal task stream, and produce no
benchmark result. Each manifest pins an upstream commit, names the real
multi-file edit surface, and uses a tiny offline fixture so the harness can be
tested before committing to a larger real-repository lane.

Evaluation must run with network disabled. The upstream checkout is supplied by
the evaluator from the pinned commit; this repository intentionally does not
vendor either external repository or its datasets.
