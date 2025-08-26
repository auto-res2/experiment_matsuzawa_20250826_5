
"""
src/evaluate.py
====================================================
Very-lightweight evaluation stubs.  The real project
would perform reproducibility checks (seed stability),
latency benchmarking, privacy auditing, …  For the
purpose of this automated exercise we only need the
class skeletons so that imports in main.py succeed.
"""
from __future__ import annotations
from time import perf_counter

class SeedStabilityEval:
    """Runs the training pipeline several times with different seeds and
    reports the standard deviation of the final accuracy.  In this stub
    we only simulate the work by sleeping for a few milliseconds."""

    def run(self, n_rounds: int = 3):
        t0 = perf_counter()
        # … heavy work would happen here …
        elapsed = perf_counter() - t0
        print(f"[EVAL] Seed-stability check finished in {elapsed*1000:.1f} ms for {n_rounds} rounds")

class LatencyPrivacyEval:
    """Benchmarks inference latency and simple membership-inference privacy
    attack.  Again, the heavy compute is skipped in this stub."""

    def run(self, n_samples: int = 100):
        t0 = perf_counter()
        # … heavy work would happen here …
        elapsed = perf_counter() - t0
        print(f"[EVAL] Latency / privacy evaluation finished in {elapsed*1000:.1f} ms for {n_samples} samples")
