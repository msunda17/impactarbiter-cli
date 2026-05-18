"""ImpactArbiter core — observability primitives for agentic AI.

Exports:
    - HardBlockException: raised when the autograd trap catches a context leak.
    - verify: decorator that wraps a routing function with a deterministic
              gradient-divergence check against a SymPy oracle.
    - StochasticEvaluator: runs the auto-heal loop N times to surface the
              probabilistic hallucination rate of a given LLM.
"""

from .decorator import HardBlockException, verify
from .evaluator import StochasticEvaluator, EvaluationReport

__all__ = [
    "HardBlockException",
    "verify",
    "StochasticEvaluator",
    "EvaluationReport",
]
