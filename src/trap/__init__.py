"""Autograd trap — gradient-based semantic divergence detector."""

from .autograd_trap import (
    TrapResult,
    format_divergence_map,
    run_paged_trap,
    run_radix_trap,
)

__all__ = [
    "TrapResult",
    "run_paged_trap",
    "run_radix_trap",
    "format_divergence_map",
]
