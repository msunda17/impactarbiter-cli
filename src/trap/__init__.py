"""Autograd trap — gradient-based semantic divergence detector."""

from .autograd_trap import (
    TrapResult,
    format_divergence_map,
    format_divergence_map_2d,
    run_paged_trap,
    run_radix_trap,
    run_radix_2d_trap,
)

__all__ = [
    "TrapResult",
    "run_paged_trap",
    "run_radix_trap",
    "run_radix_2d_trap",
    "format_divergence_map",
    "format_divergence_map_2d",
]
