"""RadixAttention oracle: SymPy AST and lambdified callable.

Ground truth (Ragged Straddle):
    absolute_idx = prefix_length + b_local_idx
    logical_block = absolute_idx // block_size
    offset = absolute_idx % block_size
"""

from __future__ import annotations

from typing import Tuple

import sympy as sp

# ─────────────────────────────────────────────────────────────────────────────
# Symbolic variables
# ─────────────────────────────────────────────────────────────────────────────
b_local_idx = sp.Symbol("b_local_idx", integer=True, nonnegative=True)
prefix_length = sp.Symbol("prefix_length", integer=True, nonnegative=True)
block_size = sp.Symbol("block_size", integer=True, positive=True)

# ─────────────────────────────────────────────────────────────────────────────
# RadixAttention routing logic (ground truth)
# ─────────────────────────────────────────────────────────────────────────────
# The absolute token position is: prefix_length + b_local_idx
# where prefix_length is the length of the shared prefix
# and b_local_idx is the local token index within the new generation
absolute_idx = prefix_length + b_local_idx

# Map absolute position to logical block and offset
logical_block = absolute_idx // block_size
offset = absolute_idx % block_size

# ─────────────────────────────────────────────────────────────────────────────
# Lambdified callable
# ─────────────────────────────────────────────────────────────────────────────
radix_oracle_callable = sp.lambdify(
    (b_local_idx, prefix_length, block_size),
    (logical_block, offset),
    "numpy",
)


def radix_oracle(
    b_local_idx: int,
    prefix_length: int,
    block_size: int = 16,
) -> Tuple[int, int]:
    """
    RadixAttention oracle: maps local token index to (logical_block, offset).

    The absolute position is prefix_length + b_local_idx, where prefix_length
    is the length of the shared prefix. The logical_block and offset are
    computed via integer division and modulo arithmetic.

    Args:
        b_local_idx: Local token index within the new generation.
        prefix_length: Length of the shared prefix.
        block_size: Block size in tokens (default 16).

    Returns:
        (logical_block, offset) tuple.
    """
    lb, off = radix_oracle_callable(b_local_idx, prefix_length, block_size)
    return int(lb), int(off)


__all__ = ["radix_oracle"]
