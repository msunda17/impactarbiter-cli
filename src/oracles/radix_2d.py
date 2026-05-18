"""2D Asymmetric Radix oracle with Ring Buffer wrapping.

Ground truth (Asymmetric per-head ring buffer):
    absolute_idx = prefix_length_h + b_local_idx
    logical_block = (absolute_idx // block_size) % total_blocks_h
    offset        = absolute_idx % block_size
    returns (head_idx, logical_block, offset)

This is the "Memorization Horizon" oracle — frontier LLMs have memorized 1D
linear routing (vLLM/SGLang style). The 2D variant introduces:
  • Per-head asymmetry: each head may have a different `prefix_length_h`.
  • Ring buffer wrapping: each head's physical block index wraps modulo
    `total_blocks_h`, so block addresses are NOT monotonically increasing.
"""

from __future__ import annotations

from typing import Tuple

import sympy as sp

# ─────────────────────────────────────────────────────────────────────────────
# Symbolic variables
# ─────────────────────────────────────────────────────────────────────────────
b_local_idx = sp.Symbol("b_local_idx", integer=True, nonnegative=True)
head_idx = sp.Symbol("head_idx", integer=True, nonnegative=True)
prefix_length_h = sp.Symbol("prefix_length_h", integer=True, nonnegative=True)
total_blocks_h = sp.Symbol("total_blocks_h", integer=True, positive=True)
block_size = sp.Symbol("block_size", integer=True, positive=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2D Asymmetric Radix routing with ring-buffer wrap
# ─────────────────────────────────────────────────────────────────────────────
absolute_idx = prefix_length_h + b_local_idx
logical_block_2d = (absolute_idx // block_size) % total_blocks_h
offset_2d = absolute_idx % block_size

# ─────────────────────────────────────────────────────────────────────────────
# Lambdified callable
# ─────────────────────────────────────────────────────────────────────────────
radix_2d_oracle_callable = sp.lambdify(
    (b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size),
    (head_idx, logical_block_2d, offset_2d),
    "numpy",
)


def radix_2d_oracle(
    b_local_idx: int,
    head_idx: int,
    prefix_length_h: int,
    total_blocks_h: int,
    block_size: int = 16,
) -> Tuple[int, int, int]:
    """2D Asymmetric Radix oracle with per-head ring buffer.

    Args:
        b_local_idx: Local token index within the new generation.
        head_idx: Attention head index (preserved in output).
        prefix_length_h: Length of the shared prefix for this head.
        total_blocks_h: Ring-buffer capacity (number of physical blocks
            allocated for this head). Block addresses wrap modulo this.
        block_size: Block size in tokens (default 16).

    Returns:
        (head_idx, logical_block, offset) tuple.
    """
    h, lb, off = radix_2d_oracle_callable(
        b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size
    )
    return int(h), int(lb), int(off)


__all__ = ["radix_2d_oracle"]
