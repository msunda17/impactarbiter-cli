"""PagedAttention oracle: SymPy AST and lambdified callable.

Ground truth:
    logical_block = token_idx // block_size
    offset = token_idx % block_size
"""

from __future__ import annotations

import sympy as sp

# ─────────────────────────────────────────────────────────────────────────────
# SymPy expressions
# ─────────────────────────────────────────────────────────────────────────────
token_idx, block_size = sp.symbols("token_idx block_size", integer=True, nonnegative=True)

# PagedAttention: simple linear mapping
logical_block_expr = token_idx // block_size
offset_expr = token_idx % block_size

# ─────────────────────────────────────────────────────────────────────────────
# Lambdified callable
# ─────────────────────────────────────────────────────────────────────────────
paged_oracle = sp.lambdify(
    (token_idx, block_size),
    (logical_block_expr, offset_expr),
    modules="numpy",
)

# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "token_idx",
    "block_size",
    "logical_block_expr",
    "offset_expr",
    "paged_oracle",
]
