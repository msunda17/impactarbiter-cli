"""
ImpactArbiter — Oracle Engine
=============================

Deterministic ground-truth physics for KV-cache memory routing.

Level 1 (default): 1D Ring-Buffer Sliding Window KV-Cache.

    total_blocks  = max_window_tokens // block_size
    logical_block = (token_idx // block_size) % total_blocks
    offset        = token_idx % block_size

The math is declared symbolically with SymPy and then `lambdify`-ed to a
PyTorch-backed callable so it can be invoked on Python ints OR on tensor
batches (used by `trap.py` under `torch.vmap`).

The module is intentionally pluggable: alternative oracles (e.g. a Level 2
2D Asymmetrical Per-Head Sliding Window) can be registered against the same
`Oracle` protocol and swapped in at runtime via `get_oracle(name)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Protocol, Tuple

import sympy
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Oracle protocol (so future fallback oracles can be injected)
# ─────────────────────────────────────────────────────────────────────────────


class Oracle(Protocol):
    name: str

    def __call__(
        self, token_idx: int, max_window_tokens: int, block_size: int
    ) -> Tuple[int, int]:
        """Return ground-truth (logical_block, offset) for a token index."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: 1D Ring-Buffer Sliding Window
# ─────────────────────────────────────────────────────────────────────────────


def _build_ring_buffer_lambdas() -> Tuple[Callable, Callable]:
    """
    Build two SymPy expressions and lambdify them with a PyTorch-friendly
    namespace. Returned callables accept either Python scalars or torch
    tensors of equivalent shape (used by `torch.vmap`).
    """
    token_idx = sympy.Symbol("token_idx", nonnegative=True, integer=True)
    max_window = sympy.Symbol("max_window_tokens", positive=True, integer=True)
    block_size = sympy.Symbol("block_size", positive=True, integer=True)

    total_blocks_expr = sympy.floor(max_window / block_size)
    logical_block_expr = sympy.Mod(
        sympy.floor(token_idx / block_size), total_blocks_expr
    )
    offset_expr = sympy.Mod(token_idx, block_size)

    # Namespace bridging SymPy's `floor` / `Mod` to both `math` (Python ints)
    # and `torch` (tensors). The dispatch on tensor-vs-scalar is what makes
    # this oracle vmap-compatible.
    def _floor(x):
        if isinstance(x, torch.Tensor):
            return torch.floor(x).long()
        return int(math.floor(x))

    def _mod(a, b):
        # `%` works for both ints and tensors and matches SymPy's Mod for
        # nonnegative dividends — which is the only regime we care about.
        return a % b

    ns = {"floor": _floor, "Mod": _mod}

    logical_block_fn = sympy.lambdify(
        (token_idx, max_window, block_size),
        logical_block_expr,
        modules=[ns, "math"],
    )
    offset_fn = sympy.lambdify(
        (token_idx, max_window, block_size),
        offset_expr,
        modules=[ns, "math"],
    )
    return logical_block_fn, offset_fn


_LOGICAL_BLOCK_FN, _OFFSET_FN = _build_ring_buffer_lambdas()


@dataclass(frozen=True)
class RingBufferOracle:
    """Level 1 oracle — 1D ring-buffer sliding window."""

    name: str = "ring_buffer_v1"

    def __call__(
        self, token_idx: int, max_window_tokens: int, block_size: int
    ) -> Tuple[int, int]:
        lb = int(_LOGICAL_BLOCK_FN(token_idx, max_window_tokens, block_size))
        off = int(_OFFSET_FN(token_idx, max_window_tokens, block_size))
        return lb, off

    # Tensor-batched path used by `trap.py` under torch.vmap.
    def batched(
        self,
        token_idx: torch.Tensor,
        max_window_tokens: int,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mw = torch.as_tensor(max_window_tokens, dtype=token_idx.dtype)
        bs = torch.as_tensor(block_size, dtype=token_idx.dtype)
        lb = _LOGICAL_BLOCK_FN(token_idx, mw, bs)
        off = _OFFSET_FN(token_idx, mw, bs)
        return lb.long(), off.long()


# ─────────────────────────────────────────────────────────────────────────────
# Registry — lets cli/agent ask for oracles by name. Level 2 plugs in here.
# ─────────────────────────────────────────────────────────────────────────────


_REGISTRY: Dict[str, Oracle] = {
    "ring_buffer_v1": RingBufferOracle(),
}


def register_oracle(oracle: Oracle) -> None:
    _REGISTRY[oracle.name] = oracle


def get_oracle(name: str = "ring_buffer_v1") -> Oracle:
    if name not in _REGISTRY:
        raise KeyError(
            f"Oracle '{name}' not registered. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


__all__ = [
    "Oracle",
    "RingBufferOracle",
    "register_oracle",
    "get_oracle",
]
