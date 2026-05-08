"""
ImpactArbiter — Autograd Trap & Fuzzer
======================================

Bridges discrete integer routing decisions (the output of the LLM-generated
mapping kernel) into a *continuous, differentiable* PyTorch graph. The
backward pass is the source of truth: if the agent's routed gradient does
not equal the Oracle's routed gradient, the agent hallucinated.

Design
------
For a fetched vector ``v = KV_cache[lb, off, :]`` and a fixed ``dummy_query
q``, the scalar loss ``L = sum(v * q)`` has gradient

    dL/d(KV_cache)[i, j, :] = q   if (i, j) == (lb, off)
                              0   otherwise

So a routing hallucination places the gradient at the wrong (block, offset)
slot — which `torch.allclose` detects.

Parallelism
-----------
We use `torch.func.vmap` + `torch.func.grad` to evaluate every test case in
the boundary fuzz batch in a single vectorized backward, avoiding a Python
for-loop over `.backward()` calls.

Thresholds
----------
- divergence < 1e-4   → soft warning (precision noise, human review)
- divergence ≥ 1e-4   → HARD BLOCK (structural hallucination)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.func import grad, vmap

from .oracle import Oracle, get_oracle


# ─────────────────────────────────────────────────────────────────────────────
# Constants & thresholds
# ─────────────────────────────────────────────────────────────────────────────

DIVERGENCE_THRESHOLD = 1e-4  # ≥ this ⇒ Hard Block
HEAD_DIM_DEFAULT = 64

# Default fuzz parameters from the spec.
DEFAULT_MAX_WINDOW_TOKENS = 100
DEFAULT_BLOCK_SIZE = 16
DEFAULT_TOKEN_TEST_CASES: Tuple[int, ...] = (15, 100, 128)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CaseResult:
    token_idx: int
    oracle_lb: int
    oracle_off: int
    agent_lb: int
    agent_off: int
    divergence: float
    verdict: str  # "PASS" | "WARN" | "HARD_BLOCK" | "CRASH"
    error: Optional[str] = None


@dataclass
class TrapReport:
    max_window_tokens: int
    block_size: int
    total_blocks: int
    cases: List[CaseResult] = field(default_factory=list)
    first_hard_block: Optional[CaseResult] = None

    @property
    def passed(self) -> bool:
        return self.first_hard_block is None and all(
            c.verdict in ("PASS", "WARN") for c in self.cases
        )

    def divergence_map(self) -> str:
        """Human-readable per-case divergence dump (also stored in DB)."""
        lines = [
            f"max_window_tokens={self.max_window_tokens} "
            f"block_size={self.block_size} total_blocks={self.total_blocks}",
            "─" * 70,
            f"{'token':>6} | {'oracle (lb, off)':>18} | "
            f"{'agent (lb, off)':>18} | {'div':>10} | verdict",
            "─" * 70,
        ]
        for c in self.cases:
            lines.append(
                f"{c.token_idx:>6} | "
                f"{('('+str(c.oracle_lb)+', '+str(c.oracle_off)+')'):>18} | "
                f"{('('+str(c.agent_lb)+', '+str(c.agent_off)+')'):>18} | "
                f"{c.divergence:>10.2e} | {c.verdict}"
                + (f"  ({c.error})" if c.error else "")
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable kernel + vmap'd grad
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_loss(kv: torch.Tensor, lb: torch.Tensor, off: torch.Tensor,
                q: torch.Tensor) -> torch.Tensor:
    """
    Scalar loss = <KV_cache[lb, off, :], q>. Differentiable in `kv`.

    We index via F.embedding on a flattened view so the kernel is
    vmap-compatible (vmap forbids data-dependent advanced indexing).
    """
    block_size = kv.shape[1]
    head_dim = kv.shape[-1]
    flat_idx = (lb * block_size + off).reshape(1)
    kv_flat = kv.reshape(-1, head_dim)
    v = torch.nn.functional.embedding(flat_idx, kv_flat).reshape(head_dim)
    return (v * q).sum()


# `argnums=0` ⇒ gradient w.r.t. the leaf KV_cache tensor.
# `in_dims=(None, 0, 0, None)` ⇒ broadcast `kv` and `q`, vmap over (lb, off).
_grad_fn = grad(_fetch_loss, argnums=0)
_batched_grad_fn = vmap(_grad_fn, in_dims=(None, 0, 0, None))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def make_kv_cache(
    total_blocks: int,
    block_size: int,
    head_dim: int = HEAD_DIM_DEFAULT,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic continuous leaf KV_cache and dummy_query."""
    g = torch.Generator().manual_seed(seed)
    kv = torch.randn(
        total_blocks, block_size, head_dim,
        dtype=torch.float32, generator=g, requires_grad=True,
    )
    q = torch.randn(head_dim, dtype=torch.float32, generator=g)
    return kv, q


def _safe_call_agent(
    target_fn: Callable,
    token_idx: int,
    max_window_tokens: int,
    block_size: int,
    total_blocks: int,
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Invoke the agent kernel, normalising signature variants gracefully."""
    try:
        try:
            out = target_fn(token_idx, max_window_tokens, block_size)
        except TypeError:
            # tolerate kernels written against the simpler 2-arg signature
            out = target_fn(token_idx, block_size)
        lb, off = out
        return int(lb), int(off), None
    except Exception as e:  # noqa: BLE001  — surface any kernel crash verbatim
        return None, None, f"{type(e).__name__}: {e}"


def run_trap(
    target_fn: Callable,
    *,
    max_window_tokens: int = DEFAULT_MAX_WINDOW_TOKENS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    token_test_cases: Sequence[int] = DEFAULT_TOKEN_TEST_CASES,
    oracle: Optional[Oracle] = None,
    head_dim: int = HEAD_DIM_DEFAULT,
) -> TrapReport:
    """
    Run the autograd trap on a target mapping function.

    The gradient comparison is done *in parallel* across all test cases via
    `torch.vmap` over `torch.func.grad`.
    """
    if oracle is None:
        oracle = get_oracle("ring_buffer_v1")

    total_blocks = max_window_tokens // block_size
    kv, q = make_kv_cache(total_blocks, block_size, head_dim=head_dim)
    # We don't need `requires_grad=True` for the functional `torch.func.grad`
    # path, but the spec mandates a continuous leaf — so we keep it as a leaf
    # and just detach inside the functional call.
    kv_leaf = kv.detach().clone().requires_grad_(False)

    report = TrapReport(
        max_window_tokens=max_window_tokens,
        block_size=block_size,
        total_blocks=total_blocks,
    )

    # ── 1. Collect agent + oracle routings (CPU-side, since agent code is
    # arbitrary Python and may not be tensor-friendly) ───────────────────────
    oracle_lbs: List[int] = []
    oracle_offs: List[int] = []
    agent_lbs: List[int] = []
    agent_offs: List[int] = []
    crashes: List[Optional[str]] = []

    for t in token_test_cases:
        olb, ooff = oracle(t, max_window_tokens, block_size)
        alb, aoff, err = _safe_call_agent(
            target_fn, t, max_window_tokens, block_size, total_blocks
        )
        oracle_lbs.append(olb)
        oracle_offs.append(ooff)
        # Clamp the agent's output to the legal index range so we can still
        # *evaluate* the autograd graph rather than just crashing — divergence
        # will then expose the misroute.
        if alb is None or aoff is None:
            agent_lbs.append(0)
            agent_offs.append(0)
        else:
            agent_lbs.append(max(0, min(alb, total_blocks - 1)))
            agent_offs.append(max(0, min(aoff, block_size - 1)))
        crashes.append(err)

    olb_t = torch.tensor(oracle_lbs, dtype=torch.long)
    ooff_t = torch.tensor(oracle_offs, dtype=torch.long)
    alb_t = torch.tensor(agent_lbs, dtype=torch.long)
    aoff_t = torch.tensor(agent_offs, dtype=torch.long)

    # ── 2. Vectorised gradient computation via torch.vmap ────────────────────
    oracle_grads = _batched_grad_fn(kv_leaf, olb_t, ooff_t, q)
    agent_grads = _batched_grad_fn(kv_leaf, alb_t, aoff_t, q)

    # ── 3. Per-case divergence + verdict ─────────────────────────────────────
    diffs = (oracle_grads - agent_grads).reshape(len(token_test_cases), -1)
    divergences = diffs.abs().max(dim=1).values.tolist()

    for i, t in enumerate(token_test_cases):
        div = float(divergences[i])
        err = crashes[i]

        if err is not None:
            verdict = "CRASH"
        elif torch.allclose(
            oracle_grads[i], agent_grads[i], rtol=1e-5, atol=1e-8
        ):
            verdict = "PASS"
        elif div < DIVERGENCE_THRESHOLD:
            verdict = "WARN"
        else:
            verdict = "HARD_BLOCK"

        case = CaseResult(
            token_idx=t,
            oracle_lb=oracle_lbs[i],
            oracle_off=oracle_offs[i],
            agent_lb=agent_lbs[i] if err is None else -1,
            agent_off=agent_offs[i] if err is None else -1,
            divergence=div,
            verdict=verdict,
            error=err,
        )
        report.cases.append(case)
        if (
            verdict in ("HARD_BLOCK", "CRASH")
            and report.first_hard_block is None
        ):
            report.first_hard_block = case

    return report


__all__ = [
    "DIVERGENCE_THRESHOLD",
    "DEFAULT_MAX_WINDOW_TOKENS",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_TOKEN_TEST_CASES",
    "CaseResult",
    "TrapReport",
    "run_trap",
    "make_kv_cache",
]
