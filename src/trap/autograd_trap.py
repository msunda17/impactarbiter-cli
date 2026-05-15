"""Autograd Trap.

Verifies that an agent-generated routing function produces the same
KV-cache write pattern as the SymPy oracle by comparing the gradients of a
dummy loss with respect to a leaf KV-cache tensor.

KV-cache shapes
---------------
- PagedAttention : ``[num_blocks, block_size, head_dim]``  (e.g. ``[100, 16, 128]``)
- RadixAttention : ``[num_heads, num_blocks, block_size, head_dim]``  (e.g. ``[8, 100, 16, 128]``)

Trap procedure
--------------
1. Two leaf tensors of identical shape, both ``requires_grad=True``.
2. Each is sliced at the agent / oracle (logical_block, offset) coordinates.
3. ``loss = sum(fetched * dummy_query)``; ``loss.backward()`` for both.
4. Divergence = ``(agent.grad - oracle.grad).abs().max().item()``.
5. If divergence > tolerance, emit an ASCII GRADIENT DIVERGENCE MAP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

DEFAULT_TOLERANCE = 1e-4
DEFAULT_BLOCK_SIZE = 16
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_BLOCKS = 100
DEFAULT_NUM_HEADS = 8


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrapResult:
    """Outcome of one trap invocation."""

    diverged: bool
    divergence_value: float
    tolerance: float
    per_token: List[dict] = field(default_factory=list)
    divergence_map: str = ""

    def __bool__(self) -> bool:  # truthy when CLEAN
        return not self.diverged


# ─────────────────────────────────────────────────────────────────────────────
# ASCII map formatter
# ─────────────────────────────────────────────────────────────────────────────
def format_divergence_map(per_token: List[dict], head_dim: int = DEFAULT_HEAD_DIM) -> str:
    """Render the GRADIENT DIVERGENCE MAP block."""
    lines = ["GRADIENT DIVERGENCE MAP — KV_cache.grad"]
    for entry in per_token:
        if not entry.get("diverged", False):
            continue
        lines.append(
            f"Token {entry['token']} | "
            f"Expected: block={entry['oracle_block']} offset={entry['oracle_offset']} | "
            f"Got: block={entry['agent_block']} offset={entry['agent_offset']}"
        )
        lines.append(
            f"Non-zero gradient at: "
            f"[{entry['agent_block']}, {entry['agent_offset']}, :] — "
            f"misrouted {head_dim} floats"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PagedAttention trap
# ─────────────────────────────────────────────────────────────────────────────
def run_paged_trap(
    agent_fn: Callable[[int, int], Tuple[int, int]],
    oracle_fn: Callable[[int, int], Tuple[int, int]],
    *,
    token_indices: List[int],
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    head_dim: int = DEFAULT_HEAD_DIM,
    tolerance: float = DEFAULT_TOLERANCE,
) -> TrapResult:
    """Run the PagedAttention autograd trap over ``token_indices``."""
    agent_cache = torch.zeros(num_blocks, block_size, head_dim, requires_grad=True)
    oracle_cache = torch.zeros(num_blocks, block_size, head_dim, requires_grad=True)

    per_token: List[dict] = []
    agent_losses: List[torch.Tensor] = []
    oracle_losses: List[torch.Tensor] = []

    for tok in token_indices:
        try:
            ab, ao = agent_fn(int(tok), int(block_size))
        except Exception as e:  # noqa: BLE001 — surface as max divergence
            per_token.append({
                "token": int(tok), "oracle_block": None, "oracle_offset": None,
                "agent_block": None, "agent_offset": None,
                "diverged": True, "error": f"{type(e).__name__}: {e}",
            })
            continue
        ob, oo = oracle_fn(int(tok), int(block_size))

        ab, ao = int(ab), int(ao)
        ob, oo = int(ob), int(oo)

        if not (0 <= ab < num_blocks and 0 <= ao < block_size):
            per_token.append({
                "token": int(tok), "oracle_block": ob, "oracle_offset": oo,
                "agent_block": ab, "agent_offset": ao,
                "diverged": True, "error": "agent index out of bounds",
            })
            continue

        dummy_q = torch.ones(head_dim)
        agent_losses.append((agent_cache[ab, ao, :] * dummy_q).sum())
        oracle_losses.append((oracle_cache[ob, oo, :] * dummy_q).sum())

        per_token.append({
            "token": int(tok),
            "oracle_block": ob, "oracle_offset": oo,
            "agent_block": ab, "agent_offset": ao,
            "diverged": (ab, ao) != (ob, oo),
        })

    if agent_losses and oracle_losses:
        agent_loss = torch.stack(agent_losses).sum()
        oracle_loss = torch.stack(oracle_losses).sum()
        agent_loss.backward()
        oracle_loss.backward()
        grad_diff = (agent_cache.grad - oracle_cache.grad).abs().max().item()
        diverged = grad_diff > tolerance
    else:
        # No valid indices to test - treat as no divergence
        grad_diff = 0.0
        diverged = False

    div_map = format_divergence_map(per_token, head_dim=head_dim) if diverged else ""
    return TrapResult(
        diverged=diverged,
        divergence_value=float(grad_diff),
        tolerance=tolerance,
        per_token=per_token,
        divergence_map=div_map,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RadixAttention trap
# ─────────────────────────────────────────────────────────────────────────────
def run_radix_trap(
    agent_fn: Callable[[int, int, int], Tuple[int, int]],
    oracle_fn: Callable[[int, int, int], Tuple[int, int]],
    *,
    cases: List[Tuple[int, int]],  # (b_local_idx, prefix_length)
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    head_dim: int = DEFAULT_HEAD_DIM,
    tolerance: float = DEFAULT_TOLERANCE,
) -> TrapResult:
    """Run the RadixAttention autograd trap over ``cases``."""
    agent_cache = torch.zeros(num_blocks, block_size, head_dim, requires_grad=True)
    oracle_cache = torch.zeros(num_blocks, block_size, head_dim, requires_grad=True)

    per_token: List[dict] = []
    agent_losses: List[torch.Tensor] = []
    oracle_losses: List[torch.Tensor] = []

    for (b_local_idx, prefix_length) in cases:
        token_label = f"(prefix={prefix_length},b_local_idx={b_local_idx})"
        try:
            ab, ao = agent_fn(int(b_local_idx), int(prefix_length), int(block_size))
        except Exception as e:  # noqa: BLE001
            per_token.append({
                "token": token_label, "oracle_block": None, "oracle_offset": None,
                "agent_block": None, "agent_offset": None,
                "diverged": True, "error": f"{type(e).__name__}: {e}",
            })
            continue
        ob, oo = oracle_fn(int(b_local_idx), int(prefix_length), int(block_size))

        ab, ao = int(ab), int(ao)
        ob, oo = int(ob), int(oo)

        if not (
            0 <= ab < num_blocks
            and 0 <= ao < block_size
        ):
            per_token.append({
                "token": token_label, "oracle_block": ob, "oracle_offset": oo,
                "agent_block": ab, "agent_offset": ao,
                "diverged": True, "error": "agent index out of bounds",
            })
            continue

        dummy_q = torch.ones(head_dim)
        agent_losses.append((agent_cache[ab, ao, :] * dummy_q).sum())
        oracle_losses.append((oracle_cache[ob, oo, :] * dummy_q).sum())

        per_token.append({
            "token": token_label,
            "oracle_block": ob, "oracle_offset": oo,
            "agent_block": ab, "agent_offset": ao,
            "diverged": (ab, ao) != (ob, oo),
        })

    if agent_losses and oracle_losses:
        agent_loss = torch.stack(agent_losses).sum()
        oracle_loss = torch.stack(oracle_losses).sum()
        agent_loss.backward()
        oracle_loss.backward()
        grad_diff = (agent_cache.grad - oracle_cache.grad).abs().max().item()
        diverged = grad_diff > tolerance
    else:
        # No valid indices to test - treat as no divergence
        grad_diff = 0.0
        diverged = False

    div_map = format_divergence_map(per_token, head_dim=head_dim) if diverged else ""
    return TrapResult(
        diverged=diverged,
        divergence_value=float(grad_diff),
        tolerance=tolerance,
        per_token=per_token,
        divergence_map=div_map,
    )


__all__ = [
    "TrapResult",
    "run_paged_trap",
    "run_radix_trap",
    "format_divergence_map",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_NUM_BLOCKS",
    "DEFAULT_NUM_HEADS",
    "DEFAULT_HEAD_DIM",
    "DEFAULT_TOLERANCE",
]
