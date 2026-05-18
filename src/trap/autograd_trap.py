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
    else:
        grad_diff = 0.0

    # Aggregate flag must OR per-token divergences so that a run where every
    # case crashed (empty ``agent_losses``) cannot silently report PASS.
    any_per_token_diverged = any(e.get("diverged", False) for e in per_token)
    diverged = (grad_diff > tolerance) or any_per_token_diverged
    if diverged and grad_diff == 0.0:
        grad_diff = 1.0

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
    else:
        grad_diff = 0.0

    any_per_token_diverged = any(e.get("diverged", False) for e in per_token)
    diverged = (grad_diff > tolerance) or any_per_token_diverged
    if diverged and grad_diff == 0.0:
        grad_diff = 1.0

    div_map = format_divergence_map(per_token, head_dim=head_dim) if diverged else ""
    return TrapResult(
        diverged=diverged,
        divergence_value=float(grad_diff),
        tolerance=tolerance,
        per_token=per_token,
        divergence_map=div_map,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2D Asymmetric Radix trap (per-head ring buffer)
# ─────────────────────────────────────────────────────────────────────────────
def format_divergence_map_2d(per_token: List[dict], head_dim: int = DEFAULT_HEAD_DIM) -> str:
    """Render the 4D GRADIENT DIVERGENCE MAP (head, block, offset)."""
    lines = ["GRADIENT DIVERGENCE MAP — KV_cache.grad (head × block × offset)"]
    for entry in per_token:
        if not entry.get("diverged", False):
            continue
        lines.append(
            f"Token {entry['token']} | "
            f"Expected: head={entry['oracle_head']} block={entry['oracle_block']} "
            f"offset={entry['oracle_offset']} | "
            f"Got: head={entry['agent_head']} block={entry['agent_block']} "
            f"offset={entry['agent_offset']}"
        )
        lines.append(
            f"Non-zero gradient at: "
            f"[{entry['agent_head']}, {entry['agent_block']}, {entry['agent_offset']}, :] — "
            f"misrouted {head_dim} floats"
        )
    return "\n".join(lines)


def run_radix_2d_trap(
    agent_fn: Callable[..., Tuple[int, int, int]],
    oracle_fn: Callable[..., Tuple[int, int, int]],
    *,
    cases: List[Tuple[int, int, int, int]],  # (b_local_idx, head_idx, prefix_length_h, total_blocks_h)
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_heads: int = DEFAULT_NUM_HEADS,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    head_dim: int = DEFAULT_HEAD_DIM,
    tolerance: float = DEFAULT_TOLERANCE,
) -> TrapResult:
    """Run the 2D Asymmetric Radix autograd trap over ``cases``.

    The KV cache is 4D: ``[num_heads, num_blocks, block_size, head_dim]``.
    Each case is ``(b_local_idx, head_idx, prefix_length_h, total_blocks_h)``.
    """
    agent_cache = torch.zeros(num_heads, num_blocks, block_size, head_dim, requires_grad=True)
    oracle_cache = torch.zeros(num_heads, num_blocks, block_size, head_dim, requires_grad=True)

    per_token: List[dict] = []
    agent_losses: List[torch.Tensor] = []
    oracle_losses: List[torch.Tensor] = []

    for (b_local_idx, head_idx, prefix_length_h, total_blocks_h) in cases:
        token_label = (
            f"(b={b_local_idx},h={head_idx},"
            f"prefix_h={prefix_length_h},N_h={total_blocks_h})"
        )
        # Compute oracle first so we can record expected values even if the
        # agent crashes or returns garbage.
        oh, ob, oo = oracle_fn(
            int(b_local_idx), int(head_idx),
            int(prefix_length_h), int(total_blocks_h), int(block_size),
        )
        oh, ob, oo = int(oh), int(ob), int(oo)

        try:
            raw = agent_fn(
                int(b_local_idx), int(head_idx),
                int(prefix_length_h), int(total_blocks_h), int(block_size),
            )
        except Exception as e:  # noqa: BLE001
            per_token.append({
                "token": token_label,
                "oracle_head": oh, "oracle_block": ob, "oracle_offset": oo,
                "agent_head": None, "agent_block": None, "agent_offset": None,
                "diverged": True,
                "error": f"agent crashed: {type(e).__name__}: {e}",
            })
            continue

        # Reject None / wrong-arity returns up front. A None index cannot
        # be sliced into the KV cache, so we treat it as a hard divergence
        # instead of silently zeroing out the gradient.
        if raw is None or not hasattr(raw, "__iter__"):
            per_token.append({
                "token": token_label,
                "oracle_head": oh, "oracle_block": ob, "oracle_offset": oo,
                "agent_head": None, "agent_block": None, "agent_offset": None,
                "diverged": True,
                "error": f"agent returned non-tuple: {raw!r}",
            })
            continue
        raw_list = list(raw)
        if len(raw_list) != 3 or any(v is None for v in raw_list):
            per_token.append({
                "token": token_label,
                "oracle_head": oh, "oracle_block": ob, "oracle_offset": oo,
                "agent_head": raw_list[0] if len(raw_list) > 0 else None,
                "agent_block": raw_list[1] if len(raw_list) > 1 else None,
                "agent_offset": raw_list[2] if len(raw_list) > 2 else None,
                "diverged": True,
                "error": f"agent returned malformed tuple {raw_list!r} "
                         f"(expected (head, block, offset))",
            })
            continue

        try:
            ah, ab, ao = int(raw_list[0]), int(raw_list[1]), int(raw_list[2])
        except (TypeError, ValueError) as e:
            per_token.append({
                "token": token_label,
                "oracle_head": oh, "oracle_block": ob, "oracle_offset": oo,
                "agent_head": raw_list[0], "agent_block": raw_list[1], "agent_offset": raw_list[2],
                "diverged": True,
                "error": f"agent returned non-integer indices: {type(e).__name__}: {e}",
            })
            continue

        if not (
            0 <= ah < num_heads
            and 0 <= ab < num_blocks
            and 0 <= ao < block_size
        ):
            per_token.append({
                "token": token_label,
                "oracle_head": oh, "oracle_block": ob, "oracle_offset": oo,
                "agent_head": ah, "agent_block": ab, "agent_offset": ao,
                "diverged": True, "error": "agent index out of bounds (ring buffer overflow or head scramble)",
            })
            continue

        dummy_q = torch.ones(head_dim)
        agent_losses.append((agent_cache[ah, ab, ao, :] * dummy_q).sum())
        oracle_losses.append((oracle_cache[oh, ob, oo, :] * dummy_q).sum())

        per_token.append({
            "token": token_label,
            "oracle_head": oh, "oracle_block": ob, "oracle_offset": oo,
            "agent_head": ah, "agent_block": ab, "agent_offset": ao,
            "diverged": (ah, ab, ao) != (oh, ob, oo),
        })

    if agent_losses and oracle_losses:
        agent_loss = torch.stack(agent_losses).sum()
        oracle_loss = torch.stack(oracle_losses).sum()
        agent_loss.backward()
        oracle_loss.backward()
        grad_diff = (agent_cache.grad - oracle_cache.grad).abs().max().item()
    else:
        grad_diff = 0.0

    # CRITICAL: aggregate flag must OR per-token divergences. If every case
    # crashed (e.g. arity mismatch) ``agent_losses`` is empty and ``grad_diff``
    # is 0.0, but the trap MUST still report failure. Previously this returned
    # a false PASS — the bug surfaced when the LLM renamed parameters and
    # silently dropped ``b_local_idx``.
    any_per_token_diverged = any(e.get("diverged", False) for e in per_token)
    diverged = (grad_diff > tolerance) or any_per_token_diverged
    if diverged and grad_diff == 0.0:
        # Surface a sentinel divergence value so downstream rendering /
        # CSV export does not show ``0.00e+00`` next to a HARD_BLOCK.
        grad_diff = 1.0

    div_map = format_divergence_map_2d(per_token, head_dim=head_dim) if diverged else ""
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
    "run_radix_2d_trap",
    "format_divergence_map",
    "format_divergence_map_2d",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_NUM_BLOCKS",
    "DEFAULT_NUM_HEADS",
    "DEFAULT_HEAD_DIM",
    "DEFAULT_TOLERANCE",
]
