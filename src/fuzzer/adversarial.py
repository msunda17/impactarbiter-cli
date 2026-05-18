"""Adversarial boundary-condition fuzzer.

Boundary cases are explicit (not randomly generated) so failures are
reproducible and reviewable. ``torch.vmap`` is used where the agent kernel is
naturally vectorizable; otherwise we fall back to a deterministic Python
batching loop with the exact same semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Boundary fixtures
# ─────────────────────────────────────────────────────────────────────────────

# PagedAttention adversarial token indices (block_size = 16).
#  • 15  → last legal slot of block 0
#  • 31  → last slot of block 1
#  • 47  → ragged boundary (block_size=16, would be block 2 offset 15)
#  • 99  → last slot of block 6 (one before a hard boundary)
#  • 100 → first slot of block 6 — classic off-by-one trap
#  • 105 → mid-block 6
#  • 128 → first slot of block 8 — clean wrap
PAGED_ADVERSARIAL_TOKENS: List[int] = [15, 31, 47, 99, 100, 105, 128]


@dataclass(frozen=True)
class RadixCase:
    """One RadixAttention boundary fixture."""
    prefix_length: int
    b_local_idx: int
    expected_block: int
    expected_offset: int
    note: str = ""


# RadixAttention test matrix (block_size = 16).
RADIX_TEST_MATRIX: List[RadixCase] = [
    RadixCase(prefix_length=47, b_local_idx=0, expected_block=2, expected_offset=15,
              note="ragged straddle — partial-block carry-over"),
    RadixCase(prefix_length=47, b_local_idx=1, expected_block=3, expected_offset=0,
              note="ragged straddle wrap into next block"),
    RadixCase(prefix_length=48, b_local_idx=0, expected_block=3, expected_offset=0,
              note="clean boundary"),
    RadixCase(prefix_length=63, b_local_idx=0, expected_block=3, expected_offset=15,
              note="last slot of block 3"),
    RadixCase(prefix_length=64, b_local_idx=0, expected_block=4, expected_offset=0,
              note="clean boundary"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Fuzz drivers
# ─────────────────────────────────────────────────────────────────────────────
def _try_vmap_paged(
    agent_fn: Callable[[int, int], Tuple[int, int]],
    tokens: torch.Tensor,
    block_size: int,
) -> Optional[List[Tuple[int, int]]]:
    """Try torch.vmap; fall back to None if the agent fn is not vectorizable."""
    try:
        def _scalar(t: torch.Tensor) -> torch.Tensor:
            lb, off = agent_fn(int(t.item()), int(block_size))
            return torch.tensor([int(lb), int(off)], dtype=torch.long)

        out = torch.vmap(_scalar)(tokens)
        return [(int(row[0]), int(row[1])) for row in out]
    except Exception:  # noqa: BLE001 — agent isn't vmap-friendly
        return None


def fuzz_paged(
    agent_fn: Callable[[int, int], Tuple[int, int]],
    oracle_fn: Callable[[int, int], Tuple[int, int]],
    *,
    block_size: int = 16,
    tokens: Optional[List[int]] = None,
) -> List[dict]:
    """Run the PagedAttention boundary fuzzer.

    Returns one dict per token with both the agent and oracle outputs and a
    ``diverged`` flag.
    """
    tokens = tokens if tokens is not None else PAGED_ADVERSARIAL_TOKENS
    tok_tensor = torch.tensor(tokens, dtype=torch.long)

    vm_results = _try_vmap_paged(agent_fn, tok_tensor, block_size)

    rows: List[dict] = []
    for i, tok in enumerate(tokens):
        if vm_results is not None:
            ab, ao = vm_results[i]
        else:
            try:
                ab, ao = agent_fn(int(tok), int(block_size))
                ab, ao = int(ab), int(ao)
            except Exception as e:  # noqa: BLE001
                rows.append({
                    "token": int(tok),
                    "agent_block": None, "agent_offset": None,
                    "oracle_block": None, "oracle_offset": None,
                    "diverged": True, "error": f"{type(e).__name__}: {e}",
                })
                continue
        ob, oo = oracle_fn(int(tok), int(block_size))
        rows.append({
            "token": int(tok),
            "agent_block": int(ab), "agent_offset": int(ao),
            "oracle_block": int(ob), "oracle_offset": int(oo),
            "diverged": (int(ab), int(ao)) != (int(ob), int(oo)),
        })
    return rows


def fuzz_radix(
    target_fn: Callable[[int, int, int], Tuple[int, int]],
    oracle_fn: Callable[[int, int, int], Tuple[int, int]],
    *,
    block_size: int = 16,
) -> List[dict]:
    """Fuzz RadixAttention with explicit boundary test matrix."""
    results = []
    for case in RADIX_TEST_MATRIX:
        try:
            agent_block, agent_offset = target_fn(
                int(case.b_local_idx), int(case.prefix_length), int(block_size)
            )
        except Exception as e:  # noqa: BLE001
            results.append({
                "prefix_length": case.prefix_length,
                "b_local_idx": case.b_local_idx,
                "agent_block": None,
                "agent_offset": None,
                "oracle_block": case.expected_block,
                "oracle_offset": case.expected_offset,
                "diverged": True,
                "error": f"{type(e).__name__}: {e}",
                "note": case.note,
            })
            continue
        results.append({
            "prefix_length": case.prefix_length,
            "b_local_idx": case.b_local_idx,
            "agent_block": int(agent_block),
            "agent_offset": int(agent_offset),
            "oracle_block": case.expected_block,
            "oracle_offset": case.expected_offset,
            "diverged": (int(agent_block), int(agent_offset)) != (case.expected_block, case.expected_offset),
            "error": None,
            "note": case.note,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2D Asymmetric Radix fixtures with per-head ring buffers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Radix2DCase:
    """One 2D Asymmetric Radix boundary fixture."""
    b_local_idx: int
    head_idx: int
    prefix_length_h: int
    total_blocks_h: int
    expected_head: int
    expected_block: int
    expected_offset: int
    note: str = ""


# RADIX_2D test matrix (block_size = 16).
# Mix of: asymmetric ragged boundaries (different prefix_length_h per head),
# clean wraps, and ring-buffer modulo wrapping cases.
RADIX_2D_TEST_MATRIX: List[Radix2DCase] = [
    # ── Asymmetric ragged boundaries ────────────────────────────────────────
    Radix2DCase(b_local_idx=0, head_idx=0, prefix_length_h=47, total_blocks_h=8,
                expected_head=0, expected_block=2, expected_offset=15,
                note="head=0 ragged straddle (prefix=47)"),
    Radix2DCase(b_local_idx=0, head_idx=1, prefix_length_h=20, total_blocks_h=8,
                expected_head=1, expected_block=1, expected_offset=4,
                note="head=1 asymmetric prefix=20 (different from head=0)"),
    Radix2DCase(b_local_idx=1, head_idx=0, prefix_length_h=47, total_blocks_h=8,
                expected_head=0, expected_block=3, expected_offset=0,
                note="head=0 wrap into next block"),
    # ── Ring-buffer modulo wrap ─────────────────────────────────────────────
    # total_blocks_h=4, block_size=16 → capacity = 64 tokens per head.
    # prefix=60, b_local=5 → abs=65 → block = 65//16 = 4, wraps to 4 % 4 = 0.
    Radix2DCase(b_local_idx=5, head_idx=0, prefix_length_h=60, total_blocks_h=4,
                expected_head=0, expected_block=0, expected_offset=1,
                note="ring-buffer wrap: abs=65, block 4 wraps to 0"),
    # prefix=64, b_local=0 → abs=64 → block = 64//16 = 4, wraps to 0.
    Radix2DCase(b_local_idx=0, head_idx=2, prefix_length_h=64, total_blocks_h=4,
                expected_head=2, expected_block=0, expected_offset=0,
                note="ring-buffer wrap at exact capacity"),
    # Deep wrap: prefix=200, b_local=0, total=4 → abs=200, block=12, wraps to 0.
    Radix2DCase(b_local_idx=0, head_idx=3, prefix_length_h=200, total_blocks_h=4,
                expected_head=3, expected_block=0, expected_offset=8,
                note="ring-buffer deep wrap (multiple revolutions)"),
    # Per-head asymmetric ring sizes
    Radix2DCase(b_local_idx=2, head_idx=1, prefix_length_h=30, total_blocks_h=2,
                expected_head=1, expected_block=0, expected_offset=0,
                note="small ring (N_h=2) wrap on head=1"),
]


def fuzz_radix_2d(
    target_fn: Callable[..., Tuple[int, int, int]],
    oracle_fn: Callable[..., Tuple[int, int, int]],
    *,
    block_size: int = 16,
) -> List[dict]:
    """Fuzz 2D Asymmetric Radix with explicit asymmetric + ring-buffer fixtures."""
    results: List[dict] = []
    for case in RADIX_2D_TEST_MATRIX:
        base_row = {
            "b_local_idx": case.b_local_idx,
            "head_idx": case.head_idx,
            "prefix_length_h": case.prefix_length_h,
            "total_blocks_h": case.total_blocks_h,
            "oracle_head": case.expected_head,
            "oracle_block": case.expected_block,
            "oracle_offset": case.expected_offset,
            "note": case.note,
        }
        try:
            raw = target_fn(
                int(case.b_local_idx),
                int(case.head_idx),
                int(case.prefix_length_h),
                int(case.total_blocks_h),
                int(block_size),
            )
        except Exception as e:  # noqa: BLE001
            results.append({
                **base_row,
                "agent_head": None, "agent_block": None, "agent_offset": None,
                "diverged": True,
                "error": f"agent crashed: {type(e).__name__}: {e}",
            })
            continue

        # Reject None / wrong-shape returns explicitly so a silently broken
        # signature adapter cannot masquerade as a passing case.
        if raw is None or not hasattr(raw, "__iter__"):
            results.append({
                **base_row,
                "agent_head": None, "agent_block": None, "agent_offset": None,
                "diverged": True,
                "error": f"agent returned non-tuple: {raw!r}",
            })
            continue
        raw_list = list(raw)
        if len(raw_list) != 3 or any(v is None for v in raw_list):
            results.append({
                **base_row,
                "agent_head": raw_list[0] if len(raw_list) > 0 else None,
                "agent_block": raw_list[1] if len(raw_list) > 1 else None,
                "agent_offset": raw_list[2] if len(raw_list) > 2 else None,
                "diverged": True,
                "error": f"agent returned malformed tuple {raw_list!r} "
                         f"(expected (head, block, offset))",
            })
            continue
        agent_head, agent_block, agent_offset = raw_list
        results.append({
            "b_local_idx": case.b_local_idx,
            "head_idx": case.head_idx,
            "prefix_length_h": case.prefix_length_h,
            "total_blocks_h": case.total_blocks_h,
            "agent_head": int(agent_head),
            "agent_block": int(agent_block),
            "agent_offset": int(agent_offset),
            "oracle_head": case.expected_head,
            "oracle_block": case.expected_block,
            "oracle_offset": case.expected_offset,
            "diverged": (
                (int(agent_head), int(agent_block), int(agent_offset))
                != (case.expected_head, case.expected_block, case.expected_offset)
            ),
            "error": None,
            "note": case.note,
        })
    return results


__all__ = [
    "PAGED_ADVERSARIAL_TOKENS",
    "RADIX_TEST_MATRIX",
    "RADIX_2D_TEST_MATRIX",
    "RadixCase",
    "Radix2DCase",
    "fuzz_paged",
    "fuzz_radix",
    "fuzz_radix_2d",
]
