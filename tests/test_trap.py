"""Programmatic claims for the autograd trap.

These four assertions are the load-bearing claims of the project:

1. Naive PagedAttention hallucination on token_idx=100 → divergence > 1e-4.
2. Correct PagedAttention on token_idx=100 → divergence < 1e-4.
3. Naive RadixAttention on prefix=47, b_local_idx=0 → divergence > 1e-4.
4. Correct RadixAttention on prefix=47, b_local_idx=0 → divergence < 1e-4.
"""

from src.oracles import paged_oracle, radix_oracle
from src.trap import run_paged_trap, run_radix_trap
from typing import Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.cli.auto_heal import _adapt_function_signature

BLOCK_SIZE = 16
TOL = 1e-4


# ── Naive (broken) routers ───────────────────────────────────────────────────
def naive_paged(token_idx: int, block_size: int):
    """Hallucinated kernel: forgets the modulo and shifts by one block."""
    return token_idx // block_size + 1, 0  # off-by-one block, wrong offset


def correct_paged(token_idx: int, block_size: int):
    return token_idx // block_size, token_idx % block_size


def naive_radix_hallucination(b_local_idx: int, prefix_length: int, block_size: int = 16) -> Tuple[int, int]:
    """Naive RadixAttention: ignores prefix, treats b_local_idx as absolute position."""
    return b_local_idx // block_size, b_local_idx % block_size


def correct_radix(b_local_idx: int, prefix_length: int, block_size: int = 16) -> Tuple[int, int]:
    """Correct RadixAttention: absolute position = prefix_length + b_local_idx."""
    absolute_position = prefix_length + b_local_idx
    return absolute_position // block_size, absolute_position % block_size


# ── Claim 1 & 2: PagedAttention at token_idx = 100 ───────────────────────────
def test_naive_paged_hallucination_diverges_at_token_100():
    res = run_paged_trap(
        naive_paged, paged_oracle,
        token_indices=[100],
        block_size=BLOCK_SIZE,
    )
    assert res.diverged, res
    assert res.divergence_value > TOL


def test_correct_paged_passes_at_token_100():
    res = run_paged_trap(
        correct_paged, paged_oracle,
        token_indices=[100],
        block_size=BLOCK_SIZE,
    )
    assert not res.diverged, res
    assert res.divergence_value < TOL


# ── Claim 3 & 4: RadixAttention at prefix=47, b_local_idx=0 ─────────────────
def test_naive_radix_hallucination_diverges_on_ragged_prefix() -> None:
    """Naive implementation diverges on ragged prefix (prefix=47, b_local_idx=0)."""
    # prefix=47, block_size=16: 47 % 16 = 15 (ragged boundary)
    # naive: 0 // 16 = 0, 0 % 16 = 0 → (0, 0)
    # correct: (47 + 0) // 16 = 2, (47 + 0) % 16 = 15 → (2, 15)
    result = run_radix_trap(
        naive_radix_hallucination,
        radix_oracle,
        cases=[(0, 47)],  # b_local_idx=0, prefix_length=47
        block_size=16,
    )
    assert result.diverged is True
    assert result.divergence_value > 1e-4


def test_correct_radix_passes_on_ragged_prefix() -> None:
    """Correct implementation passes on ragged prefix (prefix=47, b_local_idx=0)."""
    result = run_radix_trap(
        correct_radix,
        radix_oracle,
        cases=[(0, 47)],  # b_local_idx=0, prefix_length=47
        block_size=BLOCK_SIZE,
    )
    assert not result.diverged, result
    assert result.divergence_value < TOL


def test_signature_adaptation_paged() -> None:
    """Test signature adaptation for paged function."""
    def refactored_paged(prefix_length: int, b_local_idx: int, block_size: int) -> Tuple[int, int]:
        return (prefix_length // block_size + b_local_idx // block_size, b_local_idx % block_size)

    adapted = _adapt_function_signature(refactored_paged, ["token_idx", "block_size"])
    block, offset = adapted(100, 16)
    assert block == 6  # 100 // 16 = 6
    assert offset == 4  # 100 % 16 = 4


def test_signature_adaptation_radix() -> None:
    """Test signature adaptation for radix function."""
    def refactored_radix(token_idx: int, block_size: int) -> Tuple[int, int]:
        return (token_idx // block_size, token_idx % block_size)

    adapted = _adapt_function_signature(refactored_radix, ["b_local_idx", "prefix_length", "block_size"])
    block, offset = adapted(0, 47, 16)
    assert block == 2  # (47 + 0) // 16 = 2
    assert offset == 15  # (47 + 0) % 16 = 15
