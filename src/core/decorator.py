"""@verify(oracle="...") — unified decorator interface.

Wrap any routing function whose output is a ``(logical_block, offset)`` tuple
and ImpactArbiter will, on every call, compare the function's discrete write
pattern against a deterministic SymPy oracle by inspecting gradients of a
dummy KV-cache tensor.

Example:
    >>> from src.core import verify
    >>> @verify(oracle="RadixFork")
    ... def fork_for_executors(b_local_idx, prefix_length, block_size=16):
    ...     return b_local_idx // block_size, b_local_idx % block_size  # buggy
    >>> fork_for_executors(0, prefix_length=47, block_size=16)
    Traceback (most recent call last):
        ...
    src.core.decorator.HardBlockException: ...

The trap is **fully deterministic**: identical (block, offset) outputs always
produce identical gradient-divergence results. Only the upstream LLM is
stochastic — the verification layer is not.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Tuple

import torch
from rich.console import Console

from ..oracles import paged_oracle, radix_oracle, radix_2d_oracle

console = Console()

# Tolerance is shared with the autograd trap module — keep in sync.
_DEFAULT_TOLERANCE = 1e-4
_DEFAULT_BLOCK_SIZE = 16
_DEFAULT_NUM_BLOCKS = 100
_DEFAULT_NUM_HEADS = 8
_DEFAULT_HEAD_DIM = 128


class HardBlockException(RuntimeError):
    """Raised when the autograd trap catches a silent context leak.

    Carries the offending (oracle, agent) coordinates and the gradient
    divergence magnitude so callers can surface a precise root-cause without
    re-running the trap.
    """

    def __init__(
        self,
        *,
        oracle_name: str,
        oracle_block: int,
        oracle_offset: int,
        agent_block: int,
        agent_offset: int,
        divergence: float,
        tolerance: float,
        oracle_head: int | None = None,
        agent_head: int | None = None,
    ) -> None:
        self.oracle_name = oracle_name
        self.oracle_block = oracle_block
        self.oracle_offset = oracle_offset
        self.agent_block = agent_block
        self.agent_offset = agent_offset
        self.divergence = divergence
        self.tolerance = tolerance
        self.oracle_head = oracle_head
        self.agent_head = agent_head
        if oracle_head is not None and agent_head is not None:
            super().__init__(
                f"[AUTOGRAD TRAP] HARD_BLOCK on oracle={oracle_name}: "
                f"divergence={divergence:.2e} > tol={tolerance:.0e} "
                f"(expected head={oracle_head} block={oracle_block} offset={oracle_offset}, "
                f"got head={agent_head} block={agent_block} offset={agent_offset})"
            )
        else:
            super().__init__(
                f"[AUTOGRAD TRAP] HARD_BLOCK on oracle={oracle_name}: "
                f"divergence={divergence:.2e} > tol={tolerance:.0e} "
                f"(expected block={oracle_block} offset={oracle_offset}, "
                f"got block={agent_block} offset={agent_offset})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Oracle registry — keyed by the name the decorator user supplies.
# ─────────────────────────────────────────────────────────────────────────────
def _radix_fork_oracle(*args: Any, **kwargs: Any) -> Tuple[int, int]:
    """RadixFork = the Planner→Executor handoff form of RadixAttention.

    Accepted call signatures (whichever the wrapped function uses):
        (b_local_idx, prefix_length, block_size)
        (b_local_idx=..., prefix_length=..., block_size=...)
    """
    if args:
        b_local_idx = int(args[0])
        prefix_length = int(args[1]) if len(args) > 1 else int(kwargs.get("prefix_length", 0))
        block_size = (
            int(args[2]) if len(args) > 2 else int(kwargs.get("block_size", _DEFAULT_BLOCK_SIZE))
        )
    else:
        b_local_idx = int(kwargs["b_local_idx"])
        prefix_length = int(kwargs.get("prefix_length", 0))
        block_size = int(kwargs.get("block_size", _DEFAULT_BLOCK_SIZE))
    return radix_oracle(b_local_idx, prefix_length, block_size)


def _paged_oracle_adapter(*args: Any, **kwargs: Any) -> Tuple[int, int]:
    """PagedAttention oracle adapter.

    Accepted call signatures:
        (token_idx, block_size)
    """
    if args:
        token_idx = int(args[0])
        block_size = (
            int(args[1]) if len(args) > 1 else int(kwargs.get("block_size", _DEFAULT_BLOCK_SIZE))
        )
    else:
        token_idx = int(kwargs["token_idx"])
        block_size = int(kwargs.get("block_size", _DEFAULT_BLOCK_SIZE))
    return paged_oracle(token_idx, block_size)


def _radix_2d_fork_oracle(*args: Any, **kwargs: Any) -> Tuple[int, int, int]:
    """Radix2DFork oracle adapter.

    Accepted call signature (positional or keyword):
        (b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size)
    """
    def _arg(i: int, name: str, default: int = 0) -> int:
        if i < len(args):
            return int(args[i])
        return int(kwargs.get(name, default))

    b_local_idx = _arg(0, "b_local_idx")
    head_idx = _arg(1, "head_idx")
    prefix_length_h = _arg(2, "prefix_length_h")
    total_blocks_h = _arg(3, "total_blocks_h", 1)
    block_size = _arg(4, "block_size", _DEFAULT_BLOCK_SIZE)
    return radix_2d_oracle(
        b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size
    )


_ORACLE_REGISTRY: dict[str, Callable[..., Tuple[int, ...]]] = {
    "RadixFork": _radix_fork_oracle,
    "Radix": _radix_fork_oracle,
    "RadixAttention": _radix_fork_oracle,
    "Radix2DFork": _radix_2d_fork_oracle,
    "Radix2D": _radix_2d_fork_oracle,
    "Paged": _paged_oracle_adapter,
    "PagedAttention": _paged_oracle_adapter,
}


def _resolve_oracle(name: str) -> Callable[..., Tuple[int, int]]:
    if name not in _ORACLE_REGISTRY:
        raise KeyError(
            f"Unknown oracle '{name}'. Available: {sorted(_ORACLE_REGISTRY)}"
        )
    return _ORACLE_REGISTRY[name]


# ─────────────────────────────────────────────────────────────────────────────
# Autograd-backed verification core
# ─────────────────────────────────────────────────────────────────────────────
def _run_autograd_check(
    *,
    oracle_block: int,
    oracle_offset: int,
    agent_block: int,
    agent_offset: int,
    block_size: int,
    num_blocks: int = _DEFAULT_NUM_BLOCKS,
    head_dim: int = _DEFAULT_HEAD_DIM,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> float:
    """Compute |agent.grad − oracle.grad|∞ for the dummy KV-cache."""
    # Out-of-bounds agent index = guaranteed divergence; treat as max.
    if not (0 <= agent_block < num_blocks and 0 <= agent_offset < block_size):
        return float("inf")

    agent_cache = torch.zeros(num_blocks, block_size, head_dim, requires_grad=True)
    oracle_cache = torch.zeros(num_blocks, block_size, head_dim, requires_grad=True)
    dummy_q = torch.ones(head_dim)

    agent_loss = (agent_cache[agent_block, agent_offset, :] * dummy_q).sum()
    oracle_loss = (oracle_cache[oracle_block, oracle_offset, :] * dummy_q).sum()
    agent_loss.backward()
    oracle_loss.backward()

    return float((agent_cache.grad - oracle_cache.grad).abs().max().item())


def _run_autograd_check_2d(
    *,
    oracle_head: int,
    oracle_block: int,
    oracle_offset: int,
    agent_head: int,
    agent_block: int,
    agent_offset: int,
    block_size: int,
    num_heads: int = _DEFAULT_NUM_HEADS,
    num_blocks: int = _DEFAULT_NUM_BLOCKS,
    head_dim: int = _DEFAULT_HEAD_DIM,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> float:
    """4D autograd check for (head, block, offset)."""
    if not (
        0 <= agent_head < num_heads
        and 0 <= agent_block < num_blocks
        and 0 <= agent_offset < block_size
    ):
        return float("inf")

    agent_cache = torch.zeros(num_heads, num_blocks, block_size, head_dim, requires_grad=True)
    oracle_cache = torch.zeros(num_heads, num_blocks, block_size, head_dim, requires_grad=True)
    dummy_q = torch.ones(head_dim)

    agent_loss = (agent_cache[agent_head, agent_block, agent_offset, :] * dummy_q).sum()
    oracle_loss = (oracle_cache[oracle_head, oracle_block, oracle_offset, :] * dummy_q).sum()
    agent_loss.backward()
    oracle_loss.backward()

    return float((agent_cache.grad - oracle_cache.grad).abs().max().item())


# ─────────────────────────────────────────────────────────────────────────────
# Public decorator
# ─────────────────────────────────────────────────────────────────────────────
def verify(
    oracle: str,
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
    raise_on_fail: bool = True,
) -> Callable[[Callable[..., Tuple[int, int]]], Callable[..., Tuple[int, int]]]:
    """Decorator: trap silent context leaks via gradient divergence.

    Args:
        oracle: Name of the registered oracle (e.g. ``"RadixFork"``).
        tolerance: Max acceptable |Δgrad|∞ before flagging a HARD_BLOCK.
        raise_on_fail: If True, raise ``HardBlockException`` on divergence.
            If False, only emit the terminal-UI banner and return the value.
    """
    oracle_fn = _resolve_oracle(oracle)

    def _decorator(
        fn: Callable[..., Tuple[int, ...]],
    ) -> Callable[..., Tuple[int, ...]]:
        @functools.wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any) -> Tuple[int, ...]:
            agent_out = fn(*args, **kwargs)
            if (
                not isinstance(agent_out, (tuple, list))
                or len(agent_out) not in (2, 3)
            ):
                raise TypeError(
                    f"@verify-decorated function {fn.__name__!r} must return "
                    f"(logical_block, offset) or (head_idx, logical_block, offset); "
                    f"got {agent_out!r}"
                )

            oracle_out = oracle_fn(*args, **kwargs)
            is_3d = len(oracle_out) == 3 or len(agent_out) == 3

            # Resolve block_size from kwargs or last positional int.
            block_size = int(kwargs.get("block_size", _DEFAULT_BLOCK_SIZE))
            if args and isinstance(args[-1], int) and len(args) >= 3:
                block_size = int(args[-1])

            if is_3d:
                if len(agent_out) == 2:
                    raise TypeError(
                        f"@verify(oracle={oracle!r}) expected 3-tuple "
                        f"(head_idx, block, offset); got {agent_out!r}"
                    )
                agent_head, agent_block, agent_offset = (
                    int(agent_out[0]), int(agent_out[1]), int(agent_out[2])
                )
                oracle_head, oracle_block, oracle_offset = (
                    int(oracle_out[0]), int(oracle_out[1]), int(oracle_out[2])
                )
                divergence = _run_autograd_check_2d(
                    oracle_head=oracle_head,
                    oracle_block=oracle_block,
                    oracle_offset=oracle_offset,
                    agent_head=agent_head,
                    agent_block=agent_block,
                    agent_offset=agent_offset,
                    block_size=block_size,
                    tolerance=tolerance,
                )
                if divergence > tolerance:
                    console.print(
                        f"[bold red][AUTOGRAD TRAP] ❌ FAIL - HARD_BLOCK[/bold red]"
                    )
                    console.print(
                        f"[red]oracle={oracle}  divergence={divergence:.2e} > "
                        f"tol={tolerance:.0e}\n"
                        f"  Expected: head={oracle_head} block={oracle_block} offset={oracle_offset}\n"
                        f"  Got     : head={agent_head} block={agent_block} offset={agent_offset}[/red]"
                    )
                    if raise_on_fail:
                        raise HardBlockException(
                            oracle_name=oracle,
                            oracle_head=oracle_head,
                            oracle_block=oracle_block,
                            oracle_offset=oracle_offset,
                            agent_head=agent_head,
                            agent_block=agent_block,
                            agent_offset=agent_offset,
                            divergence=divergence,
                            tolerance=tolerance,
                        )
                else:
                    console.print(
                        f"[green][AUTOGRAD TRAP] ✅ PASS[/green]  "
                        f"oracle={oracle} divergence={divergence:.2e} ≤ "
                        f"tol={tolerance:.0e}"
                    )
                return (agent_head, agent_block, agent_offset)

            # 2-tuple (legacy 1D) path
            agent_block, agent_offset = int(agent_out[0]), int(agent_out[1])
            oracle_block, oracle_offset = int(oracle_out[0]), int(oracle_out[1])

            divergence = _run_autograd_check(
                oracle_block=oracle_block,
                oracle_offset=oracle_offset,
                agent_block=agent_block,
                agent_offset=agent_offset,
                block_size=block_size,
                tolerance=tolerance,
            )

            if divergence > tolerance:
                console.print(
                    f"[bold red][AUTOGRAD TRAP] ❌ FAIL - HARD_BLOCK[/bold red]"
                )
                console.print(
                    f"[red]oracle={oracle}  divergence={divergence:.2e} > "
                    f"tol={tolerance:.0e}\n"
                    f"  Expected: block={oracle_block} offset={oracle_offset}\n"
                    f"  Got     : block={agent_block} offset={agent_offset}[/red]"
                )
                if raise_on_fail:
                    raise HardBlockException(
                        oracle_name=oracle,
                        oracle_block=oracle_block,
                        oracle_offset=oracle_offset,
                        agent_block=agent_block,
                        agent_offset=agent_offset,
                        divergence=divergence,
                        tolerance=tolerance,
                    )
            else:
                console.print(
                    f"[green][AUTOGRAD TRAP] ✅ PASS[/green]  "
                    f"oracle={oracle} divergence={divergence:.2e} ≤ "
                    f"tol={tolerance:.0e}"
                )
            return (agent_block, agent_offset)

        _wrapped.__impactarbiter_oracle__ = oracle  # type: ignore[attr-defined]
        return _wrapped

    return _decorator


__all__ = ["verify", "HardBlockException"]
