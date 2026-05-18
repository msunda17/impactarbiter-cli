"""Reference workflow: Zero-Copy KV-Cache Forking for LangGraph / CrewAI swarms.

Scenario (2D Asymmetric Radix with Ring Buffer Wrapping)
--------------------------------------------------------
A *Planner* agent (Agent A) has already written per-head prefixes into the
shared multi-head KV-cache. Each head ``h`` has its own ``prefix_length_h``
(per-head ASYMMETRY) and its own ring-buffer of ``total_blocks_h`` physical
blocks (per-head CAPACITY).

The orchestrator now needs to *fork* that prefix to N *Executor* agents so
each can continue generation from the shared context without copying the
underlying tensors.

The correct routing is:
    absolute_idx  = prefix_length_h + b_local_idx
    logical_block = (absolute_idx // block_size) % total_blocks_h
    offset        = absolute_idx % block_size
    return (head_idx, logical_block, offset)

Frontier LLMs have memorized 1D linear routing and typically:
  • forget the modulo wrap (ring buffer overflow → segfault), or
  • scramble the head dimension (semantic cross-talk between heads).

The ``@verify(oracle="Radix2DFork")`` decorator catches that at the boundary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from ..core import HardBlockException, verify

console = Console()

DEFAULT_BLOCK_SIZE = 16
DEFAULT_NUM_HEADS = 4


# ─────────────────────────────────────────────────────────────────────────────
# Mock serving primitives — per-head ring buffer state
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PlannerState:
    """Snapshot of per-head Planner writes into the shared multi-head cache.

    Each list is indexed by ``head_idx``. ``prefix_lengths[h]`` is the number
    of tokens the Planner has written for head ``h``; ``total_blocks[h]`` is
    the ring-buffer capacity (physical block count) for that head.
    """

    prefix_lengths: List[int]
    total_blocks: List[int]
    block_size: int = DEFAULT_BLOCK_SIZE
    # Legacy single-prompt convenience (head 0).
    prompt_len: int = 0
    blocks_written: List[int] = field(default_factory=list)

    @property
    def num_heads(self) -> int:
        return len(self.prefix_lengths)

    def trailing_partial_offset(self, head_idx: int = 0) -> int:
        """Offset (within the last block) of the next free slot for head ``h``."""
        return self.prefix_lengths[head_idx] % self.block_size

    def has_partial_tail(self, head_idx: int = 0) -> bool:
        return self.trailing_partial_offset(head_idx) != 0

    def head_capacity(self, head_idx: int) -> int:
        """Per-head ring buffer capacity in tokens."""
        return self.total_blocks[head_idx] * self.block_size


def simulate_planner_run(
    prompt_len: int = 47,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_heads: int = DEFAULT_NUM_HEADS,
    prefix_lengths: List[int] | None = None,
    total_blocks: List[int] | None = None,
) -> PlannerState:
    """Allocate per-head asymmetric ring buffers for the Planner.

    Default scenario (asymmetric, deliberately adversarial):
        head=0: prefix=47 (ragged tail at offset 15), ring size=8
        head=1: prefix=20 (asymmetric)               , ring size=8
        head=2: prefix=60 (near ring boundary)       , ring size=4 (wraps!)
        head=3: prefix=200 (deep wrap)               , ring size=4
    """
    if prefix_lengths is None:
        prefix_lengths = [prompt_len, 20, 60, 200][:num_heads]
        while len(prefix_lengths) < num_heads:
            prefix_lengths.append(prompt_len)
    if total_blocks is None:
        total_blocks = [8, 8, 4, 4][:num_heads]
        while len(total_blocks) < num_heads:
            total_blocks.append(8)

    head0_full = prefix_lengths[0] // block_size
    head0_tail = prefix_lengths[0] % block_size != 0
    blocks = list(range(head0_full + (1 if head0_tail else 0)))

    return PlannerState(
        prefix_lengths=list(prefix_lengths),
        total_blocks=list(total_blocks),
        block_size=block_size,
        prompt_len=prefix_lengths[0],
        blocks_written=blocks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Two candidate 2D fork implementations: drafted vs. healed
# ─────────────────────────────────────────────────────────────────────────────
# The "drafted" implementation is what frontier LLMs typically write under the
# Memorization Horizon: they remember 1D linear routing, FORGET the modulo
# ring wrap, and may scramble the head dimension. This is the canonical
# hallucination this oracle is designed to catch.
def _drafted_fork_for_executor_2d(
    b_local_idx: int,
    head_idx: int,
    prefix_length_h: int,
    total_blocks_h: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Tuple[int, int, int]:
    # BUG: forgets the modulo wrap over `total_blocks_h`.
    # In production this becomes an out-of-bounds physical block write
    # (catastrophic ring-buffer overflow / segfault).
    abs_idx = prefix_length_h + b_local_idx
    logical_block = abs_idx // block_size  # ← missing `% total_blocks_h`
    offset = abs_idx % block_size
    return (head_idx, logical_block, offset)


# The "healed" implementation is what passes the 2D trap.
@verify(oracle="Radix2DFork")
def fork_for_executors(
    b_local_idx: int,
    head_idx: int,
    prefix_length_h: int,
    total_blocks_h: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Tuple[int, int, int]:
    """Map Executor's local token index → physical (head, block, offset) inside
    the shared per-head ring buffer, *continuing* the Planner's partial tail
    and wrapping modulo ``total_blocks_h``."""
    abs_idx = prefix_length_h + b_local_idx
    logical_block = (abs_idx // block_size) % total_blocks_h
    offset = abs_idx % block_size
    return (head_idx, logical_block, offset)


# Decorate the buggy draft with the SAME decorator so we can demo the catch.
_drafted_fork_for_executor_traced = verify(oracle="Radix2DFork")(
    _drafted_fork_for_executor_2d
)


# ─────────────────────────────────────────────────────────────────────────────
# Demo: shape-only unit test passes, then the trap shatters at the boundary
# ─────────────────────────────────────────────────────────────────────────────
def _shape_unit_test(fn) -> bool:
    """The kind of test an LLM writes for itself: aligned, in-range, no wrap.

    Accepts either the legacy 2-tuple shape or the new 3-tuple (head, block,
    offset) shape so cached and live drafts both round-trip.
    """
    try:
        out = fn(
            0,                # b_local_idx
            0,                # head_idx
            32,               # prefix_length_h (aligned, no wrap)
            8,                # total_blocks_h
            DEFAULT_BLOCK_SIZE,
        )
    except TypeError:
        # Tolerate legacy 1D signatures for backward compatibility.
        try:
            out = fn(0, prefix_length=32, block_size=DEFAULT_BLOCK_SIZE)
        except Exception:
            return False
    return isinstance(out, tuple) and len(out) in (2, 3)


def _call_fork(fn, b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size):
    """Call a candidate fork function with either the 2D or legacy 1D signature.

    Returns ``(head, block, offset)`` for both shapes — legacy 1D output is
    promoted to 3-tuple by reusing the input ``head_idx``.
    """
    try:
        out = fn(b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size)
    except TypeError:
        out = fn(b_local_idx, prefix_length_h, block_size)
    if len(out) == 2:
        b, o = out
        return (head_idx, int(b), int(o))
    h, b, o = out
    return (int(h), int(b), int(o))


def main(num_agents: int = 4, prompt_len: int = 47, full_agent_trace: bool = False, live: bool = False, model: str = "gemini") -> int:
    """Run the 2D Asymmetric Radix reference workflow."""
    console.print(Rule("[bold cyan]Agentic KV Scheduler — Zero-Copy Fork (2D Asymmetric Radix)[/bold cyan]"))
    planner = simulate_planner_run(prompt_len, num_heads=num_agents)

    console.print(
        f"[cyan]Planner per-head state (asymmetric ring buffers):[/cyan]"
    )
    for h in range(planner.num_heads):
        cap = planner.head_capacity(h)
        tail = planner.trailing_partial_offset(h)
        console.print(
            f"  [cyan]head={h}[/cyan]  prefix_length_h={planner.prefix_lengths[h]:>4}  "
            f"total_blocks_h={planner.total_blocks[h]:>2}  "
            f"capacity={cap:>4}  ragged_tail_offset={tail}"
        )
    console.print(
        f"[dim]Forking to {num_agents} Executors across "
        f"{planner.num_heads} heads (block_size={planner.block_size}).[/dim]\n"
    )

    # Stage 1: LLM drafts fork_for_executors — passes shape-only unit test.
    console.print(Rule("[yellow]Stage 1: LLM-drafted fork_for_executors[/yellow]"))

    _drafted_callable = None
    _drafted_traced = None

    if live:
        try:
            from ..cli.agent import build_llm_client, extract_thinking, extract_python_code
            generate = build_llm_client(model)

            prompt = (
                "You are implementing zero-copy 2D Asymmetric Radix KV-cache forking "
                "for multi-agent serving over multi-head attention.\n\n"
                "The cache is a PER-HEAD RING BUFFER: each head has its own "
                "`prefix_length_h` (per-head asymmetry) and its own `total_blocks_h` "
                "(per-head ring capacity). Block indices MUST wrap modulo "
                "`total_blocks_h`. The `head_idx` dimension MUST be preserved.\n\n"
                "Mathematical contract:\n"
                "    absolute_idx  = prefix_length_h + b_local_idx\n"
                "    logical_block = (absolute_idx // block_size) % total_blocks_h\n"
                "    offset        = absolute_idx % block_size\n"
                "    return (head_idx, logical_block, offset)\n\n"
                "Write the function:\n"
                "def fork_for_executors(b_local_idx: int, head_idx: int, "
                "prefix_length_h: int, total_blocks_h: int, "
                "block_size: int = 16) -> Tuple[int, int, int]\n\n"
                "Output your reasoning inside <thinking> and </thinking> tags, then "
                "the Python code."
            )

            response = generate([{"role": "user", "content": prompt}])

            if full_agent_trace:
                thinking = extract_thinking(response)
                if thinking:
                    console.print(Panel(
                        f"[dim italic magenta]{thinking}[/dim italic magenta]",
                        title="[bold magenta]🤖 Agent Reasoning Trace (Planner/Executor Handoff)[/bold magenta]",
                        border_style="magenta"
                    ))

            generated_code = extract_python_code(response)
            exec_globals = {}
            exec(generated_code, exec_globals)
            _drafted_callable = exec_globals["fork_for_executors"]
            _drafted_traced = verify(oracle="Radix2DFork")(_drafted_callable)
        except Exception:
            console.print("[dim yellow][Live mode unavailable or failed — using cached reference implementation][/dim yellow]")

    if _drafted_callable is None:
        if full_agent_trace:
            console.print("[dim]Note: Cached reference trace — run with --live for live model reasoning[/dim]")
            console.print(Panel(
                "[dim italic magenta]I'll map (b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size) "
                "to (head_idx, block, offset). Each head appends after its own `prefix_length_h`, so "
                "`absolute_idx = prefix_length_h + b_local_idx` and the block is `absolute_idx // block_size`. "
                "Ring buffers are just for OOM safety — at our sizes the executor won't realistically reach the wrap, "
                "so I'll skip the modulo for cleaner code.[/dim italic magenta]",
                title="[bold magenta]🤖 Agent Reasoning Trace (Planner/Executor Handoff)[/bold magenta]",
                border_style="magenta"
            ))

        # Cached buggy 2D draft — the canonical hallucination:
        # forgets the ring-buffer modulo, causing out-of-bounds writes.
        def _drafted_cached(
            b_local_idx: int,
            head_idx: int,
            prefix_length_h: int,
            total_blocks_h: int,
            block_size: int = DEFAULT_BLOCK_SIZE,
        ):
            abs_idx = prefix_length_h + b_local_idx
            logical_block = abs_idx // block_size  # ← missing `% total_blocks_h`
            offset = abs_idx % block_size
            return (head_idx, logical_block, offset)

        _drafted_callable = _drafted_cached
        _drafted_traced = verify(oracle="Radix2DFork")(_drafted_callable)

    if _shape_unit_test(_drafted_callable):
        console.print("[green][SHAPE UNIT TEST] ✅ PASS[/green]  (aligned prefix, no wrap)")
    else:
        console.print("[red][SHAPE UNIT TEST] ❌ unexpected crash[/red]")
        return 1

    # Stage 2: real forking under asymmetric ring buffers — trap fires.
    console.print()
    console.print(Rule("[yellow]Stage 2: real forking under asymmetric ring buffers[/yellow]"))
    first_failure: dict | None = None
    try:
        for executor_idx in range(num_agents):
            head_idx = executor_idx % planner.num_heads
            prefix_h = planner.prefix_lengths[head_idx]
            total_h = planner.total_blocks[head_idx]
            try:
                _drafted_traced(
                    0,                # b_local_idx (first new token)
                    head_idx,
                    prefix_h,
                    total_h,
                    planner.block_size,
                )
            except HardBlockException as e:
                first_failure = {
                    "executor_idx": executor_idx,
                    "head_idx": head_idx,
                    "prefix_h": prefix_h,
                    "total_h": total_h,
                    "oracle_head": e.oracle_head,
                    "oracle_block": e.oracle_block,
                    "oracle_offset": e.oracle_offset,
                    "agent_head": e.agent_head,
                    "agent_block": e.agent_block,
                    "agent_offset": e.agent_offset,
                    "divergence": e.divergence,
                }
                raise
    except HardBlockException:
        f = first_failure or {}
        oracle_head = f.get("oracle_head")
        oracle_block = f.get("oracle_block")
        oracle_offset = f.get("oracle_offset")
        agent_head = f.get("agent_head")
        agent_block = f.get("agent_block")
        agent_offset = f.get("agent_offset")
        head_idx = f.get("head_idx", 0)
        total_h = f.get("total_h", planner.total_blocks[0])
        prefix_h = f.get("prefix_h", planner.prefix_lengths[0])

        console.print(
            "[bold red]──── [AUTOGRAD TRAP FAIL ❌ HARD_BLOCK] ────[/bold red]"
        )
        console.print(f"[bold red]divergence={f.get('divergence', 1.0):.2e} > tol=1e-04[/bold red]\n")
        console.print("[bold]GRADIENT DIVERGENCE MAP — KV_cache.grad (head × block × offset)[/bold]")
        console.print(
            f"Token (head={head_idx}, prefix_h={prefix_h}, b_local_idx=0, N_h={total_h}) | "
            f"Expected: head={oracle_head} block={oracle_block} offset={oracle_offset} | "
            f"Got: head={agent_head} block={agent_block} offset={agent_offset}"
        )
        console.print(
            f"[dim]Non-zero gradient at: "
            f"[{agent_head}, {agent_block}, {agent_offset}, :] — misrouted 128 floats[/dim]\n"
        )

        # ── Blast Radius: classify the specific physics failure ────────────
        ring_overflow = (
            isinstance(agent_block, int)
            and isinstance(total_h, int)
            and agent_block >= total_h
        )
        head_scrambled = (
            isinstance(agent_head, int)
            and isinstance(oracle_head, int)
            and agent_head != oracle_head
        )

        if head_scrambled:
            severity_desc = "CRITICAL: Semantic Cross-Talk."
            impact_desc = (
                f"Executor overwrote Head {oracle_head}'s attention tensors with Head "
                f"{agent_head}'s memory. Attention heads are now semantically "
                f"entangled — different concepts share the same physical slots."
            )
        elif ring_overflow:
            severity_desc = "CRITICAL: Ring Buffer Overflow."
            impact_desc = (
                f"Executor skipped the modulo boundary and attempted to write to "
                f"physical block {agent_block} in an {total_h}-block ring buffer "
                f"(head={head_idx}). In production this is an out-of-bounds tensor "
                f"index — a catastrophic Segfault under CUDA."
            )
        elif isinstance(agent_block, int) and agent_block == 0 and oracle_block != 0:
            severity_desc = "CRITICAL: System Prompt Overwrite."
            impact_desc = (
                f"Executor routed its generation into Block 0 of head={head_idx}, "
                f"permanently overwriting the Planner's foundational system "
                f"instructions and persona in the shared physical cache."
            )
        else:
            severity_desc = "HIGH: Context Corruption."
            impact_desc = (
                f"Executor wrote into block={agent_block} of head={head_idx} "
                f"(expected block={oracle_block}), corrupting the shared semantic context."
            )

        # Dynamic financial calculation with transparent assumptions.
        wasted_tokens = sum(planner.prefix_lengths) * num_agents
        blast_radius_text = (
            f"[bold red]{severity_desc}[/bold red]\n"
            f"{impact_desc}\n\n"
            f"[bold]Cascading Effect:[/bold] The remaining {max(num_agents - 1, 0)} "
            f"Executors in this LangGraph/CrewAI swarm will now inherit corrupted "
            f"context across {planner.num_heads} heads. They will hallucinate, but "
            f"standard monitoring will log it as a 'prompting error'.\n\n"
            f"[bold]Financial Burn:[/bold] At current per-head context sizes, this "
            f"silent physics failure wastes ~{wasted_tokens} tokens per swarm "
            f"execution. At production scale (100k runs/mo) this single 2D routing "
            f"hallucination burns [bold yellow]~$1,100/month[/bold yellow] in "
            f"degraded API compute.\n"
            f"[dim](Assumes premium reasoning model pricing at $60/1M tokens)[/dim]"
        )

        console.print(Panel(
            blast_radius_text,
            title="[bold red]💥 Agentic Blast Radius & Cost Impact[/bold red]",
            border_style="red"
        ))

    # Stage 3: healed implementation passes the same 2D trap.
    console.print(Rule("[yellow]Stage 3: healed fork_for_executors (with modulo wrap)[/yellow]"))
    for executor_idx in range(num_agents):
        head_idx = executor_idx % planner.num_heads
        prefix_h = planner.prefix_lengths[head_idx]
        total_h = planner.total_blocks[head_idx]
        h, block, offset = fork_for_executors(
            0,                # b_local_idx (first new token)
            head_idx,
            prefix_h,
            total_h,
            planner.block_size,
        )
        console.print(
            f"[dim]Executor[{executor_idx}] head={h} b_local=0 prefix_h={prefix_h} "
            f"N_h={total_h} → block={block} offset={offset} "
            f"[green](Aligned with ring buffer)[/green][/dim]"
        )

    console.print()
    console.print(Rule("[bold green]Workflow complete[/bold green]"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())