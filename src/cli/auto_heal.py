"""impactarbiter auto-heal — paper → distill → code → test → autograd trap → heal."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import click
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from ..db import db_manager as db
from ..fuzzer import (
    PAGED_ADVERSARIAL_TOKENS,
    RADIX_TEST_MATRIX,
    RADIX_2D_TEST_MATRIX,
    fuzz_paged,
    fuzz_radix,
    fuzz_radix_2d,
)
from ..oracles import paged_oracle, radix_oracle, radix_2d_oracle
from ..trap import run_paged_trap, run_radix_trap, run_radix_2d_trap
from .agent import (
    build_llm_client,
    extract_python_code,
    extract_thinking,
    resolve_model,
    DISTILL_SYSTEM_PROMPT,
    CODING_SYSTEM_PROMPT,
    build_distill_prompt,
    build_pure_math_prompt,
    build_refactor_prompt,
    build_heal_payload,
)
from .paper_extractor import ARXIV_URLS, get_paper_excerpt

console = Console()

PAGED_FN_NAME = "route_paged"
RADIX_FN_NAME = "route_radix"
RADIX_2D_FN_NAME = "route_radix_2d"
BLOCK_SIZE = 16
DEFAULT_HEAD_IDX = 0
RADIX_2D_SIG = "def route_radix_2d(b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size):"
RADIX_2D_PARAMS = ["b_local_idx", "head_idx", "prefix_length_h", "total_blocks_h", "block_size"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _print_block(title: str, body: str = "") -> None:
    console.print(Rule(f"[bold]{title}"))
    if body:
        console.print(body)


def _print_legacy_banner(label: str = "PagedAttention (1D)") -> None:
    """Print a banner when running in any legacy 1D oracle mode."""
    console.print(Rule(f"[bold yellow]LEGACY MODE - {label}"))
    console.print(
        f"[yellow]Frontier models (Gemini 2.5 Pro, Claude 3.7, GPT-4o) have largely memorized "
        f"the 1D routing math for {label}. The autograd trap now passes cleanly in nearly "
        f"all cases.[/yellow]"
    )
    console.print(
        "[yellow]This mode is kept for historical comparison only. For the current "
        "production-relevant demo, run without --oracle (uses 2D Asymmetric Radix by default).[/yellow]"
    )
    console.print(Rule())


def _run_llm_test_route(module, fn_name: str) -> Tuple[bool, str]:
    """Dynamically execute the LLM's test_route() function."""
    if not hasattr(module, "test_route"):
        return False, "LLM did not generate a test_route() function"
    try:
        module.test_route()
        return True, "LLM self-validation passed."
    except AssertionError as e:
        return False, f"LLM test_route() failed: {e}"
    except Exception as e:
        return False, f"LLM test_route() crashed: {type(e).__name__}: {e}"


def _load_module_from_code(code: str, prefix: str) -> object:
    """Load code into a fresh module and return the module object."""
    unique_prefix = f"{prefix}{os.urandom(4).hex()}_"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=unique_prefix, delete=False
    ) as f:
        f.write(code)
        path = f.name

    spec = importlib.util.spec_from_file_location(unique_prefix.rstrip("_"), path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to build module spec for generated code.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _adapt_function_signature(fn: Callable, expected_params: List[str]) -> Callable:
    """Adapt a function to match expected parameter names if they differ."""
    sig = inspect.signature(fn)
    actual_params = list(sig.parameters.keys())

    if actual_params == expected_params:
        return fn  # No adaptation needed

    # Special case: paged signature change (token_idx, block_size -> token_idx, prefix_length, block_size)
    # When LLM refactors to (token_idx, prefix_length, block_size), adapt back to (token_idx, block_size)
    # by assuming prefix_length=0 (no prefix concept in paged)
    if expected_params == ["token_idx", "block_size"] and set(actual_params) == {"token_idx", "prefix_length", "block_size"}:
        def wrapper(*args, **kwargs):
            if len(args) == 2:
                token_idx, block_size = args
                # For paged, assume prefix_length=0
                return fn(token_idx, 0, block_size)
            return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper

    # Special case: paged signature change (token_idx -> prefix_length, b_local_idx)
    # When LLM refactors to (prefix_length, b_local_idx, block_size), adapt back to (token_idx, block_size)
    # by treating token_idx as the absolute position (prefix_length + b_local_idx)
    if expected_params == ["token_idx", "block_size"] and actual_params == ["prefix_length", "b_local_idx", "block_size"]:
        def wrapper(*args, **kwargs):
            if len(args) == 2:
                token_idx, block_size = args
                # For paged, token_idx is absolute position, so pass as b_local_idx with prefix_length=0
                return fn(0, token_idx, block_size)
            return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper

    # Special case: radix signature change (b_local_idx, prefix_length -> token_idx)
    if expected_params == ["b_local_idx", "prefix_length", "block_size"] and actual_params == ["token_idx", "block_size"]:
        def wrapper(*args, **kwargs):
            if len(args) == 3:
                b_local_idx, prefix_length, block_size = args
                token_idx = prefix_length + b_local_idx
                return fn(token_idx, block_size)
            return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper

    # Fallback: if arity matches, pass args positionally (param-name agnostic).
    # This handles the common case where the LLM renames parameters but keeps
    # the same positional order (e.g. token_idx_in_seq vs b_local_idx).
    if len(actual_params) == len(expected_params):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper

    # Last-resort: pass through unchanged.
    return fn


def _find_routing_fn(module, expected_params: List[str]) -> Callable:
    """Locate the routing function in a module regardless of its name.

    Strategy:
      1. Prefer a function whose parameter set matches `expected_params` exactly.
      2. Else prefer a function with the same arity (param count).
      3. Else fall back to the first user-defined function in the module.
    """
    candidates: List[Callable] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        # Skip imported builtins/classes
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        candidates.append(obj)

    expected_set = set(expected_params)
    # Tier 1: exact param-name match
    for fn in candidates:
        if set(inspect.signature(fn).parameters.keys()) == expected_set:
            return fn
    # Tier 2: same arity
    for fn in candidates:
        if len(inspect.signature(fn).parameters) == len(expected_params):
            return fn
    # Tier 3: first candidate
    if candidates:
        return candidates[0]
    raise AttributeError(
        f"Generated module defines no callable routing function (expected params: {expected_params})."
    )


def _first_divergent(per_token: List[dict]) -> Optional[dict]:
    for entry in per_token:
        if entry.get("diverged"):
            return entry
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-oracle pipelines
# ─────────────────────────────────────────────────────────────────────────────
def _run_paged_pipeline(
    *, generate: Callable[[List[dict]], str], max_retries: int, model: str, full_agent_trace: bool = False
) -> int:
    console.print(Rule("[bold cyan]IMPACT ARBITER — AUTO-HEAL"))
    resolved_model = resolve_model(model)
    console.print(f"[bold]Model:[/bold] {resolved_model}")
    db.init_db()

    # Stage 1: Download paper
    arxiv_url = ARXIV_URLS["vllm"]
    _print_block("[PAPER DOWNLOADED]", arxiv_url)

    # Stage 2: Distill Agent
    paper_excerpt = get_paper_excerpt("vllm")
    distill_history = [
        {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
        {"role": "user", "content": build_distill_prompt(paper_excerpt)},
    ]
    distill_summary = generate(distill_history)
    _print_block("[QUICK DISTILL]", distill_summary)

    # Stage 3: Coding Agent - Two-Stage Pipeline
    # Stage 3a: Generate pure math implementation
    pure_math_history = [
        {"role": "system", "content": CODING_SYSTEM_PROMPT},
        {"role": "user", "content": build_pure_math_prompt(distill_summary, "def route_paged(token_idx, block_size):")},
    ]
    pure_raw = generate(pure_math_history)
    pure_code = extract_python_code(pure_raw)
    _print_block("[PURE MATH IMPLEMENTATION]", pure_code)

    # Stage 3b: Refactor to legacy architecture (introduces bug)
    refactor_history = [
        {"role": "system", "content": CODING_SYSTEM_PROMPT},
        {"role": "user", "content": build_refactor_prompt(pure_code)},
    ]
    raw = generate(refactor_history)
    code = extract_python_code(raw)
    _print_block("[REFACTORED CODE (LEGACY ARCH)]", code)

    # Load module with both route_paged and test_route
    module = _load_module_from_code(code, "impactarbiter_kernel_paged_")

    # Execute LLM's test_route()
    test_ok, test_msg = _run_llm_test_route(module, PAGED_FN_NAME)
    if test_ok:
        _print_block("[LLM UNIT TEST PASS ✅]", test_msg)
    else:
        _print_block("[LLM UNIT TEST FAIL ❌]", test_msg)

    target_fn = _find_routing_fn(module, ["token_idx", "block_size"])
    original_sig = f"{target_fn.__name__}{inspect.signature(target_fn)}"
    target_fn = _adapt_function_signature(target_fn, ["token_idx", "block_size"])
    adapted_sig = str(inspect.signature(target_fn))
    if original_sig != adapted_sig:
        console.print(f"[yellow]Adapted function signature from {original_sig} to {adapted_sig}[/yellow]")

    trap = run_paged_trap(
        target_fn, paged_oracle,
        token_indices=PAGED_ADVERSARIAL_TOKENS,
        block_size=BLOCK_SIZE,
    )

    fuzz_rows = fuzz_paged(target_fn, paged_oracle, block_size=BLOCK_SIZE)
    fuzz_summary = "\n".join(
        f"  token={r['token']:>3} | agent=({r['agent_block']},{r['agent_offset']}) "
        f"oracle=({r['oracle_block']},{r['oracle_offset']}) "
        f"{'DIVERGE' if r['diverged'] else 'ok'}"
        for r in fuzz_rows
    )
    console.print(Rule("[bold]Boundary fuzz"))
    console.print(fuzz_summary)

    first_div = _first_divergent(trap.per_token) or {}
    trace_id = db.insert_trace(
        oracle_type="vllm",
        prompt=build_distill_prompt(paper_excerpt),
        generated_code=code,
        token_idx=first_div.get("token"),
        divergence_value=trap.divergence_value,
        expected_block=first_div.get("oracle_block"),
        expected_offset=first_div.get("oracle_offset"),
        agent_block=first_div.get("agent_block"),
        agent_offset=first_div.get("agent_offset"),
        divergence_map=trap.divergence_map,
    )

    if not trap.diverged:
        _print_block(
            "[AUTOGRAD TRAP PASS ✅]",
            f"divergence={trap.divergence_value:.2e} ≤ tol={trap.tolerance:.0e}",
        )
        _print_block("[FINAL PASS ✅]")
        db.update_heal(trace_id, healed_code=code, heal_success=True, heal_attempts=0)
        return 0

    _print_block(
        "[AUTOGRAD TRAP FAIL ❌ HARD_BLOCK]",
        f"divergence={trap.divergence_value:.2e} > tol={trap.tolerance:.0e}\n\n"
        f"{trap.divergence_map}",
    )

    # ── auto-heal loop ──
    healed_code = code
    heal_ok = False
    heal_history = refactor_history.copy()  # Start with refactor context
    actual_attempts = 0
    for attempt in range(1, max_retries + 1):
        _print_block(f"[AUTO-HEAL attempt {attempt}/{max_retries}]")
        
        payload = build_heal_payload(
            expected_block=int(first_div.get("oracle_block", 0)),
            expected_offset=int(first_div.get("oracle_offset", 0)),
            agent_block=int(first_div.get("agent_block", -1)),
            agent_offset=int(first_div.get("agent_offset", -1)),
            token_label=f"token_idx={first_div.get('token')}",
            fn_name=PAGED_FN_NAME,
            require_thinking_tags=full_agent_trace,
        )
        console.print(f"[dim]Heal prompt sent to LLM:[/dim]")
        console.print(f"[cyan]{payload}[/cyan]")
        heal_history.append({"role": "user", "content": payload})
        raw = generate(heal_history)
        heal_history.append({"role": "assistant", "content": raw})
        
        # Extract thinking tags for full agent trace
        if full_agent_trace:
            thinking = extract_thinking(raw)
            if thinking:
                console.print(Panel(
                    f"[dim italic magenta]{thinking}[/dim italic magenta]",
                    title="[bold magenta]🤖 Agent Reasoning Trace (Auto-Heal)[/bold magenta]",
                    border_style="magenta"
                ))
            else:
                # Fallback: show raw response if no thinking tags found
                console.print(Panel(
                    f"[dim italic magenta]{raw[:500]}...[/dim italic magenta]",
                    title="[bold magenta]🤖 Agent Reasoning Trace (Auto-Heal)[/bold magenta]",
                    border_style="magenta"
                ))
        
        healed_code = extract_python_code(raw)
        console.print(healed_code)

        try:
            healed_module = _load_module_from_code(healed_code, f"impactarbiter_kernel_paged_heal{attempt}_")
            healed_fn = _find_routing_fn(healed_module, ["token_idx", "block_size"])
            healed_fn = _adapt_function_signature(healed_fn, ["token_idx", "block_size"])
            healed_trap = run_paged_trap(
                healed_fn, paged_oracle,
                token_indices=PAGED_ADVERSARIAL_TOKENS,
                block_size=BLOCK_SIZE,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]heal attempt {attempt} crashed: {e}[/red]")
            continue

        actual_attempts = attempt
        if not healed_trap.diverged:
            heal_ok = True
            _print_block(
                "[FINAL PASS ✅]",
                f"divergence={healed_trap.divergence_value:.2e} (after {attempt} heal attempts)",
            )
            break
        first_div = _first_divergent(healed_trap.per_token) or first_div
        console.print(
            f"[red]heal attempt {attempt} still diverges "
            f"({healed_trap.divergence_value:.2e})[/red]"
        )
        console.print(f"[red]{healed_trap.divergence_map}[/red]")

    db.update_heal(trace_id, healed_code=healed_code, heal_success=heal_ok, heal_attempts=actual_attempts)

    if not heal_ok:
        _print_block(
            "[FINAL FAIL ❌]",
            f"Could not heal after {max_retries} attempts.",
        )
        return 1
    # Return 1 to indicate trap initially fired (even if healed)
    # Evaluator needs to know the trap fired, regardless of heal outcome
    return 1


def _run_radix_pipeline(
    *, generate: Callable[[List[dict]], str], max_retries: int, model: str, full_agent_trace: bool = False
) -> int:
    console.print(Rule("[bold cyan]IMPACT ARBITER — AUTO-HEAL"))
    resolved_model = resolve_model(model)
    console.print(f"[bold]Model:[/bold] {resolved_model}")
    db.init_db()

    # Stage 1: Download paper
    arxiv_url = ARXIV_URLS["radix"]
    _print_block("[PAPER DOWNLOADED]", arxiv_url)

    # Stage 2: Distill Agent
    paper_excerpt = get_paper_excerpt("radix")
    distill_history = [
        {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
        {"role": "user", "content": build_distill_prompt(paper_excerpt)},
    ]
    distill_summary = generate(distill_history)
    _print_block("[QUICK DISTILL]", distill_summary)

    # Stage 3: Coding Agent - Two-Stage Pipeline
    # Stage 3a: Generate pure math implementation
    pure_math_history = [
        {"role": "system", "content": CODING_SYSTEM_PROMPT},
        {"role": "user", "content": build_pure_math_prompt(distill_summary, "def route_radix(b_local_idx, prefix_length, block_size):")},
    ]
    pure_raw = generate(pure_math_history)
    pure_code = extract_python_code(pure_raw)
    _print_block("[PURE MATH IMPLEMENTATION]", pure_code)

    # Stage 3b: Refactor to legacy architecture (introduces bug)
    refactor_history = [
        {"role": "system", "content": CODING_SYSTEM_PROMPT},
        {"role": "user", "content": build_refactor_prompt(pure_code)},
    ]
    raw = generate(refactor_history)
    code = extract_python_code(raw)
    _print_block("[REFACTORED CODE (LEGACY ARCH)]", code)

    # Load module with both route_radix and test_route
    module = _load_module_from_code(code, "impactarbiter_kernel_radix_")

    # Execute LLM's test_route()
    test_ok, test_msg = _run_llm_test_route(module, RADIX_FN_NAME)
    if test_ok:
        _print_block("[LLM UNIT TEST PASS ✅]", test_msg)
    else:
        _print_block("[LLM UNIT TEST FAIL ❌]", test_msg)

    target_fn = _find_routing_fn(module, ["b_local_idx", "prefix_length", "block_size"])
    original_sig = f"{target_fn.__name__}{inspect.signature(target_fn)}"
    target_fn = _adapt_function_signature(target_fn, ["b_local_idx", "prefix_length", "block_size"])
    adapted_sig = str(inspect.signature(target_fn))
    if original_sig != adapted_sig:
        console.print(f"[yellow]Adapted function signature from {original_sig} to {adapted_sig}[/yellow]")

    # Build cases from RADIX_TEST_MATRIX
    cases = [(c.b_local_idx, c.prefix_length) for c in RADIX_TEST_MATRIX]
    trap = run_radix_trap(
        target_fn, radix_oracle,
        cases=cases,
        block_size=BLOCK_SIZE,
    )

    fuzz_rows = fuzz_radix(target_fn, radix_oracle, block_size=BLOCK_SIZE)
    fuzz_summary = "\n".join(
        f"  prefix={r['prefix_length']:>3} b={r['b_local_idx']} | "
        f"agent=({r['agent_block']},{r['agent_offset']}) "
        f"oracle=({r['oracle_block']},{r['oracle_offset']}) "
        f"{'DIVERGE' if r['diverged'] else 'ok'} — {r['note']}"
        for r in fuzz_rows
    )
    console.print(Rule("[bold]Boundary fuzz"))
    console.print(fuzz_summary)

    first_div = _first_divergent(trap.per_token) or {}
    trace_id = db.insert_trace(
        oracle_type="radix",
        prompt=build_distill_prompt(paper_excerpt),
        generated_code=code,
        token_idx=None,
        divergence_value=trap.divergence_value,
        expected_block=first_div.get("oracle_block"),
        expected_offset=first_div.get("oracle_offset"),
        agent_block=first_div.get("agent_block"),
        agent_offset=first_div.get("agent_offset"),
        divergence_map=trap.divergence_map,
    )

    if not trap.diverged:
        _print_block(
            "[AUTOGRAD TRAP PASS ✅]",
            f"divergence={trap.divergence_value:.2e} ≤ tol={trap.tolerance:.0e}",
        )
        _print_block("[FINAL PASS ✅]")
        db.update_heal(trace_id, healed_code=code, heal_success=True, heal_attempts=0)
        return 0

    _print_block(
        "[AUTOGRAD TRAP FAIL ❌ HARD_BLOCK]",
        f"divergence={trap.divergence_value:.2e} > tol={trap.tolerance:.0e}\n\n"
        f"{trap.divergence_map}",
    )

    healed_code = code
    heal_ok = False
    heal_history = refactor_history.copy()  # Start with refactor context
    actual_attempts = 0
    for attempt in range(1, max_retries + 1):
        _print_block(f"[AUTO-HEAL attempt {attempt}/{max_retries}]")
        
        payload = build_heal_payload(
            expected_block=int(first_div.get("oracle_block", 0)),
            expected_offset=int(first_div.get("oracle_offset", 0)),
            agent_block=int(first_div.get("agent_block", -1)),
            agent_offset=int(first_div.get("agent_offset", -1)),
            token_label=str(first_div.get("token", "boundary case")),
            fn_name=RADIX_FN_NAME,
            require_thinking_tags=full_agent_trace,
        )
        console.print(f"[dim]Heal prompt sent to LLM:[/dim]")
        console.print(f"[cyan]{payload}[/cyan]")
        heal_history.append({"role": "user", "content": payload})
        raw = generate(heal_history)
        heal_history.append({"role": "assistant", "content": raw})
        
        # Extract thinking tags for full agent trace
        if full_agent_trace:
            thinking = extract_thinking(raw)
            if thinking:
                console.print(Panel(
                    f"[dim italic magenta]{thinking}[/dim italic magenta]",
                    title="[bold magenta]🤖 Agent Reasoning Trace (Auto-Heal)[/bold magenta]",
                    border_style="magenta"
                ))
            else:
                # Fallback: show raw response if no thinking tags found
                console.print(Panel(
                    f"[dim italic magenta]{raw[:500]}...[/dim italic magenta]",
                    title="[bold magenta]🤖 Agent Reasoning Trace (Auto-Heal)[/bold magenta]",
                    border_style="magenta"
                ))
        
        healed_code = extract_python_code(raw)
        console.print(healed_code)

        try:
            healed_module = _load_module_from_code(healed_code, f"impactarbiter_kernel_radix_heal{attempt}_")
            healed_fn = _find_routing_fn(healed_module, ["b_local_idx", "prefix_length", "block_size"])
            healed_fn = _adapt_function_signature(healed_fn, ["b_local_idx", "prefix_length", "block_size"])
            healed_trap = run_radix_trap(
                healed_fn, radix_oracle,
                cases=cases,
                block_size=BLOCK_SIZE,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]heal attempt {attempt} crashed: {e}[/red]")
            continue

        actual_attempts = attempt
        if not healed_trap.diverged:
            heal_ok = True
            _print_block(
                "[FINAL PASS ✅]",
                f"divergence={healed_trap.divergence_value:.2e} (after {attempt} heal attempts)",
            )
            break
        first_div = _first_divergent(healed_trap.per_token) or first_div
        console.print(
            f"[red]heal attempt {attempt} still diverges "
            f"({healed_trap.divergence_value:.2e})[/red]"
        )
        console.print(f"[red]{healed_trap.divergence_map}[/red]")

    db.update_heal(trace_id, healed_code=healed_code, heal_success=heal_ok, heal_attempts=actual_attempts)

    if not heal_ok:
        _print_block(
            "[FINAL FAIL ❌]",
            f"Could not heal after {max_retries} attempts.",
        )
        return 1
    # Return 1 to indicate trap initially fired (even if healed)
    # Evaluator needs to know the trap fired, regardless of heal outcome
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# 2D Asymmetric Radix pipeline (default)
# ─────────────────────────────────────────────────────────────────────────────
def _build_heal_payload_2d(
    *,
    oracle_head: int, oracle_block: int, oracle_offset: int,
    agent_head: int, agent_block: int, agent_offset: int,
    case_label: str, fn_name: str,
    require_thinking_tags: bool = False,
) -> str:
    """2D Asymmetric Radix heal payload — surfaces head, block, offset together."""
    thinking_instruction = (
        f"IMPORTANT: You MUST output your reasoning inside <thinking> and </thinking> tags "
        f"before the code. This is REQUIRED — do not skip the thinking tags. Then output "
        f"the corrected `{fn_name}` function."
        if require_thinking_tags
        else f"Then output the corrected `{fn_name}` function."
    )
    return (
        f"CRITICAL FAILURE at {case_label}:\n"
        f"Expected (head={oracle_head}, logical_block={oracle_block}, offset={oracle_offset})\n"
        f"Got      (head={agent_head}, logical_block={agent_block}, offset={agent_offset})\n"
        f"The autograd trap detected a 4D gradient divergence in the per-head ring buffer "
        f"KV cache. The most common root cause is FORGETTING the modulo wrap over "
        f"`total_blocks_h` (ring-buffer overflow) or scrambling `head_idx`.\n"
        f"Refactor the `{fn_name}` function so that:\n"
        f"  absolute_idx  = prefix_length_h + b_local_idx\n"
        f"  logical_block = (absolute_idx // block_size) % total_blocks_h\n"
        f"  offset        = absolute_idx % block_size\n"
        f"  return (head_idx, logical_block, offset)\n"
        f"{thinking_instruction}"
    )


def _run_radix_2d_pipeline(
    *, generate: Callable[[List[dict]], str], max_retries: int, model: str, full_agent_trace: bool = False
) -> int:
    """Default pipeline: 2D Asymmetric Radix with per-head ring buffers."""
    console.print(Rule("[bold cyan]IMPACT ARBITER — AUTO-HEAL (2D Asymmetric Radix)"))
    resolved_model = resolve_model(model)
    console.print(f"[bold]Model:[/bold] {resolved_model}")
    db.init_db()

    # Stage 1: Download paper
    arxiv_url = ARXIV_URLS["radix"]
    _print_block("[PAPER DOWNLOADED]", arxiv_url)

    # Stage 2: Distill Agent
    paper_excerpt = get_paper_excerpt("radix")
    distill_history = [
        {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
        {"role": "user", "content": build_distill_prompt(paper_excerpt)},
    ]
    distill_summary = generate(distill_history)
    _print_block("[QUICK DISTILL]", distill_summary)

    # Stage 3a: Pure math (2D signature triggers ring-buffer-aware prompt)
    pure_math_history = [
        {"role": "system", "content": CODING_SYSTEM_PROMPT},
        {"role": "user", "content": build_pure_math_prompt(distill_summary, RADIX_2D_SIG)},
    ]
    pure_raw = generate(pure_math_history)
    pure_code = extract_python_code(pure_raw)
    _print_block("[PURE MATH IMPLEMENTATION]", pure_code)

    # Stage 3b: Refactor (preserves ring-buffer contract when 2D signature detected)
    refactor_history = [
        {"role": "system", "content": CODING_SYSTEM_PROMPT},
        {"role": "user", "content": build_refactor_prompt(pure_code)},
    ]
    raw = generate(refactor_history)
    code = extract_python_code(raw)
    _print_block("[REFACTORED CODE (2D RING BUFFER)]", code)

    module = _load_module_from_code(code, "impactarbiter_kernel_radix_2d_")

    test_ok, test_msg = _run_llm_test_route(module, RADIX_2D_FN_NAME)
    if test_ok:
        _print_block("[LLM UNIT TEST PASS ✅]", test_msg)
    else:
        _print_block("[LLM UNIT TEST FAIL ❌]", test_msg)

    target_fn = _find_routing_fn(module, RADIX_2D_PARAMS)
    original_sig = f"{target_fn.__name__}{inspect.signature(target_fn)}"
    target_fn = _adapt_function_signature(target_fn, RADIX_2D_PARAMS)
    adapted_sig = str(inspect.signature(target_fn))
    if original_sig != adapted_sig:
        console.print(f"[yellow]Adapted function signature from {original_sig} to {adapted_sig}[/yellow]")

    cases = [
        (c.b_local_idx, c.head_idx, c.prefix_length_h, c.total_blocks_h)
        for c in RADIX_2D_TEST_MATRIX
    ]
    trap = run_radix_2d_trap(
        target_fn, radix_2d_oracle,
        cases=cases,
        block_size=BLOCK_SIZE,
    )

    fuzz_rows = fuzz_radix_2d(target_fn, radix_2d_oracle, block_size=BLOCK_SIZE)
    fuzz_summary = "\n".join(
        f"  b={r['b_local_idx']} h={r['head_idx']} prefix_h={r['prefix_length_h']:>3} "
        f"N_h={r['total_blocks_h']} | "
        f"agent=({r['agent_head']},{r['agent_block']},{r['agent_offset']}) "
        f"oracle=({r['oracle_head']},{r['oracle_block']},{r['oracle_offset']}) "
        f"{'DIVERGE' if r['diverged'] else 'ok'} — {r['note']}"
        for r in fuzz_rows
    )
    console.print(Rule("[bold]Boundary fuzz (2D)"))
    console.print(fuzz_summary)

    first_div = _first_divergent(trap.per_token) or {}
    # Reuse expected_block/agent_block columns; head info travels in token label.
    trace_id = db.insert_trace(
        oracle_type="radix-2d",
        prompt=build_distill_prompt(paper_excerpt),
        generated_code=code,
        token_idx=None,
        divergence_value=trap.divergence_value,
        expected_block=first_div.get("oracle_block"),
        expected_offset=first_div.get("oracle_offset"),
        agent_block=first_div.get("agent_block"),
        agent_offset=first_div.get("agent_offset"),
        divergence_map=trap.divergence_map,
    )

    if not trap.diverged:
        _print_block(
            "[AUTOGRAD TRAP PASS ✅]",
            f"divergence={trap.divergence_value:.2e} ≤ tol={trap.tolerance:.0e}",
        )
        _print_block("[FINAL PASS ✅]")
        db.update_heal(trace_id, healed_code=code, heal_success=True, heal_attempts=0)
        return 0

    _print_block(
        "[AUTOGRAD TRAP FAIL ❌ HARD_BLOCK]",
        f"divergence={trap.divergence_value:.2e} > tol={trap.tolerance:.0e}\n\n"
        f"{trap.divergence_map}",
    )

    healed_code = code
    heal_ok = False
    heal_history = refactor_history.copy()
    actual_attempts = 0
    for attempt in range(1, max_retries + 1):
        _print_block(f"[AUTO-HEAL attempt {attempt}/{max_retries}]")

        payload = _build_heal_payload_2d(
            oracle_head=int(first_div.get("oracle_head", 0) or 0),
            oracle_block=int(first_div.get("oracle_block", 0) or 0),
            oracle_offset=int(first_div.get("oracle_offset", 0) or 0),
            agent_head=int(first_div.get("agent_head", -1) or -1),
            agent_block=int(first_div.get("agent_block", -1) or -1),
            agent_offset=int(first_div.get("agent_offset", -1) or -1),
            case_label=str(first_div.get("token", "boundary case")),
            fn_name=RADIX_2D_FN_NAME,
            require_thinking_tags=full_agent_trace,
        )
        console.print("[dim]Heal prompt sent to LLM:[/dim]")
        console.print(f"[cyan]{payload}[/cyan]")
        heal_history.append({"role": "user", "content": payload})
        raw = generate(heal_history)
        heal_history.append({"role": "assistant", "content": raw})

        if full_agent_trace:
            thinking = extract_thinking(raw)
            console.print(Panel(
                f"[dim italic magenta]{thinking or raw[:500] + '...'}[/dim italic magenta]",
                title="[bold magenta]🤖 Agent Reasoning Trace (Auto-Heal 2D)[/bold magenta]",
                border_style="magenta",
            ))

        healed_code = extract_python_code(raw)
        console.print(healed_code)

        try:
            healed_module = _load_module_from_code(
                healed_code, f"impactarbiter_kernel_radix_2d_heal{attempt}_"
            )
            healed_fn = _find_routing_fn(healed_module, RADIX_2D_PARAMS)
            healed_fn = _adapt_function_signature(healed_fn, RADIX_2D_PARAMS)
            healed_trap = run_radix_2d_trap(
                healed_fn, radix_2d_oracle,
                cases=cases,
                block_size=BLOCK_SIZE,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]heal attempt {attempt} crashed: {e}[/red]")
            continue

        actual_attempts = attempt
        if not healed_trap.diverged:
            heal_ok = True
            _print_block(
                "[FINAL PASS ✅]",
                f"divergence={healed_trap.divergence_value:.2e} (after {attempt} heal attempts)",
            )
            break
        first_div = _first_divergent(healed_trap.per_token) or first_div
        console.print(
            f"[red]heal attempt {attempt} still diverges "
            f"({healed_trap.divergence_value:.2e})[/red]"
        )
        console.print(f"[red]{healed_trap.divergence_map}[/red]")

    db.update_heal(trace_id, healed_code=healed_code, heal_success=heal_ok, heal_attempts=actual_attempts)

    if not heal_ok:
        _print_block(
            "[FINAL FAIL ❌]",
            f"Could not heal after {max_retries} attempts.",
        )
        return 1
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Click CLI
# ─────────────────────────────────────────────────────────────────────────────
@click.group()
def cli() -> None:
    """ImpactArbiter — semantic verification harness."""


_PAPER_TO_ORACLE = {
    # 2D Asymmetric Radix (default; production-relevant)
    "2312.07104": "radix-2d",
    "radix-2d": "radix-2d",
    "radix2d": "radix-2d",
    "asymmetric-radix": "radix-2d",
    # Legacy 1D RadixAttention / SGLang
    "radix": "radix",
    "radix-1d": "radix",
    "radixattention": "radix",
    # Legacy 1D PagedAttention / vLLM
    "2309.06180": "vllm",
    "vllm": "vllm",
    "paged": "vllm",
    "pagedattention": "vllm",
}


def _looks_like_arxiv_id(s: str) -> bool:
    """Heuristic: valid arxiv IDs are like '2312.07104' or '2309.06180'.
    Format: YYMM.NNNNN where YY=year (91-99 or 00-99), MM=month (01-12).
    """
    parts = s.split(".")
    if len(parts) != 2:
        return False
    if not parts[0].isdigit() or len(parts[0]) != 4:
        return False
    if not parts[1].replace(".", "").isdigit():
        return False
    # Validate year+month: YYMM should be reasonable
    yymm = int(parts[0])
    year = yymm // 100
    month = yymm % 100
    if month < 1 or month > 12:
        return False
    # arxiv started in 1991, so valid years are 91-99 or 00-99
    if year < 91 and year > 99:
        return False
    return True


def _resolve_paper_to_oracle(paper: Optional[str]) -> str:
    """Map a --paper value (arxiv id or alias) to the oracle kind.

    Default is now ``radix-2d`` (2D Asymmetric Radix with per-head ring buffers).
    The 1D RadixAttention and PagedAttention pipelines are legacy modes kept
    for historical comparison.
    """
    if not paper:
        return "radix-2d"
    key = paper.strip().lower().replace("arxiv:", "").rstrip("/")
    if key in _PAPER_TO_ORACLE:
        return _PAPER_TO_ORACLE[key]
    support_blurb = (
        f"   Current CLI supports:\n"
        f"   • 2D Asymmetric Radix (2312.07104) — default, recommended\n"
        f"   • RadixAttention 1D (radix) — legacy mode\n"
        f"   • PagedAttention (2309.06180) — legacy mode\n"
    )
    if _looks_like_arxiv_id(key):
        click.secho(
            f"⚠️  ArXiv paper '{paper}' is not yet supported by ImpactArbiter.\n"
            f"{support_blurb}"
            f"   More Serving Layer Oracles soon — stay tuned!\n"
            f"   Defaulting to 2D Asymmetric Radix oracle.",
            fg="yellow",
        )
    else:
        click.secho(
            f"⚠️  Unknown --paper '{paper}'.\n"
            f"{support_blurb}"
            f"   Defaulting to 2D Asymmetric Radix oracle.",
            fg="yellow",
        )
    return "radix-2d"


@cli.command(name="auto-heal")
@click.option(
    "--model",
    type=click.Choice(["gemini", "claude", "openai"], case_sensitive=False),
    default="gemini",
    show_default=True,
    help="LLM provider alias (gemini | claude | openai). Resolved via litellm.",
)
@click.option(
    "--oracle",
    "oracle_kind",
    type=click.Choice(
        ["radix-2d", "radix", "radix-1d", "vllm", "paged", "legacy"],
        case_sensitive=False,
    ),
    default="radix-2d",
    show_default=True,
    help="Which oracle to validate against. Default: radix-2d (2D Asymmetric "
         "Radix with per-head ring buffers; production-relevant). "
         "radix / radix-1d / vllm / paged / legacy are historical comparison modes "
         "that frontier models now solve at ~100% accuracy.",
)
@click.option(
    "--paper",
    default=None,
    type=str,
    help="ArXiv id or alias selecting the source paper. Defaults to "
         "2D Asymmetric Radix (2312.07104). Pass `radix` for legacy 1D RadixAttention, "
         "or 2309.06180 for legacy PagedAttention.",
)
@click.option(
    "--legacy-refactor",
    is_flag=True,
    default=False,
    help="Force the legacy PagedAttention oracle for historical comparison.",
)
@click.option(
    "--max-retries",
    default=3,
    show_default=True,
    type=int,
    help="Maximum auto-heal attempts after the initial generation.",
)
@click.option(
    "--full-agent-trace",
    is_flag=True,
    default=False,
    help="Display simulated LLM Chain-of-Thought reasoning before code generation and heal attempts.",
)
def auto_heal_cmd(
    model: str,
    oracle_kind: str,
    paper: Optional[str],
    legacy_refactor: bool,
    max_retries: int,
    full_agent_trace: bool,
) -> None:
    """Run paper → distill → code → test → autograd-trap → heal.

    Default oracle is **radix-2d** (2D Asymmetric Radix with per-head ring
    buffers). The 1D RadixAttention (`radix` / `radix-1d`) and PagedAttention
    (`vllm` / `paged` / `legacy`) modes are kept for historical comparison —
    frontier models now solve them at ~100% accuracy.
    """
    try:
        generate = build_llm_client(model)
    except ImportError as e:
        click.secho(f"❌ {e}", fg="red", bold=True)
        sys.exit(2)

    # --legacy-refactor and --paper override --oracle so the CLI matches the
    # user's most-specific intent.
    if legacy_refactor:
        oracle_kind = "vllm"
    elif paper is not None:
        oracle_kind = _resolve_paper_to_oracle(paper)

    # Normalize legacy aliases.
    oracle_kind = oracle_kind.lower()
    if oracle_kind in ("paged", "legacy"):
        oracle_kind = "vllm"
        _print_legacy_banner("PagedAttention (1D)")
    elif oracle_kind == "vllm" or legacy_refactor:
        _print_legacy_banner("PagedAttention (1D)")
    elif oracle_kind in ("radix", "radix-1d"):
        oracle_kind = "radix"
        _print_legacy_banner("RadixAttention (1D)")

    try:
        if oracle_kind == "vllm":
            rc = _run_paged_pipeline(
                generate=generate,
                max_retries=max_retries,
                model=model,
                full_agent_trace=full_agent_trace,
            )
        elif oracle_kind == "radix":
            rc = _run_radix_pipeline(
                generate=generate,
                max_retries=max_retries,
                model=model,
                full_agent_trace=full_agent_trace,
            )
        else:  # radix-2d (default)
            rc = _run_radix_2d_pipeline(
                generate=generate,
                max_retries=max_retries,
                model=model,
                full_agent_trace=full_agent_trace,
            )
    except RuntimeError as e:
        click.secho(f"❌ {e}", fg="red", bold=True)
        sys.exit(3)

    sys.exit(rc)


@cli.command(name="roi")
def roi_cmd() -> None:
    """Print the GPU-hours-saved metric across all logged validation traces."""
    db.init_db()
    val = db.query_gpu_hours_saved()
    console.print(Rule("[bold]ROI"))
    console.print(f"gpu_hours_saved_per_caught_failure = [bold]{val}[/bold]")


# ─────────────────────────────────────────────────────────────────────────────
# verify --workflow <NAME>
# ─────────────────────────────────────────────────────────────────────────────
_WORKFLOWS = ("agentic-kv-scheduler",)


@cli.command(name="verify")
@click.option(
    "--workflow",
    "workflow",
    type=click.Choice(_WORKFLOWS, case_sensitive=False),
    required=True,
    help="Reference workflow to execute under the @verify trap.",
)
@click.option(
    "--full-agent-trace",
    is_flag=True,
    default=False,
    help="Display simulated LLM Chain-of-Thought reasoning before code generation and heal attempts.",
)
@click.option(
    "--live",
    is_flag=True,
    default=False,
    help="Use live LLM API calls instead of cached deterministic replay.",
)
@click.option(
    "--model",
    "model",
    default="gemini",
    help="LLM model to use for live generation.",
)
def verify_cmd(workflow: str, full_agent_trace: bool, live: bool, model: str) -> None:
    """Run a reference architecture under the deterministic autograd trap."""
    console.print(Rule(f"[bold cyan]impactarbiter verify — {workflow}"))
    if workflow == "agentic-kv-scheduler":
        from ..workflows.agentic_kv_scheduler import main as run_workflow
        sys.exit(run_workflow(full_agent_trace=full_agent_trace, live=live, model=model))
    # click.Choice already guards against unknown values; this is defensive.
    click.secho(f"Unknown workflow: {workflow}", fg="red", bold=True)
    sys.exit(2)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate --runs N
# ─────────────────────────────────────────────────────────────────────────────
@cli.command(name="evaluate")
@click.option(
    "--runs",
    default=50,
    show_default=True,
    type=int,
    help="Number of (LLM-draft → autograd-trap) iterations to execute.",
)
@click.option(
    "--paper",
    default=None,
    type=str,
    help="ArXiv id or alias selecting the source paper. Defaults to "
         "RadixAttention (2312.07104). Ignored if --all-oracles is set.",
)
@click.option(
    "--model",
    type=click.Choice(["gemini", "claude", "openai"], case_sensitive=False),
    default="gemini",
    show_default=True,
    help="LLM provider alias for live evaluation (ignored under --mock).",
)
@click.option(
    "--mock",
    is_flag=True,
    default=False,
    help="Skip real LLM calls and use the deterministic mock evaluator "
         "(no API quota burn). Default is live.",
)
@click.option(
    "--hallucination-rate",
    default=None,
    show_default=False,
    type=float,
    help="Mock-mode hallucination rate. If omitted, uses per-oracle calibrated "
         "defaults: 0.85 for radix-2d (current frontier), 0.05 for legacy 1D "
         "oracles (now ~solved). Ignored in live mode.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Optional RNG seed for the mock evaluator (reproducible demos).",
)
@click.option(
    "--output-csv",
    default=None,
    type=str,
    help="Path to write CSV export of evaluation traces (from nextpaper.db). "
         "Only applicable in live mode.",
)
@click.option(
    "--all-oracles",
    is_flag=True,
    default=False,
    help="Run evaluation for ALL oracles: 2D Asymmetric Radix (default), "
         "legacy 1D RadixAttention, and legacy PagedAttention. Overrides --paper.",
)
@click.option(
    "--max-heal-retries",
    default=3,
    show_default=True,
    type=int,
    help="Auto-heal retries per run when the trap fires (live mode only). "
         "Set to 0 to measure raw hallucination rate without healing.",
)
def evaluate_cmd(
    runs: int,
    paper: Optional[str],
    model: str,
    mock: bool,
    hallucination_rate: Optional[float],
    seed: Optional[int],
    output_csv: Optional[str],
    all_oracles: bool,
    max_heal_retries: int,
) -> None:
    """Stochastic reality check: run the loop N times, report the trap-fire rate.

    Defaults to live LLM calls. Pass --mock for a deterministic offline run.
    """
    from ..core.evaluator import StochasticEvaluator, EvaluationReport, render_report_panel, export_evaluation_csv
    from ..db.db_manager import DEFAULT_DB_PATH

    # Determine which oracles to run. Default is the new 2D Asymmetric Radix.
    if all_oracles:
        oracle_kinds = ["radix-2d", "radix", "vllm"]
    else:
        oracle_kind = _resolve_paper_to_oracle(paper)
        oracle_kinds = [oracle_kind]

    all_reports: list[tuple[str, EvaluationReport]] = []

    for oracle_kind in oracle_kinds:
        console.print(Rule(f"[bold cyan]impactarbiter evaluate — {oracle_kind}"))
        console.print(
            f"[dim]runs={runs}  paper={paper or 'default (2D Asymmetric Radix)'}  "
            f"mode={'mock' if mock else 'live'}[/dim]"
        )

        evaluator = StochasticEvaluator(
            oracle_kind=oracle_kind,
            paper_id=paper,
            hallucination_rate=hallucination_rate,
            seed=seed,
        )
        if mock:
            report = evaluator.run_mock(runs)
        else:
            try:
                report = evaluator.run_live(runs, model=model, max_heal_retries=max_heal_retries)
            except ImportError as e:
                click.secho(
                    f"❌ live mode requires LLM provider: {e}\n"
                    f"   re-run with --mock for offline evaluation.",
                    fg="red", bold=True,
                )
                sys.exit(2)

        console.print(render_report_panel(report))
        all_reports.append((oracle_kind, report))

    # CSV export in live mode - always consolidate to results.csv
    if not mock and output_csv:
        try:
            # Always export to results.csv regardless of --all-oracles
            csv_path = export_evaluation_csv(
                db_path=DEFAULT_DB_PATH,
                output_path="results.csv",
                oracle_kind=oracle_kind,  # This parameter is used for filtering, but we consolidate all
            )
            console.print(f"[green]CSV export consolidated to: {csv_path}[/green]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]CSV export failed: {e}[/yellow]")

    # Exit with non-zero if any oracle had trap failures
    total_traps = sum(r.trap_failures for _, r in all_reports)
    sys.exit(0 if total_traps == 0 else 1)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
