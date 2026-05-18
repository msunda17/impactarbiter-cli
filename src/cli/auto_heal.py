"""impactarbiter auto-heal — paper → distill → code → test → autograd trap → heal."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import tempfile
from typing import Callable, List, Optional, Tuple

import click
from rich.console import Console
from rich.rule import Rule

from ..db import db_manager as db
from ..fuzzer import PAGED_ADVERSARIAL_TOKENS, RADIX_TEST_MATRIX, fuzz_paged, fuzz_radix
from ..oracles import paged_oracle, radix_oracle
from ..trap import run_paged_trap, run_radix_trap
from .agent import (
    build_llm_client,
    extract_python_code,
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
BLOCK_SIZE = 16
DEFAULT_HEAD_IDX = 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _print_block(title: str, body: str = "") -> None:
    console.print(Rule(f"[bold]{title}"))
    if body:
        console.print(body)


def _print_legacy_banner() -> None:
    """Print a banner when running in legacy PagedAttention mode."""
    console.print(Rule("[bold yellow]LEGACY MODE - PagedAttention"))
    console.print(
        "[yellow]Frontier models have largely memorized the simple 1D PagedAttention routing math.[/yellow]"
    )
    console.print(
        "[yellow]The autograd trap now passes cleanly in all cases. This mode is kept for historical comparison only.[/yellow]"
    )
    console.print(
        "[yellow]For the current production-relevant demo, run without --oracle (uses Radix by default).[/yellow]"
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
    *, generate: Callable[[List[dict]], str], max_retries: int, model: str
) -> int:
    console.print(Rule("[bold cyan]IMPACT ARBITER — AUTO-HEAL"))
    console.print(f"[bold]Model:[/bold] {model}")
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
        db.update_heal(trace_id, healed_code=code, heal_success=True)
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
    for attempt in range(1, max_retries + 1):
        _print_block(f"[AUTO-HEAL attempt {attempt}/{max_retries}]")
        payload = build_heal_payload(
            expected_block=int(first_div.get("oracle_block", 0)),
            expected_offset=int(first_div.get("oracle_offset", 0)),
            agent_block=int(first_div.get("agent_block", -1)),
            agent_offset=int(first_div.get("agent_offset", -1)),
            token_label=f"token_idx={first_div.get('token')}",
            fn_name=PAGED_FN_NAME,
        )
        heal_history.append({"role": "user", "content": payload})
        raw = generate(heal_history)
        heal_history.append({"role": "assistant", "content": raw})
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

        if not healed_trap.diverged:
            heal_ok = True
            _print_block(
                "[FINAL PASS ✅]",
                f"divergence={healed_trap.divergence_value:.2e} (after {attempt} heal attempts)",
            )
            break
        first_div = _first_divergent(healed_trap.per_token) or first_div
        console.print(
            f"[yellow]heal attempt {attempt} still diverges "
            f"({healed_trap.divergence_value:.2e})[/yellow]"
        )

    db.update_heal(trace_id, healed_code=healed_code, heal_success=heal_ok)

    if not heal_ok:
        _print_block(
            "[FINAL FAIL ❌]",
            f"Could not heal after {max_retries} attempts.",
        )
        return 1
    return 0


def _run_radix_pipeline(
    *, generate: Callable[[List[dict]], str], max_retries: int, model: str
) -> int:
    console.print(Rule("[bold cyan]IMPACT ARBITER — AUTO-HEAL"))
    console.print(f"[bold]Model:[/bold] {model}")
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
        db.update_heal(trace_id, healed_code=code, heal_success=True)
        return 0

    _print_block(
        "[AUTOGRAD TRAP FAIL ❌ HARD_BLOCK]",
        f"divergence={trap.divergence_value:.2e} > tol={trap.tolerance:.0e}\n\n"
        f"{trap.divergence_map}",
    )

    healed_code = code
    heal_ok = False
    heal_history = refactor_history.copy()  # Start with refactor context
    for attempt in range(1, max_retries + 1):
        _print_block(f"[AUTO-HEAL attempt {attempt}/{max_retries}]")
        payload = build_heal_payload(
            expected_block=int(first_div.get("oracle_block", 0)),
            expected_offset=int(first_div.get("oracle_offset", 0)),
            agent_block=int(first_div.get("agent_block", -1)),
            agent_offset=int(first_div.get("agent_offset", -1)),
            token_label=str(first_div.get("token", "boundary case")),
            fn_name=RADIX_FN_NAME,
        )
        heal_history.append({"role": "user", "content": payload})
        raw = generate(heal_history)
        heal_history.append({"role": "assistant", "content": raw})
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

        if not healed_trap.diverged:
            heal_ok = True
            _print_block(
                "[FINAL PASS ✅]",
                f"divergence={healed_trap.divergence_value:.2e} (after {attempt} heal attempts)",
            )
            break
        first_div = _first_divergent(healed_trap.per_token) or first_div
        console.print(
            f"[yellow]heal attempt {attempt} still diverges "
            f"({healed_trap.divergence_value:.2e})[/yellow]"
        )

    db.update_heal(trace_id, healed_code=healed_code, heal_success=heal_ok)

    if not heal_ok:
        _print_block(
            "[FINAL FAIL ❌]",
            f"Could not heal after {max_retries} attempts.",
        )
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Click CLI
# ─────────────────────────────────────────────────────────────────────────────
@click.group()
def cli() -> None:
    """ImpactArbiter — semantic verification harness."""


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
    type=click.Choice(["radix", "vllm", "paged", "legacy"], case_sensitive=False),
    default="radix",
    show_default=True,
    help="Which oracle to validate against (radix | vllm | paged | legacy). "
         "Default: radix (recommended). vllm/paged/legacy are historical comparison modes.",
)
@click.option(
    "--max-retries",
    default=3,
    show_default=True,
    type=int,
    help="Maximum auto-heal attempts after the initial generation.",
)
def auto_heal_cmd(
    model: str,
    oracle_kind: str,
    max_retries: int,
) -> None:
    """Run paper → distill → code → test → autograd-trap → heal.

    Default oracle is radix (recommended for production-relevant demo).
    Use --oracle vllm/paged/legacy for historical comparison mode.
    """
    try:
        generate = build_llm_client(model)
    except ImportError as e:
        click.secho(f"❌ {e}", fg="red", bold=True)
        sys.exit(2)

    # Normalize legacy aliases (paged, legacy -> vllm)
    if oracle_kind in ("paged", "legacy"):
        oracle_kind = "vllm"
        _print_legacy_banner()

    try:
        if oracle_kind == "vllm":
            rc = _run_paged_pipeline(
                generate=generate,
                max_retries=max_retries,
                model=model,
            )
        else:
            rc = _run_radix_pipeline(
                generate=generate,
                max_retries=max_retries,
                model=model,
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


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
