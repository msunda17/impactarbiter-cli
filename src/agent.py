"""
ImpactArbiter — Auto-Heal Orchestrator
======================================

Drives the LLM generation ↔ Autograd-Trap feedback loop.

Provider abstraction: we use `litellm` so any of OpenAI / Anthropic / Vertex
/ Bedrock / local Ollama can be selected by passing the corresponding
`model` string (e.g. ``gpt-4o``, ``anthropic/claude-3-5-sonnet-20241022``,
``vertex_ai/gemini-1.5-pro``).
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .oracle import get_oracle
from .trap import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_WINDOW_TOKENS,
    DEFAULT_TOKEN_TEST_CASES,
    CaseResult,
    TrapReport,
    run_trap,
)

TARGET_FN_NAME = "ring_buffer_paged_mapping"

SYSTEM_PROMPT = (
    "You are a Senior ML Infrastructure Engineer specialising in CUDA, "
    "PyTorch and KV-cache memory management for vLLM-style PagedAttention. "
    "Your output MUST be ONLY a single raw Python function definition — "
    "no markdown fences, no imports, no commentary, no examples."
)

INITIAL_TASK = (
    f"Implement the function "
    f"`{TARGET_FN_NAME}(token_idx, max_window_tokens, block_size)` "
    "for a 1D Ring-Buffer Sliding Window KV-Cache. It must return the tuple "
    "(logical_block, offset) using strict integer floor-division and modulo "
    "arithmetic. IMPORTANT: Your function MUST have a `return` statement at the end. "
    "Do not include any imports, docstrings, comments, examples, tests or markdown — "
    "output the complete function definition including the return statement."
)


# ─────────────────────────────────────────────────────────────────────────────
# Code extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_python_code(text: str) -> str:
    """Strip markdown fences if the LLM ignored instructions."""
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    code = (m.group(1) if m else text).strip()
    return code


def is_code_complete(code: str) -> bool:
    """
    Heuristic check: code should end with a return statement and have balanced parentheses.
    """
    # Check for return statement
    if "return" not in code:
        return False
    # Check for balanced parentheses
    if code.count("(") != code.count(")"):
        return False
    # Check for balanced brackets
    if code.count("[") != code.count("]"):
        return False
    # Check for balanced braces
    if code.count("{") != code.count("}"):
        return False
    return True


def load_function_from_code(code: str, fn_name: str = TARGET_FN_NAME) -> Callable:
    """Write `code` to a temp module, import it, return `fn_name` callable."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="arbiter_kernel_", delete=False
    ) as tmp:
        tmp.write(code)
        path = tmp.name
    try:
        spec = importlib.util.spec_from_file_location("arbiter_kernel", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["arbiter_kernel"] = module
        spec.loader.exec_module(module)
        if not hasattr(module, fn_name):
            raise AttributeError(
                f"Generated code does not define `{fn_name}`."
            )
        return getattr(module, fn_name)
    finally:
        # we keep the file around for the trace; only remove on success-path
        # in caller if desired. For determinism, delete now.
        try:
            os.unlink(path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# LLM client (litellm)
# ─────────────────────────────────────────────────────────────────────────────


def build_litellm_client(model: str) -> Callable[[List[dict]], str]:
    """
    Returns a `generate(messages) -> str` closure backed by litellm.
    Falls back to a clear error if litellm isn't installed.
    """
    try:
        import litellm  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "litellm is required for the auto-heal loop. "
            "Install with: pip install litellm"
        ) from e

    def _generate(messages: List[dict]) -> str:
        resp = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=2048,  # Increased to avoid truncation
        )
        return resp["choices"][0]["message"]["content"]

    return _generate


# ─────────────────────────────────────────────────────────────────────────────
# Feedback payload (exactly as specified)
# ─────────────────────────────────────────────────────────────────────────────


def build_feedback_payload(case: CaseResult, divergence_map: str) -> str:
    """
    Constructs the structured CRITICAL FAILURE payload required by the spec.
    """
    return (
        f"CRITICAL FAILURE: Your implementation caused a gradient divergence "
        f"of {case.divergence:.4e} at token_idx [{case.token_idx}]. "
        f"Oracle expected (logical_block: {case.oracle_lb}, "
        f"offset: {case.oracle_off}) but your code routed to "
        f"(logical_block: {case.agent_lb}, offset: {case.agent_off}). "
        f"The KV_cache.grad indicates a memory offset error. Refactor to "
        f"ensure block-level alignment before wrapping.\n\n"
        f"Full divergence map:\n{divergence_map}\n\n"
        f"Output ONLY the corrected `{TARGET_FN_NAME}` function, no markdown."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Auto-heal loop
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HealResult:
    passed: bool
    attempts: int
    initial_code: str
    final_code: str
    initial_report: TrapReport
    final_report: TrapReport
    history: List[dict] = field(default_factory=list)


def auto_heal(
    *,
    model: str,
    max_retries: int = 3,
    max_window_tokens: int = DEFAULT_MAX_WINDOW_TOKENS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    token_test_cases: Tuple[int, ...] = DEFAULT_TOKEN_TEST_CASES,
    on_event: Optional[Callable[[str, dict], None]] = None,
    generate_fn: Optional[Callable[[List[dict]], str]] = None,
) -> HealResult:
    """
    Runs: generate → trap → (on hard-block) feedback → regenerate, until
    PASS or `max_retries` exhausted.

    `on_event(name, payload)` is the streaming hook used by `cli.py` to
    print the demo flow in real time.
    """
    emit = on_event or (lambda *_: None)
    generate = generate_fn or build_litellm_client(model)
    oracle = get_oracle("ring_buffer_v1")

    history: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INITIAL_TASK},
    ]

    initial_code: Optional[str] = None
    initial_report: Optional[TrapReport] = None
    last_code: str = ""
    last_report: Optional[TrapReport] = None

    for attempt in range(1, max_retries + 1):
        emit("attempt_start", {"attempt": attempt, "max": max_retries})

        raw = generate(history)
        code = extract_python_code(raw)
        history.append({"role": "assistant", "content": raw})
        emit("generation", {"attempt": attempt, "code": code})

        try:
            target_fn = load_function_from_code(code)
        except Exception as e:  # noqa: BLE001
            obs = (
                f"CRITICAL FAILURE: Your code failed to load: "
                f"{type(e).__name__}: {e}. Output ONLY a valid "
                f"`{TARGET_FN_NAME}` function, no markdown."
            )
            history.append({"role": "user", "content": obs})
            if initial_code is None:
                initial_code = code
            last_code = code
            emit("load_error", {"attempt": attempt, "error": str(e)})
            continue

        report = run_trap(
            target_fn,
            max_window_tokens=max_window_tokens,
            block_size=block_size,
            token_test_cases=token_test_cases,
            oracle=oracle,
        )

        if initial_code is None:
            initial_code = code
            initial_report = report
        last_code = code
        last_report = report

        emit("trap_report", {"attempt": attempt, "report": report})

        if report.passed:
            return HealResult(
                passed=True,
                attempts=attempt,
                initial_code=initial_code,
                final_code=code,
                initial_report=initial_report or report,
                final_report=report,
                history=history,
            )

        # Hard block ⇒ feedback payload, loop.
        case = report.first_hard_block or report.cases[-1]
        payload = build_feedback_payload(case, report.divergence_map())
        history.append({"role": "user", "content": payload})
        emit("feedback", {"attempt": attempt, "payload": payload, "case": case})

    return HealResult(
        passed=False,
        attempts=max_retries,
        initial_code=initial_code or "",
        final_code=last_code,
        initial_report=initial_report or last_report,  # type: ignore[arg-type]
        final_report=last_report,  # type: ignore[arg-type]
        history=history,
    )


__all__ = [
    "TARGET_FN_NAME",
    "SYSTEM_PROMPT",
    "INITIAL_TASK",
    "HealResult",
    "auto_heal",
    "build_feedback_payload",
    "extract_python_code",
    "load_function_from_code",
    "build_litellm_client",
]
