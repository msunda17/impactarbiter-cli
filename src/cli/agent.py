"""LLM agent abstraction (litellm) and code-extraction helpers."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
from typing import Callable, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Model-id resolution
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_ALIASES = {
    "gemini": os.environ.get("IMPACTARBITER_GEMINI_MODEL", "vertex_ai/gemini-2.5-pro"),
    "claude": os.environ.get("IMPACTARBITER_CLAUDE_MODEL", "anthropic/claude-3-5-sonnet-20241022"),
    "openai": os.environ.get("IMPACTARBITER_OPENAI_MODEL", "gpt-4o"),
}


def resolve_model(name: str) -> str:
    """Resolve a friendly alias to a litellm model id."""
    if name in _MODEL_ALIASES:
        return _MODEL_ALIASES[name]
    return name  # caller passed a full litellm id


# ─────────────────────────────────────────────────────────────────────────────
# litellm client
# ─────────────────────────────────────────────────────────────────────────────
def build_llm_client(model_alias: str) -> Callable[[List[dict]], str]:
    """Return a ``generate(messages) -> str`` callable backed by litellm."""
    try:
        import litellm  # type: ignore
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "litellm is required. Install with: pip install litellm"
        ) from e

    full_model = resolve_model(model_alias)

    def _generate(messages: List[dict]) -> str:
        try:
            resp = litellm.completion(
                model=full_model,
                messages=messages,
                temperature=0.0,
            )
            return resp["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"LLM provider call failed for model '{full_model}': "
                f"{type(e).__name__}: {e}"
            ) from e

    return _generate


# ─────────────────────────────────────────────────────────────────────────────
# Code extraction & dynamic loading
# ─────────────────────────────────────────────────────────────────────────────
_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(raw: str) -> str:
    """Pull a Python function body out of an LLM response (handles ``` fences)."""
    if not raw:
        return ""
    m = _FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


def load_function_from_code(code: str, fn_name: str) -> Callable:
    """Compile ``code`` in a fresh module and return ``module.<fn_name>``."""
    unique_prefix = f"impactarbiter_kernel_{os.urandom(4).hex()}_"
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

    if not hasattr(module, fn_name):
        raise AttributeError(
            f"Generated code does not define a function named '{fn_name}'."
        )
    return getattr(module, fn_name)


# ─────────────────────────────────────────────────────────────────────────────
# Two-Stage RAG Pipeline Prompts
# ─────────────────────────────────────────────────────────────────────────────
DISTILL_SYSTEM_PROMPT = (
    "You are an ML researcher. Extract and summarize the KV cache routing "
    "logic from this paper excerpt into a clear system specification."
)

CODING_SYSTEM_PROMPT = (
    "You are an ML Infra Engineer. Output ONLY raw Python code containing "
    "the requested functions — no markdown fences, no commentary."
)

def build_distill_prompt(paper_excerpt: str) -> str:
    """Step 1: Extract clean mathematical specification from paper."""
    return (
        f"--- PAPER EXCERPT ---\n{paper_excerpt}\n--- END EXCERPT ---\n\n"
        "You are an ML researcher. Extract ONLY the KV cache routing logic "
        "for Planner-to-Executor handoff in multi-agent serving. "
        "Produce a concise, precise system specification that includes:\n"
        "1. How the Executor inherits the Planner's prefix.\n"
        "2. How the mapping handles ragged boundaries and partial blocks when the prefix ends mid-block.\n"
        "3. Any mention of per-head asymmetry or dynamic multi-turn handoffs under ragged batches.\n"
        "Output ONLY the clean specification. No extra commentary."
    )


def build_pure_math_prompt(distill_summary: str, fn_signature: str) -> str:
    """Step 2: Pure implementation from distilled math — intentionally hard for PagedAttention."""
    return (
        f"--- DISTILLED SYSTEM SPECIFICATION ---\n{distill_summary}\n--- END SPECIFICATION ---\n\n"
        f"Implement the following function exactly as described in the specification above:\n"
        f"{fn_signature}\n"
        "Return a tuple (logical_block, offset).\n\n"
        "This is a real production Planner-to-Executor handoff in agentic serving:\n"
        "• The Planner's prefix may end mid-block on some heads.\n"
        "• Different heads may have different prefix lengths (asymmetric).\n"
        "• The Executor must correctly continue writing into the shared partial block under ragged, multi-turn conditions.\n"
        "• The sequence is dynamic, ragged, and multi-turn under live agentic load with multiple agents and batches.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "• Reason strictly from the paper's math. Do not use memorized vLLM, SGLang, or standard PagedAttention code.\n"
        "• Return ONLY the complete, minimal, executable Python function. No docstrings, no comments, no explanations, no test functions.\n"
        "Output ONLY the raw Python code."
    )


def build_refactor_prompt(pure_code: str) -> str:
    """Step 3: Refactor into production-style code — keeps it hard."""
    return (
        f"--- PURE MATH IMPLEMENTATION ---\n{pure_code}\n--- END IMPLEMENTATION ---\n\n"
        "Now refactor this pure math implementation into clean, production-ready Python code "
        "for a real serving engine (vLLM/SGLang style). "
        "Keep the exact same logic, but make it modular and ready for integration into a larger kernel "
        "that handles multi-agent handoffs under ragged batches and dynamic multi-turn conditions.\n"
        "Additionally, include a `test_route()` function (no arguments) that uses `assert` "
        "statements to validate the routing function on block-aligned positions only "
        "(e.g., prefix_length and token offsets that are exact multiples of block_size). "
        "Do NOT assert on intra-block or ragged positions. Calling test_route() must not raise.\n"
        "Return ONLY the routing function and `test_route()`. No docstrings, no comments, no extra text."
    )


def build_heal_payload(
    expected_block: int,
    expected_offset: int,
    agent_block: int,
    agent_offset: int,
    token_label: str,
    fn_name: str,
) -> str:
    """Step 4: Clean gradient feedback — actionable for both oracles."""
    return (
        f"CRITICAL FAILURE at {token_label}:\n"
        f"Expected (logical_block={expected_block}, offset={expected_offset})\n"
        f"Got      (logical_block={agent_block}, offset={agent_offset})\n"
        f"The autograd trap detected a gradient divergence. "
        f"This indicates a memory routing error where the Executor wrote to the wrong physical slot "
        f"in the shared KV cache (partial-block continuation failed under ragged boundary).\n"
        f"Refactor the `{fn_name}` function to correctly handle the case where the Planner's prefix "
        f"ends mid-block on some heads while maintaining correct handoff for all heads under dynamic multi-turn conditions.\n"
        f"Use only the mathematical specification from the paper.\n"
        f"Output ONLY the corrected `{fn_name}` function. No extra text."
    )

__all__ = [
    "resolve_model",
    "build_llm_client",
    "extract_python_code",
    "load_function_from_code",
    "DISTILL_SYSTEM_PROMPT",
    "CODING_SYSTEM_PROMPT",
    "build_distill_prompt",
    "build_pure_math_prompt",
    "build_refactor_prompt",
    "build_heal_payload",
]
