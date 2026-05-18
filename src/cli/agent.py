"""LLM agent abstraction (litellm) and code-extraction helpers."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
import tempfile
import warnings
from typing import Callable, List, Optional

# Suppress Pydantic serialization warnings from litellm
warnings.filterwarnings("ignore", message="PydanticSerialization")
# Also suppress all UserWarnings from pydantic
logging.getLogger("pydantic").setLevel(logging.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# Model-id resolution
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_ALIASES = {
    "gemini": os.environ.get("IMPACTARBITER_GEMINI_MODEL", "vertex_ai/gemini-2.5-pro"),
    "claude": os.environ.get("IMPACTARBITER_CLAUDE_MODEL", "anthropic/claude-sonnet-4.6"),
    "openai": os.environ.get("IMPACTARBITER_OPENAI_MODEL", "gpt-5.5"),
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

def extract_python_code(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    # First, remove thinking tags to avoid interference
    text_without_thinking = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    
    match = re.search(r"```python\s*([\s\S]*?)\s*```", text_without_thinking)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*([\s\S]*?)\s*```", text_without_thinking)
    if match:
        return match.group(1).strip()
    return text_without_thinking.strip()


def extract_thinking(text: str) -> Optional[str]:
    """Extract text between <thinking> and </thinking> tags."""
    match = re.search(r"<thinking>\s*([\s\S]*?)\s*</thinking>", text)
    if match:
        return match.group(1).strip()
    return None


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
    "You are an ML Infra Engineer. You MUST output your internal reasoning and chain-of-thought "
    "step-by-step inside <thinking> and </thinking> tags before you output your final Python code. "
    "This is REQUIRED - do not skip the thinking tags. After the thinking tags, output ONLY raw Python code "
    "containing the requested functions — no markdown fences, no commentary outside the thinking tags."
)

def build_distill_prompt(paper_excerpt: str) -> str:
    """Step 1: Extract clean mathematical specification from paper."""
    return (
        f"--- PAPER EXCERPT ---\n{paper_excerpt}\n--- END EXCERPT ---\n\n"
        "You are an ML researcher. Extract ONLY the KV cache routing logic "
        "for Planner-to-Executor handoff in multi-agent serving. "
        "Produce a concise, precise system specification of the mathematical routing. "
        "Output ONLY the clean specification. No extra commentary."
    )


def _is_2d_signature(fn_signature: str) -> bool:
    """Detect the 2D Asymmetric Radix signature from the user-supplied prototype."""
    return "head_idx" in fn_signature and "total_blocks_h" in fn_signature


def build_pure_math_prompt(distill_summary: str, fn_signature: str) -> str:
    if _is_2d_signature(fn_signature):
        return (
            f"--- DISTILLED SYSTEM SPECIFICATION ---\n{distill_summary}\n--- END SPECIFICATION ---\n\n"
            f"Implement the 2D Asymmetric Radix routing function exactly as described:\n"
            f"{fn_signature}\n"
            "Return a plain tuple (head_idx, logical_block, offset).\n\n"
            "MEMORY MODEL:\n"
            "• The KV cache is a per-head ring buffer. Each head has its own circular "
            "buffer of `total_blocks_h` physical blocks.\n"
            "• Each head has its own prefix length `prefix_length_h`.\n"
            "• The `head_idx` dimension must be preserved unchanged.\n\n"
            "STRICT SIGNATURE CONSTRAINT — DO NOT VIOLATE:\n"
            "• Use the EXACT function name and parameter names from the signature above. "
            "Do NOT rename, reorder, drop, or merge parameters.\n"
            "• Do NOT wrap the function in a class or introduce a NamedTuple.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "• Reason strictly from the spec above. Do not use memorized vLLM/SGLang code.\n"
            "• Return ONLY the complete, minimal, executable Python function. "
            "No docstrings, no comments, no explanations, no test functions.\n"
            "Output ONLY the raw Python code."
        )

    # Legacy 1D path
    return (
        f"--- DISTILLED SYSTEM SPECIFICATION ---\n{distill_summary}\n--- END SPECIFICATION ---\n\n"
        f"Implement the following function exactly as described in the specification above:\n"
        f"{fn_signature}\n"
        "Return a tuple (logical_block, offset).\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "• Reason strictly from the paper's math. Do not use memorized vLLM, SGLang, or standard PagedAttention code.\n"
        "• Return ONLY the complete, minimal, executable Python function. No docstrings, no comments, no explanations, no test functions.\n"
        "Output ONLY the raw Python code."
    )


def build_refactor_prompt(pure_code: str) -> str:
    is_2d = "head_idx" in pure_code and "total_blocks_h" in pure_code
    if is_2d:
        return (
            f"--- PURE MATH IMPLEMENTATION ---\n{pure_code}\n--- END IMPLEMENTATION ---\n\n"
            "Refactor this into clean, production-ready Python code for a real "
            "multi-head serving engine. Preserve the 2D Asymmetric Radix contract.\n\n"
            "STRICT SIGNATURE CONSTRAINT — DO NOT VIOLATE:\n"
            "• The refactored function MUST be named exactly `route_radix_2d`.\n"
            "• It MUST accept five positional parameters in this exact order and with "
            "these exact names:\n"
            "    (b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size)\n"
            "• Do NOT rename parameters, drop parameters, reorder them, or wrap the "
            "function in a class. Do NOT introduce a NamedTuple return type — return "
            "a plain tuple `(head_idx, logical_block, offset)`.\n\n"
            "Additionally, include a `test_route()` function (no arguments) that uses "
            "`assert` statements to validate `route_radix_2d` across at least one "
            "ring-buffer wrap case and one asymmetric per-head case. Calling test_route() "
            "must not raise.\n"
            "Return ONLY `route_radix_2d` and `test_route()`. "
            "No docstrings, no comments, no extra text."
        )
        
    return (
        f"--- PURE MATH IMPLEMENTATION ---\n{pure_code}\n--- END IMPLEMENTATION ---\n\n"
        "Refactor this into clean, production-ready Python code for a real serving engine. "
        "Make it modular and ready for integration into a larger kernel.\n"
        "Additionally, include a `test_route()` function (no arguments) that uses `assert` "
        "statements to validate the routing function. Calling test_route() must not raise.\n"
        "Return ONLY the routing function and `test_route()`. "
        "No docstrings, no comments, no extra text."
    )


def build_heal_payload(
    expected_block: int,
    expected_offset: int,
    agent_block: int,
    agent_offset: int,
    token_label: str,
    fn_name: str,
    *,
    require_thinking_tags: bool = False,
) -> str:
    """Step 4: Clean gradient feedback — actionable for both oracles."""
    thinking_instruction = (
        f"IMPORTANT: You MUST output your reasoning inside <thinking> and </thinking> tags before the code. "
        f"This is REQUIRED - do not skip the thinking tags. Then output the corrected `{fn_name}` function."
        if require_thinking_tags
        else f"Then output the corrected `{fn_name}` function."
    )
    return (
        f"CRITICAL FAILURE at {token_label}:\n"
        f"Expected (logical_block={expected_block}, offset={expected_offset})\n"
        f"Got      (logical_block={agent_block}, offset={agent_offset})\n"
        f"The autograd trap detected a gradient divergence. "
        f"This indicates a memory routing error where the Executor wrote to the wrong physical slot "
        f"in the shared KV cache.\n"
        f"Refactor the `{fn_name}` function to correctly handle the continuous physical constraints.\n"
        f"Use only the mathematical specification from the paper.\n"
        f"{thinking_instruction}"
    )

__all__ = [
    "resolve_model",
    "build_llm_client",
    "extract_python_code",
    "extract_thinking",
    "load_function_from_code",
    "DISTILL_SYSTEM_PROMPT",
    "CODING_SYSTEM_PROMPT",
    "build_distill_prompt",
    "build_pure_math_prompt",
    "build_refactor_prompt",
    "build_heal_payload",
]
