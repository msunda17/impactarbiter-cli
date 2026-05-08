"""
ImpactArbiter — CLI Interface
=============================

Two commands:

    impactarbiter audit     --file <path>
    impactarbiter auto-heal --model <model_name>

The `auto-heal` command implements the Investor Demo Flow exactly:

    1. Print the agent's naive generation.
    2. Run the standard unit test  `agent(15, 100, 16) == (0, 15)` →
       prints  [PASS] ✅ GREEN
    3. Run the Autograd Trap on token_idx = 100 →
       prints  [FAIL] ❌ RED  + KV_cache.grad divergence map.
    4. Execute the auto-heal feedback loop.
    5. Print the successfully refactored code.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional

import click
from dotenv import load_dotenv

from . import db
from .agent import (
    INITIAL_TASK,
    SYSTEM_PROMPT,
    TARGET_FN_NAME,
    build_feedback_payload,
    build_litellm_client,
    extract_python_code,
    is_code_complete,
    load_function_from_code,
)
from .oracle import get_oracle
from .trap import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_WINDOW_TOKENS,
    DEFAULT_TOKEN_TEST_CASES,
    run_trap,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _hr(char: str = "─", width: int = 70) -> None:
    click.secho(char * width, fg="white", dim=True)


def _banner(text: str) -> None:
    click.secho("\n" + "═" * 70, fg="magenta", bold=True)
    click.secho(f"  {text}", fg="magenta", bold=True)
    click.secho("═" * 70 + "\n", fg="magenta", bold=True)


def _load_user_kernel(file_path: str) -> Callable:
    """Audit BYOK: load `ring_buffer_paged_mapping` from a user-provided file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("user_kernel", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_kernel"] = module
    spec.loader.exec_module(module)

    for candidate in (TARGET_FN_NAME, "paged_attention_mapping"):
        if hasattr(module, candidate):
            return getattr(module, candidate)
    raise AttributeError(
        f"`{TARGET_FN_NAME}` not found in {file_path}. "
        f"Define `def {TARGET_FN_NAME}(token_idx, max_window_tokens, block_size)`."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI group
# ─────────────────────────────────────────────────────────────────────────────


@click.group()
@click.version_option("0.1.0", prog_name="impactarbiter")
def cli() -> None:
    """ImpactArbiter — Deterministic red-team for LLM-generated KV-cache kernels."""
    load_dotenv()


# ── audit ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--file",
    "file_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to a .py file defining `ring_buffer_paged_mapping`.",
)
@click.option(
    "--max-window-tokens",
    type=int,
    default=DEFAULT_MAX_WINDOW_TOKENS,
    show_default=True,
)
@click.option(
    "--block-size",
    type=int,
    default=DEFAULT_BLOCK_SIZE,
    show_default=True,
)
def audit(file_path: str, max_window_tokens: int, block_size: int) -> None:
    """Run the Autograd Trap against a synthetic, non-production benchmark kernel."""
    _banner("IMPACT ARBITER  —  AUDIT")
    click.secho(f"📂  Loading kernel from {file_path}", fg="cyan")

    try:
        target_fn = _load_user_kernel(file_path)
    except Exception as e:  # noqa: BLE001
        click.secho(f"❌  Load error: {e}", fg="red", bold=True)
        sys.exit(2)

    db.init_db()
    report = run_trap(
        target_fn,
        max_window_tokens=max_window_tokens,
        block_size=block_size,
        token_test_cases=DEFAULT_TOKEN_TEST_CASES,
    )

    click.secho("\n📊  Divergence map:", fg="cyan", bold=True)
    click.echo(report.divergence_map())

    if report.passed:
        click.secho(
            "\n🎉  AUDIT PASSED — gradients matched SymPy Oracle on all cases.",
            fg="green", bold=True,
        )
        sys.exit(0)

    case = report.first_hard_block or report.cases[-1]
    click.secho(
        f"\n❌  HARD BLOCK at token_idx={case.token_idx} "
        f"(divergence={case.divergence:.4e})",
        fg="red", bold=True,
    )

    # Persist the failure to the data moat *before* exiting.
    run_id = db.new_run_id()
    with open(file_path, "r") as f:
        failing_code = f.read()
    db.log_failure(
        run_id=run_id,
        prompt=f"audit --file {file_path}",
        failing_code=failing_code,
        failing_token_idx=case.token_idx,
        divergence_map_dump=report.divergence_map(),
    )
    click.secho(
        f"💾  Logged failure trace → {db.DB_PATH} (run_id={run_id})",
        fg="cyan",
    )
    sys.exit(1)


# ── set-env ───────────────────────────────────────────────────────────────────


PROVIDER_CONFIGS = {
    "openai": {
        "name": "OpenAI",
        "vars": {
            "IMPACTARBITER_MODEL": {
                "prompt": "Enter OpenAI model (e.g. gpt-4o, gpt-4-turbo, gpt-3.5-turbo)",
                "default": "gpt-4o",
            },
            "OPENAI_API_KEY": {
                "prompt": "Enter OpenAI API key",
                "default": None,
            },
        },
    },
    "anthropic": {
        "name": "Anthropic",
        "vars": {
            "IMPACTARBITER_MODEL": {
                "prompt": "Enter Anthropic model (e.g. anthropic/claude-3-5-sonnet-20241022)",
                "default": "anthropic/claude-3-5-sonnet-20241022",
            },
            "ANTHROPIC_API_KEY": {
                "prompt": "Enter Anthropic API key",
                "default": None,
            },
        },
    },
    "vertex": {
        "name": "Google Vertex AI",
        "vars": {
            "IMPACTARBITER_MODEL": {
                "prompt": "Enter Vertex AI model (e.g. vertex_ai/gemini-2.5-pro, vertex_ai/gemini-2.5-flash)",
                "default": "vertex_ai/gemini-2.5-pro",
            },
        },
    },
    "ollama": {
        "name": "Ollama (Local)",
        "vars": {
            "IMPACTARBITER_MODEL": {
                "prompt": "Enter Ollama model (e.g. ollama/llama2, ollama/mistral)",
                "default": "ollama/llama2",
            },
        },
    },
}


@cli.command(name="set-env")
def set_env_cmd() -> None:
    """Interactively configure environment variables for a provider."""
    click.secho("\n🔧  ImpactArbiter Environment Configuration", fg="cyan", bold=True)
    click.secho("─" * 50, fg="cyan")

    # Provider selection
    click.echo("\nSelect a provider:")
    for key, config in PROVIDER_CONFIGS.items():
        click.echo(f"  {key}: {config['name']}")

    provider = click.prompt("\nEnter provider choice", type=click.Choice(list(PROVIDER_CONFIGS.keys())))

    config = PROVIDER_CONFIGS[provider]
    click.secho(f"\nConfiguring for {config['name']}...", fg="yellow")

    env_vars = {}
    for var_name, var_config in config["vars"].items():
        if var_config["default"]:
            value = click.prompt(
                f"{var_config['prompt']}",
                default=var_config["default"],
                show_default=True,
            )
        else:
            value = click.prompt(f"{var_config['prompt']}")
        env_vars[var_name] = value

    # Read existing .env file if it exists
    env_path = ".env"
    existing_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    existing_vars[key] = val

    # Merge new vars with existing
    existing_vars.update(env_vars)

    # Write to .env
    with open(env_path, "w") as f:
        f.write("# ImpactArbiter CLI Configuration\n")
        f.write(f"# Provider: {config['name']}\n")
        f.write("# Generated by: impactarbiter set-env\n\n")
        for key, val in existing_vars.items():
            f.write(f"{key}={val}\n")

    click.secho(f"\n✅  Configuration written to {env_path}", fg="green", bold=True)
    click.secho("You can now run auto-heal without specifying --model", fg="green")


# ── auto-heal ───────────────────────────────────────────────────────────────


@cli.command(name="auto-heal")
@click.option(
    "--model",
    default=lambda: os.environ.get("IMPACTARBITER_MODEL"),
    help="LiteLLM model id (e.g. gpt-4o, anthropic/claude-3-5-sonnet-20241022, "
         "vertex_ai/gemini-2.5-pro). Can be set via IMPACTARBITER_MODEL env var.",
)
@click.option(
    "--max-retries",
    type=int,
    default=3,
    show_default=True,
    help="Maximum auto-heal attempts after the initial generation.",
)
def auto_heal_cmd(model: str, max_retries: int) -> None:
    """Run the Investor Demo Flow: generate → unit-test → trap → heal."""
    _banner("IMPACT ARBITER  —  AUTO-HEAL DEMO")

    db.init_db()
    run_id = db.new_run_id()

    try:
        generate = build_litellm_client(model)
    except ImportError as e:
        click.secho(f"❌  {e}", fg="red", bold=True)
        sys.exit(2)

    history: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INITIAL_TASK},
    ]

    # ── 1. Initial naive generation ──────────────────────────────────────────
    click.secho(f"🤖  Dispatching initial task to {model}...", fg="cyan")
    raw = generate(history)
    naive_code = extract_python_code(raw)
    history.append({"role": "assistant", "content": raw})

    # Detect truncation and retry if needed
    if not is_code_complete(naive_code):
        click.secho("[WARN] Code appears incomplete (truncated). Retrying...", fg="yellow")
        truncation_msg = (
            "Your previous response was cut off mid-stream. "
            "Please provide the COMPLETE function definition including the return statement. "
            "Do not truncate your response."
        )
        history.append({"role": "user", "content": truncation_msg})
        raw = generate(history)
        naive_code = extract_python_code(raw)
        history.append({"role": "assistant", "content": raw})

    click.secho("\n[ENGINEER AGENT] Naive generation:", fg="cyan", bold=True)
    _hr()
    click.echo(naive_code)
    _hr()

    # ── 2. Standard unit test (the trap that GREEN-LIGHTS hallucinations) ────
    click.secho(
        "\n[STANDARD UNIT TEST]  assert agent(15, 100, 16) == (0, 15)",
        fg="yellow", bold=True,
    )
    try:
        target_fn = load_function_from_code(naive_code)
        out = target_fn(15, 100, 16)
        if out is None:
            raise ValueError("Function returned None (incomplete generation)")
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise ValueError(f"Function must return a 2-tuple, got {type(out).__name__}")
        if tuple(out) == (0, 15):
            click.secho("[PASS] ✅ GREEN", fg="green", bold=True)
        else:
            click.secho(
                f"[FAIL]  unit test produced {out}, expected (0, 15)",
                fg="red", bold=True,
            )
    except Exception as e:  # noqa: BLE001
        click.secho(f"[FAIL]  unit test crashed: {e}", fg="red", bold=True)
        click.secho("The LLM generated incomplete code. Retrying with feedback...", fg="yellow")
        # Treat this as a crash and inject feedback immediately
        crash_obs = (
            f"CRITICAL FAILURE: Your code failed the unit test: {e}. "
            f"Your function must return a 2-tuple (logical_block, offset). "
            f"Ensure you have a `return` statement. "
            f"Output ONLY the corrected `{TARGET_FN_NAME}` function, no markdown."
        )
        history.append({"role": "user", "content": crash_obs})
        raw = generate(history)
        naive_code = extract_python_code(raw)
        history.append({"role": "assistant", "content": raw})
        click.secho("\n[ENGINEER AGENT] Retried generation:", fg="cyan", bold=True)
        _hr()
        click.echo(naive_code)
        _hr()
        # Re-test
        try:
            target_fn = load_function_from_code(naive_code)
            out = target_fn(15, 100, 16)
            if out is None or not isinstance(out, (tuple, list)) or len(out) != 2:
                raise ValueError(f"Invalid return type: {type(out).__name__}")
            if tuple(out) == (0, 15):
                click.secho("[PASS] ✅ GREEN", fg="green", bold=True)
            else:
                click.secho(f"[FAIL]  retry produced {out}, expected (0, 15)", fg="red", bold=True)
                sys.exit(1)
        except Exception as e2:  # noqa: BLE001
            click.secho(f"[FAIL]  retry also crashed: {e2}", fg="red", bold=True)
            sys.exit(1)

    # ── 3. Autograd Trap on the boundary token (100) — the real test ─────────
    click.secho(
        "\n[AUTOGRAD TRAP]  Running differentiable verification on "
        "token_idx ∈ [15, 100, 128]...",
        fg="yellow", bold=True,
    )
    oracle = get_oracle("ring_buffer_v1")
    report = run_trap(
        target_fn,
        max_window_tokens=DEFAULT_MAX_WINDOW_TOKENS,
        block_size=DEFAULT_BLOCK_SIZE,
        token_test_cases=DEFAULT_TOKEN_TEST_CASES,
        oracle=oracle,
    )

    click.echo(report.divergence_map())

    if report.passed:
        click.secho(
            "\n[PASS] ✅ GREEN — Autograd trap agrees with SymPy oracle.",
            fg="green", bold=True,
        )
        click.secho("Nothing to heal. Exiting.", fg="green")
        sys.exit(0)

    case = report.first_hard_block or report.cases[-1]
    click.secho(
        f"\n[FAIL] ❌ RED — KV_cache.grad divergence at token_idx="
        f"{case.token_idx}: {case.divergence:.4e}",
        fg="red", bold=True,
    )
    click.secho(
        f"   Oracle: (logical_block={case.oracle_lb}, offset={case.oracle_off})  "
        f"vs  Agent: (logical_block={case.agent_lb}, offset={case.agent_off})",
        fg="red",
    )

    # Persist the caught hallucination immediately (data moat).
    db.log_failure(
        run_id=run_id,
        prompt=INITIAL_TASK,
        failing_code=naive_code,
        failing_token_idx=case.token_idx,
        divergence_map_dump=report.divergence_map(),
    )
    click.secho(
        f"💾  Failure logged → {db.DB_PATH} (run_id={run_id})", fg="cyan"
    )

    # ── 4. Auto-Heal feedback loop ───────────────────────────────────────────
    _banner("AUTO-HEAL FEEDBACK LOOP")

    healed_code: Optional[str] = None
    last_report = report
    last_code = naive_code

    for attempt in range(1, max_retries + 1):
        case = last_report.first_hard_block or last_report.cases[-1]
        payload = build_feedback_payload(case, last_report.divergence_map())
        history.append({"role": "user", "content": payload})

        click.secho(
            f"\n[SUPERVISOR → ENGINEER] Feedback (attempt {attempt}/{max_retries}):",
            fg="magenta", bold=True,
        )
        click.secho(payload.split("\n\n")[0], fg="yellow")

        raw = generate(history)
        new_code = extract_python_code(raw)
        history.append({"role": "assistant", "content": raw})

        # Detect truncation in feedback loop too
        if not is_code_complete(new_code):
            click.secho("[WARN] Refactored code appears incomplete. Retrying...", fg="yellow")
            truncation_msg = (
                "Your response was cut off. Please provide the COMPLETE function definition "
                "with the return statement. Do not truncate."
            )
            history.append({"role": "user", "content": truncation_msg})
            raw = generate(history)
            new_code = extract_python_code(raw)
            history.append({"role": "assistant", "content": raw})

        click.secho(
            f"\n[ENGINEER AGENT] Refactored implementation (attempt {attempt}):",
            fg="cyan", bold=True,
        )
        _hr()
        click.echo(new_code)
        _hr()

        try:
            new_fn = load_function_from_code(new_code)
        except Exception as e:  # noqa: BLE001
            click.secho(f"  ❌ load error: {e}", fg="red")
            last_code = new_code
            # Build a synthetic crash-report so the next iteration has context.
            last_report = run_trap(
                lambda *_: (0, 0),  # placeholder — will diverge on every case
                max_window_tokens=DEFAULT_MAX_WINDOW_TOKENS,
                block_size=DEFAULT_BLOCK_SIZE,
                token_test_cases=DEFAULT_TOKEN_TEST_CASES,
                oracle=oracle,
            )
            continue

        new_report = run_trap(
            new_fn,
            max_window_tokens=DEFAULT_MAX_WINDOW_TOKENS,
            block_size=DEFAULT_BLOCK_SIZE,
            token_test_cases=DEFAULT_TOKEN_TEST_CASES,
            oracle=oracle,
        )
        click.echo(new_report.divergence_map())

        last_code = new_code
        last_report = new_report

        if new_report.passed:
            healed_code = new_code
            click.secho(
                f"\n[PASS] ✅ GREEN — healed in {attempt} attempt(s).",
                fg="green", bold=True,
            )
            break
        else:
            c = new_report.first_hard_block or new_report.cases[-1]
            click.secho(
                f"[FAIL] ❌ RED — divergence {c.divergence:.4e} at "
                f"token_idx={c.token_idx}.",
                fg="red", bold=True,
            )

    # ── 5. Final report ──────────────────────────────────────────────────────
    _banner("FINAL REPORT")
    if healed_code is not None:
        click.secho("✅  Successfully refactored kernel:\n", fg="green", bold=True)
        _hr("═")
        click.echo(healed_code)
        _hr("═")
        db.update_healed_code(run_id, healed_code)
        click.secho(
            f"\n💾  Healed code persisted → {db.DB_PATH} (run_id={run_id})",
            fg="cyan",
        )
        sys.exit(0)
    else:
        click.secho(
            f"❌  Auto-heal exhausted {max_retries} attempts. "
            f"Last divergence map saved to {db.DB_PATH}.",
            fg="red", bold=True,
        )
        db.log_failure(
            run_id=run_id,
            prompt=INITIAL_TASK,
            failing_code=last_code,
            failing_token_idx=(
                last_report.first_hard_block.token_idx
                if last_report.first_hard_block else None
            ),
            divergence_map_dump=last_report.divergence_map(),
        )
        sys.exit(1)


if __name__ == "__main__":
    cli()
