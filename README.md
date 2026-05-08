# ImpactArbiter CLI

**ImpactArbiter MVP** — a deterministic ML infra fuzzer that validates LLM-generated KV-cache kernel implementations against a ground-truth SymPy Oracle using differentiable PyTorch autograd traps.

It exposes three commands:
- **`audit`** — BYOK (Bring Your Own Kernel): fuzz a local Python file directly against the Oracle.
- **`auto-heal`** — Agentic Mode: an LLM generates an implementation, the ImpactArbiter validates it against the Oracle, and autograd traces are fed back as structured feedback until the kernel passes or retries are exhausted.
- **`set-env`** — Interactive configuration: set up provider credentials and model preferences via `.env` file.

---

## How It Works

```
  ┌─────────────────────────┐
  │      EngineerAgent      │  ← stateful LLM agent, maintains message_history
  │   (Senior ML Infra SWE) │
  └────────────┬────────────┘
               │ proposed implementation
               ▼
  ┌─────────────────────────┐
  │    SymPy Oracle Engine  │  ← ground-truth logical_block & offset
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  PyTorch Gradient Proxy │  ← differentiable KV-cache memory access
  └────────────┬────────────┘
               │
               Gradient Match? ──NO──▶ Supervisor constructs feedback payload
               │                    │
              YES          ◀────────┘ (injected into EngineerAgent history)
               ▼
  ✅  VALIDATION PASSED  →  trace saved to nextpaper.db
```

The validator fuzzes **three token boundary cases** (`15, 100, 128`), computes `KV_cache` gradients for both oracle and agent output using `torch.func.vmap`, and checks gradient alignment. Any divergence constructs a deterministic feedback payload that is re-injected into the agent's conversation history.

---

## Installation

### Prerequisites
- Python **3.10+**
- `pip` or a virtual environment

### Install core package

```bash
pip install -e .
```

### Install with LLM provider support

```bash
# Google Cloud Vertex AI (Gemini)
pip install -e ".[vertex]"

# OpenAI
pip install -e ".[openai]"

# Anthropic
pip install -e ".[anthropic]"

# All providers at once
pip install -e ".[all-providers]"

# Development tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

After installation, the `impactarbiter` binary is available globally:

```bash
impactarbiter --help
```

---

## Configuration

### Interactive Setup

Use the `set-env` command to configure your provider and model:

```bash
impactarbiter set-env
```

This will prompt you to:
1. Select a provider (openai, anthropic, vertex, ollama)
2. Enter required variables (model name, API keys)
3. Write configuration to `.env`

Configuration is stored in `.env` in your project directory. You can also manually create it:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

### Environment Variables

| Variable | Description | Required |
|---|---|---|
| `IMPACTARBITER_MODEL` | LiteLLM model identifier (e.g. `vertex_ai/gemini-2.5-pro`) | Yes |
| `IMPACTARBITER_DB` | Path to SQLite database (defaults to `nextpaper.db`) | No |
| `OPENAI_API_KEY` | OpenAI API key (for OpenAI models) | Provider-specific |
| `ANTHROPIC_API_KEY` | Anthropic API key (for Anthropic models) | Provider-specific |

---

## Commands

### `audit` — BYOK Mode

Fuzz a local Python file directly against the Oracle. The file must expose a function named **exactly** `ring_buffer_paged_mapping(token_idx, max_window_tokens, block_size)` returning `(logical_block, offset)`.

```bash
impactarbiter audit --file path/to/your_kernel.py
```

**Example kernel:**

```python
# my_kernel.py
def ring_buffer_paged_mapping(token_idx, max_window_tokens, block_size):
    logical_block = token_idx // block_size
    offset = token_idx % block_size
    return logical_block, offset
```

**Output on pass:**
```
════════════════════════════════════════════════════
══════════════════          AUDIT REPORT
════════════════════════════════════════════════════

✅  All tests PASSED
```

**Output on fail:**
```
❌  RED — KV_cache.grad divergence at token_idx=100: 2.6502e00
   Oracle: (logical_block=0, offset=4)  vs  Agent: (logical_block=0, offset=0)
💾  Failure logged → nextpaper.db (run_id=abc123)
```

---

### `auto-heal` — Agentic Mode

Instantiates an LLM agent, runs the validation loop, and saves failure traces to `nextpaper.db`.

```bash
# Using .env configuration
impactarbiter auto-heal

# Override model from CLI
impactarbiter auto-heal --model vertex_ai/gemini-2.5-pro

# Set max retry attempts
impactarbiter auto-heal --max-retries 5
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `$IMPACTARBITER_MODEL` | LiteLLM model identifier |
| `--max-retries` | `3` | Max agent generation attempts |

**Example terminal output:**

```
════════════════════════════════════════════════════
══════════════════        AUTO-HEAL DEMO
════════════════════════════════════════════════════

🤖  Dispatching initial task to vertex_ai/gemini-2.5-pro...

[ENGINEER AGENT] Naive generation:
────────────────────────────────────────────────────────────
def ring_buffer_paged_mapping(token_idx, max_window_tokens, block_size):
    ...
────────────────────────────────────────────────────────────

[STANDARD UNIT TEST]  assert agent(15, 100, 16) == (0, 15)
[PASS] ✅ GREEN

[AUTOGRAD TRAP]  Running differentiable verification...
[FAIL] ❌ RED — KV_cache.grad divergence at token_idx=100

══════════════════        AUTO-HEAL FEEDBACK LOOP
════════════════════════════════════════════════════

[SUPERVISOR → ENGINEER] Feedback (attempt 1/3):
CRITICAL FAILURE: Your implementation caused a gradient divergence...

[ENGINEER AGENT] Refactored implementation (attempt 1):
────────────────────────────────────────────────────────────
def ring_buffer_paged_mapping(token_idx, max_window_tokens, block_size):
    num_blocks = max_window_tokens // block_size
    absolute_block_idx = token_idx // block_size
    logical_block = absolute_block_idx % num_blocks
    offset = token_idx % block_size
    return (logical_block, offset)
────────────────────────────────────────────────────────────

[PASS] ✅ GREEN — healed in 1 attempt(s).

✅  Successfully refactored kernel
💾  Healed code persisted → nextpaper.db
```

On completion, `nextpaper.db` contains:
- `run_id` — unique identifier for this run
- `prompt` — initial task sent to the agent
- `failing_code` — the hallucinated implementation
- `failing_token_idx` — where divergence occurred
- `divergence_map_dump` — full gradient divergence map
- `healed_code` — the corrected implementation (if successful)
- `timestamp` — when the run occurred

---

### `set-env` — Configuration

Interactive setup for provider credentials and model selection.

```bash
impactarbiter set-env
```

This will:
1. Prompt you to select a provider (openai, anthropic, vertex, ollama)
2. Prompt for required variables based on provider
3. Write configuration to `.env`

You can then run `auto-heal` without specifying `--model`.

---

## Development

```bash
pip install -e ".[dev]"

# Lint
ruff check src/

# Type-check
mypy src/

# Run tests
pytest
```

---

## Project Structure

```
impactarbiter-cli/
├── src/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point, audit/auto-heal/set-env commands
│   ├── oracle.py           # SymPy ring-buffer oracle
│   ├── trap.py             # PyTorch autograd trap with vmap
│   ├── db.py               # SQLite persistence layer
│   └── agent.py            # LLM agent with litellm client
├── .env.example            # Example configuration
├── pyproject.toml          # Installable package config
├── nextpaper.db            # SQLite database (auto-generated)
└── README.md
