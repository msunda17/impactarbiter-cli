# ImpactArbiter
> 📖 Read the full launch post and watch the 2D tensor collision demo here: https://maniksundar.substack.com/p/the-physics-illusion-why-llms-still

## Problem Statement

LLM-generated unit tests for KV-cache routing kernels suffer from a silent failure mode: the LLM hallucinates the same bug in both the implementation and the test, causing the test to pass while the kernel remains incorrect. This happens because LLMs reason from the same flawed mental model when writing both code and tests. ImpactArbiter addresses this by using a two-stage RAG pipeline: first, a Distill Agent extracts and summarizes the routing logic from the actual research paper; second, a Coding Agent writes the implementation and test based on that summary. The generated code is then run through a PyTorch autograd trap that compares gradient signatures against SymPy oracles. The trap catches bugs that unit tests miss, even when the LLM's own test_route() assertions pass.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install in Development Mode

```bash
pip install -e .
```

### 3. Configure API Keys (Choose Your Provider)

**OpenAI:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
# On Windows (PowerShell): $env:OPENAI_API_KEY="your-openai-api-key"
```

**Claude (Anthropic):**
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
# On Windows (PowerShell): $env:ANTHROPIC_API_KEY="your-anthropic-api-key"
```

**Gemini / Vertex AI:**
```bash
# Ensure gcloud is authenticated and project is set
gcloud auth login
gcloud config set project impactagent
export GOOGLE_CLOUD_PROJECT="impactagent"
# On Windows (PowerShell): $env:GOOGLE_CLOUD_PROJECT="impactagent"
```

**For persistent configuration, add these to your `.env` file:**
```
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_CLOUD_PROJECT=impactagent
```

## Install

```bash
pip install impactarbiter
```

## Demo Command

```bash
impactarbiter auto-heal --oracle radix --model gemini
```

### Additional Flags

- `--full-agent-trace`: Display LLM Chain-of-Thought reasoning before code generation and heal attempts
- `--live`: Use live LLM API calls instead of cached deterministic replay (requires API key)
- `--mock`: Run offline evaluation with deterministic replay (default if no API key)

Example with live LLM generation and full trace:
```bash
impactarbiter verify --workflow agentic-kv-scheduler --full-agent-trace --live
```

## Sample Output (2D Asymmetric Ring Buffer)

```
─────────────────── IMPACT ARBITER — AUTO-HEAL ───────────────────
Model: vertex_ai/gemini-2.5-pro
[PAPER DOWNLOADED]
https://arxiv.org/pdf/2312.07104.pdf

[QUICK DISTILL]
### KV Cache Routing Specification: Planner-Executor Handoff
...
[GENERATED CODE & TESTS]
def route_radix_2d(b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size):
    k = prefix_length_h + b_local_idx
    logical_block = k // block_size
    offset = k % block_size
    return (head_idx, logical_block, offset)

[LLM UNIT TEST PASS ✅]
LLM self-validation passed.

[AUTOGRAD TRAP FAIL ❌ HARD_BLOCK]
divergence=1.00e+00 > tol=1e-04

GRADIENT DIVERGENCE MAP — KV_cache.grad (head × block × offset)
Token (b=5,h=0,prefix_h=60,N_h=4) | Expected: head=0 block=0 offset=1 | Got: head=0 block=4 offset=1
Non-zero gradient at: [0, 4, 1, :] — misrouted 128 floats

[AUTO-HEAL attempt 1/3]
def route_radix_2d(b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size):
    absolute_idx = prefix_length_h + b_local_idx
    logical_block = (absolute_idx // block_size) % total_blocks_h
    offset = absolute_idx % block_size
    return (head_idx, logical_block, offset)

[FINAL PASS ✅]
divergence=0.00e+00 (after 1 heal attempts)
```

### On LLM non-determinism and trap reliability

> The autograd trap itself is fully deterministic, which means identical code always produces identical gradient divergence results.  
> What varies is whether the LLM generates correct or incorrect routing logic on a given run.  
> This mirrors real production reality: agent-generated serving code sometimes passes, but often silently fails on ragged boundaries.  
> ImpactArbiter gives you deterministic verification of whichever code the agent produces, so you're not relying on hoping the model "got it right this time."

> In practice, Gemini 2.5 Pro generates incorrect routing on roughly 65% of attempts for the critical 2D ring-buffer wrap cases. The trap catches every incorrect implementation with zero false negatives.

## Coverage & Field Results

### 2D RadixAttention Test Matrix (Default)

Recommended for production-relevant demo.

| b_local_idx | head_idx | prefix_h | N_h | expected block | expected offset | note |
|------------|---------|---------|-----|---------------|----------------|------|
| 0 | 0 | 47 | 8 | 2 | 15 | ragged straddle — partial-block carry-over |
| 5 | 0 | 60 | 4 | 0 | 1 | ring-buffer wrap: abs=65, block 4 wraps to 0 |
| 0 | 3 | 200 | 4 | 0 | 8 | ring-buffer deep wrap (multiple revolutions) |

### Legacy PagedAttention

Boundary fixtures `[15, 99, 100, 105, 128]` are maintained for historical comparison mode.

### Summary statistics from most recent evaluation run:

| Oracle | Total Runs | Trap Fired | Healed Successfully |
|--------|-----------|------------|---------------------|
| radix-2d | 32 | 21 | 21 |
| radix (1D) | 15 | 9 | 9 |
| vllm (Paged) | 15 | 0 | 0 |

## Repository layout

```
src/
├── oracles/         # SymPy ASTs + lambdified callables
├── trap/            # autograd trap & ASCII divergence map
├── fuzzer/          # explicit boundary fixtures
├── cli/             # auto-heal pipeline + litellm agent + paper extractor
└── db/              # nextpaper.db (SQLite) validation_traces
tests/
├── test_paged_oracle.py
├── test_radix_oracle.py
└── test_trap.py
```

## Running the tests

```bash
pytest tests/ -v
```

The four load-bearing claims in `tests/test_trap.py` must all pass.

## Contributing

ImpactArbiter is an open project — contributions of new oracles, fuzz cases, and bug reports are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor guide. A short summary:

- **Reporting issues**: Open a GitHub issue using the relevant template under `.github/ISSUE_TEMPLATE/`. Bug reports must include the exact CLI command, the model used, the divergence map (or stack trace), and the contents of `nextpaper.db` row(s) when relevant.
- **Contributing a new oracle**: Every oracle must ship as a triple — (1) a SymPy AST plus a `lambdified` callable in `src/oracles/`, (2) a deterministic autograd trap in `src/trap/`, and (3) explicit boundary fixtures in `src/fuzzer/`. PRs without all three will be sent back.
- **Discussing methodology**: Open a methodology issue — these are reviewed weekly and used to calibrate the mock hallucination rates and per-oracle session windows.

### Issue types

| Template | When to use |
|----------|-------------|
| `bug_report.md` | The trap, auto-heal, evaluator, or CSV export produces incorrect or crashing behavior. |
| `oracle_contribution.md` | You want to propose a new attention/routing oracle (e.g. FlashInfer, MLA, sliding-window). |
| `methodology.md` | You disagree with a hallucination-rate calibration, session-window definition, or trap tolerance. |

### Reporting a bug — minimum reproducible report

A bug is only actionable when we can replay the failure deterministically. Please include:

1. **Command line** — The exact `impactarbiter ...` invocation, including all flags.
2. **Environment** — Python version, OS, and whether `--live` or `--mock` was used.
3. **Model identity** (if `--live`) — e.g. `vertex_ai/gemini-2.5-pro`. Do not paste API keys.
4. **Failure surface** — Paste either the gradient divergence map, the auto-heal stack trace, or the `results.csv` summary block.
5. **Expected vs actual** — What the oracle predicts vs what the agent / trap returned.

If the bug is in the trap itself (false PASS or false FAIL), attach the offending agent function and the corresponding boundary case from `src/fuzzer/`. Trap correctness bugs are treated as P0.

## License

MIT.
