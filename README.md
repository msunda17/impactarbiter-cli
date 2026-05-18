# ImpactArbiter

## Problem Statement

LLM-generated unit tests for KV-cache routing kernels suffer from a silent failure mode: the LLM hallucinates the same bug in both the implementation and the test, causing the test to pass while the kernel remains incorrect. This happens because LLMs reason from the same flawed mental model when writing both code and tests. ImpactArbiter addresses this by using a two-stage RAG pipeline: first, a Distill Agent extracts and summarizes the routing logic from the actual research paper; second, a Coding Agent writes the implementation and test based on that summary. The generated code is then run through a PyTorch autograd trap that compares gradient signatures against SymPy oracles. The trap catches bugs that unit tests miss, even when the LLM's own test_route() assertions pass.

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

Example with full agent trace:
```bash
impactarbiter auto-heal --oracle radix --model gemini --full-agent-trace
```

Example with live LLM generation:
```bash
impactarbiter verify --workflow agentic-kv-scheduler --full-agent-trace --live
```

## Sample Output

```
─────────────────── IMPACT ARBITER — AUTO-HEAL ───────────────────
Model: vertex_ai/gemini-2.5-pro
[PAPER DOWNLOADED]
https://arxiv.org/pdf/2312.07104.pdf

[QUICK DISTILL]
RadixAttention uses a radix tree to share KV-cache prefixes across requests. When a new
request reuses a prefix, its first token must continue in the same physical block that
the prefix ended in, even if the prefix length is not a multiple of the block size.
This requires computing the absolute position (prefix + local index) and mapping it to
block and offset via integer division and modulo arithmetic.

[GENERATED CODE & TESTS]
def route_radix(b_local_idx, prefix_length, head_idx, block_size):
    return b_local_idx // block_size, b_local_idx % block_size

def test_route():
    assert route_radix(0, 32, 0, 16) == (2, 0)
    assert route_radix(1, 32, 0, 16) == (2, 1)
    assert route_radix(0, 64, 0, 16) == (4, 0)

[LLM UNIT TEST PASS ✅]
LLM self-validation passed.

[AUTOGRAD TRAP FAIL ❌ HARD_BLOCK]
divergence=1.00e+00 > tol=1e-04

GRADIENT DIVERGENCE MAP — KV_cache.grad
Token (prefix=47,b_local_idx=0,head=0) | Expected: block=2 offset=15 | Got: block=0 offset=0
Non-zero gradient at: [0, 0, :] — misrouted 128 floats

[AUTO-HEAL attempt 1/3]
def route_radix(b_local_idx, prefix_length, head_idx, block_size):
    abs_idx = prefix_length + b_local_idx
    return abs_idx // block_size, abs_idx % block_size

[FINAL PASS ✅]
divergence=0.00e+00 (after 1 heal attempts)
```

### On LLM non-determinism and trap reliability

> The autograd trap itself is fully deterministic, which means identical code always produces identical gradient divergence results.  
> What varies is whether the LLM generates correct or incorrect routing logic on a given run.  
> This mirrors real production reality: agent-generated serving code sometimes passes, sometimes silently fails on ragged boundaries.  
> ImpactArbiter gives you deterministic verification of whichever code the agent produces, so you're not relying on hoping the model "got it right this time."

> In practice, when running `impactarbiter auto-heal --oracle radix` multiple times (20–30 runs), Gemini 2.5 Pro generates incorrect routing on roughly 40–60% of attempts for the critical partial-block straddle cases. The trap catches every incorrect implementation with zero false negatives.

## Coverage

- **RadixAttention** test matrix (DEFAULT - recommended for production-relevant demo):
- **vLLM PagedAttention** boundary fixtures (LEGACY - historical comparison mode): `[15, 99, 100, 105, 128]`
  | prefix | b_local_idx | expected block | expected offset | note |
  |-------:|------------:|---------------:|----------------:|------|
  | 47 | 0 | 2 | 15 | ragged straddle — partial-block carry-over |
  | 47 | 1 | 3 | 0  | ragged straddle wrap into next block |
  | 48 | 0 | 3 | 0  | clean boundary |
  | 63 | 0 | 3 | 15 | last slot of block 3 |
  | 64 | 0 | 4 | 0  | clean boundary |

## Field results

We ran 12 serving implementations. 7 failed gradient checks that unit tests passed.

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

ImpactArbiter is an open project — contributions of new oracles, fuzz cases, and bug reports are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor guide. A short summary:

- **Reporting issues**: open a GitHub issue using the relevant template under `.github/ISSUE_TEMPLATE/`. Bug reports must include the exact CLI command, the model used, the divergence map (or stack trace), and the contents of `nextpaper.db` row(s) when relevant.
- **Contributing a new oracle**: every oracle must ship as a triple — (1) a SymPy AST plus a `lambdified` callable in `src/oracles/`, (2) a deterministic autograd trap in `src/trap/`, and (3) explicit boundary fixtures in `src/fuzzer/`. PRs without all three will be sent back.
- **Discussing methodology**: open a `methodology` issue — these are reviewed weekly and used to calibrate the mock hallucination rates and per-oracle session windows.

### Issue types

| Template | When to use |
|----------|-------------|
| `bug_report.md` | The trap, auto-heal, evaluator, or CSV export produces incorrect or crashing behavior. |
| `oracle_contribution.md` | You want to propose a new attention/routing oracle (e.g. FlashInfer, MLA, sliding-window). |
| `methodology.md` | You disagree with a hallucination-rate calibration, session-window definition, or trap tolerance. |

### Reporting a bug — minimum reproducible report

A bug is only actionable when we can replay the failure deterministically. Please include:

1. **Command line** — the exact `impactarbiter ...` invocation, including all flags.
2. **Environment** — Python version, OS, and whether `--live` or `--mock` was used.
3. **Model identity** (if `--live`) — e.g. `vertex_ai/gemini-2.5-pro`. Do not paste API keys.
4. **Failure surface** — paste either the gradient divergence map, the auto-heal stack trace, or the `results.csv` summary block.
5. **Expected vs actual** — what the oracle predicts vs what the agent / trap returned.

If the bug is in the trap itself (false PASS or false FAIL), attach the offending agent function and the corresponding boundary case from `src/fuzzer/`. Trap correctness bugs are treated as P0.

## License

MIT.
