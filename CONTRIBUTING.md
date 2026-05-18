# Contributing to ImpactArbiter

Thanks for considering a contribution. ImpactArbiter is a verification tool — every new oracle, trap, or boundary condition you add directly hardens the project's ability to catch silent KV-cache routing bugs.

## Reporting Issues

All issues are filed through GitHub Issues. Blank issues are disabled; please pick the right template:

- **`bug_report`** — `.github/ISSUE_TEMPLATE/bug_report.md`. Use when the trap, auto-heal pipeline, evaluator, or CSV export produces incorrect or crashing behavior.
- **`oracle_contribution`** — `.github/ISSUE_TEMPLATE/oracle_contribution.md`. Use *before* opening a PR for a new attention / routing oracle so the design can be calibrated.
- **`methodology`** — `.github/ISSUE_TEMPLATE/methodology.md`. Use when you disagree with a hallucination-rate calibration, session-window definition, or trap tolerance. Methodology issues require empirical evidence (≥ 20 live runs).

### What every bug report must include

A bug is only actionable if we can replay it deterministically. Provide:

1. The exact `impactarbiter ...` command, including all flags.
2. OS, Python version, and `--live` vs `--mock`.
3. Model identity if `--live` (e.g. `vertex_ai/gemini-2.5-pro`). **Never paste API keys.**
4. The failure surface — pick one:
   - the gradient divergence map,
   - the full Python stack trace, or
   - the offending `results.csv` rows + summary block.
5. Expected vs actual: oracle prediction vs agent / trap output.

### Trap correctness bugs are P0

If you suspect the autograd trap itself is wrong (false PASS or false FAIL), include the offending agent function and the boundary case from `src/fuzzer/` that should have caught it. These take priority over feature work.

### Triage labels

- `bug` — incorrect behavior with a reproducer
- `oracle` — new attention / routing mechanism
- `methodology` — calibration / window / tolerance debate
- `enhancement` — additive change, no behavior break
- `P0` — trap correctness or data-integrity bug

## How to Propose a New Oracle

ImpactArbiter's oracles are SymPy-based ground-truth specifications for KV-cache routing mechanisms. To add a new oracle:

1. **Identify the routing logic**: Study the research paper or system specification to understand how tokens map to physical blocks and offsets.

2. **Create the oracle module** in `src/oracles/`:
   - Define symbolic variables for all inputs (e.g., `token_idx`, `block_size`, `prefix_length`)
   - Write SymPy expressions for `logical_block` and `offset`
   - Lambdify the expressions into a callable: `oracle = sp.lambdify((inputs), (logical_block, offset), 'numpy')`
   - Export the callable in `__all__`

3. **Add boundary fixtures** to `src/fuzzer/adversarial.py`:
   - Define explicit adversarial test cases (tokens or prefix/b_local_idx pairs)
   - Include expected (block, offset) outputs for each case
   - Focus on ragged boundaries, off-by-one cases, and edge conditions

4. **Implement the trap** in `src/trap/autograd_trap.py`:
   - Define the KV-cache tensor shape for the oracle
   - Create a trap function that runs the autograd divergence check
   - Return a `TrapResult` with divergence value and ASCII map

5. **Add tests** to `tests/test_trap.py`:
   - Write naive (broken) and correct implementations
   - Assert that naive implementations diverge (> 1e-4)
   - Assert that correct implementations pass (< 1e-4)

## The ImpactDistill 5-Step Pipeline

ImpactArbiter follows a five-step validation pipeline:

1. **Paper Download**: Extract the research paper PDF from ArXiv using pypdf
2. **Distill Agent**: An LLM summarizes the KV-cache routing logic into prose (no code)
3. **Coding Agent**: A second LLM writes the routing function + test_route() based on the distilled summary
4. **LLM Self-Validation**: Dynamically execute the LLM's test_route() (passes due to hallucination)
5. **Autograd Trap**: Run the generated code through PyTorch autograd trap to detect gradient divergence

This two-stage RAG approach ensures the LLM reasons from actual research papers rather than hardcoded prompts, while the autograd trap catches bugs that the LLM's own unit tests miss.

## How to Submit Boundary Condition Test Cases

When adding new boundary fixtures:

- Use explicit, deterministic values (not randomly generated)
- Include ragged boundaries (prefix not multiple of block_size)
- Test off-by-one cases (e.g., token_idx = block_size - 1, block_size, block_size + 1)
- Document the expected behavior with a note explaining the edge condition

Example:
```python
RadixCase(
    prefix_length=47, b_local_idx=0, 
    expected_block=2, expected_offset=15,
    note="ragged straddle — partial-block carry-over"
)
```

## Root Validation Requirement Before Merge

Before merging any new oracle or boundary fixture:

1. All tests in `tests/test_trap.py` must pass
2. The oracle must be callable from the trap with correct tensor shapes
3. The fuzzer must produce deterministic, reproducible results
4. The autograd divergence map must clearly identify misrouted tokens
5. Documentation must explain the routing physics in prose

No business or sales terminology in code or outputs. Focus on the technical correctness of the routing logic.

## Code style

- No emojis in code or in CLI output unless they are part of an existing rich block (e.g. `[FINAL PASS ✅]`).
- No marketing or sales terminology in module docstrings, log lines, or error messages.
- Keep oracle math in SymPy first; only call `.lambdify` at module load time.
