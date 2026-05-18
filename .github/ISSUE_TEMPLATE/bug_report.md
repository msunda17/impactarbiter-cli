---
name: Bug report
about: Report a crash, false PASS, false FAIL, or other incorrect behavior
title: "[BUG] <short description>"
labels: bug
assignees: ''
---

## Summary

<!-- One sentence: what is broken? -->

## Affected component

- [ ] Autograd trap (`src/trap/`) — false PASS or false FAIL
- [ ] Auto-heal pipeline (`src/cli/auto_heal.py`)
- [ ] Stochastic evaluator (`src/core/evaluator.py`)
- [ ] CSV export (`results.csv`)
- [ ] Oracle (`src/oracles/`)
- [ ] Fuzzer (`src/fuzzer/`)
- [ ] CLI / agent (`src/cli/agent.py`)
- [ ] Other:

## Reproduction

**Command:**

```bash
impactarbiter <exact command with all flags>
```

**Environment:**

- OS:
- Python version:
- Mode: `--live` / `--mock`
- Model (if `--live`): e.g. `vertex_ai/gemini-2.5-pro` (do **not** paste API keys)
- impactarbiter version / commit SHA:

## Failure surface

<!--
Paste ONE of:
  - The gradient divergence map ("GRADIENT DIVERGENCE MAP — KV_cache.grad ...")
  - The full Python stack trace
  - The relevant `results.csv` rows + summary block
-->

```
<paste here>
```

## Expected vs actual

- **Expected:** <oracle prediction or correct behavior>
- **Actual:** <what the trap / agent / CLI returned>

## Trap correctness (only if applicable)

If you suspect the trap itself is wrong (false PASS or false FAIL), include:

- The offending agent function (verbatim)
- The boundary case from `src/fuzzer/` that should have caught it
- Why you believe the trap should have flagged this case

> Trap correctness bugs are treated as P0.

## Additional context

<!-- Any related runs, screenshots, or `nextpaper.db` rows. -->
