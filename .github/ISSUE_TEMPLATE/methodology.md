---
name: Methodology discussion
about: Disagree with a hallucination-rate calibration, session-window definition, or trap tolerance
title: "[METHODOLOGY] <topic>"
labels: methodology, discussion
assignees: ''
---

## What part of the methodology are you challenging?

- [ ] Mock hallucination rate (`_DEFAULT_MOCK_HALLUCINATION_RATE` / `_LEGACY_1D_MOCK_HALLUCINATION_RATE`)
- [ ] Session-window definition in `_group_by_session` (currently 30 minutes)
- [ ] Autograd trap tolerance (`tol=1e-4`)
- [ ] Boundary fuzz case selection
- [ ] CSV summary statistics format
- [ ] Other:

## Current behavior

<!-- Quote the relevant constant / line. e.g. "_DEFAULT_MOCK_HALLUCINATION_RATE = 0.85 in src/core/evaluator.py" -->

## Proposed change

<!-- What value, definition, or behavior do you want instead? -->

## Evidence

Methodology issues need data, not opinions. Please attach:

- [ ] Empirical results from `impactarbiter evaluate --live --runs N` (N ≥ 20)
- [ ] Per-oracle breakdown (radix-2d / radix / vllm)
- [ ] Model identity and date of run
- [ ] Why the current value gives a misleading picture

## Impact

- Which downstream metric does this change? (`results.csv` summary? auto-heal retry budget? trap PASS/FAIL boundary?)
- Does this require a migration / regeneration of historical traces in `nextpaper.db`?
