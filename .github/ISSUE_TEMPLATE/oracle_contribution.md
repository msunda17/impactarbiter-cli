---
name: Oracle contribution proposal
about: Propose a new attention / KV-cache routing oracle (e.g. FlashInfer, MLA, sliding-window)
title: "[ORACLE] <name of attention mechanism>"
labels: oracle, enhancement
assignees: ''
---

## Oracle name

<!-- e.g. "FlashInfer paged ragged" or "Multi-head Latent Attention (MLA)" -->

## Source paper

- arXiv ID / URL:
- Section / equation that defines the routing function:
- Why this oracle is worth adding (what failure mode does it expose that the existing radix-2d / radix / vllm oracles do not?):

## Routing contract

Define the function the agent must produce. Use the same shape as existing oracles:

**Signature:**

```python
def route_<name>(<params>) -> tuple[int, ...]:
    ...
```

**Mathematical specification (from the paper):**

```
absolute_idx  = ...
logical_block = ...
offset        = ...
return (...)
```

**Parameters and constraints:**

| Param | Type | Range | Meaning |
|-------|------|-------|---------|
|       |      |       |         |

## The required triple

Every new oracle must ship with all three of the following. PRs missing any one of these will be sent back.

### 1. Symbolic oracle — `src/oracles/<name>.py`

- [ ] SymPy AST defining the routing function
- [ ] `lambdified` callable used as the ground truth
- [ ] Unit tests in `tests/test_<name>_oracle.py` proving symbolic vs numeric agreement on at least 5 boundary cases

### 2. Autograd trap — `src/trap/<name>_trap.py`

- [ ] PyTorch trap that compares `KV_cache.grad` between agent and oracle
- [ ] Deterministic (no `torch.rand` without a seed)
- [ ] Returns `divergence_value` and `diverged: bool`
- [ ] Produces a gradient divergence map identical in shape to `run_radix_2d_trap`

### 3. Boundary fuzz cases — `src/fuzzer/adversarial.py`

- [ ] At least one **ragged straddle** case (partial-block carry-over)
- [ ] At least one **ring-buffer wrap** case (if applicable)
- [ ] At least one **asymmetric per-head** case (if applicable)
- [ ] One case that the LLM is *expected* to get wrong, with a brief justification

## Hallucination calibration

How often do current frontier LLMs (Gemini 2.5 Pro / GPT-4 / Claude) get this routing wrong?

- [ ] I have run the oracle through `impactarbiter evaluate --live --runs 20` and observed an empirical hallucination rate of: \_\_\_\_\_%
- [ ] I have not yet measured this — please calibrate during review.

## CLI integration

- [ ] Added to the `--oracle` choices in `src/cli/auto_heal.py`
- [ ] Added to `_run_<name>_pipeline` in `src/cli/auto_heal.py`
- [ ] Added to `StochasticEvaluator` in `src/core/evaluator.py`
- [ ] Added to `export_evaluation_csv` `all_oracles` list
- [ ] Added a `_build_heal_payload_<name>` if the failure surface differs from the existing 1D / 2D shapes

## Mock hallucination

- [ ] Added a `_mock_<name>_route(*, hallucinate)` factory in `src/core/evaluator.py` so `--mock` evaluations work without burning API quota.

## Anything else?
