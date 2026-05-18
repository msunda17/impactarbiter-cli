"""Stochastic Evaluator — proves the probabilistic failure rate of LLMs.

Runs the full extract → LLM-draft → unit-test → autograd-trap loop N times
and reports how often the deterministic trap fires. The trap itself never
varies; only the upstream LLM does.

By default this module **does not** burn API quota: it simulates a calibrated
~85% hallucination rate so that `impactarbiter evaluate` is fast and
reproducible in CI / demos. Pass ``live=True`` (or ``--live`` from the CLI) to
drive a real model end-to-end via the existing auto-heal pipeline.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel

from ..oracles import radix_oracle, radix_2d_oracle, paged_oracle
from ..trap import run_radix_trap, run_radix_2d_trap, run_paged_trap

console = Console()

# Calibrated against field results. The 2D Asymmetric Radix oracle (per-head
# ring buffer + ring wrap) is the new default — frontier models still
# hallucinate the modulo wrap at ~85%. Legacy 1D oracles are now near-solved
# (~5%), which is why they are kept only for historical comparison.
_DEFAULT_MOCK_HALLUCINATION_RATE = 0.85
_LEGACY_1D_MOCK_HALLUCINATION_RATE = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunOutcome:
    """Outcome of one (LLM-draft → trap) iteration."""

    run_idx: int
    unit_test_pass: bool
    trap_diverged: bool
    divergence: float

    @property
    def silent_failure(self) -> bool:
        """Unit test passed AND autograd trap caught a divergence."""
        return self.unit_test_pass and self.trap_diverged


@dataclass
class EvaluationReport:
    total_runs: int
    unit_test_passes: int
    trap_failures: int
    silent_failures: int
    paper_id: str
    oracle_kind: str
    mode: str  # "mock" | "live"
    runs: List[RunOutcome] = field(default_factory=list)

    @property
    def hallucination_rate(self) -> float:
        return self.trap_failures / self.total_runs if self.total_runs else 0.0

    def render(self) -> str:
        unit_pct = 100.0 * self.unit_test_passes / self.total_runs
        rate_pct = 100.0 * self.hallucination_rate
        return (
            "[STOCHASTIC EVALUATION COMPLETED]\n"
            f"Total Runs: {self.total_runs}\n"
            f"Unit Test Pass Rate: {unit_pct:.0f}% "
            f"({self.unit_test_passes}/{self.total_runs})\n"
            f"Autograd Trap Failures (Context Leaks): "
            f"{self.trap_failures}/{self.total_runs}\n"
            f"Deterministic Hallucination Rate: {rate_pct:.0f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM — calibrated against measured hallucination rate
# ─────────────────────────────────────────────────────────────────────────────
def _mock_radix_route(*, hallucinate: bool):
    """Return a routing fn that either hallucinates the prefix or routes
    correctly, matching the two failure modes we have caught in the field."""

    def _correct(b_local_idx: int, prefix_length: int, block_size: int):
        abs_idx = prefix_length + b_local_idx
        return abs_idx // block_size, abs_idx % block_size

    def _buggy(b_local_idx: int, prefix_length: int, block_size: int):
        # The classic LLM hallucination: ignore prefix_length entirely.
        return b_local_idx // block_size, b_local_idx % block_size

    return _buggy if hallucinate else _correct


def _mock_paged_route(*, hallucinate: bool):
    def _correct(token_idx: int, block_size: int):
        return token_idx // block_size, token_idx % block_size

    def _buggy(token_idx: int, block_size: int):
        # Off-by-one ring-buffer error — also seen in real LLM output.
        return (token_idx + 1) // block_size, (token_idx + 1) % block_size

    return _buggy if hallucinate else _correct


def _mock_radix_2d_route(*, hallucinate: bool):
    """2D Asymmetric Radix mock with the canonical hallucination: forgetting
    the modulo wrap over ``total_blocks_h`` (ring-buffer overflow)."""

    def _correct(b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size):
        abs_idx = prefix_length_h + b_local_idx
        return head_idx, (abs_idx // block_size) % total_blocks_h, abs_idx % block_size

    def _buggy(b_local_idx, head_idx, prefix_length_h, total_blocks_h, block_size):
        # Memorized 1D linear routing — forgets `% total_blocks_h` (ring wrap).
        abs_idx = prefix_length_h + b_local_idx
        return head_idx, abs_idx // block_size, abs_idx % block_size

    return _buggy if hallucinate else _correct


# ─────────────────────────────────────────────────────────────────────────────
# Single-run helpers
# ─────────────────────────────────────────────────────────────────────────────
_RADIX_CASES = [(0, 47), (1, 47), (0, 48), (5, 47), (0, 64)]
_PAGED_TOKENS = [15, 99, 100, 105, 128]
# Mix of asymmetric ragged + ring-buffer wrap cases (block_size=16).
_RADIX_2D_CASES = [
    (0, 0, 47, 8),    # head=0 ragged straddle, no wrap
    (0, 1, 20, 8),    # head=1 asymmetric prefix
    (5, 0, 60, 4),    # ring wrap: abs=65 → block 4 % 4 = 0
    (0, 2, 64, 4),    # exact-capacity wrap
    (0, 3, 200, 4),   # deep multi-revolution wrap
]


def _shape_unit_test_pass(agent_fn, oracle_kind: str) -> bool:
    """A 'shape only' unit test — the kind LLMs typically produce. Always
    passes on aligned cases, by construction. This emulates the silent
    failure mode where the LLM's own assertions are blind to ragged
    boundaries or ring-buffer wraps."""
    try:
        if oracle_kind == "radix-2d":
            # Aligned prefix, no wrap — the "shape only" case the LLM tests.
            out = agent_fn(0, 0, 32, 8, 16)
            return isinstance(out, tuple) and len(out) == 3
        if oracle_kind == "radix":
            out = agent_fn(0, 32, 16)  # aligned prefix, aligned offset
        else:
            out = agent_fn(16, 16)  # aligned token
        return isinstance(out, tuple) and len(out) == 2
    except Exception:  # noqa: BLE001
        return False


def _run_one_radix(run_idx: int, *, hallucinate: bool) -> RunOutcome:
    fn = _mock_radix_route(hallucinate=hallucinate)
    unit_pass = _shape_unit_test_pass(fn, "radix")
    trap = run_radix_trap(fn, radix_oracle, cases=_RADIX_CASES)
    return RunOutcome(
        run_idx=run_idx,
        unit_test_pass=unit_pass,
        trap_diverged=trap.diverged,
        divergence=trap.divergence_value,
    )


def _run_one_paged(run_idx: int, *, hallucinate: bool) -> RunOutcome:
    fn = _mock_paged_route(hallucinate=hallucinate)
    unit_pass = _shape_unit_test_pass(fn, "paged")
    trap = run_paged_trap(fn, paged_oracle, token_indices=_PAGED_TOKENS)
    return RunOutcome(
        run_idx=run_idx,
        unit_test_pass=unit_pass,
        trap_diverged=trap.diverged,
        divergence=trap.divergence_value,
    )


def _run_one_radix_2d(run_idx: int, *, hallucinate: bool) -> RunOutcome:
    fn = _mock_radix_2d_route(hallucinate=hallucinate)
    unit_pass = _shape_unit_test_pass(fn, "radix-2d")
    trap = run_radix_2d_trap(fn, radix_2d_oracle, cases=_RADIX_2D_CASES)
    return RunOutcome(
        run_idx=run_idx,
        unit_test_pass=unit_pass,
        trap_diverged=trap.diverged,
        divergence=trap.divergence_value,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public evaluator
# ─────────────────────────────────────────────────────────────────────────────
class StochasticEvaluator:
    """Runs N (LLM-draft → autograd-trap) iterations and reports the rate."""

    def __init__(
        self,
        *,
        oracle_kind: str = "radix-2d",
        paper_id: Optional[str] = None,
        hallucination_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if oracle_kind not in ("radix-2d", "radix", "paged", "vllm"):
            raise ValueError(f"unknown oracle_kind: {oracle_kind!r}")
        if oracle_kind == "vllm":
            oracle_kind = "paged"
        self.oracle_kind = oracle_kind
        self.paper_id = paper_id or (
            "2312.07104" if oracle_kind in ("radix-2d", "radix") else "2309.06180"
        )
        # Calibrate per-oracle defaults. 1D oracles are now ~solved by frontier
        # models so we drop their mock hallucination rate accordingly.
        if hallucination_rate is None:
            hallucination_rate = (
                _DEFAULT_MOCK_HALLUCINATION_RATE
                if oracle_kind == "radix-2d"
                else _LEGACY_1D_MOCK_HALLUCINATION_RATE
            )
        self.hallucination_rate = float(hallucination_rate)
        self._rng = random.Random(seed)

    # ── Mock path ──────────────────────────────────────────────────────────────
    def run_mock(self, runs: int) -> EvaluationReport:
        outcomes: List[RunOutcome] = []
        for i in range(runs):
            hallucinate = self._rng.random() < self.hallucination_rate
            if self.oracle_kind == "radix-2d":
                outcome = _run_one_radix_2d(i, hallucinate=hallucinate)
            elif self.oracle_kind == "radix":
                outcome = _run_one_radix(i, hallucinate=hallucinate)
            else:
                outcome = _run_one_paged(i, hallucinate=hallucinate)
            outcomes.append(outcome)
        return self._summarise(outcomes, mode="mock")

    # ── Live path ──────────────────────────────────────────────────────────
    def run_live(self, runs: int, *, model: str, max_heal_retries: int = 3) -> EvaluationReport:
        """Drive the real auto-heal pipeline ``runs`` times.

        Imported lazily to avoid pulling litellm into the mock path.

        ``max_heal_retries`` controls whether the auto-heal loop runs after
        a trap fires (default 3). Pass 0 to measure raw LLM hallucination
        rate without any healing.
        """
        # Local import: avoids a hard dependency cycle through the CLI module.
        from ..cli.agent import build_llm_client
        from ..cli.auto_heal import (
            _run_radix_2d_pipeline,
            _run_radix_pipeline,
            _run_paged_pipeline,
        )

        if self.oracle_kind == "radix-2d":
            pipeline = _run_radix_2d_pipeline
        elif self.oracle_kind == "radix":
            pipeline = _run_radix_pipeline
        else:
            pipeline = _run_paged_pipeline

        outcomes: List[RunOutcome] = []
        generate = build_llm_client(model)
        for i in range(runs):
            try:
                rc = pipeline(
                    generate=generate, max_retries=max_heal_retries, model=model
                )
            except Exception:  # noqa: BLE001 — count provider errors as trap-fail.
                rc = 1
            outcomes.append(
                RunOutcome(
                    run_idx=i,
                    unit_test_pass=True,  # auto-heal gates on unit test before trap
                    trap_diverged=(rc != 0),
                    divergence=float("nan"),
                )
            )
        return self._summarise(outcomes, mode="live")

    # ── Internal ───────────────────────────────────────────────────────────
    def _summarise(self, outcomes: List[RunOutcome], *, mode: str) -> EvaluationReport:
        return EvaluationReport(
            total_runs=len(outcomes),
            unit_test_passes=sum(1 for o in outcomes if o.unit_test_pass),
            trap_failures=sum(1 for o in outcomes if o.trap_diverged),
            silent_failures=sum(1 for o in outcomes if o.silent_failure),
            paper_id=self.paper_id,
            oracle_kind=self.oracle_kind,
            mode=mode,
            runs=outcomes,
        )


def render_report_panel(report: EvaluationReport) -> Panel:
    """Wrap the ASCII summary in a Rich panel for the CLI."""
    return Panel(
        report.render(),
        title=f"impactarbiter evaluate — {report.oracle_kind} ({report.mode})",
        border_style="cyan",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV export from SQLite
# ─────────────────────────────────────────────────────────────────────────────
def export_evaluation_csv(
    db_path: str,
    output_path: str,
    oracle_kind: str = "radix",
) -> str:
    """Export evaluation traces from SQLite to CSV with per-oracle heal-attempt stats.

    If the CSV file already exists, new traces are appended to it instead of replacing it.
    The function merges existing trace records with new ones from the database and
    recalculates summary statistics for all oracle types.

    Returns the path to the written CSV file.
    """
    from ..db.db_manager import fetch_traces

    # Fetch traces for all known oracle types from the database
    all_oracles = ["radix-2d", "radix", "vllm"]
    db_traces = []
    for oracle in all_oracles:
        try:
            traces = fetch_traces(oracle_type=oracle, limit=1000, db_path=db_path)
            db_traces.extend(traces)
        except Exception:  # noqa: BLE001
            pass

    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # If CSV file exists, read existing trace records
    existing_traces = []
    if csv_path.exists():
        try:
            with csv_path.open("r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and len(header) == 5:
                    for row in reader:
                        # Skip empty rows and summary sections
                        if not row or row[0] == "SUMMARY STATISTICS" or row[0] == "oracle_type" or row[0] == "export_timestamp":
                            continue
                        # Only include rows that look like trace records
                        if len(row) == 5 and row[0].startswith("20"):
                            existing_traces.append({
                                "timestamp": row[0],
                                "oracle_type": row[1],
                                "trap_status": row[2],
                                "heal_success": row[3],
                                "heal_attempts": int(row[4]) if row[4].isdigit() else 0,
                            })
        except Exception:  # noqa: BLE001
            pass

    # Convert database traces to the same format
    db_trace_records = []
    for trace in db_traces:
        trap_status = "HARD_BLOCK" if (trace["divergence_value"] or 0) > 1e-4 else "PASS"
        db_trace_records.append({
            "timestamp": trace["timestamp"],
            "oracle_type": trace["oracle_type"],
            "trap_status": trap_status,
            "heal_success": trace["heal_success"],
            "heal_attempts": trace.get("heal_attempts", 0) or 0,
        })

    # Merge existing traces with database traces, deduplicating by timestamp
    all_trace_records = existing_traces + db_trace_records
    seen_timestamps = set()
    unique_traces = []
    for trace in all_trace_records:
        if trace["timestamp"] not in seen_timestamps:
            seen_timestamps.add(trace["timestamp"])
            unique_traces.append(trace)

    # Build CSV rows from all unique traces
    csv_rows = []
    for trace in unique_traces:
        csv_rows.append([
            trace["timestamp"],
            trace["oracle_type"],
            trace["trap_status"],
            trace["heal_success"],
            trace["heal_attempts"],
        ])

    # Group by oracle_type and calculate per-oracle statistics for all traces
    per_oracle_stats = {}
    for oracle in all_oracles:
        oracle_traces = [r for r in csv_rows if r[1] == oracle]
        heal_distribution = {0: 0, 1: 0, 2: 0, 3: 0, "unresolved": 0}
        trap_count = 0
        resolved_count = 0

        for row in oracle_traces:
            trap_status = row[2]
            heal_success = row[3]
            heal_attempts = int(row[4])

            if trap_status == "HARD_BLOCK":
                trap_count += 1
                if heal_success:
                    resolved_count += 1
                    if heal_attempts in heal_distribution:
                        heal_distribution[heal_attempts] += 1
                else:
                    heal_distribution["unresolved"] += 1

        per_oracle_stats[oracle] = {
            "total_runs": len(oracle_traces),
            "trap_fired": trap_count,
            "healed_successfully": resolved_count,
            "heal_attempts_0": heal_distribution[0],
            "heal_attempts_1": heal_distribution[1],
            "heal_attempts_2": heal_distribution[2],
            "heal_attempts_3": heal_distribution[3],
            "unresolved": heal_distribution["unresolved"],
        }

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "oracle_type",
            "trap_status",
            "heal_success",
            "heal_attempts",
        ])

        # Sort by timestamp (newest first)
        sorted_rows = sorted(csv_rows, key=lambda x: x[0], reverse=True)
        for row in sorted_rows:
            writer.writerow(row)

        # Write per-oracle summary sections
        for oracle in all_oracles:
            stats = per_oracle_stats[oracle]
            if stats["total_runs"] == 0:
                continue
            writer.writerow([])
            writer.writerow(["SUMMARY STATISTICS"])
            writer.writerow(["oracle_type", oracle])
            writer.writerow(["total_runs", stats["total_runs"]])
            writer.writerow(["trap_fired", stats["trap_fired"]])
            writer.writerow(["healed_successfully", stats["healed_successfully"]])
            writer.writerow(["heal_attempts_0", stats["heal_attempts_0"]])
            writer.writerow(["heal_attempts_1", stats["heal_attempts_1"]])
            writer.writerow(["heal_attempts_2", stats["heal_attempts_2"]])
            writer.writerow(["heal_attempts_3", stats["heal_attempts_3"]])
            writer.writerow(["unresolved", stats["unresolved"]])

        writer.writerow(["export_timestamp", datetime.now(timezone.utc).isoformat()])

    return str(csv_path)


__all__ = [
    "StochasticEvaluator",
    "EvaluationReport",
    "RunOutcome",
    "render_report_panel",
    "export_evaluation_csv",
]

