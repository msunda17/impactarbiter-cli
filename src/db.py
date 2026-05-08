"""
ImpactArbiter — Data Moat
=========================

Local SQLite logger. Every caught hallucination is persisted to
``nextpaper.db`` *before* the CLI exits, so we accumulate a private corpus
of (prompt, failing_code, divergence_map, healed_code) tuples — the moat.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

DB_PATH = os.environ.get("IMPACTARBITER_DB", "nextpaper.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS failure_traces (
    run_id              TEXT PRIMARY KEY,
    prompt              TEXT NOT NULL,
    failing_code        TEXT NOT NULL,
    failing_token_idx   INTEGER,
    divergence_map_dump TEXT NOT NULL,
    healed_code         TEXT,
    timestamp           TEXT NOT NULL
);
"""


@contextmanager
def _connect(db_path: str = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    """Idempotent schema bootstrap."""
    with _connect(db_path):
        pass


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def log_failure(
    *,
    run_id: str,
    prompt: str,
    failing_code: str,
    failing_token_idx: Optional[int],
    divergence_map_dump: str,
    healed_code: Optional[str] = None,
    db_path: str = DB_PATH,
) -> None:
    """Insert (or upsert) a failure trace."""
    ts = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO failure_traces (
                run_id, prompt, failing_code, failing_token_idx,
                divergence_map_dump, healed_code, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                healed_code = excluded.healed_code,
                divergence_map_dump = excluded.divergence_map_dump,
                timestamp = excluded.timestamp
            """,
            (
                run_id, prompt, failing_code, failing_token_idx,
                divergence_map_dump, healed_code, ts,
            ),
        )


def update_healed_code(
    run_id: str, healed_code: str, db_path: str = DB_PATH
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE failure_traces SET healed_code = ? WHERE run_id = ?",
            (healed_code, run_id),
        )


__all__ = [
    "DB_PATH",
    "init_db",
    "new_run_id",
    "log_failure",
    "update_healed_code",
]
