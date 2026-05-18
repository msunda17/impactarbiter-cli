"""nextpaper.db — SQLite store for validation traces.

Schema (table ``validation_traces``):
    id                INTEGER PRIMARY KEY AUTOINCREMENT
    timestamp         TEXT (UTC ISO-8601)
    oracle_type       TEXT (e.g. "radix", "vllm")
    prompt            TEXT
    generated_code    TEXT
    token_idx         INTEGER (the failing token, may be NULL)
    divergence_value  REAL
    expected_block    INTEGER
    expected_offset   INTEGER
    agent_block       INTEGER
    agent_offset      INTEGER
    divergence_map    TEXT
    healed_code       TEXT
    heal_success      INTEGER (0/1)
    heal_attempts     INTEGER (number of heal attempts: 0, 1, 2, or 3)
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, List, Optional

DEFAULT_DB_PATH = os.environ.get("IMPACTARBITER_DB", "nextpaper.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS validation_traces (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    oracle_type       TEXT NOT NULL,
    prompt            TEXT NOT NULL,
    generated_code    TEXT NOT NULL,
    token_idx         INTEGER,
    divergence_value  REAL,
    expected_block    INTEGER,
    expected_offset   INTEGER,
    agent_block       INTEGER,
    agent_offset      INTEGER,
    divergence_map    TEXT,
    healed_code       TEXT,
    heal_success      INTEGER DEFAULT 0,
    heal_attempts     INTEGER DEFAULT 0
);
"""


@contextmanager
def _connect(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Create or open the SQLite database."""
    conn = sqlite3.connect(db_path)
    # Add heal_attempts column if it doesn't exist (migration)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT heal_attempts FROM validation_traces LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        conn.execute("ALTER TABLE validation_traces ADD COLUMN heal_attempts INTEGER DEFAULT 0")
        conn.commit()
    conn.execute(_SCHEMA)
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Idempotent schema bootstrap."""
    with _connect(db_path):
        pass


def insert_trace(
    *,
    oracle_type: str,
    prompt: str,
    generated_code: str,
    token_idx: Optional[int],
    divergence_value: float,
    expected_block: Optional[int],
    expected_offset: Optional[int],
    agent_block: Optional[int],
    agent_offset: Optional[int],
    divergence_map: str,
    healed_code: Optional[str] = None,
    heal_success: bool = False,
    heal_attempts: int = 0,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Insert a validation trace; returns the new row id."""
    ts = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO validation_traces (
                timestamp, oracle_type, prompt, generated_code, token_idx,
                divergence_value, expected_block, expected_offset,
                agent_block, agent_offset, divergence_map,
                healed_code, heal_success, heal_attempts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts, oracle_type, prompt, generated_code, token_idx,
                float(divergence_value),
                expected_block, expected_offset,
                agent_block, agent_offset, divergence_map,
                healed_code, 1 if heal_success else 0, heal_attempts,
            ),
        )
        return int(cur.lastrowid)


def update_heal(
    trace_id: int,
    *,
    healed_code: str,
    heal_success: bool,
    heal_attempts: int = 1,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE validation_traces SET healed_code = ?, heal_success = ?, heal_attempts = ? WHERE id = ?",
            (healed_code, 1 if heal_success else 0, heal_attempts, trace_id),
        )


def query_gpu_hours_saved(db_path: str = DEFAULT_DB_PATH) -> int:
    """Return ``COUNT(id) * 400`` — the GPU-hours-saved-per-caught-failure metric."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(id) * 400 AS gpu_hours_saved_per_caught_failure FROM validation_traces"
        ).fetchone()
        return int(row[0] if row and row[0] is not None else 0)


def fetch_traces(
    *,
    oracle_type: Optional[str] = None,
    limit: int = 100,
    db_path: str = DEFAULT_DB_PATH,
) -> List[dict]:
    """Return the most recent traces (optionally filtered by oracle_type)."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if oracle_type:
            cur = conn.execute(
                "SELECT * FROM validation_traces WHERE oracle_type = ? "
                "ORDER BY id DESC LIMIT ?",
                (oracle_type, int(limit)),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM validation_traces ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
        return [dict(row) for row in cur.fetchall()]


__all__ = [
    "DEFAULT_DB_PATH",
    "init_db",
    "insert_trace",
    "update_heal",
    "query_gpu_hours_saved",
    "fetch_traces",
]
