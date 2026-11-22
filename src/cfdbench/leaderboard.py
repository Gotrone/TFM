"""SQLite-backed leaderboard.

Schema (table ``entries``): one row per (submission, split) with the metrics stored
as JSON, so that new metrics can be added without a migration.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .metrics import METRICS, RANKING_METRIC

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    split TEXT NOT NULL,
    metrics TEXT NOT NULL,
    fit_time_s REAL,
    predict_time_s REAL,
    n_test INTEGER,
    dataset_version TEXT,
    dataset_checksum TEXT,
    source TEXT NOT NULL DEFAULT 'baseline',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_split ON entries(split);
"""


@contextmanager
def connect(db_path: str | Path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_entries(db_path: str | Path, rows: list[dict], author: str = "", source: str = "baseline") -> None:
    with connect(db_path) as conn:
        for r in rows:
            conn.execute(
                "INSERT INTO entries (model, author, split, metrics, fit_time_s, predict_time_s, n_test,"
                " dataset_version, dataset_checksum, source, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    r["model"],
                    author,
                    r["split"],
                    json.dumps(r["metrics"]),
                    r.get("fit_time_s"),
                    r.get("predict_time_s"),
                    r.get("n_test"),
                    r.get("dataset_version"),
                    r.get("dataset_checksum"),
                    source,
                    r["created_at"],
                ),
            )


def get_leaderboard(db_path: str | Path, split: str, metric: str = RANKING_METRIC) -> list[dict]:
    """Entries of a split ranked by ``metric`` (ascending: lower is better)."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric '{metric}', choose from {METRICS}")
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM entries WHERE split = ?", (split,)).fetchall()
    out = []
    for r in rows:
        m = json.loads(r["metrics"])
        out.append(
            {
                "id": r["id"],
                "model": r["model"],
                "author": r["author"],
                "source": r["source"],
                "split": r["split"],
                "metrics": m,
                "fit_time_s": r["fit_time_s"],
                "predict_time_s": r["predict_time_s"],
                "dataset_version": r["dataset_version"],
                "created_at": r["created_at"],
            }
        )
    out.sort(key=lambda e: e["metrics"][metric])
    for rank, e in enumerate(out, 1):
        e["rank"] = rank
    return out


def reset(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM entries")
