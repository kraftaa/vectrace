"""SQLite models and schema helpers for VecTrace."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_PATH = "./vectrace.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    document_id: str
    chunk_index: int
    strategy: str
    chunk_size: int | None
    text_preview: str
    pipeline_run_id: str | None = None


@dataclass(frozen=True)
class VectorRecord:
    id: str
    collection_name: str
    chunk_id: str
    embedding_model: str
    model_version: str | None = None
    batch_id: str | None = None
    pipeline_run_id: str | None = None


def create_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    # WAL mode reduces writer contention for CLI + ingestion workloads.
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def load_schema_sql() -> str:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing schema file: {SCHEMA_PATH}")
    return SCHEMA_PATH.read_text(encoding="utf-8")


_DOCUMENT_DEEP_LINK_COLUMNS: tuple[tuple[str, str], ...] = (
    ("source_url", "TEXT"),
    ("source_page", "INTEGER"),
    ("source_section", "TEXT"),
)

# Column identifiers cannot be parameterized in SQLite — they must be
# interpolated into the SQL string. These guards keep that interpolation
# safe even if someone later sources column definitions from config.
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_ALLOWED_COLUMN_TYPES = frozenset({"TEXT", "INTEGER", "REAL", "BLOB", "NUMERIC"})


def _apply_idempotent_alters(conn: sqlite3.Connection) -> None:
    """Add new optional columns to existing DBs.

    Until a schema_version migration framework lands, this is how we
    forward-port older databases. Only the "duplicate column" error is
    swallowed; "database is locked", schema corruption, and other genuine
    failures must propagate so callers don't get a misleading success.
    """
    for column, column_type in _DOCUMENT_DEEP_LINK_COLUMNS:
        if not _SAFE_IDENTIFIER_RE.match(column):
            raise ValueError(f"Unsafe column identifier: {column!r}")
        if column_type not in _ALLOWED_COLUMN_TYPES:
            raise ValueError(f"Unsafe column type: {column_type!r}")
        try:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {column} {column_type}")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc):
                raise


def initialize_db(db_path: str = DEFAULT_DB_PATH) -> None:
    conn = create_connection(db_path)
    try:
        conn.executescript(load_schema_sql())
        _apply_idempotent_alters(conn)
        conn.commit()
    finally:
        conn.close()
