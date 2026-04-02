"""SQLite models and schema helpers for VecTrace."""

from __future__ import annotations

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
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def load_schema_sql() -> str:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing schema file: {SCHEMA_PATH}")
    return SCHEMA_PATH.read_text(encoding="utf-8")


def initialize_db(db_path: str = DEFAULT_DB_PATH) -> None:
    conn = create_connection(db_path)
    try:
        conn.executescript(load_schema_sql())
        conn.commit()
    finally:
        conn.close()
