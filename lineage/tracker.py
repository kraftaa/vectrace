"""Lineage tracker: records provenance during ingestion and embedding."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Iterable

from lineage.models import ChunkRecord, VectorRecord, create_connection, initialize_db


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class LineageTracker:
    def __init__(self, db_path: str = "./vectrace.db", autoinit: bool = False):
        if autoinit:
            initialize_db(db_path)
        self.conn = create_connection(db_path)
        self.batch_id = str(uuid.uuid4())
        self.pipeline_run_id: str | None = None

    def __enter__(self) -> "LineageTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.pipeline_run_id is not None:
            status = "success" if exc_type is None else "failed"
            self.complete_pipeline(status=status)
        self.close()

    def start_pipeline(self, name: str) -> str:
        if self.pipeline_run_id is not None:
            raise RuntimeError(
                "A pipeline run is already active. Call complete_pipeline() before start_pipeline()."
            )
        run_id = str(uuid.uuid4())
        self.batch_id = str(uuid.uuid4())
        started_at = _utc_now_iso()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO pipeline_runs (id, name, started_at, status)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, name, started_at, "running"),
            )
        self.pipeline_run_id = run_id
        return run_id

    def complete_pipeline(self, status: str = "success") -> None:
        if not self.pipeline_run_id:
            return
        run_id = self.pipeline_run_id
        with self.conn:
            self.conn.execute(
                """
                UPDATE pipeline_runs
                SET completed_at = ?, status = ?
                WHERE id = ?
                """,
                (_utc_now_iso(), status, run_id),
            )
        self.pipeline_run_id = None

    def record_document(
        self,
        doc_id: str,
        source_path: str,
        source_type: str,
        version: str = "v1",
        content_hash: str | None = None,
        source_url: str | None = None,
        source_page: int | None = None,
        source_section: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO documents (
                    id, source_path, source_type, version, content_hash,
                    source_url, source_page, source_section
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_path = excluded.source_path,
                    source_type = excluded.source_type,
                    version = excluded.version,
                    content_hash = excluded.content_hash,
                    source_url = excluded.source_url,
                    source_page = excluded.source_page,
                    source_section = excluded.source_section
                """,
                (
                    doc_id,
                    source_path,
                    source_type,
                    version,
                    content_hash,
                    source_url,
                    source_page,
                    source_section,
                ),
            )

    def record_chunk(
        self,
        chunk_id: str,
        document_id: str,
        chunk_index: int,
        strategy: str,
        chunk_size: int | None,
        text_preview: str,
    ) -> None:
        self.record_chunks(
            [
                ChunkRecord(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    text_preview=text_preview[:500],
                    pipeline_run_id=self.pipeline_run_id,
                )
            ]
        )

    def record_chunks(self, chunks: Iterable[ChunkRecord]) -> None:
        rows = [
            (
                chunk.id,
                chunk.document_id,
                chunk.pipeline_run_id,
                chunk.chunk_index,
                chunk.strategy,
                chunk.chunk_size,
                chunk.text_preview[:500],
            )
            for chunk in chunks
        ]
        if not rows:
            return
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO chunks (
                    id, document_id, pipeline_run_id, chunk_index, strategy, chunk_size, text_preview
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    document_id = excluded.document_id,
                    pipeline_run_id = excluded.pipeline_run_id,
                    chunk_index = excluded.chunk_index,
                    strategy = excluded.strategy,
                    chunk_size = excluded.chunk_size,
                    text_preview = excluded.text_preview
                """,
                rows,
            )

    def record_vector(
        self,
        vector_id: str,
        collection_name: str,
        chunk_id: str,
        embedding_model: str,
        model_version: str | None = None,
    ) -> None:
        self.record_vectors(
            [
                VectorRecord(
                    id=vector_id,
                    collection_name=collection_name,
                    chunk_id=chunk_id,
                    embedding_model=embedding_model,
                    model_version=model_version,
                    batch_id=self.batch_id,
                    pipeline_run_id=self.pipeline_run_id,
                )
            ]
        )

    def record_vectors(self, vectors: Iterable[VectorRecord]) -> None:
        now = _utc_now_iso()
        rows = [
            (
                vector.id,
                vector.collection_name,
                vector.chunk_id,
                vector.pipeline_run_id,
                vector.embedding_model,
                vector.model_version,
                vector.batch_id,
                now,
            )
            for vector in vectors
        ]
        if not rows:
            return
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO vectors (
                    id, collection_name, chunk_id, pipeline_run_id,
                    embedding_model, model_version, batch_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(collection_name, id) DO UPDATE SET
                    chunk_id = excluded.chunk_id,
                    pipeline_run_id = excluded.pipeline_run_id,
                    embedding_model = excluded.embedding_model,
                    model_version = excluded.model_version,
                    batch_id = excluded.batch_id,
                    created_at = excluded.created_at
                """,
                rows,
            )

    def record_retrieval_event(
        self,
        query_id: str,
        query_text: str,
        collection_name: str,
        vector_id: str,
        rank: int | None = None,
        score: float | None = None,
        final_answer: str | None = None,
        metadata_json: str | None = None,
        event_id: str | None = None,
    ) -> str:
        if rank is not None and rank <= 0:
            raise ValueError("rank must be > 0")
        stored_event_id = event_id or str(uuid.uuid4())
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO retrieval_events (
                    id, query_id, query_text, final_answer, collection_name, vector_id,
                    rank, score, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stored_event_id,
                    query_id,
                    query_text,
                    final_answer,
                    collection_name,
                    vector_id,
                    rank,
                    score,
                    metadata_json,
                    _utc_now_iso(),
                ),
            )
        return stored_event_id

    def close(self) -> None:
        self.conn.close()
