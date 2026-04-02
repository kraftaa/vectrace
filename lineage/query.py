"""Lineage query helpers."""

from __future__ import annotations

import json
from typing import Any

from lineage.models import create_connection


class AmbiguousVectorIDError(ValueError):
    """Raised when the same vector ID exists in multiple collections."""


class LineageQuery:
    def __init__(self, db_path: str = "./vectrace.db"):
        self.conn = create_connection(db_path)

    def __enter__(self) -> "LineageQuery":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_lineage(
        self, vector_id: str, collection_name: str | None = None
    ) -> dict[str, Any] | None:
        base_sql = """
            SELECT
                v.id AS vector_id,
                v.collection_name,
                v.embedding_model,
                v.model_version,
                v.batch_id,
                v.pipeline_run_id,
                v.created_at AS vector_created_at,
                c.id AS chunk_id,
                c.chunk_index,
                c.strategy AS chunk_strategy,
                c.chunk_size,
                c.text_preview,
                d.id AS document_id,
                d.source_path,
                d.source_type,
                d.version AS document_version,
                d.content_hash
            FROM vectors v
            JOIN chunks c ON v.chunk_id = c.id
            JOIN documents d ON c.document_id = d.id
            WHERE v.id = ?
        """
        params: list[Any] = [vector_id]
        if collection_name:
            base_sql += " AND v.collection_name = ?"
            params.append(collection_name)
        rows = self.conn.execute(base_sql, params).fetchall()

        if not rows:
            return None
        if collection_name is None and len(rows) > 1:
            collections = ", ".join(sorted({row["collection_name"] for row in rows}))
            raise AmbiguousVectorIDError(
                f"Vector ID '{vector_id}' exists in multiple collections: {collections}"
            )

        row = rows[0]
        return {
            "vector": {
                "id": row["vector_id"],
                "collection_name": row["collection_name"],
                "embedding_model": row["embedding_model"],
                "model_version": row["model_version"],
                "batch_id": row["batch_id"],
                "pipeline_run_id": row["pipeline_run_id"],
                "created_at": row["vector_created_at"],
            },
            "chunk": {
                "id": row["chunk_id"],
                "index": row["chunk_index"],
                "strategy": row["chunk_strategy"],
                "size": row["chunk_size"],
                "text_preview": row["text_preview"],
            },
            "document": {
                "id": row["document_id"],
                "source_path": row["source_path"],
                "source_type": row["source_type"],
                "version": row["document_version"],
                "content_hash": row["content_hash"],
            },
        }

    def get_vectors_by_document(self, document_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
                v.id,
                v.collection_name,
                v.embedding_model,
                v.model_version,
                v.pipeline_run_id,
                v.created_at
            FROM vectors v
            JOIN chunks c ON v.chunk_id = c.id
            WHERE c.document_id = ?
            ORDER BY v.created_at DESC
            """,
            (document_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_latest_retrieval_event(
        self, vector_id: str, collection_name: str
    ) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT
                id,
                query_id,
                query_text,
                final_answer,
                collection_name,
                vector_id,
                rank,
                score,
                metadata_json,
                created_at
            FROM retrieval_events
            WHERE vector_id = ? AND collection_name = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (vector_id, collection_name),
        ).fetchone()
        if row is None:
            return None
        metadata = None
        if row["metadata_json"]:
            try:
                metadata = json.loads(row["metadata_json"])
            except json.JSONDecodeError:
                metadata = row["metadata_json"]
        return {
            "id": row["id"],
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "final_answer": row["final_answer"],
            "collection_name": row["collection_name"],
            "vector_id": row["vector_id"],
            "rank": row["rank"],
            "score": row["score"],
            "metadata": metadata,
            "created_at": row["created_at"],
        }

    def close(self) -> None:
        self.conn.close()
