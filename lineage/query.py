"""Lineage query helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from lineage.models import create_connection

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "who",
    "with",
    "you",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 1 and token not in _STOP_WORDS}


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
        return self._row_to_retrieval_event(row)

    def get_retrieval_event(self, event_id: str) -> dict[str, Any] | None:
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
            WHERE id = ?
            LIMIT 1
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_retrieval_event(row)

    def find_retrieval_events(
        self,
        query_text: str,
        final_answer: str | None = None,
        collection_name: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        sql = """
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
            WHERE query_text = ?
        """
        params: list[Any] = [query_text]
        if final_answer is not None:
            sql += " AND final_answer = ?"
            params.append(final_answer)
        if collection_name is not None:
            sql += " AND collection_name = ?"
            params.append(collection_name)
        sql += """
            ORDER BY
                created_at DESC,
                CASE WHEN rank IS NULL THEN 1 ELSE 0 END ASC,
                rank ASC,
                id DESC
            LIMIT ?
        """
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_retrieval_event(row) for row in rows]

    def find_trace_candidates(
        self, question: str, collection_name: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        question_terms = _tokenize(question)
        if not question_terms:
            return []

        search_terms = sorted(question_terms)[:8]
        sql = """
            SELECT
                v.id AS vector_id,
                v.collection_name,
                c.id AS chunk_id,
                c.chunk_index,
                c.text_preview,
                d.source_path
            FROM vectors v
            JOIN chunks c ON v.chunk_id = c.id
            JOIN documents d ON c.document_id = d.id
            WHERE v.collection_name = ?
        """
        params: list[Any] = [collection_name]
        if search_terms:
            like_clause = " OR ".join(["LOWER(c.text_preview) LIKE ?"] * len(search_terms))
            sql += f" AND ({like_clause})"
            params.extend([f"%{term}%" for term in search_terms])
        prefetch_limit = max(limit * 80, 300)
        sql += """
            ORDER BY
                CASE WHEN c.chunk_index IS NULL THEN 1 ELSE 0 END ASC,
                c.chunk_index ASC,
                v.id ASC
            LIMIT ?
        """
        params.append(prefetch_limit)

        rows = self.conn.execute(
            sql,
            params,
        ).fetchall()
        scored: list[dict[str, Any]] = []
        for row in rows:
            preview = row["text_preview"] or ""
            chunk_terms = _tokenize(preview)
            overlap = sorted(question_terms.intersection(chunk_terms))
            if not overlap:
                continue
            score = len(overlap) / len(question_terms)
            scored.append(
                {
                    "vector_id": row["vector_id"],
                    "collection_name": row["collection_name"],
                    "chunk_id": row["chunk_id"],
                    "chunk_index": row["chunk_index"],
                    "text_preview": preview,
                    "source_path": row["source_path"],
                    "score": round(score, 6),
                    "overlap_terms": overlap,
                }
            )
        scored.sort(
            key=lambda item: (
                -item["score"],
                item["chunk_index"] if item["chunk_index"] is not None else 0,
                item["vector_id"],
            )
        )
        return scored[:limit]

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _row_to_retrieval_event(row: Any) -> dict[str, Any]:
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
