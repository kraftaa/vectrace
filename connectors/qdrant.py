"""Qdrant connector with lineage tracking injection."""

from __future__ import annotations

import json
import sys
import uuid
from typing import Any, Iterable

from lineage.models import ChunkRecord, VectorRecord
from lineage.tracker import LineageTracker

try:
    from qdrant_client import QdrantClient
except ImportError:  # pragma: no cover - optional dependency
    QdrantClient = None

try:
    from qdrant_client.http import models as qdrant_models
except ImportError:  # pragma: no cover - optional dependency
    qdrant_models = None


def _require_qdrant() -> None:
    if QdrantClient is None:
        raise RuntimeError(
            "qdrant-client is not installed. Install with: python3 -m pip install -e '.[qdrant]'"
        )


def _infer_source_type(path: str) -> str:
    lowered = path.lower()
    if lowered.startswith("s3://"):
        return "s3"
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return "http"
    if lowered.startswith("notion://"):
        return "notion"
    return "local"


def test_connection(qdrant_url: str, collection_name: str, api_key: str | None = None) -> None:
    _require_qdrant()
    client = QdrantClient(url=qdrant_url, api_key=api_key)
    try:
        collections = client.get_collections()
        known = {c.name for c in collections.collections}
        if collection_name not in known:
            raise RuntimeError(
                f"Collection '{collection_name}' not found in {qdrant_url}. "
                f"Available collections: {', '.join(sorted(known)) or '(none)'}"
            )
    finally:
        client.close()


class TrackedQdrant:
    """Drop-in Qdrant wrapper that records vector lineage to SQLite."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        db_path: str = "./vectrace.db",
        api_key: str | None = None,
        client: Any | None = None,
        qdrant_location: str | None = None,
    ):
        if client is not None:
            self.client = client
            self._owns_client = False
        elif qdrant_location is not None:
            _require_qdrant()
            self.client = QdrantClient(location=qdrant_location, api_key=api_key)
            self._owns_client = True
        else:
            _require_qdrant()
            self.client = QdrantClient(url=qdrant_url, api_key=api_key)
            self._owns_client = True
        self.tracker = LineageTracker(db_path=db_path, autoinit=True)
        self.last_tracking_errors: list[str] = []

    def __enter__(self) -> "TrackedQdrant":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def upsert_with_lineage(
        self,
        collection_name: str,
        points: Iterable[Any],
        document_id: str,
        document_path: str,
        chunk_strategy: str = "fixed-size",
        embedding_model: str = "unknown",
        model_version: str | None = None,
        source_type: str | None = None,
        document_version: str = "v1",
        content_hash: str | None = None,
        pipeline_name: str = "qdrant_upsert",
    ) -> None:
        point_list = list(points)
        if not point_list:
            return
        point_ids = [point.id for point in point_list]

        self.tracker.start_pipeline(pipeline_name)

        try:
            chunk_rows: list[ChunkRecord] = []
            vector_rows: list[VectorRecord] = []
            for idx, point in enumerate(point_list):
                payload = getattr(point, "payload", {}) or {}
                full_text = str(payload.get("text", ""))
                text_preview = full_text[:500]
                chunk_index = payload.get("chunk_index", idx)
                try:
                    chunk_index = int(chunk_index)
                except (TypeError, ValueError):
                    chunk_index = idx

                vector_id = str(point.id)
                chunk_id = f"{document_id}:chunk:{vector_id}"
                chunk_rows.append(
                    ChunkRecord(
                        id=chunk_id,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        strategy=chunk_strategy,
                        chunk_size=len(full_text),
                        text_preview=text_preview,
                        pipeline_run_id=self.tracker.pipeline_run_id,
                    )
                )
                vector_rows.append(
                    VectorRecord(
                        id=vector_id,
                        collection_name=collection_name,
                        chunk_id=chunk_id,
                        embedding_model=embedding_model,
                        model_version=model_version,
                        batch_id=self.tracker.batch_id,
                        pipeline_run_id=self.tracker.pipeline_run_id,
                    )
                )

            self.client.upsert(collection_name=collection_name, points=point_list)
        except Exception:
            self.tracker.complete_pipeline(status="failed")
            raise

        try:
            self.tracker.record_document(
                doc_id=document_id,
                source_path=document_path,
                source_type=source_type or _infer_source_type(document_path),
                version=document_version,
                content_hash=content_hash,
            )
            self.tracker.record_chunks(chunk_rows)
            self.tracker.record_vectors(vector_rows)
        except Exception:
            # Best-effort rollback to reduce cross-store drift when lineage write fails.
            self._rollback_qdrant_points(collection_name=collection_name, point_ids=point_ids)
            self.tracker.complete_pipeline(status="failed")
            raise
        self.tracker.complete_pipeline(status="success")

    def search_with_tracking(
        self,
        collection_name: str,
        query_text: str,
        query_vector: list[float],
        limit: int = 5,
        final_answer: str | None = None,
        query_id: str | None = None,
        metadata: dict | None = None,
        **search_kwargs: Any,
    ) -> tuple[list[Any], str]:
        """Run ``client.search()`` and record each hit as a retrieval event.

        Lineage writes are best-effort: a SQLite failure here will not raise,
        because the search itself has already returned results to the caller's
        RAG pipeline. Failures are reported on stderr and collected in
        ``self.last_tracking_errors`` for programmatic handling.

        Returns ``(hits, query_id)``. ``query_id`` is auto-generated when omitted
        so the caller can later attach a final answer or correlate logs.
        """
        hits = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            **search_kwargs,
        )
        resolved_query_id = query_id or str(uuid.uuid4())
        base_metadata: dict[str, Any] = {"trace_mode": "exact"}
        if metadata:
            base_metadata.update(metadata)
        self.last_tracking_errors = []

        for rank, hit in enumerate(hits, start=1):
            hit_metadata = dict(base_metadata)
            payload = getattr(hit, "payload", None)
            if isinstance(payload, dict):
                evidence_text = payload.get("text") or payload.get("evidence_text")
                if isinstance(evidence_text, str) and evidence_text:
                    hit_metadata["evidence_text"] = evidence_text[:500]
            try:
                self.tracker.record_retrieval_event(
                    query_id=resolved_query_id,
                    query_text=query_text,
                    final_answer=final_answer,
                    collection_name=collection_name,
                    vector_id=str(hit.id),
                    rank=rank,
                    score=float(hit.score) if hit.score is not None else None,
                    metadata_json=json.dumps(hit_metadata, sort_keys=True),
                )
            except Exception as exc:
                error_msg = (
                    f"vectrace: failed to record retrieval event for vector "
                    f"'{hit.id}' (rank={rank}): {exc}"
                )
                self.last_tracking_errors.append(error_msg)
                print(error_msg, file=sys.stderr)
        return hits, resolved_query_id

    def _rollback_qdrant_points(self, collection_name: str, point_ids: list[Any]) -> None:
        if not point_ids:
            return

        selectors: list[Any] = [list(point_ids)]
        if qdrant_models is not None and hasattr(qdrant_models, "PointIdsList"):
            try:
                selectors.append(qdrant_models.PointIdsList(points=list(point_ids)))
            except Exception:
                pass

        attempts = [
            lambda sel: self.client.delete(collection_name=collection_name, points_selector=sel),
            lambda sel: self.client.delete(collection_name, sel),
        ]

        try:
            for selector in selectors:
                for attempt in attempts:
                    try:
                        attempt(selector)
                        return
                    except TypeError:
                        continue
                    except Exception:
                        continue
        except Exception:
            pass
        # Rollback is best-effort; caller handles the primary lineage exception.

    def close(self) -> None:
        self.tracker.close()
        if self._owns_client:
            self.client.close()
