"""Demo-data seeding and bootstrap-event recording for VecTrace."""

from __future__ import annotations

import json
import sqlite3

from lineage.models import ChunkRecord, VectorRecord
from lineage.query import LineageQuery
from lineage.tracker import LineageTracker


def db_has_vectors(db_path: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return bool(row and row[0] > 0)
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def vector_exists(db_path: str, vector_id: str, collection: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM vectors WHERE id = ? AND collection_name = ? LIMIT 1",
            (vector_id, collection),
        ).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def infer_next_start_index(db_path: str, collection: str, prefix: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id FROM vectors WHERE collection_name = ? AND id LIKE ?",
            (collection, f"{prefix}%"),
        ).fetchall()
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()

    max_index = -1
    for row in rows:
        vector_id = row[0]
        if not isinstance(vector_id, str) or not vector_id.startswith(prefix):
            continue
        suffix = vector_id[len(prefix):]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def seed_demo_lineage(db_path: str, vector_id: str, collection: str) -> None:
    # Make demo IDs vector-specific so repeated onboarding cannot collide.
    safe_vector = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in vector_id)
    doc_id = f"demo_doc_{safe_vector}"
    chunk_id = f"{doc_id}:chunk:0"

    with LineageTracker(db_path, autoinit=False) as tracker:
        tracker.start_pipeline("onboard_demo")
        tracker.record_document(
            doc_id=doc_id,
            source_path="s3://demo/customer_refunds_policy.pdf",
            source_type="s3",
            version="v1",
            content_hash="sha256:demo",
        )
        tracker.record_chunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            chunk_index=0,
            strategy="semantic",
            chunk_size=62,
            text_preview="Customers can request a refund within 30 days if the item is defective.",
        )
        tracker.record_vector(
            vector_id=vector_id,
            collection_name=collection,
            chunk_id=chunk_id,
            embedding_model="text-embedding-3-small",
            model_version="2024-06-01",
        )
        tracker.complete_pipeline("success")


def seed_bulk_demo_lineage(
    db_path: str,
    collection: str,
    vectors: int,
    docs: int,
    start_index: int,
    prefix: str,
) -> tuple[str, str]:
    if vectors <= 0:
        raise ValueError("--vectors must be > 0")
    if docs <= 0:
        raise ValueError("--docs must be > 0")
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")

    topics = [
        "refund eligibility",
        "warranty duration",
        "shipping delay policy",
        "account suspension process",
        "chargeback dispute workflow",
        "subscription cancellation",
        "security incident response",
        "data retention policy",
        "onboarding checklist",
        "service-level agreement",
    ]

    with LineageTracker(db_path, autoinit=False) as tracker:
        tracker.start_pipeline("seed_demo_bulk")
        for doc_idx in range(docs):
            doc_id = f"demo_doc_{doc_idx:03d}"
            tracker.record_document(
                doc_id=doc_id,
                source_path=f"s3://demo/{doc_id}.pdf",
                source_type="s3",
                version=f"v{1 + (doc_idx % 3)}",
                content_hash=f"sha256:demo-{doc_idx:03d}",
            )

        chunk_rows: list[ChunkRecord] = []
        vector_rows: list[VectorRecord] = []
        first_vector_id = ""
        last_vector_id = ""
        for offset in range(vectors):
            seq = start_index + offset
            doc_idx = seq % docs
            doc_id = f"demo_doc_{doc_idx:03d}"
            topic = topics[seq % len(topics)]
            vector_id = f"{prefix}{seq:05d}"
            # Include vector_id so repeated seed runs with different prefixes append cleanly.
            chunk_id = f"{doc_id}:chunk:{vector_id}"
            text_preview = (
                f"Demo chunk {seq} on {topic}. This synthetic record is generated for VecTrace demos."
            )
            chunk_rows.append(
                ChunkRecord(
                    id=chunk_id,
                    document_id=doc_id,
                    chunk_index=seq,
                    strategy="semantic",
                    chunk_size=len(text_preview),
                    text_preview=text_preview,
                    pipeline_run_id=tracker.pipeline_run_id,
                )
            )
            vector_rows.append(
                VectorRecord(
                    id=vector_id,
                    collection_name=collection,
                    chunk_id=chunk_id,
                    embedding_model="text-embedding-3-small",
                    model_version="2024-06-01",
                    batch_id=tracker.batch_id,
                    pipeline_run_id=tracker.pipeline_run_id,
                )
            )
            if not first_vector_id:
                first_vector_id = vector_id
            last_vector_id = vector_id

        tracker.record_chunks(chunk_rows)
        tracker.record_vectors(vector_rows)
        tracker.complete_pipeline("success")
    return first_vector_id, last_vector_id


def parse_metadata_json(metadata_json: str | None) -> tuple[dict | list | None, str | None]:
    if metadata_json is None:
        return None, None
    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        return None, f"--metadata-json must be valid JSON: {exc}"
    return parsed, None


def record_bootstrap_events(
    db_path: str,
    question: str,
    final_answer: str,
    collection: str,
    top_k: int,
    query_id: str,
    input_metadata: dict | list | None = None,
) -> tuple[list[str], list[dict]]:
    with LineageQuery(db_path) as query:
        candidates = query.find_trace_candidates(
            question=question,
            collection_name=collection,
            limit=top_k,
        )

    if not candidates:
        return [], []

    event_ids: list[str] = []
    with LineageTracker(db_path, autoinit=False) as tracker:
        for rank, candidate in enumerate(candidates, start=1):
            metadata_obj: dict[str, object] = {}
            if input_metadata is not None:
                if isinstance(input_metadata, dict):
                    metadata_obj.update(input_metadata)
                else:
                    metadata_obj["input_metadata"] = input_metadata
            metadata_obj.update(
                {
                    "trace_mode": "bootstrap",
                    "score_type": "lexical_overlap_bootstrap",
                    "match_terms": candidate["overlap_terms"],
                    "chunk_id": candidate["chunk_id"],
                    "source_path": candidate["source_path"],
                    "evidence_text": candidate["text_preview"],
                }
            )
            event_id = tracker.record_retrieval_event(
                query_id=query_id,
                query_text=question,
                final_answer=final_answer,
                collection_name=candidate["collection_name"],
                vector_id=candidate["vector_id"],
                rank=rank,
                score=float(candidate["score"]),
                metadata_json=json.dumps(metadata_obj, sort_keys=True),
            )
            event_ids.append(event_id)
    return event_ids, candidates
