#!/usr/bin/env python3
"""VecTrace CLI."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import uuid

from lineage.models import ChunkRecord, VectorRecord, initialize_db
from lineage.query import AmbiguousVectorIDError, LineageQuery
from lineage.tracker import LineageTracker
from output.html import generate_report


def _build_parser() -> argparse.ArgumentParser:
    epilog = """Start here (core):
  vectrace ask-trace --db ./vectrace.db --collection support_kb --question "Can I get a refund after 90 days?"
  vectrace trace --db ./vectrace.db --vector-id vec_101 --collection support_kb
  vectrace report --db ./vectrace.db --vector-id vec_101 --collection support_kb --output ./trace.html

Advanced / integration:
  vectrace record-retrieval --db ./vectrace.db --collection support_kb --vector-id vec_101 --query-text "..."
  vectrace record-qa --db ./vectrace.db --collection support_kb --question "..." --final-answer "..."
"""
    parser = argparse.ArgumentParser(
        description="VecTrace: trace where your RAG answers come from.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, title="commands")

    init_parser = subparsers.add_parser("init", help="[advanced] Initialize VecTrace SQLite schema.")
    init_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")

    onboard_parser = subparsers.add_parser(
        "onboard", help="[advanced] Initialize, seed demo trace data, and generate a shareable report."
    )
    onboard_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")
    onboard_parser.add_argument(
        "--vector-id", default="vec_demo_001", help="Demo vector ID for seeded trace data."
    )
    onboard_parser.add_argument(
        "--collection", default="support_kb", help="Collection name for the demo vector."
    )
    onboard_parser.add_argument(
        "--output",
        default="trace-demo.html",
        help="Output HTML report path generated during onboarding.",
    )
    onboard_parser.add_argument(
        "--redact-preview",
        action="store_true",
        help="Redact chunk preview text in the generated demo report.",
    )
    onboard_parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Only initialize schema; do not seed demo trace data or generate report.",
    )

    seed_demo_parser = subparsers.add_parser(
        "seed-demo", help="[advanced] Append larger synthetic trace data into a database for demos."
    )
    seed_demo_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")
    seed_demo_parser.add_argument(
        "--collection", default="support_kb", help="Collection name for synthetic vectors."
    )
    seed_demo_parser.add_argument(
        "--vectors", type=int, default=100, help="Number of vectors to append."
    )
    seed_demo_parser.add_argument(
        "--docs", type=int, default=10, help="Number of synthetic source documents."
    )
    seed_demo_parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Starting numeric index for generated vector IDs. Auto-inferred when omitted.",
    )
    seed_demo_parser.add_argument(
        "--prefix", default="vec_demo_", help="Vector ID prefix for generated records."
    )

    record_retrieval_parser = subparsers.add_parser(
        "record-retrieval",
        help="[advanced] Record retrieval telemetry (query text, rank/score, final answer) for a vector.",
    )
    record_retrieval_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")
    record_retrieval_parser.add_argument(
        "--query-id",
        default=None,
        help="Optional retrieval query identifier. Auto-generated if omitted.",
    )
    record_retrieval_parser.add_argument(
        "--query-text", required=True, help="User query text used for retrieval."
    )
    record_retrieval_parser.add_argument(
        "--final-answer", default=None, help="Final model answer text (optional)."
    )
    record_retrieval_parser.add_argument(
        "--collection", required=True, help="Collection name containing the vector."
    )
    record_retrieval_parser.add_argument(
        "--vector-id", required=True, help="Retrieved vector ID."
    )
    record_retrieval_parser.add_argument(
        "--rank", type=int, default=1, help="Retrieval rank for this vector (1-based)."
    )
    record_retrieval_parser.add_argument(
        "--score", type=float, default=None, help="Retrieval score for this vector."
    )
    record_retrieval_parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON metadata string (for additional retrieval context).",
    )
    record_retrieval_parser.add_argument(
        "--evidence-text",
        default=None,
        help="Optional exact evidence text snippet supporting the final answer.",
    )

    record_qa_parser = subparsers.add_parser(
        "record-qa",
        help="[advanced] Auto-record retrieval events from question/answer by matching question terms to chunk text.",
    )
    record_qa_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")
    record_qa_parser.add_argument(
        "--question", required=True, help="User question text."
    )
    record_qa_parser.add_argument(
        "--final-answer",
        required=True,
        help="Final answer text to attach to recorded retrieval events.",
    )
    record_qa_parser.add_argument(
        "--collection", required=True, help="Collection name to search for candidate vectors."
    )
    record_qa_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many matched vectors to record as retrieval events.",
    )
    record_qa_parser.add_argument(
        "--query-id",
        default=None,
        help="Optional retrieval query identifier. Auto-generated if omitted.",
    )
    record_qa_parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON metadata object merged into each recorded event.",
    )

    ask_trace_parser = subparsers.add_parser(
        "ask-trace",
        help=(
            "[core] One-command bootstrap flow: map question -> likely vectors, "
            "record retrieval events, and generate trace report/json."
        ),
    )
    ask_trace_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")
    ask_trace_parser.add_argument(
        "--question", required=True, help="User question text."
    )
    ask_trace_parser.add_argument(
        "--collection", required=True, help="Collection name to search for candidate vectors."
    )
    ask_trace_parser.add_argument(
        "--final-answer",
        default=None,
        help=(
            "Optional final answer text. If omitted, uses top candidate snippet "
            "as bootstrap extractive answer."
        ),
    )
    ask_trace_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many matched vectors to record as retrieval events.",
    )
    ask_trace_parser.add_argument(
        "--match-index",
        type=int,
        default=1,
        help="1-based match index from recorded events to render.",
    )
    ask_trace_parser.add_argument(
        "--query-id",
        default=None,
        help="Optional retrieval query identifier. Auto-generated if omitted.",
    )
    ask_trace_parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON metadata object merged into each recorded event.",
    )
    ask_trace_parser.add_argument(
        "--output",
        default="ask-trace.html",
        help="Output HTML path.",
    )
    ask_trace_parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON output path to write selected trace payload.",
    )
    ask_trace_parser.add_argument(
        "--redact-preview",
        action="store_true",
        help="Redact chunk preview text before rendering report.",
    )
    ask_trace_parser.add_argument(
        "--redact-retrieval",
        action="store_true",
        help="Redact retrieval query/final-answer fields in output.",
    )
    ask_trace_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for terminal payload.",
    )

    trace_qa_parser = subparsers.add_parser(
        "trace-qa",
        help="[advanced] Find recorded retrieval events by question/answer and show matched trace records.",
    )
    trace_qa_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")
    trace_qa_parser.add_argument("--question", required=True, help="Exact query text.")
    trace_qa_parser.add_argument(
        "--answer",
        default=None,
        help="Optional exact final answer filter.",
    )
    trace_qa_parser.add_argument(
        "--collection",
        default=None,
        help="Optional collection filter.",
    )
    trace_qa_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of matching retrieval events to return.",
    )
    trace_qa_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    trace_qa_parser.add_argument(
        "--redact-preview",
        action="store_true",
        help="Redact chunk preview text from output.",
    )
    trace_qa_parser.add_argument(
        "--redact-retrieval",
        action="store_true",
        help="Redact retrieval query/final-answer fields in output.",
    )

    report_qa_parser = subparsers.add_parser(
        "report-qa",
        help="[advanced] Find recorded retrieval events by question/answer and generate an HTML trace report.",
    )
    report_qa_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")
    report_qa_parser.add_argument("--question", required=True, help="Exact query text.")
    report_qa_parser.add_argument(
        "--answer",
        default=None,
        help="Optional exact final answer filter.",
    )
    report_qa_parser.add_argument(
        "--collection",
        default=None,
        help="Optional collection filter.",
    )
    report_qa_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many retrieval matches to search before selecting one.",
    )
    report_qa_parser.add_argument(
        "--match-index",
        type=int,
        default=1,
        help="1-based match index to render from the retrieval match list.",
    )
    report_qa_parser.add_argument(
        "--output",
        default="qa-trace.html",
        help="Output HTML path.",
    )
    report_qa_parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON output path to write the selected QA trace payload.",
    )
    report_qa_parser.add_argument(
        "--redact-preview",
        action="store_true",
        help="Redact chunk preview text before rendering report.",
    )
    report_qa_parser.add_argument(
        "--redact-retrieval",
        action="store_true",
        help="Redact retrieval query/final-answer fields in output.",
    )

    trace_parser = subparsers.add_parser(
        "trace", help="[core] Trace a vector back to chunk and source document."
    )
    trace_parser.add_argument("--vector-id", required=True, help="Vector ID to trace.")
    trace_parser.add_argument(
        "--collection", default=None, help="Optional collection name for disambiguation."
    )
    trace_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for trace details.",
    )
    trace_parser.add_argument(
        "--plain",
        action="store_true",
        help="Use plain text formatting for terminal/log output.",
    )
    trace_parser.add_argument(
        "--redact-preview",
        action="store_true",
        help="Redact chunk preview text from output.",
    )
    trace_parser.add_argument(
        "--redact-retrieval",
        action="store_true",
        help="Redact retrieval query/final-answer fields in output.",
    )
    trace_parser.add_argument(
        "--include-retrieval",
        action="store_true",
        help="Include latest recorded retrieval telemetry for this vector.",
    )
    trace_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")

    report_parser = subparsers.add_parser("report", help="[core] Generate an HTML trace report.")
    report_parser.add_argument("--vector-id", required=True, help="Vector ID to trace.")
    report_parser.add_argument(
        "--collection", default=None, help="Optional collection name for disambiguation."
    )
    report_parser.add_argument(
        "--redact-preview",
        action="store_true",
        help="Redact chunk preview text before rendering report.",
    )
    report_parser.add_argument(
        "--redact-retrieval",
        action="store_true",
        help="Redact retrieval query/final-answer fields in output.",
    )
    report_parser.add_argument(
        "--include-retrieval",
        action="store_true",
        help="Include latest recorded retrieval telemetry for this vector in the report.",
    )
    report_parser.add_argument("--output", default="trace.html", help="Output HTML path.")
    report_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")

    connect_parser = subparsers.add_parser(
        "connect", help="[advanced] Test Qdrant connectivity and verify collection exists."
    )
    connect_parser.add_argument(
        "--qdrant-url", default="http://localhost:6333", help="Qdrant URL."
    )
    connect_parser.add_argument("--qdrant-collection", required=True, help="Collection name.")
    connect_parser.add_argument("--api-key", default=None, help="Optional Qdrant API key.")

    return parser


def _prepare_lineage_output(lineage: dict, redact_preview: bool) -> dict:
    prepared = {
        "vector": dict(lineage["vector"]),
        "chunk": dict(lineage["chunk"]),
        "document": dict(lineage["document"]),
    }
    if redact_preview:
        preview = prepared["chunk"].get("text_preview")
        prepared["chunk"]["text_preview"] = (
            f"[REDACTED:{len(preview)} chars]" if preview else "[REDACTED]"
        )
    return prepared


def _redact_text(value: object) -> str:
    text = "" if value is None else str(value)
    return f"[REDACTED:{len(text)} chars]" if text else "[REDACTED]"


def _prepare_retrieval_output(
    retrieval: dict | None,
    redact_preview: bool,
    redact_retrieval: bool = False,
) -> dict | None:
    if retrieval is None:
        return None
    prepared = dict(retrieval)
    prepared["trace_mode"] = _trace_mode_from_retrieval(retrieval)
    if redact_retrieval:
        prepared["query_text"] = _redact_text(prepared.get("query_text"))
        prepared["final_answer"] = _redact_text(prepared.get("final_answer"))
    metadata = prepared.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        if redact_retrieval:
            for key in (
                "query_text",
                "final_answer",
                "question",
                "answer",
                "user_query",
                "assistant_answer",
            ):
                if key in metadata:
                    metadata[key] = _redact_text(metadata.get(key))
        if redact_preview and "evidence_text" in metadata:
            metadata["evidence_text"] = _redact_text(metadata.get("evidence_text"))
        prepared["metadata"] = metadata
    return prepared


def _build_evidence(
    trace_data: dict,
    retrieval: dict | None = None,
    redact_preview: bool = False,
    redact_retrieval: bool = False,
) -> dict:
    vector = trace_data["vector"]
    chunk = trace_data["chunk"]
    document = trace_data["document"]
    raw_evidence_text = chunk.get("text_preview")
    if retrieval:
        metadata = retrieval.get("metadata")
        if isinstance(metadata, dict) and metadata.get("evidence_text"):
            raw_evidence_text = metadata["evidence_text"]
    support = _assess_answer_support(retrieval=retrieval, evidence_text=raw_evidence_text)
    support_reason = support["reason"]
    support_details = support["details"]
    if redact_retrieval:
        method = None
        if isinstance(support_details, dict):
            method = support_details.get("method")
        support_reason = "Assessment computed; retrieval context redacted."
        support_details = {
            "method": method or "rule_plus_overlap",
            "redacted": True,
        }
    if redact_preview:
        evidence_text = _redact_text(raw_evidence_text)
    else:
        evidence_text = raw_evidence_text
    return {
        "vector_id": vector.get("id"),
        "collection_name": vector.get("collection_name"),
        "trace_mode": _trace_mode_from_retrieval(retrieval),
        "chunk_id": chunk.get("id"),
        "chunk_index": chunk.get("index"),
        "chunk_text": evidence_text,
        "source_document_id": document.get("id"),
        "source_path": document.get("source_path"),
        "retrieval_rank": retrieval.get("rank") if retrieval else None,
        "retrieval_score": retrieval.get("score") if retrieval else None,
        "support_status": support["status"],
        "support_reason": support_reason,
        "support_details": support_details,
    }


def _trace_mode_from_retrieval(retrieval: dict | None) -> str | None:
    if retrieval is None:
        return None
    metadata = retrieval.get("metadata")
    if isinstance(metadata, dict):
        mode = metadata.get("trace_mode")
        if isinstance(mode, str) and mode.strip():
            return mode.strip()
        if metadata.get("score_type") == "lexical_overlap_bootstrap":
            return "bootstrap"
    return "exact"


def _extract_day_constraint(text: str) -> tuple[str, int] | None:
    lowered = text.lower()
    patterns = [
        (r"\bafter\s+(\d+)\s+days?\b", "after"),
        (r"\bwithin\s+(\d+)\s+days?\b", "within"),
        (r"\bup to\s+(\d+)\s+days?\b", "within"),
    ]
    for pattern, mode in patterns:
        match = re.search(pattern, lowered)
        if match:
            try:
                return mode, int(match.group(1))
            except ValueError:
                return None
    return None


def _tokenize_support_text(text: str) -> set[str]:
    stop_words = {
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
        "get",
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
        "what",
        "with",
        "you",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in stop_words
    }


def _answer_polarity(answer: str) -> str:
    normalized = answer.strip().lower()
    if normalized.startswith(("yes", "yeah", "yep")):
        return "yes"
    if normalized.startswith(("no", "nope")):
        return "no"
    return "unknown"


def _assess_answer_support(retrieval: dict | None, evidence_text: str | None) -> dict:
    if retrieval is None:
        return {
            "status": "unclear",
            "reason": "No retrieval context was provided.",
            "details": {"method": "rule_plus_overlap"},
        }

    final_answer = retrieval.get("final_answer")
    query_text = retrieval.get("query_text")
    if not final_answer:
        return {
            "status": "unclear",
            "reason": "Final answer is missing.",
            "details": {"method": "rule_plus_overlap"},
        }
    if not evidence_text:
        return {
            "status": "unclear",
            "reason": "Evidence snippet is missing.",
            "details": {"method": "rule_plus_overlap"},
        }

    answer_polarity = _answer_polarity(str(final_answer))
    details: dict[str, object] = {"method": "rule_plus_overlap", "answer_polarity": answer_polarity}

    if query_text:
        query_constraint = _extract_day_constraint(str(query_text))
        evidence_constraint = _extract_day_constraint(str(evidence_text))
        if query_constraint and evidence_constraint:
            details["query_day_constraint"] = {
                "mode": query_constraint[0],
                "days": query_constraint[1],
            }
            details["evidence_day_constraint"] = {
                "mode": evidence_constraint[0],
                "days": evidence_constraint[1],
            }
            if (
                query_constraint[0] == "after"
                and evidence_constraint[0] == "within"
                and query_constraint[1] > evidence_constraint[1]
            ):
                if answer_polarity == "yes":
                    return {
                        "status": "unsupported",
                        "reason": (
                            f"Question asks about after {query_constraint[1]} days, "
                            f"but evidence limits to within {evidence_constraint[1]} days."
                        ),
                        "details": details,
                    }
                if answer_polarity == "no":
                    return {
                        "status": "supported",
                        "reason": (
                            f"Evidence limits refunds to within {evidence_constraint[1]} days; "
                            f"question asks about after {query_constraint[1]} days."
                        ),
                        "details": details,
                    }

    answer_tokens = _tokenize_support_text(str(final_answer))
    evidence_tokens = _tokenize_support_text(str(evidence_text))
    if answer_polarity in {"yes", "no"} and len(answer_tokens) <= 2:
        details["answer_tokens"] = sorted(answer_tokens)
        return {
            "status": "unclear",
            "reason": "Answer is polarity-only; not enough lexical content for overlap scoring.",
            "details": details,
        }
    if not answer_tokens:
        return {
            "status": "unclear",
            "reason": "Final answer lacks enough terms for comparison.",
            "details": details,
        }
    overlap = sorted(answer_tokens.intersection(evidence_tokens))
    overlap_ratio = len(overlap) / len(answer_tokens)
    details["answer_tokens"] = sorted(answer_tokens)
    details["overlap_terms"] = overlap
    details["overlap_ratio"] = round(overlap_ratio, 4)

    if overlap_ratio >= 0.7:
        return {
            "status": "supported",
            "reason": "Answer terms align with retrieved evidence snippet.",
            "details": details,
        }
    if overlap_ratio == 0:
        return {
            "status": "unsupported",
            "reason": "Answer terms have weak overlap with retrieved evidence snippet.",
            "details": details,
        }
    return {
        "status": "unclear",
        "reason": "Evidence overlap is partial; manual review recommended.",
        "details": details,
    }


def _db_has_vectors(db_path: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return bool(row and row[0] > 0)
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def _vector_exists(db_path: str, vector_id: str, collection: str) -> bool:
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


def _infer_next_start_index(db_path: str, collection: str, prefix: str) -> int:
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
        suffix = vector_id[len(prefix) :]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def _seed_demo_lineage(db_path: str, vector_id: str, collection: str) -> None:
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


def _seed_bulk_demo_lineage(
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


def _parse_metadata_json(metadata_json: str | None) -> tuple[dict | list | None, str | None]:
    if metadata_json is None:
        return None, None
    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        return None, f"--metadata-json must be valid JSON: {exc}"
    return parsed, None


def _record_bootstrap_events(
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


def _print_lineage(
    lineage: dict,
    plain: bool = False,
    retrieval: dict | None = None,
    evidence: dict | None = None,
) -> None:
    vector = lineage["vector"]
    chunk = lineage["chunk"]
    document = lineage["document"]

    if plain:
        print(f"vector.id={vector['id']}")
        print(f"vector.collection={vector['collection_name']}")
        print(f"vector.model={vector['embedding_model']}")
        print(f"vector.model_version={vector['model_version'] or '-'}")
        print(f"vector.batch_id={vector['batch_id'] or '-'}")
        print(f"vector.pipeline_run_id={vector['pipeline_run_id'] or '-'}")
        print(f"vector.created_at={vector['created_at']}")
        print(f"chunk.id={chunk['id']}")
        print(f"chunk.index={chunk['index']}")
        print(f"chunk.strategy={chunk['strategy']}")
        print(f"chunk.size={chunk['size']}")
        print(f"chunk.preview={(chunk['text_preview'] or '')[:120]}")
        print(f"document.id={document['id']}")
        print(f"document.source_path={document['source_path']}")
        print(f"document.source_type={document['source_type']}")
        print(f"document.version={document['version'] or '-'}")
        print(f"document.content_hash={document['content_hash'] or '-'}")
        if retrieval:
            print(f"retrieval.id={retrieval['id']}")
            print(f"retrieval.query_id={retrieval['query_id']}")
            print(f"retrieval.rank={retrieval['rank'] if retrieval['rank'] is not None else '-'}")
            print(f"retrieval.score={retrieval['score'] if retrieval['score'] is not None else '-'}")
            print(f"retrieval.trace_mode={retrieval.get('trace_mode') or '-'}")
            print(f"retrieval.query_text={retrieval['query_text']}")
            print(f"retrieval.final_answer={retrieval['final_answer'] or '-'}")
            metadata = retrieval.get("metadata")
            print(
                "retrieval.metadata="
                + (json.dumps(metadata, sort_keys=True) if metadata is not None else "-")
            )
            print(f"retrieval.created_at={retrieval['created_at']}")
        if evidence:
            print(f"evidence.support_status={evidence.get('support_status')}")
            print(f"evidence.support_reason={evidence.get('support_reason')}")
        return

    print("VECTOR TRACE")
    print("=" * 48)
    print(f"Vector ID      : {vector['id']}")
    print(f"Collection     : {vector['collection_name']}")
    print(f"Model          : {vector['embedding_model']}")
    print(f"Model Version  : {vector['model_version'] or '-'}")
    print(f"Batch ID       : {vector['batch_id'] or '-'}")
    print(f"Pipeline Run   : {vector['pipeline_run_id'] or '-'}")
    print(f"Created        : {vector['created_at']}")
    print()
    print(f"Chunk ID       : {chunk['id']}")
    print(f"Chunk Index    : {chunk['index']}")
    print(f"Chunk Strategy : {chunk['strategy']}")
    print(f"Chunk Size     : {chunk['size']}")
    print(f"Chunk Preview  : {(chunk['text_preview'] or '')[:120]}")
    print()
    print(f"Document ID    : {document['id']}")
    print(f"Source Path    : {document['source_path']}")
    print(f"Source Type    : {document['source_type']}")
    print(f"Version        : {document['version'] or '-'}")
    print(f"Content Hash   : {document['content_hash'] or '-'}")
    if retrieval:
        print()
        print("Retrieval Context")
        print(f"Event ID       : {retrieval['id']}")
        print(f"Query ID       : {retrieval['query_id']}")
        print(f"Rank           : {retrieval['rank'] if retrieval['rank'] is not None else '-'}")
        print(f"Score          : {retrieval['score'] if retrieval['score'] is not None else '-'}")
        print(f"Trace Mode     : {retrieval.get('trace_mode') or '-'}")
        print(f"Query Text     : {retrieval['query_text']}")
        print(f"Final Answer   : {retrieval['final_answer'] or '-'}")
        metadata = retrieval.get("metadata")
        if metadata is not None:
            print(f"Metadata       : {json.dumps(metadata, sort_keys=True)}")
        print(f"Recorded At    : {retrieval['created_at']}")
    if evidence:
        print()
        print("Evidence Assessment")
        print(f"Support        : {evidence.get('support_status')}")
        print(f"Reason         : {evidence.get('support_reason')}")


def _print_db_error(exc: sqlite3.OperationalError, db_path: str) -> None:
    print(
        f"Trace database error: {exc}. "
        f"Run `vectrace init --db {db_path}` first.",
        file=sys.stderr,
    )


def _write_json_output(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _format_matched_vector(event: dict) -> str:
    return (
        f"Matched vector: {event['vector_id']} (collection={event['collection_name']}, "
        f"rank={event['rank']}, score={event['score']})"
    )


def _print_report_outputs(
    output_path: str,
    json_output_path: str | None = None,
    event: dict | None = None,
) -> None:
    print(f"Report generated at {output_path}")
    if json_output_path:
        print(f"JSON generated at {json_output_path}")
    if event:
        print(_format_matched_vector(event))


def cli(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        initialize_db(args.db)
        print(f"Initialized VecTrace at {args.db}")
        return 0

    if args.command == "onboard":
        initialize_db(args.db)
        print(f"Initialized VecTrace at {args.db}")
        if args.skip_seed:
            print("Skipped demo seed (--skip-seed).")
            return 0

        if _vector_exists(args.db, args.vector_id, args.collection):
            print(
                f"Demo vector '{args.vector_id}' already exists in '{args.collection}'; "
                "seed step skipped."
            )
        elif _db_has_vectors(args.db):
            print(
                "Detected existing trace records but requested demo vector is missing; "
                "seeding requested demo vector."
            )
            _seed_demo_lineage(args.db, args.vector_id, args.collection)
            print(f"Seeded demo trace vector '{args.vector_id}' in collection '{args.collection}'.")
        else:
            _seed_demo_lineage(args.db, args.vector_id, args.collection)
            print(f"Seeded demo trace vector '{args.vector_id}' in collection '{args.collection}'.")

        with LineageQuery(args.db) as query:
            try:
                lineage = query.get_lineage(
                    vector_id=args.vector_id, collection_name=args.collection
                )
            except AmbiguousVectorIDError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            except sqlite3.OperationalError as exc:
                _print_db_error(exc, args.db)
                return 2
        if not lineage:
            print(
                f"Demo vector '{args.vector_id}' not found in '{args.collection}'. "
                "Use --vector-id/--collection that match existing data.",
                file=sys.stderr,
            )
            return 2
        prepared = _prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        generate_report(prepared, args.output, redact_preview=args.redact_preview)
        print(f"Generated demo report at {args.output}")
        print("Next:")
        print(
            f"  vectrace trace --vector-id {args.vector_id} --collection {args.collection} --db {args.db}"
        )
        print(
            f"  vectrace report --vector-id {args.vector_id} --collection {args.collection} "
            f"--db {args.db} --output {args.output}"
        )
        return 0

    if args.command == "seed-demo":
        initialize_db(args.db)
        start_index = (
            args.start_index
            if args.start_index is not None
            else _infer_next_start_index(args.db, args.collection, args.prefix)
        )
        try:
            first_id, last_id = _seed_bulk_demo_lineage(
                db_path=args.db,
                collection=args.collection,
                vectors=args.vectors,
                docs=args.docs,
                start_index=start_index,
                prefix=args.prefix,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"Seeded {args.vectors} synthetic vectors in collection '{args.collection}' "
            f"into {args.db}."
        )
        print(f"Vector range: {first_id} .. {last_id}")
        print("Try:")
        print(
            f"  vectrace trace --vector-id {first_id} --collection {args.collection} --db {args.db}"
        )
        return 0

    if args.command == "record-retrieval":
        initialize_db(args.db)
        if args.rank <= 0:
            print("--rank must be > 0", file=sys.stderr)
            return 2
        if not _vector_exists(args.db, args.vector_id, args.collection):
            print(
                f"Vector '{args.vector_id}' not found in collection '{args.collection}'. "
                "Record vector trace data first.",
                file=sys.stderr,
            )
            return 2

        metadata_json: str | None = None
        metadata_obj: dict | list | None = None
        if args.metadata_json is not None:
            try:
                metadata_obj = json.loads(args.metadata_json)
            except json.JSONDecodeError as exc:
                print(f"--metadata-json must be valid JSON: {exc}", file=sys.stderr)
                return 2
        if args.evidence_text is not None:
            if metadata_obj is None:
                metadata_obj = {}
            if not isinstance(metadata_obj, dict):
                print(
                    "--evidence-text requires --metadata-json to be an object when provided.",
                    file=sys.stderr,
                )
                return 2
            metadata_obj["evidence_text"] = args.evidence_text
        if metadata_obj is None:
            metadata_obj = {"trace_mode": "exact"}
        elif isinstance(metadata_obj, dict):
            metadata_obj.setdefault("trace_mode", "exact")
        if metadata_obj is not None:
            metadata_json = json.dumps(metadata_obj, sort_keys=True)

        query_id = args.query_id or str(uuid.uuid4())
        with LineageTracker(args.db, autoinit=False) as tracker:
            try:
                event_id = tracker.record_retrieval_event(
                    query_id=query_id,
                    query_text=args.query_text,
                    final_answer=args.final_answer,
                    collection_name=args.collection,
                    vector_id=args.vector_id,
                    rank=args.rank,
                    score=args.score,
                    metadata_json=metadata_json,
                )
            except sqlite3.IntegrityError as exc:
                print(f"Could not record retrieval event: {exc}", file=sys.stderr)
                return 2

        print(f"Recorded retrieval event {event_id} for vector '{args.vector_id}'.")
        print(
            f"Try: vectrace trace --vector-id {args.vector_id} --collection {args.collection} "
            f"--db {args.db} --include-retrieval"
        )
        return 0

    if args.command == "record-qa":
        initialize_db(args.db)
        if args.top_k <= 0:
            print("--top-k must be > 0", file=sys.stderr)
            return 2

        input_metadata, metadata_error = _parse_metadata_json(args.metadata_json)
        if metadata_error is not None:
            print(metadata_error, file=sys.stderr)
            return 2

        query_id = args.query_id or str(uuid.uuid4())
        try:
            event_ids, candidates = _record_bootstrap_events(
                db_path=args.db,
                question=args.question,
                final_answer=args.final_answer,
                collection=args.collection,
                top_k=args.top_k,
                query_id=query_id,
                input_metadata=input_metadata,
            )
        except sqlite3.OperationalError as exc:
            _print_db_error(exc, args.db)
            return 2

        if not event_ids:
            print(
                f"No candidate vectors found for question in collection '{args.collection}'. "
                "Seed/import traces first, or use `record-retrieval` with explicit vector IDs.",
                file=sys.stderr,
            )
            return 1

        print(
            f"Recorded {len(event_ids)} retrieval event(s) for query_id={query_id} "
            f"in collection '{args.collection}'."
        )
        for rank, candidate in enumerate(candidates, start=1):
            print(
                f"  rank={rank} vector_id={candidate['vector_id']} "
                f"score={candidate['score']}"
            )
        print("Next:")
        print(
            f"  vectrace trace-qa --db {args.db} --question \"{args.question}\" "
            f"--answer \"{args.final_answer}\" --collection {args.collection}"
        )
        print(
            f"  vectrace report-qa --db {args.db} --question \"{args.question}\" "
            f"--answer \"{args.final_answer}\" --collection {args.collection} "
            "--output ./qa-trace.html --json-output ./qa-trace.json"
        )
        return 0

    if args.command == "ask-trace":
        initialize_db(args.db)
        if args.top_k <= 0:
            print("--top-k must be > 0", file=sys.stderr)
            return 2
        if args.match_index <= 0:
            print("--match-index must be > 0", file=sys.stderr)
            return 2

        input_metadata, metadata_error = _parse_metadata_json(args.metadata_json)
        if metadata_error is not None:
            print(metadata_error, file=sys.stderr)
            return 2

        query_id = args.query_id or str(uuid.uuid4())
        bootstrap_answer: str | None = args.final_answer

        try:
            with LineageQuery(args.db) as query:
                candidates = query.find_trace_candidates(
                    question=args.question,
                    collection_name=args.collection,
                    limit=args.top_k,
                )
        except sqlite3.OperationalError as exc:
            _print_db_error(exc, args.db)
            return 2

        if not candidates:
            print(
                f"No candidate vectors found for question in collection '{args.collection}'. "
                "Seed/import traces first, or use `record-retrieval` with explicit vector IDs.",
                file=sys.stderr,
            )
            return 1

        if bootstrap_answer is None:
            top_snippet = str(candidates[0].get("text_preview") or "").strip()
            bootstrap_answer = top_snippet or "No extractive answer available from matched snippets."

        try:
            event_ids, candidates = _record_bootstrap_events(
                db_path=args.db,
                question=args.question,
                final_answer=bootstrap_answer,
                collection=args.collection,
                top_k=args.top_k,
                query_id=query_id,
                input_metadata=input_metadata,
            )
        except sqlite3.OperationalError as exc:
            _print_db_error(exc, args.db)
            return 2

        if args.match_index > len(event_ids):
            print(
                f"--match-index {args.match_index} exceeds recorded matches ({len(event_ids)}).",
                file=sys.stderr,
            )
            return 2

        selected_event_id = event_ids[args.match_index - 1]
        with LineageQuery(args.db) as query:
            event = query.get_retrieval_event(selected_event_id)
            if event is None:
                print(
                    f"Recorded retrieval event '{selected_event_id}' could not be loaded.",
                    file=sys.stderr,
                )
                return 1
            lineage = query.get_lineage(
                vector_id=event["vector_id"], collection_name=event["collection_name"]
            )

        if not lineage:
            print(
                f"Trace not found for matched vector '{event['vector_id']}' "
                f"in collection '{event['collection_name']}'.",
                file=sys.stderr,
            )
            return 1

        prepared = _prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        retrieval_output = _prepare_retrieval_output(
            event,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        evidence = _build_evidence(
            prepared,
            retrieval=event,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        selected = {
            "retrieval": retrieval_output,
            "trace": prepared,
            "lineage": prepared,
            "evidence": evidence,
        }

        if args.json_output:
            payload = {"schema_version": "1.0", "match": selected}
            _write_json_output(args.json_output, payload)
        generate_report(
            prepared,
            args.output,
            retrieval=retrieval_output,
            evidence=evidence,
            redact_preview=args.redact_preview,
        )

        if args.format == "json":
            payload = {"schema_version": "1.0", "match": selected, "query_id": query_id}
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"Recorded {len(event_ids)} retrieval event(s) for query_id={query_id} "
                f"in collection '{args.collection}'."
            )
            if args.final_answer is None:
                print("Final answer was bootstrapped from top matched snippet.")
            _print_report_outputs(
                output_path=args.output,
                json_output_path=args.json_output,
                event=event,
            )
        return 0

    if args.command == "trace":
        if args.format == "json" and args.plain:
            print("--plain cannot be used with --format json.", file=sys.stderr)
            return 2
        retrieval = None
        with LineageQuery(args.db) as query:
            try:
                lineage = query.get_lineage(
                    vector_id=args.vector_id, collection_name=args.collection
                )
                if lineage and args.include_retrieval:
                    collection = lineage["vector"]["collection_name"]
                    retrieval = query.get_latest_retrieval_event(
                        vector_id=args.vector_id, collection_name=collection
                    )
            except AmbiguousVectorIDError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            except sqlite3.OperationalError as exc:
                _print_db_error(exc, args.db)
                return 2
        if not lineage:
            print(f"No trace found for vector id '{args.vector_id}'.", file=sys.stderr)
            return 1
        prepared = _prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        retrieval_output = _prepare_retrieval_output(
            retrieval,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        evidence = _build_evidence(
            prepared,
            retrieval=retrieval,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        if args.format == "json":
            payload = {
                "schema_version": "1.0",
                "trace": prepared,
                "lineage": prepared,
                "retrieval": retrieval_output,
                "evidence": evidence,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_lineage(
                prepared,
                plain=args.plain,
                retrieval=retrieval_output,
                evidence=evidence,
            )
        return 0

    if args.command == "trace-qa":
        if args.top_k <= 0:
            print("--top-k must be > 0", file=sys.stderr)
            return 2

        with LineageQuery(args.db) as query:
            try:
                events = query.find_retrieval_events(
                    query_text=args.question,
                    final_answer=args.answer,
                    collection_name=args.collection,
                    limit=args.top_k,
                )
            except sqlite3.OperationalError as exc:
                _print_db_error(exc, args.db)
                return 2

            matched: list[dict] = []
            for event in events:
                lineage = query.get_lineage(
                    vector_id=event["vector_id"], collection_name=event["collection_name"]
                )
                if lineage is None:
                    continue
                prepared = _prepare_lineage_output(lineage, redact_preview=args.redact_preview)
                matched.append(
                    {
                        "retrieval": _prepare_retrieval_output(
                            event,
                            redact_preview=args.redact_preview,
                            redact_retrieval=args.redact_retrieval,
                        ),
                        "trace": prepared,
                        "lineage": prepared,
                        "evidence": _build_evidence(
                            prepared,
                            retrieval=event,
                            redact_preview=args.redact_preview,
                            redact_retrieval=args.redact_retrieval,
                        ),
                    }
                )

        if not matched:
            print(
                "No retrieval events matched that question/answer. "
                "Record retrieval events first with `vectrace record-retrieval`.",
                file=sys.stderr,
            )
            return 1

        if args.format == "json":
            payload = {"schema_version": "1.0", "matches": matched}
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        print(f"Matched {len(matched)} retrieval event(s)")
        for idx, item in enumerate(matched, start=1):
            retrieval = item["retrieval"]
            print()
            print(f"[Match {idx}] vector_id={retrieval['vector_id']} collection={retrieval['collection_name']}")
            _print_lineage(
                item["trace"],
                plain=False,
                retrieval=retrieval,
                evidence=item.get("evidence"),
            )
        return 0

    if args.command == "report-qa":
        if args.top_k <= 0:
            print("--top-k must be > 0", file=sys.stderr)
            return 2
        if args.match_index <= 0:
            print("--match-index must be > 0", file=sys.stderr)
            return 2

        with LineageQuery(args.db) as query:
            try:
                events = query.find_retrieval_events(
                    query_text=args.question,
                    final_answer=args.answer,
                    collection_name=args.collection,
                    limit=max(args.top_k, args.match_index),
                )
            except sqlite3.OperationalError as exc:
                _print_db_error(exc, args.db)
                return 2

            if not events:
                print(
                    "No retrieval events matched that question/answer. "
                    "Record retrieval events first with `vectrace record-retrieval`.",
                    file=sys.stderr,
                )
                return 1
            if args.match_index > len(events):
                print(
                    f"--match-index {args.match_index} exceeds available matches ({len(events)}).",
                    file=sys.stderr,
                )
                return 2

            event = events[args.match_index - 1]
            lineage = query.get_lineage(
                vector_id=event["vector_id"], collection_name=event["collection_name"]
            )

        if not lineage:
            print(
                f"Trace not found for matched vector '{event['vector_id']}' "
                f"in collection '{event['collection_name']}'.",
                file=sys.stderr,
            )
            return 1

        prepared = _prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        retrieval_output = _prepare_retrieval_output(
            event,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        evidence = _build_evidence(
            prepared,
            retrieval=event,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        selected = {
            "retrieval": retrieval_output,
            "trace": prepared,
            "lineage": prepared,
            "evidence": evidence,
        }
        if args.json_output:
            payload = {"schema_version": "1.0", "match": selected}
            _write_json_output(args.json_output, payload)
        generate_report(
            prepared,
            args.output,
            retrieval=retrieval_output,
            evidence=evidence,
            redact_preview=args.redact_preview,
        )
        _print_report_outputs(
            output_path=args.output,
            json_output_path=args.json_output,
            event=event,
        )
        return 0

    if args.command == "report":
        retrieval = None
        with LineageQuery(args.db) as query:
            try:
                lineage = query.get_lineage(
                    vector_id=args.vector_id, collection_name=args.collection
                )
                if lineage and args.include_retrieval:
                    collection = lineage["vector"]["collection_name"]
                    retrieval = query.get_latest_retrieval_event(
                        vector_id=args.vector_id, collection_name=collection
                    )
            except AmbiguousVectorIDError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            except sqlite3.OperationalError as exc:
                _print_db_error(exc, args.db)
                return 2
        if not lineage:
            print(f"No trace found for vector id '{args.vector_id}'.", file=sys.stderr)
            return 1
        prepared = _prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        retrieval_output = _prepare_retrieval_output(
            retrieval,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        evidence = _build_evidence(
            prepared,
            retrieval=retrieval,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        generate_report(
            prepared,
            args.output,
            retrieval=retrieval_output,
            evidence=evidence,
            redact_preview=args.redact_preview,
        )
        _print_report_outputs(output_path=args.output)
        return 0

    if args.command == "connect":
        from connectors.qdrant import test_connection

        try:
            test_connection(
                qdrant_url=args.qdrant_url,
                collection_name=args.qdrant_collection,
                api_key=args.api_key,
            )
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"Connected to Qdrant: {args.qdrant_url} / {args.qdrant_collection}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
