#!/usr/bin/env python3
"""VecTrace CLI."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid

from lineage.models import ChunkRecord, VectorRecord, initialize_db
from lineage.query import AmbiguousVectorIDError, LineageQuery
from lineage.tracker import LineageTracker
from output.html import generate_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VecTrace: trace where your RAG answers come from."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize VecTrace SQLite schema.")
    init_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")

    onboard_parser = subparsers.add_parser(
        "onboard", help="Initialize, seed demo trace data, and generate a shareable report."
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
        "seed-demo", help="Append larger synthetic trace data into a database for demos."
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
        help="Record retrieval telemetry (query text, rank/score, final answer) for a vector.",
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

    trace_parser = subparsers.add_parser(
        "trace", help="Trace a vector back to chunk and source document."
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
        "--include-retrieval",
        action="store_true",
        help="Include latest recorded retrieval telemetry for this vector.",
    )
    trace_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")

    report_parser = subparsers.add_parser("report", help="Generate an HTML trace report.")
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
        "--include-retrieval",
        action="store_true",
        help="Include latest recorded retrieval telemetry for this vector in the report.",
    )
    report_parser.add_argument("--output", default="trace.html", help="Output HTML path.")
    report_parser.add_argument("--db", default="./vectrace.db", help="Trace database path.")

    connect_parser = subparsers.add_parser(
        "connect", help="Test Qdrant connectivity and verify collection exists."
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


def _print_lineage(lineage: dict, plain: bool = False, retrieval: dict | None = None) -> None:
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
            print(f"retrieval.query_text={retrieval['query_text']}")
            print(f"retrieval.final_answer={retrieval['final_answer'] or '-'}")
            metadata = retrieval.get("metadata")
            print(
                "retrieval.metadata="
                + (json.dumps(metadata, sort_keys=True) if metadata is not None else "-")
            )
            print(f"retrieval.created_at={retrieval['created_at']}")
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
        print(f"Query Text     : {retrieval['query_text']}")
        print(f"Final Answer   : {retrieval['final_answer'] or '-'}")
        metadata = retrieval.get("metadata")
        if metadata is not None:
            print(f"Metadata       : {json.dumps(metadata, sort_keys=True)}")
        print(f"Recorded At    : {retrieval['created_at']}")


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
                print(
                    f"Trace database error: {exc}. "
                    f"Run `vectrace init --db {args.db}` first.",
                    file=sys.stderr,
                )
                return 2
        if not lineage:
            print(
                f"Demo vector '{args.vector_id}' not found in '{args.collection}'. "
                "Use --vector-id/--collection that match existing data.",
                file=sys.stderr,
            )
            return 2
        prepared = _prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        generate_report(prepared, args.output)
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
        if args.metadata_json is not None:
            try:
                metadata_obj = json.loads(args.metadata_json)
            except json.JSONDecodeError as exc:
                print(f"--metadata-json must be valid JSON: {exc}", file=sys.stderr)
                return 2
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
                print(
                    f"Trace database error: {exc}. "
                    f"Run `vectrace init --db {args.db}` first.",
                    file=sys.stderr,
                )
                return 2
        if not lineage:
            print(f"No trace found for vector id '{args.vector_id}'.", file=sys.stderr)
            return 1
        prepared = _prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        if args.format == "json":
            payload = {
                "schema_version": "1.0",
                "trace": prepared,
                "lineage": prepared,
                "retrieval": retrieval,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_lineage(prepared, plain=args.plain, retrieval=retrieval)
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
                print(
                    f"Trace database error: {exc}. "
                    f"Run `vectrace init --db {args.db}` first.",
                    file=sys.stderr,
                )
                return 2
        if not lineage:
            print(f"No trace found for vector id '{args.vector_id}'.", file=sys.stderr)
            return 1
        prepared = _prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        generate_report(prepared, args.output, retrieval=retrieval)
        print(f"Report generated at {args.output}")
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
