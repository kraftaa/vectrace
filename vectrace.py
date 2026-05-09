#!/usr/bin/env python3
"""VecTrace CLI."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid

from lineage.models import initialize_db
from lineage.query import AmbiguousVectorIDError, LineageQuery
from lineage.seed import (
    db_has_vectors,
    infer_next_start_index,
    parse_metadata_json,
    record_bootstrap_events,
    seed_bulk_demo_lineage,
    seed_demo_lineage,
    vector_exists,
)
from lineage.tracker import LineageTracker
from output.html import generate_report
from output.preparation import (
    build_evidence,
    prepare_lineage_output,
    prepare_retrieval_output,
)
from output.printing import (
    print_db_error,
    print_lineage,
    print_report_outputs,
    write_json_output,
)


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

        if vector_exists(args.db, args.vector_id, args.collection):
            print(
                f"Demo vector '{args.vector_id}' already exists in '{args.collection}'; "
                "seed step skipped."
            )
        elif db_has_vectors(args.db):
            print(
                "Detected existing trace records but requested demo vector is missing; "
                "seeding requested demo vector."
            )
            seed_demo_lineage(args.db, args.vector_id, args.collection)
            print(f"Seeded demo trace vector '{args.vector_id}' in collection '{args.collection}'.")
        else:
            seed_demo_lineage(args.db, args.vector_id, args.collection)
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
                print_db_error(exc, args.db)
                return 2
        if not lineage:
            print(
                f"Demo vector '{args.vector_id}' not found in '{args.collection}'. "
                "Use --vector-id/--collection that match existing data.",
                file=sys.stderr,
            )
            return 2
        prepared = prepare_lineage_output(lineage, redact_preview=args.redact_preview)
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
            else infer_next_start_index(args.db, args.collection, args.prefix)
        )
        try:
            first_id, last_id = seed_bulk_demo_lineage(
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
        if not vector_exists(args.db, args.vector_id, args.collection):
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

        input_metadata, metadata_error = parse_metadata_json(args.metadata_json)
        if metadata_error is not None:
            print(metadata_error, file=sys.stderr)
            return 2

        query_id = args.query_id or str(uuid.uuid4())
        try:
            event_ids, candidates = record_bootstrap_events(
                db_path=args.db,
                question=args.question,
                final_answer=args.final_answer,
                collection=args.collection,
                top_k=args.top_k,
                query_id=query_id,
                input_metadata=input_metadata,
            )
        except sqlite3.OperationalError as exc:
            print_db_error(exc, args.db)
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

        input_metadata, metadata_error = parse_metadata_json(args.metadata_json)
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
            print_db_error(exc, args.db)
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
            event_ids, candidates = record_bootstrap_events(
                db_path=args.db,
                question=args.question,
                final_answer=bootstrap_answer,
                collection=args.collection,
                top_k=args.top_k,
                query_id=query_id,
                input_metadata=input_metadata,
            )
        except sqlite3.OperationalError as exc:
            print_db_error(exc, args.db)
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

        prepared = prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        retrieval_output = prepare_retrieval_output(
            event,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        evidence = build_evidence(
            prepared,
            retrieval=event,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
            raw_evidence_text=lineage["chunk"].get("text_preview"),
        )
        selected = {
            "retrieval": retrieval_output,
            "trace": prepared,
            "lineage": prepared,
            "evidence": evidence,
        }

        if args.json_output:
            payload = {"schema_version": "1.0", "match": selected}
            write_json_output(args.json_output, payload)
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
            print_report_outputs(
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
                print_db_error(exc, args.db)
                return 2
        if not lineage:
            print(f"No trace found for vector id '{args.vector_id}'.", file=sys.stderr)
            return 1
        prepared = prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        retrieval_output = prepare_retrieval_output(
            retrieval,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        evidence = build_evidence(
            prepared,
            retrieval=retrieval,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
            raw_evidence_text=lineage["chunk"].get("text_preview"),
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
            print_lineage(
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
                print_db_error(exc, args.db)
                return 2

            matched: list[dict] = []
            for event in events:
                lineage = query.get_lineage(
                    vector_id=event["vector_id"], collection_name=event["collection_name"]
                )
                if lineage is None:
                    continue
                prepared = prepare_lineage_output(lineage, redact_preview=args.redact_preview)
                matched.append(
                    {
                        "retrieval": prepare_retrieval_output(
                            event,
                            redact_preview=args.redact_preview,
                            redact_retrieval=args.redact_retrieval,
                        ),
                        "trace": prepared,
                        "lineage": prepared,
                        "evidence": build_evidence(
                            prepared,
                            retrieval=event,
                            redact_preview=args.redact_preview,
                            redact_retrieval=args.redact_retrieval,
                            raw_evidence_text=lineage["chunk"].get("text_preview"),
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
            print_lineage(
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
                print_db_error(exc, args.db)
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

        prepared = prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        retrieval_output = prepare_retrieval_output(
            event,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        evidence = build_evidence(
            prepared,
            retrieval=event,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
            raw_evidence_text=lineage["chunk"].get("text_preview"),
        )
        selected = {
            "retrieval": retrieval_output,
            "trace": prepared,
            "lineage": prepared,
            "evidence": evidence,
        }
        if args.json_output:
            payload = {"schema_version": "1.0", "match": selected}
            write_json_output(args.json_output, payload)
        generate_report(
            prepared,
            args.output,
            retrieval=retrieval_output,
            evidence=evidence,
            redact_preview=args.redact_preview,
        )
        print_report_outputs(
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
                print_db_error(exc, args.db)
                return 2
        if not lineage:
            print(f"No trace found for vector id '{args.vector_id}'.", file=sys.stderr)
            return 1
        prepared = prepare_lineage_output(lineage, redact_preview=args.redact_preview)
        retrieval_output = prepare_retrieval_output(
            retrieval,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
        )
        evidence = build_evidence(
            prepared,
            retrieval=retrieval,
            redact_preview=args.redact_preview,
            redact_retrieval=args.redact_retrieval,
            raw_evidence_text=lineage["chunk"].get("text_preview"),
        )
        generate_report(
            prepared,
            args.output,
            retrieval=retrieval_output,
            evidence=evidence,
            redact_preview=args.redact_preview,
        )
        print_report_outputs(output_path=args.output)
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
