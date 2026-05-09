"""Terminal and JSON-file output helpers for VecTrace CLI."""

from __future__ import annotations

import json
import sqlite3
import sys


def print_lineage(
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


def print_db_error(exc: sqlite3.OperationalError, db_path: str) -> None:
    print(
        f"Trace database error: {exc}. "
        f"Run `vectrace init --db {db_path}` first.",
        file=sys.stderr,
    )


def write_json_output(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def format_matched_vector(event: dict) -> str:
    return (
        f"Matched vector: {event['vector_id']} (collection={event['collection_name']}, "
        f"rank={event['rank']}, score={event['score']})"
    )


def print_report_outputs(
    output_path: str,
    json_output_path: str | None = None,
    event: dict | None = None,
) -> None:
    print(f"Report generated at {output_path}")
    if json_output_path:
        print(f"JSON generated at {json_output_path}")
    if event:
        print(format_matched_vector(event))
