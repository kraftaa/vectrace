"""Output payload preparation: lineage shaping, evidence building, redaction."""

from __future__ import annotations

from output.support import assess_answer_support


def prepare_lineage_output(lineage: dict, redact_preview: bool) -> dict:
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


def redact_text(value: object) -> str:
    text = "" if value is None else str(value)
    return f"[REDACTED:{len(text)} chars]" if text else "[REDACTED]"


def trace_mode_from_retrieval(retrieval: dict | None) -> str | None:
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


def prepare_retrieval_output(
    retrieval: dict | None,
    redact_preview: bool,
    redact_retrieval: bool = False,
) -> dict | None:
    if retrieval is None:
        return None
    prepared = dict(retrieval)
    prepared["trace_mode"] = trace_mode_from_retrieval(retrieval)
    if redact_retrieval:
        prepared["query_text"] = redact_text(prepared.get("query_text"))
        prepared["final_answer"] = redact_text(prepared.get("final_answer"))
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
                    metadata[key] = redact_text(metadata.get(key))
        if redact_preview and "evidence_text" in metadata:
            metadata["evidence_text"] = redact_text(metadata.get("evidence_text"))
        prepared["metadata"] = metadata
    elif redact_retrieval and metadata is not None:
        prepared["metadata"] = redact_text(metadata)
    return prepared


def build_evidence(
    trace_data: dict,
    retrieval: dict | None = None,
    redact_preview: bool = False,
    redact_retrieval: bool = False,
    raw_evidence_text: str | None = None,
) -> dict:
    vector = trace_data["vector"]
    chunk = trace_data["chunk"]
    document = trace_data["document"]
    resolved_raw_evidence_text = raw_evidence_text
    if resolved_raw_evidence_text is None:
        resolved_raw_evidence_text = chunk.get("text_preview")
    if retrieval:
        metadata = retrieval.get("metadata")
        if isinstance(metadata, dict) and metadata.get("evidence_text"):
            resolved_raw_evidence_text = metadata["evidence_text"]
    support = assess_answer_support(retrieval=retrieval, evidence_text=resolved_raw_evidence_text)
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
        evidence_text = redact_text(resolved_raw_evidence_text)
    else:
        evidence_text = resolved_raw_evidence_text
    return {
        "vector_id": vector.get("id"),
        "collection_name": vector.get("collection_name"),
        "trace_mode": trace_mode_from_retrieval(retrieval),
        "chunk_id": chunk.get("id"),
        "chunk_index": chunk.get("index"),
        "chunk_text": evidence_text,
        "source_document_id": document.get("id"),
        "source_path": document.get("source_path"),
        "source_url": document.get("source_url"),
        "source_page": document.get("source_page"),
        "source_section": document.get("source_section"),
        "retrieval_rank": retrieval.get("rank") if retrieval else None,
        "retrieval_score": retrieval.get("score") if retrieval else None,
        "support_status": support["status"],
        "support_reason": support_reason,
        "support_details": support_details,
    }
