"""HTML report generation for vector trace records."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path


_SAFE_URL_SCHEMES: tuple[str, ...] = (
    "https://",
    "http://",
    "s3://",
    "file://",
    "ftp://",
)


def _safe(value: object) -> str:
    if value is None:
        return "-"
    return escape(str(value))


def _path_link(path: object) -> str:
    if path is None:
        return "-"
    text = str(path)
    label = escape(text)
    href: str | None = None
    if text.startswith(("http://", "https://", "file://", "s3://")):
        href = text
    elif text.startswith("/"):
        href = f"file://{text}"
    elif len(text) >= 3 and text[1] == ":" and text[2] in ("\\", "/"):
        normalized = text.replace("\\", "/")
        href = f"file:///{normalized}"
    if href is not None:
        href_escaped = escape(href, quote=True)
        return (
            f'<a href="{href_escaped}" target="_blank" rel="noopener noreferrer">'
            f"<code>{label}</code></a>"
        )
    return f"<code>{label}</code>"


def _deep_link(url: object, page: object) -> str:
    """Render a clickable deep-link to ``url``, adding ``#page=N`` for PDFs.

    The fragment lets browsers' built-in PDF viewers jump directly to the cited
    page; non-PDF targets still get a clickable link.
    """
    if url is None:
        return "-"
    text = str(url)
    if not text:
        return "-"
    # Only render an <a href> for known-safe schemes; anything else (e.g.
    # javascript:, data:) is shown as plain text to prevent stored XSS when
    # source_url is sourced from untrusted DB writes.
    if not text.lower().lstrip().startswith(_SAFE_URL_SCHEMES):
        return escape(text)
    href = text
    label = escape(text)
    if page is not None:
        try:
            page_int = int(page)
            if page_int > 0:
                href = f"{href}#page={page_int}"
                label = f"{label} <span class='badge'>page {page_int}</span>"
        except (TypeError, ValueError):
            pass
    href_escaped = escape(href, quote=True)
    return (
        f'<a href="{href_escaped}" target="_blank" rel="noopener noreferrer">{label}</a>'
    )


def _why_this_match_panel(evidence: dict) -> str:
    details = evidence.get("support_details") or {}
    if not isinstance(details, dict):
        details = {}
    overlap_terms = details.get("overlap_terms")
    overlap_terms_display = "-"
    if isinstance(overlap_terms, list) and overlap_terms:
        overlap_terms_display = ", ".join(escape(str(t)) for t in overlap_terms)
    elif isinstance(overlap_terms, list):
        overlap_terms_display = "(none)"

    overlap_ratio = details.get("overlap_ratio")
    overlap_ratio_display = (
        f"{overlap_ratio:.2f}" if isinstance(overlap_ratio, (int, float)) else "-"
    )

    polarity = details.get("answer_polarity") or "-"

    constraint_rows = ""
    query_constraint = details.get("query_day_constraint")
    evidence_constraint = details.get("evidence_day_constraint")
    if isinstance(query_constraint, dict) and isinstance(evidence_constraint, dict):
        constraint_rows = (
            f'\n        <dt>Query Constraint</dt><dd><span class="badge">'
            f'{_safe(query_constraint.get("mode"))} {_safe(query_constraint.get("days"))} days'
            f"</span></dd>"
            f'\n        <dt>Evidence Constraint</dt><dd><span class="badge">'
            f'{_safe(evidence_constraint.get("mode"))} {_safe(evidence_constraint.get("days"))} days'
            f"</span></dd>"
        )

    return f"""
    <section class="card">
      <h2>Why This Match</h2>
      <dl>
        <dt>Trace Mode</dt><dd><span class="badge">{_safe(evidence.get("trace_mode"))}</span></dd>
        <dt>Rank</dt><dd>{_safe(evidence.get("retrieval_rank"))}</dd>
        <dt>Score</dt><dd>{_safe(evidence.get("retrieval_score"))}</dd>
        <dt>Answer Polarity</dt><dd><span class="badge">{_safe(polarity)}</span></dd>
        <dt>Overlap Ratio</dt><dd>{_safe(overlap_ratio_display)}</dd>
        <dt>Overlap Terms</dt><dd>{overlap_terms_display}</dd>{constraint_rows}
      </dl>
    </section>
"""


def generate_report(
    lineage: dict,
    output_path: str,
    retrieval: dict | None = None,
    evidence: dict | None = None,
    redact_preview: bool = False,
) -> None:
    """Generate an HTML report from trace data."""
    if not lineage:
        raise ValueError("lineage is required to render a report")

    vector = lineage["vector"]
    chunk = lineage["chunk"]
    document = lineage["document"]
    retrieval_section = ""
    if retrieval:
        metadata = retrieval.get("metadata")
        if redact_preview and isinstance(metadata, dict) and "evidence_text" in metadata:
            metadata = dict(metadata)
            text = "" if metadata.get("evidence_text") is None else str(metadata.get("evidence_text"))
            metadata["evidence_text"] = f"[REDACTED:{len(text)} chars]" if text else "[REDACTED]"
        if isinstance(metadata, (dict, list)):
            metadata_display = json.dumps(metadata, indent=2, sort_keys=True)
        else:
            metadata_display = metadata
        evidence_text = None
        if isinstance(metadata, dict):
            evidence_text = metadata.get("evidence_text")
        evidence_row = ""
        if evidence_text:
            evidence_row = (
                f'\n      <div class="preview"><strong>Evidence Text</strong>\\n{_safe(evidence_text)}</div>'
            )
        retrieval_section = f"""
    <section class="card">
      <h2>Retrieval Context</h2>
      <dl>
        <dt>Event ID</dt><dd><code>{_safe(retrieval.get("id"))}</code></dd>
        <dt>Query ID</dt><dd><code>{_safe(retrieval.get("query_id"))}</code></dd>
        <dt>Vector ID</dt><dd><code>{_safe(retrieval.get("vector_id"))}</code></dd>
        <dt>Trace Mode</dt><dd><span class="badge">{_safe(retrieval.get("trace_mode"))}</span></dd>
        <dt>Rank</dt><dd>{_safe(retrieval.get("rank"))}</dd>
        <dt>Score</dt><dd>{_safe(retrieval.get("score"))}</dd>
        <dt>Recorded At</dt><dd>{_safe(retrieval.get("created_at"))}</dd>
      </dl>
      <div class="preview"><strong>Query</strong>\n{_safe(retrieval.get("query_text"))}</div>
      <div class="preview"><strong>Final Answer</strong>\n{_safe(retrieval.get("final_answer"))}</div>
{evidence_row}
      <div class="preview"><strong>Metadata</strong>\n{_safe(metadata_display)}</div>
    </section>
"""
    evidence_section = ""
    if evidence:
        support_status = evidence.get("support_status")
        support_reason = evidence.get("support_reason")
        support_details = evidence.get("support_details")
        support_details_display = "-"
        if isinstance(support_details, (dict, list)):
            support_details_display = json.dumps(support_details, indent=2, sort_keys=True)
        elif support_details is not None:
            support_details_display = str(support_details)
        evidence_section = f"""
    <section class="card">
      <h2>Answer Evidence</h2>
      <dl>
        <dt>Support</dt><dd><span class="badge">{_safe(support_status)}</span></dd>
        <dt>Assessment</dt><dd>{_safe(support_reason)}</dd>
        <dt>Vector ID</dt><dd><code>{_safe(evidence.get("vector_id"))}</code></dd>
        <dt>Collection</dt><dd><span class="badge">{_safe(evidence.get("collection_name"))}</span></dd>
        <dt>Trace Mode</dt><dd><span class="badge">{_safe(evidence.get("trace_mode"))}</span></dd>
        <dt>Chunk ID</dt><dd><code>{_safe(evidence.get("chunk_id"))}</code></dd>
        <dt>Chunk Index</dt><dd>{_safe(evidence.get("chunk_index"))}</dd>
        <dt>Source Path</dt><dd>{_path_link(evidence.get("source_path"))}</dd>
        <dt>Source URL</dt><dd>{_deep_link(evidence.get("source_url"), evidence.get("source_page"))}</dd>
        <dt>Source Section</dt><dd>{_safe(evidence.get("source_section"))}</dd>
      </dl>
      <div class="preview"><strong>Retrieved Text Snippet</strong>\n{_safe(evidence.get("chunk_text"))}</div>
      <div class="preview"><strong>Assessment Details</strong>\n{_safe(support_details_display)}</div>
    </section>
"""
    why_match_section = _why_this_match_panel(evidence) if evidence else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VecTrace Trace Report</title>
  <style>
    :root {{
      --bg: #f4f5f0;
      --card: #ffffff;
      --ink: #202226;
      --muted: #5c6672;
      --accent: #0f766e;
      --accent-soft: #d9f3ef;
      --stroke: #dde3e8;
      --mono: "SFMono-Regular", Menlo, Consolas, Monaco, monospace;
      --sans: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background: radial-gradient(circle at 85% 8%, #d7efe8 0%, var(--bg) 50%);
      line-height: 1.55;
    }}
    main {{
      max-width: 980px;
      margin: 2rem auto;
      padding: 0 1rem 2rem;
    }}
    .hero {{
      background: linear-gradient(140deg, #0f766e, #115e59);
      color: #fff;
      border-radius: 16px;
      padding: 1.5rem;
      box-shadow: 0 14px 40px rgba(17, 94, 89, 0.24);
    }}
    .hero h1 {{
      margin: 0 0 0.2rem;
      font-size: 1.5rem;
    }}
    .hero p {{
      margin: 0;
      opacity: 0.94;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--stroke);
      border-radius: 12px;
      padding: 1.1rem 1.2rem;
      margin-top: 1rem;
    }}
    h2 {{
      margin: 0 0 0.6rem;
      font-size: 1.08rem;
      color: #0f172a;
    }}
    dl {{
      display: grid;
      grid-template-columns: minmax(145px, 170px) 1fr;
      gap: 0.45rem 0.8rem;
      margin: 0;
    }}
    dt {{
      color: var(--muted);
      font-weight: 600;
    }}
    dd {{
      margin: 0;
      word-break: break-word;
    }}
    code {{
      font-family: var(--mono);
      font-size: 0.9em;
      background: #f7fafc;
      padding: 0.1rem 0.35rem;
      border-radius: 5px;
      border: 1px solid #ebf0f4;
    }}
    .badge {{
      background: var(--accent-soft);
      color: #115e59;
      border: 1px solid #bce7df;
      font-family: var(--mono);
      font-size: 0.84rem;
      padding: 0.12rem 0.45rem;
      border-radius: 99px;
    }}
    .preview {{
      margin-top: 0.8rem;
      background: #f9fbfc;
      border: 1px solid #e8edf2;
      border-radius: 8px;
      padding: 0.7rem 0.8rem;
      font-family: var(--mono);
      font-size: 0.85rem;
      white-space: pre-wrap;
    }}
    footer {{
      margin-top: 1.6rem;
      color: var(--muted);
      font-size: 0.85rem;
      text-align: center;
    }}
    @media (max-width: 640px) {{
      dl {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>VecTrace Trace Report</h1>
      <p>Generated at {_safe(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</p>
    </section>
{retrieval_section}
{evidence_section}
{why_match_section}

    <section class="card">
      <h2>Vector</h2>
      <dl>
        <dt>ID</dt><dd><code>{_safe(vector.get("id"))}</code></dd>
        <dt>Collection</dt><dd><span class="badge">{_safe(vector.get("collection_name"))}</span></dd>
        <dt>Model</dt><dd><span class="badge">{_safe(vector.get("embedding_model"))}</span></dd>
        <dt>Model Version</dt><dd>{_safe(vector.get("model_version"))}</dd>
        <dt>Batch ID</dt><dd><code>{_safe(vector.get("batch_id"))}</code></dd>
        <dt>Pipeline Run</dt><dd><code>{_safe(vector.get("pipeline_run_id"))}</code></dd>
        <dt>Created</dt><dd>{_safe(vector.get("created_at"))}</dd>
      </dl>
    </section>

    <section class="card">
      <h2>Chunk</h2>
      <dl>
        <dt>ID</dt><dd><code>{_safe(chunk.get("id"))}</code></dd>
        <dt>Index</dt><dd>{_safe(chunk.get("index"))}</dd>
        <dt>Strategy</dt><dd><span class="badge">{_safe(chunk.get("strategy"))}</span></dd>
        <dt>Size</dt><dd>{_safe(chunk.get("size"))}</dd>
      </dl>
      <div class="preview">{_safe(chunk.get("text_preview"))}</div>
    </section>

    <section class="card">
      <h2>Source Document</h2>
      <dl>
        <dt>ID</dt><dd><code>{_safe(document.get("id"))}</code></dd>
        <dt>Path</dt><dd>{_path_link(document.get("source_path"))}</dd>
        <dt>Type</dt><dd><span class="badge">{_safe(document.get("source_type"))}</span></dd>
        <dt>Version</dt><dd>{_safe(document.get("version"))}</dd>
        <dt>Content Hash</dt><dd><code>{_safe(document.get("content_hash"))}</code></dd>
        <dt>Source URL</dt><dd>{_deep_link(document.get("source_url"), document.get("source_page"))}</dd>
        <dt>Source Section</dt><dd>{_safe(document.get("source_section"))}</dd>
      </dl>
    </section>

    <footer>Generated by VecTrace</footer>
  </main>
</body>
</html>
"""

    Path(output_path).write_text(html, encoding="utf-8")
