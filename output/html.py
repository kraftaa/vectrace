"""HTML report generation for vector trace records."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path


def _safe(value: object) -> str:
    if value is None:
        return "-"
    return escape(str(value))


def generate_report(lineage: dict, output_path: str, retrieval: dict | None = None) -> None:
    """Generate an HTML report from trace data."""
    if not lineage:
        raise ValueError("lineage is required to render a report")

    vector = lineage["vector"]
    chunk = lineage["chunk"]
    document = lineage["document"]
    retrieval_section = ""
    if retrieval:
        metadata = retrieval.get("metadata")
        retrieval_section = f"""
    <section class="card">
      <h2>Retrieval Context</h2>
      <dl>
        <dt>Event ID</dt><dd><code>{_safe(retrieval.get("id"))}</code></dd>
        <dt>Query ID</dt><dd><code>{_safe(retrieval.get("query_id"))}</code></dd>
        <dt>Rank</dt><dd>{_safe(retrieval.get("rank"))}</dd>
        <dt>Score</dt><dd>{_safe(retrieval.get("score"))}</dd>
        <dt>Recorded At</dt><dd>{_safe(retrieval.get("created_at"))}</dd>
      </dl>
      <div class="preview"><strong>Query</strong>\n{_safe(retrieval.get("query_text"))}</div>
      <div class="preview"><strong>Final Answer</strong>\n{_safe(retrieval.get("final_answer"))}</div>
      <div class="preview"><strong>Metadata</strong>\n{_safe(metadata)}</div>
    </section>
"""

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
        <dt>Path</dt><dd><code>{_safe(document.get("source_path"))}</code></dd>
        <dt>Type</dt><dd><span class="badge">{_safe(document.get("source_type"))}</span></dd>
        <dt>Version</dt><dd>{_safe(document.get("version"))}</dd>
        <dt>Content Hash</dt><dd><code>{_safe(document.get("content_hash"))}</code></dd>
      </dl>
    </section>
{retrieval_section}

    <footer>Generated by VecTrace</footer>
  </main>
</body>
</html>
"""

    Path(output_path).write_text(html, encoding="utf-8")
