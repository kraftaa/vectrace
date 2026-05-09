from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from output.html import generate_report


class HtmlReportTests(unittest.TestCase):
    def test_report_escapes_html_content(self) -> None:
        lineage = {
            "vector": {
                "id": "0",
                "collection_name": "support_kb",
                "embedding_model": "m",
                "model_version": None,
                "batch_id": None,
                "pipeline_run_id": None,
                "created_at": "2026-04-01T00:00:00+00:00",
            },
            "chunk": {
                "id": "chunk_0",
                "index": 0,
                "strategy": "semantic",
                "size": 10,
                "text_preview": "<script>alert('xss')</script>",
            },
            "document": {
                "id": "doc_1",
                "source_path": "/tmp/x",
                "source_type": "local",
                "version": "v1",
                "content_hash": None,
            },
        }

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "lineage.html"
            generate_report(lineage, str(out))
            rendered = out.read_text(encoding="utf-8")

        self.assertIn("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;", rendered)
        self.assertNotIn("<script>alert('xss')</script>", rendered)

    def test_report_renders_retrieval_context(self) -> None:
        lineage = {
            "vector": {
                "id": "0",
                "collection_name": "support_kb",
                "embedding_model": "m",
                "model_version": None,
                "batch_id": None,
                "pipeline_run_id": None,
                "created_at": "2026-04-01T00:00:00+00:00",
            },
            "chunk": {
                "id": "chunk_0",
                "index": 0,
                "strategy": "semantic",
                "size": 10,
                "text_preview": "preview",
            },
            "document": {
                "id": "doc_1",
                "source_path": "/tmp/x",
                "source_type": "local",
                "version": "v1",
                "content_hash": None,
            },
        }
        retrieval = {
            "id": "evt_1",
            "query_id": "q_1",
            "trace_mode": "exact",
            "query_text": "<bad> query",
            "final_answer": "answer",
            "collection_name": "support_kb",
            "vector_id": "0",
            "rank": 1,
            "score": 0.99,
            "metadata": {"trace_id": "abc"},
            "created_at": "2026-04-01T00:00:01+00:00",
        }

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "trace.html"
            generate_report(lineage, str(out), retrieval=retrieval)
            rendered = out.read_text(encoding="utf-8")

        self.assertIn("Retrieval Context", rendered)
        self.assertIn("evt_1", rendered)
        self.assertIn("&lt;bad&gt; query", rendered)
        self.assertIn("exact", rendered)

    def test_report_renders_answer_evidence_section(self) -> None:
        lineage = {
            "vector": {
                "id": "0",
                "collection_name": "support_kb",
                "embedding_model": "m",
                "model_version": None,
                "batch_id": None,
                "pipeline_run_id": None,
                "created_at": "2026-04-01T00:00:00+00:00",
            },
            "chunk": {
                "id": "chunk_0",
                "index": 0,
                "strategy": "semantic",
                "size": 10,
                "text_preview": "snippet words",
            },
            "document": {
                "id": "doc_1",
                "source_path": "/tmp/x",
                "source_type": "local",
                "version": "v1",
                "content_hash": None,
            },
        }
        evidence = {
            "vector_id": "0",
            "collection_name": "support_kb",
            "trace_mode": "bootstrap",
            "chunk_id": "chunk_0",
            "chunk_index": 0,
            "chunk_text": "snippet words",
            "source_document_id": "doc_1",
            "source_path": "/tmp/x",
            "retrieval_rank": 1,
            "retrieval_score": 0.5,
            "support_status": "supported",
            "support_reason": "Answer terms align with retrieved evidence snippet.",
            "support_details": {"method": "rule_plus_overlap", "overlap_ratio": 1.0},
        }
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "trace.html"
            generate_report(lineage, str(out), evidence=evidence)
            rendered = out.read_text(encoding="utf-8")

        self.assertIn("Answer Evidence", rendered)
        self.assertIn("Retrieved Text Snippet", rendered)
        self.assertIn("snippet words", rendered)
        self.assertIn("Assessment Details", rendered)
        self.assertIn("rule_plus_overlap", rendered)
        self.assertIn("bootstrap", rendered)

    def test_report_renders_deep_link_and_why_this_match_panel(self) -> None:
        lineage = {
            "vector": {
                "id": "vec_dl",
                "collection_name": "support_kb",
                "embedding_model": "m",
                "model_version": None,
                "batch_id": None,
                "pipeline_run_id": None,
                "created_at": "2026-05-08T00:00:00+00:00",
            },
            "chunk": {
                "id": "chunk_dl",
                "index": 0,
                "strategy": "semantic",
                "size": 12,
                "text_preview": "refund within 30 days",
            },
            "document": {
                "id": "doc_dl",
                "source_path": "s3://bucket/policy.pdf",
                "source_type": "s3",
                "version": "v1",
                "content_hash": None,
                "source_url": "https://example.com/policy.pdf",
                "source_page": 7,
                "source_section": "3.2 Refunds",
            },
        }
        evidence = {
            "vector_id": "vec_dl",
            "collection_name": "support_kb",
            "trace_mode": "exact",
            "chunk_id": "chunk_dl",
            "chunk_index": 0,
            "chunk_text": "refund within 30 days",
            "source_document_id": "doc_dl",
            "source_path": "s3://bucket/policy.pdf",
            "source_url": "https://example.com/policy.pdf",
            "source_page": 7,
            "source_section": "3.2 Refunds",
            "retrieval_rank": 2,
            "retrieval_score": 0.81,
            "support_status": "unsupported",
            "support_reason": "Question asks about after 90 days, but evidence limits to within 30 days.",
            "support_details": {
                "method": "rule_plus_overlap",
                "answer_polarity": "yes",
                "overlap_ratio": 0.42,
                "overlap_terms": ["refund", "days"],
                "query_day_constraint": {"mode": "after", "days": 90},
                "evidence_day_constraint": {"mode": "within", "days": 30},
            },
        }
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "trace.html"
            generate_report(lineage, str(out), evidence=evidence)
            rendered = out.read_text(encoding="utf-8")

        self.assertIn("Why This Match", rendered)
        self.assertIn("Overlap Ratio", rendered)
        self.assertIn("0.42", rendered)
        self.assertIn("refund, days", rendered)
        self.assertIn("after 90 days", rendered)
        self.assertIn("within 30 days", rendered)
        self.assertIn("https://example.com/policy.pdf#page=7", rendered)
        self.assertIn("3.2 Refunds", rendered)

    def test_deep_link_rejects_javascript_scheme(self) -> None:
        from output.html import _deep_link

        for hostile in (
            "javascript:alert(1)",
            "JavaScript:alert(1)",
            "  javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "vbscript:msgbox(1)",
        ):
            rendered = _deep_link(hostile, None)
            self.assertNotIn("<a ", rendered, msg=f"unsafe scheme rendered as link: {hostile!r}")
            self.assertNotIn("href=", rendered, msg=f"unsafe scheme rendered as link: {hostile!r}")
            self.assertNotIn("<script", rendered.lower())

    def test_deep_link_renders_safe_schemes_as_anchor(self) -> None:
        from output.html import _deep_link

        rendered = _deep_link("https://example.com/policy.pdf", 7)
        self.assertIn('href="https://example.com/policy.pdf#page=7"', rendered)
        self.assertIn("<a", rendered)


if __name__ == "__main__":
    unittest.main()
