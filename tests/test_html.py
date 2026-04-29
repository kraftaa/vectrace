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


if __name__ == "__main__":
    unittest.main()
